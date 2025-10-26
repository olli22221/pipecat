#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import math
import os
import queue as queue_module
import time
from typing import Optional

import numpy as np
from loguru import logger

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    OutputImageRawFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

try:
    import librosa
    import cv2
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Ditto, you need to `pip install opencv-python librosa`. "
    )
    raise Exception(f"Missing module: {e}")


class DittoTalkingHeadService(FrameProcessor):
    """
    Ditto Talking Head Service for Pipecat - Real-time Streaming Mode

    This service uses the Ditto talking head model to generate realistic
    talking head videos from an input image and audio using real-time streaming.

    Requirements:
    - CUDA-capable GPU (A100 or similar recommended)
    - Ditto installed from https://github.com/antgroup/ditto-talkinghead
    - opencv-python, librosa for processing

    Args:
        ditto_path: Path to Ditto installation directory
        data_root: Path to Ditto model checkpoints
        cfg_pkl: Path to Ditto config file (use online config like v0.4_hubert_cfg_pytorch_online.pkl)
        source_image_path: Path to the source image (the avatar face to animate)
        chunk_size: Audio chunk size tuple (history, current, future) frames
                   - Default (3, 5, 2) = ~200ms latency, 50% overlap
        save_frames_dir: Optional directory path to save generated frames as PNG files
        target_fps: Target frame rate for PTS calculation and rate limiting (default: 50)
                   - Note: Actual generation FPS depends on SDK speed
        compress_frames: Enable JPEG compression of frames before transport (default: True)
                        - Reduces ~8MB raw RGB to ~200KB JPEG per frame (~40x compression)
        jpeg_quality: JPEG quality level 1-100 (default: 75)
                     - Higher = better quality but larger size
                     - 75 is good balance of quality vs bandwidth
        **kwargs: Additional arguments passed to FrameProcessor
    """

    def __init__(
        self,
        *,
        ditto_path: str,
        data_root: str,
        cfg_pkl: str,
        source_image_path: str,
        chunk_size: tuple = (3, 5, 2),
        save_frames_dir: Optional[str] = None,
        target_fps: int = 50,
        compress_frames: bool = True,
        jpeg_quality: int = 75,
        **kwargs
    ):
        super().__init__(**kwargs)

        self._ditto_path = ditto_path
        self._data_root = data_root
        self._cfg_pkl = cfg_pkl
        self._source_image_path = source_image_path
        self._chunk_size = chunk_size
        self._save_frames_dir = save_frames_dir
        self._target_fps = target_fps
        self._compress_frames = compress_frames
        self._jpeg_quality = jpeg_quality
        
        # Initialize ALL queues and state variables
        self._video_frame_queue = asyncio.Queue()  # For transferring frames from capture to playback
        self._audio_queue = None  # Will be initialized when audio task is created
        self._current_audio_frames = []  # Audio frames for the current chunk being processed
        self._frames_in_current_chunk = 0  # Track position within chunk (chunk_size[1] frames per chunk)

        # SDK and initialization state
        self._sdk = None
        self._initialized = False

        # Audio processing state
        self._audio_buffer = bytearray()
        self._audio_history = np.array([], dtype=np.float32)
        self._audio_task = None  # Audio processing task
        self._event_id = None  # Track current utterance
        self._is_interrupting = False  # Interrupt flag
        self._resampler = create_stream_resampler()  # Audio resampler

        # Timing and synchronization
        self._video_frame_count = 0  # Total video frames generated
        self._audio_sample_rate = 16000  # Sample rate we use for audio processing
        self._first_audio_pts = None  # PTS of first audio frame (nanoseconds)
        
        # Frame reading/playback state
        self._frame_reader_running = False
        self._frame_reader_task = None
        self._video_playback_task = None
        self._frame_capture_queue = None  # Will be set in start()
        
        # Create save directory if specified
        if self._save_frames_dir:
            os.makedirs(self._save_frames_dir, exist_ok=True)
            logger.info(f"Frames will be saved to: {self._save_frames_dir}")
            
        # Validate paths
        if not os.path.exists(source_image_path):
            raise ValueError(f"Source image not found: {source_image_path}")
        if not os.path.exists(ditto_path):
            raise ValueError(f"Ditto path not found: {ditto_path}")
        if not os.path.exists(data_root):
            raise ValueError(f"Data root not found: {data_root}")
        if not os.path.exists(cfg_pkl):
            raise ValueError(f"Config file not found: {cfg_pkl}")

    async def start(self, frame: StartFrame):
        """Initialize the Ditto SDK and start background tasks for real-time streaming."""
        if self._initialized:
            return

        logger.info(f"{self}: Initializing Ditto Talking Head service (ONLINE MODE)")
        logger.info(f"{self}: Source image (avatar): {self._source_image_path}")
        logger.info(f"{self}: Data root: {self._data_root}")
        logger.info(f"{self}: Config: {self._cfg_pkl}")

        try:
            # Import Ditto's StreamSDK
            import sys
            sys.path.insert(0, self._ditto_path)
            from stream_pipeline_online import StreamSDK

            # Initialize SDK with online_mode=True
            logger.info(f"{self}: Initializing Ditto StreamSDK in online mode...")
            self._sdk = StreamSDK(self._cfg_pkl, self._data_root, online_mode=True)

            # Force online_mode
            self._sdk.online_mode = True
            if hasattr(self._sdk, 'audio2motion'):
                self._sdk.audio2motion.online_mode = True

            logger.info(f"{self}: Final SDK online_mode: {self._sdk.online_mode}")

            # Setup SDK paths
            import tempfile
            temp_output = os.path.join(tempfile.gettempdir(), f"ditto_output_{id(self)}.mp4")

            # Call setup with overlap_v2 to reduce accumulation requirement
            logger.info(f"{self}: Calling SDK.setup()...")
            self._sdk.setup(
                source_path=self._source_image_path,
                output_path=temp_output,
                overlap_v2=79  # Reduce accumulation requirement for short utterances
            )
            logger.info(f"{self}: SDK.setup() completed")

            # Create a wrapper class that intercepts calls to the writer
            logger.info(f"{self}: Wrapping writer to capture frames...")

            frame_capture_queue = queue_module.Queue()
            frame_save_dir = self._save_frames_dir
            frame_count = [0]
            original_writer = self._sdk.writer

            class WriterWrapper:
                """Wrapper that intercepts all calls to the writer."""
                def __init__(self, original):
                    self.original = original
                    
                def __call__(self, frame_rgb, fmt="rgb"):
                    """Intercept and capture frames."""
                    if isinstance(frame_rgb, np.ndarray):
                        frame_capture_queue.put(frame_rgb.copy())
                        frame_count[0] += 1
                        
                        if frame_save_dir:
                            frame_path = os.path.join(frame_save_dir, f"frame_{frame_count[0]:06d}.png")
                            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(frame_path, frame_bgr)
                        
                        if frame_count[0] == 1:
                            logger.info(f"🎉 FIRST FRAME CAPTURED!")
                        elif frame_count[0] % 10 == 0:
                            logger.info(f"Captured {frame_count[0]} frames")
                    
                    # Call original writer
                    return self.original(frame_rgb, fmt)
                
                def close(self):
                    """Forward close to original."""
                    if hasattr(self.original, 'close'):
                        return self.original.close()

            # Replace the writer object entirely
            self._sdk.writer = WriterWrapper(original_writer)
            logger.info(f"{self}: Writer wrapped successfully")

            # Store the frame capture queue
            self._frame_capture_queue = frame_capture_queue

            # Start background tasks
            self._frame_reader_running = True
            self._frame_reader_task = self.create_task(self._read_frames_from_sdk())
            logger.info(f"{self}: Started frame reader task")

            self._video_playback_task = self.create_task(self._consume_and_push_video())
            logger.info(f"{self}: Started video playback task")

            # Start audio processing task
            await self._create_audio_task()

            self._initialized = True
            logger.info(f"{self}: ✅ Ditto service initialized successfully")

        except Exception as e:
            logger.error(f"{self}: ❌ Failed to initialize Ditto: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to initialize Ditto: {e}")

    async def stop(self, frame: EndFrame):
        """Clean up resources"""
        self._frame_reader_running = False

        # Cancel tasks
        await self._cancel_audio_task()

        if self._frame_reader_task:
            self._frame_reader_task.cancel()
            try:
                await self._frame_reader_task
            except asyncio.CancelledError:
                pass
            self._frame_reader_task = None

        if self._video_playback_task:
            self._video_playback_task.cancel()
            try:
                await self._video_playback_task
            except asyncio.CancelledError:
                pass
            self._video_playback_task = None

        self._audio_buffer = bytearray()
        self._sdk = None
        self._initialized = False

    async def cancel(self, frame: CancelFrame):
        """Cancel any ongoing operations"""
        self._is_interrupting = True
        await self._cancel_audio_task()
        self._audio_buffer = bytearray()

    async def _create_audio_task(self):
        """Create the audio processing task if it doesn't exist."""
        if not self._audio_task:
            self._audio_queue = asyncio.Queue()
            self._audio_task = self.create_task(self._audio_task_handler())
            logger.info(f"{self}: Audio processing task created")

    async def _cancel_audio_task(self):
        """Cancel the audio processing task if it exists."""
        if self._audio_task:
            self._audio_task.cancel()
            try:
                await self._audio_task
            except asyncio.CancelledError:
                pass
            self._audio_task = None
            logger.debug(f"{self}: Audio task cancelled")

    async def _audio_task_handler(self):
        """Handle processing audio frames from the queue.

        Continuously processes audio frames, accumulates them in a buffer,
        and sends chunks to Ditto SDK. Uses timeout to detect end of speech.
        """
        VAD_STOP_SECS = 0.5  # Time to wait before finalizing audio

        while True:
            try:
                # Wait for audio frames with timeout
                frame = await asyncio.wait_for(
                    self._audio_queue.get(),
                    timeout=VAD_STOP_SECS
                )

                if self._is_interrupting:
                    logger.debug(f"{self}: Interrupting, breaking audio task")
                    break

                if isinstance(frame, TTSAudioRawFrame):
                    # Starting new inference
                    if self._event_id is None:
                        self._event_id = str(frame.id)
                        logger.info(f"{self}: Starting new utterance {self._event_id}")

                    # Capture PTS from first audio frame
                    if self._first_audio_pts is None and frame.pts is not None:
                        self._first_audio_pts = frame.pts
                        logger.info(f"{self}: Captured first audio PTS: {frame.pts} ns")

                    # Collect this audio frame
                    self._current_audio_frames.append(frame)

                    # Resample audio to 16kHz
                    audio_resampled = await self._resampler.resample(
                        frame.audio, frame.sample_rate, 16000
                    )

                    # Resampler returns bytes, so extend buffer directly
                    self._audio_buffer.extend(audio_resampled)

                    # Process chunks as they accumulate
                    chunk_size_bytes = 6480 * 2  # 6480 samples * 2 bytes per sample
                    while len(self._audio_buffer) >= chunk_size_bytes:
                        chunk_bytes = bytes(self._audio_buffer[:chunk_size_bytes])
                        self._audio_buffer = self._audio_buffer[chunk_size_bytes:]

                        # Convert back to float32 for SDK
                        chunk_array = np.frombuffer(chunk_bytes, dtype=np.int16)
                        chunk_float = chunk_array.astype(np.float32) / 32768.0

                        # Pass collected audio frames with the chunk
                        audio_frames_for_chunk = self._current_audio_frames.copy()
                        logger.info(f"{self}: Passing {len(audio_frames_for_chunk)} audio frames to chunk processing")
                        self._current_audio_frames.clear()

                        await self._process_single_chunk(chunk_float, audio_frames_for_chunk)

                self._audio_queue.task_done()

            except asyncio.TimeoutError:
                # No audio received for VAD_STOP_SECS - finalize
                if self._event_id is not None:
                    logger.info(f"{self}: Timeout detected, finalizing utterance {self._event_id}")
                    await self._finalize_audio()
                    self._event_id = None
                    self._audio_buffer.clear()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames from the pipeline."""
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self.start(frame)

        elif isinstance(frame, TTSStartedFrame):
            logger.info(f"{self}: Starting new TTS utterance")
            self._is_interrupting = False
            self._audio_buffer.clear()
            self._current_audio_frames.clear()
            # Reset timing tracking
            self._video_frame_count = 0
            self._first_audio_pts = None
            self._frames_in_current_chunk = 0
            # Note: event_id will be set when first audio frame arrives

        elif isinstance(frame, TTSAudioRawFrame):
            if not self._initialized or self._is_interrupting:
                pass
            else:
                # Queue audio for processing
                await self._audio_queue.put(frame)
                # Don't push audio downstream yet - we'll push it with video
                return

        elif isinstance(frame, TTSStoppedFrame):
            # The timeout in _audio_task_handler will handle finalization
            logger.debug(f"{self}: TTS stopped")

        elif isinstance(frame, (InterruptionFrame, UserStartedSpeakingFrame)):
            logger.info(f"{self}: Interruption detected, clearing state")
            await self._handle_interruption()

        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self.stop(frame)

        await self.push_frame(frame, direction)

    async def _handle_interruption(self):
        """Handle user interruption by stopping audio processing and restarting."""
        self._is_interrupting = True
        self._audio_buffer.clear()
        self._current_audio_frames.clear()
        self._event_id = None

        # Reset timing
        self._video_frame_count = 0
        self._first_audio_pts = None
        self._frames_in_current_chunk = 0

        # Cancel and restart audio task
        await self._cancel_audio_task()
        self._is_interrupting = False
        await self._create_audio_task()

        # Clear video queue
        while not self._video_frame_queue.empty():
            try:
                self._video_frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def _process_single_chunk(self, audio_chunk: np.ndarray, audio_frames: list = None):
        """Process a single 6480-sample chunk immediately."""
        # Add history padding (1920 samples from previous chunk)
        if len(self._audio_history) >= 1920:
            padded_audio = np.concatenate([self._audio_history[-1920:], audio_chunk])
        else:
            # First chunk - pad with zeros
            padding = np.zeros(1920, dtype=np.float32)
            padded_audio = np.concatenate([padding, audio_chunk])

        logger.info(f"{self}: Processing chunk - received {len(audio_frames) if audio_frames else 0} audio frames")
        logger.debug(f"{self}: Running SDK chunk (total: {len(padded_audio)} samples)")

        # Store audio frames BEFORE running SDK so they're ready when frames are generated
        # These will only be attached to the FIRST frame of this chunk
        # Note: We DON'T reset _frames_in_current_chunk here because previous chunk's frames
        # might still be in the queue. The frame reader will reset it when it detects a new chunk.
        if audio_frames:
            self._current_audio_frames.extend(audio_frames)
            logger.info(f"{self}: Stored {len(audio_frames)} audio frames, _current_audio_frames now has {len(self._current_audio_frames)} frames")

        # Run SDK processing in executor (non-blocking)
        # This will generate video frames that get captured by our wrapper
        start_time = time.time()
        await asyncio.get_event_loop().run_in_executor(
            None,
            self._sdk.run_chunk,
            padded_audio,
            self._chunk_size
        )
        end_time = time.time()

        # Calculate and log generation speed
        generation_time = end_time - start_time
        frames_generated = self._chunk_size[1]  # Current frames (typically 5)
        fps = frames_generated / generation_time if generation_time > 0 else 0
        logger.info(f"{self}: Generated {frames_generated} frames in {generation_time:.3f}s ({fps:.1f} fps)")

        # Update history for next chunk
        self._audio_history = audio_chunk

    async def _finalize_audio(self):
        """Process any remaining audio in the buffer after TTS stops."""
        if len(self._audio_buffer) == 0:
            return

        num_samples = len(self._audio_buffer) // 2  # Convert bytes to samples
        logger.info(f"{self}: Finalizing audio processing, buffer has {num_samples} samples ({len(self._audio_buffer)} bytes)")

        # Add future context padding (1280 samples = 2560 bytes)
        padding_samples = 1280  # 2 frames * 640
        logger.info(f"{self}: Adding {padding_samples} samples of future padding")

        # Get last sample from buffer for padding
        if len(self._audio_buffer) >= 2:
            # Extract last int16 sample
            last_int16 = np.frombuffer(self._audio_buffer[-2:], dtype=np.int16)[0]
            # Create padding
            padding_array = np.full(padding_samples, last_int16, dtype=np.int16)
            self._audio_buffer.extend(padding_array.tobytes())

        num_samples_padded = len(self._audio_buffer) // 2
        logger.info(f"{self}: After padding, buffer has {num_samples_padded} samples")

        # Process remaining chunks
        chunk_size_bytes = 6480 * 2  # 6480 samples * 2 bytes per sample
        iteration = 0

        while len(self._audio_buffer) >= chunk_size_bytes:
            iteration += 1
            logger.debug(f"{self}: Iteration {iteration}: buffer size = {len(self._audio_buffer) // 2} samples")

            # Extract chunk
            chunk_bytes = bytes(self._audio_buffer[:chunk_size_bytes])
            self._audio_buffer = self._audio_buffer[chunk_size_bytes:]

            # Convert to float32 for SDK
            chunk_array = np.frombuffer(chunk_bytes, dtype=np.int16)
            chunk_float = chunk_array.astype(np.float32) / 32768.0

            await self._process_single_chunk(chunk_float)

        # Process final partial chunk if any remains
        if len(self._audio_buffer) > 0:
            remaining_samples = len(self._audio_buffer) // 2
            logger.info(f"{self}: Processing final partial chunk ({remaining_samples} samples)")

            # Convert what we have
            chunk_array = np.frombuffer(bytes(self._audio_buffer), dtype=np.int16)
            chunk_float = chunk_array.astype(np.float32) / 32768.0

            # Pad to minimum size if needed
            if len(chunk_float) < 6480:
                padding_needed = 6480 - len(chunk_float)
                final_padding = np.full(padding_needed, chunk_float[-1], dtype=np.float32)
                chunk_float = np.concatenate([chunk_float, final_padding])

            await self._process_single_chunk(chunk_float)

        # Clear buffer
        self._audio_buffer.clear()
        logger.info(f"{self}: Finalization complete")

    async def _read_frames_from_sdk(self):
        """Read frames from SDK's capture queue and add to video playback queue with PTS."""
        logger.info(f"{self}: Frame reader task started - reading from capture queue")

        frames_read = 0
        try:
            while self._frame_reader_running:
                try:
                    # Read from capture queue (populated by our wrapper)
                    frame = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self._frame_capture_queue.get(timeout=0.1)
                    )

                    if frame is None:
                        logger.info(f"{self}: Received None from capture queue, stopping reader")
                        break

                    # Calculate frame timing based on frame count and target FPS
                    # Each frame represents 1/target_fps seconds (50fps = 20ms, 25fps = 40ms)
                    frame_time_offset_s = self._video_frame_count / self._target_fps

                    # Only attach audio frames to the FIRST frame of each chunk
                    # Each chunk generates chunk_size[1] frames (typically 5)
                    # Check if we have audio frames ready - if so, this is the start of a new chunk
                    audio_frames_to_push = []
                    if len(self._current_audio_frames) > 0:
                        # New chunk detected - we have audio frames ready
                        # Reset counter and attach audio to this first frame
                        self._frames_in_current_chunk = 0
                        logger.info(f"{self}: New chunk detected! _current_audio_frames has {len(self._current_audio_frames)} frames, resetting counter")
                        audio_frames_to_push = self._current_audio_frames.copy()
                        self._current_audio_frames.clear()
                        logger.info(f"{self}: Attached {len(audio_frames_to_push)} audio frames to first video frame of chunk")

                    # Increment frame counter within chunk
                    self._frames_in_current_chunk += 1

                    frames_read += 1
                    self._video_frame_count += 1

                    if frames_read == 1:
                        logger.info(f"{self}: 🎉 FIRST FRAME CAPTURED!")
                    elif frames_read % 10 == 0:
                        logger.info(f"{self}: Read {frames_read} frames from SDK")

                    # Add frame to video playback queue with timing info and audio frames
                    await self._video_frame_queue.put((frame, frame_time_offset_s, audio_frames_to_push))
                    logger.debug(f"{self}: Added frame to video queue with {len(audio_frames_to_push)} audio frames (qsize={self._video_frame_queue.qsize()})")

                except queue_module.Empty:
                    await asyncio.sleep(0.01)
                    continue
                except Exception as e:
                    logger.error(f"{self}: Error reading frame: {e}")
                    import traceback
                    traceback.print_exc()
                    break

        except asyncio.CancelledError:
            logger.info(f"{self}: Frame reader task cancelled (read {frames_read} frames)")
        except Exception as e:
            logger.error(f"{self}: Frame reader error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            logger.info(f"{self}: Frame reader finished (total frames: {frames_read})")

    async def _consume_and_push_video(self):
        """Consume video frames from queue and push them with rate limiting."""
        logger.info(f"{self}: Video playback task started")

        frames_pushed = 0
        chunk_buffer = []  # Buffer for accumulating chunk frames
        chunk_audio_frames = None  # Audio frames for the current chunk
        last_push_time = None

        try:
            while True:
                try:
                    # Get frame from the queue (with timeout to check for cancellation)
                    frame_data = await asyncio.wait_for(
                        self._video_frame_queue.get(),
                        timeout=0.1
                    )

                    if frame_data is None:
                        # Push any remaining buffered frames before stopping
                        if chunk_buffer:
                            await self._push_chunk_bundle(chunk_buffer, chunk_audio_frames, frames_pushed)
                            frames_pushed += len(chunk_buffer)
                        logger.info(f"{self}: Received None, stopping video playback")
                        break

                    frame, frame_time_offset_s, audio_frames = frame_data

                    # Compress frame if enabled
                    if self._compress_frames:
                        # Convert RGB to BGR for OpenCV
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                        # Encode as JPEG
                        success, jpeg_data = cv2.imencode(
                            '.jpg',
                            frame_bgr,
                            [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
                        )

                        if not success:
                            logger.error(f"{self}: Failed to encode frame as JPEG")
                            continue

                        # Use compressed JPEG data
                        frame_data_to_send = jpeg_data.tobytes()

                        # Create OutputImageRawFrame with JPEG data
                        output_frame = OutputImageRawFrame(
                            image=frame_data_to_send,
                            size=(frame.shape[1], frame.shape[0]),  # (width, height)
                            format="JPEG"
                        )

                        if frames_pushed == 0:
                            # Log compression stats for first frame
                            raw_size = frame.nbytes
                            compressed_size = len(frame_data_to_send)
                            ratio = raw_size / compressed_size
                            logger.info(f"{self}: 🗜️ JPEG compression: {raw_size:,} bytes → {compressed_size:,} bytes ({ratio:.1f}x reduction)")
                    else:
                        # No compression - use raw RGB
                        output_frame = OutputImageRawFrame(
                            image=frame,
                            size=(frame.shape[1], frame.shape[0]),  # (width, height)
                            format="RGB"
                        )

                    # Set PTS if we have audio timing
                    if self._first_audio_pts is not None:
                        frame_pts_ns = self._first_audio_pts + int(frame_time_offset_s * 1_000_000_000)
                        output_frame.pts = frame_pts_ns

                    # If this frame has audio, it's the first frame of a new chunk
                    if audio_frames and len(audio_frames) > 0:
                        # Push previous chunk if any
                        if chunk_buffer:
                            # Rate limiting: wait to match target FPS
                            if last_push_time is not None:
                                # Calculate how long the chunk should take to play
                                chunk_duration_s = len(chunk_buffer) / self._target_fps
                                elapsed = time.time() - last_push_time
                                wait_time = chunk_duration_s - elapsed

                                if wait_time > 0:
                                    logger.debug(f"{self}: Rate limiting - waiting {wait_time:.3f}s before pushing chunk")
                                    await asyncio.sleep(wait_time)

                            await self._push_chunk_bundle(chunk_buffer, chunk_audio_frames, frames_pushed)
                            frames_pushed += len(chunk_buffer)
                            last_push_time = time.time()

                        # Start new chunk
                        chunk_buffer = [output_frame]
                        chunk_audio_frames = audio_frames
                    else:
                        # Add to current chunk
                        chunk_buffer.append(output_frame)

                    # If chunk is complete (chunk_size[1] frames, typically 5)
                    if len(chunk_buffer) >= self._chunk_size[1]:
                        # Rate limiting: wait to match target FPS
                        if last_push_time is not None:
                            chunk_duration_s = len(chunk_buffer) / self._target_fps
                            elapsed = time.time() - last_push_time
                            wait_time = chunk_duration_s - elapsed

                            if wait_time > 0:
                                logger.debug(f"{self}: Rate limiting - waiting {wait_time:.3f}s before pushing chunk")
                                await asyncio.sleep(wait_time)

                        await self._push_chunk_bundle(chunk_buffer, chunk_audio_frames, frames_pushed)
                        frames_pushed += len(chunk_buffer)
                        last_push_time = time.time()

                        # Reset chunk buffer
                        chunk_buffer = []
                        chunk_audio_frames = None

                except asyncio.TimeoutError:
                    # No frame available, push any buffered frames
                    if chunk_buffer:
                        await self._push_chunk_bundle(chunk_buffer, chunk_audio_frames, frames_pushed)
                        frames_pushed += len(chunk_buffer)
                        last_push_time = time.time()
                        chunk_buffer = []
                        chunk_audio_frames = None
                    continue
                except Exception as e:
                    logger.error(f"{self}: Error pushing video frame: {e}")
                    import traceback
                    traceback.print_exc()
                    break

        except asyncio.CancelledError:
            logger.info(f"{self}: Video playback task cancelled")
        except Exception as e:
            logger.error(f"{self}: Error in video playback: {e}")
            import traceback
            traceback.print_exc()
        finally:
            logger.info(f"{self}: Video playback finished (pushed {frames_pushed} frames)")

    async def _push_chunk_bundle(self, video_frames: list, audio_frames: list, total_pushed: int):
        """Push a bundle of video frames with their corresponding audio frames."""
        if not video_frames:
            return

        frames_to_push = []

        # Add audio frames first (only attached to first frame of chunk)
        if audio_frames:
            for audio_frame in audio_frames:
                frames_to_push.append(audio_frame)

        # Add all video frames
        for video_frame in video_frames:
            frames_to_push.append(video_frame)

        # Push entire bundle at once
        if len(frames_to_push) > 0:
            await asyncio.gather(*[self.push_frame(frame) for frame in frames_to_push])

            if total_pushed == 0:
                logger.info(f"{self}: 🎥 FIRST CHUNK PUSHED! ({len(video_frames)} video frames + {len(audio_frames) if audio_frames else 0} audio frames)")
            elif total_pushed % 10 == 0:
                logger.info(f"{self}: Pushed {total_pushed} frames total")

            logger.debug(f"{self}: Pushed chunk bundle: {len(audio_frames) if audio_frames else 0} audio + {len(video_frames)} video = {len(frames_to_push)} total frames")