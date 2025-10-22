#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import math
import os
import queue as queue_module
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
        target_fps: Target frame rate for idle frame generation (default: 30 fps)
                   - Ensures smooth video even during silence
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
        target_fps: int = 30,
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
        
        # Initialize ALL queues and state variables
        self._video_frame_queue = asyncio.Queue()  # For transferring frames from capture to playback
        self._audio_queue = None  # Will be initialized when audio task is created

        # Audio-to-video frame mapping for synchronization
        # We need 1:1 correspondence between audio and video frames
        # Video: 20fps = 50ms per frame
        # Audio: Need to accumulate and re-chunk to 50ms portions
        self._synced_audio_queue = asyncio.Queue()  # Queue of audio frames synchronized with video (50ms each)
        self._audio_accumulator = bytearray()  # Accumulates raw audio bytes for re-chunking
        self._audio_sample_rate = 16000  # Ditto uses 16kHz
        self._video_frame_duration_ms = 50  # 20fps = 50ms per frame
        self._samples_per_video_frame = int((self._audio_sample_rate * self._video_frame_duration_ms) / 1000)  # 800 samples

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
        
        # Frame reading/playback state
        self._frame_reader_running = False
        self._frame_reader_task = None
        self._video_playback_task = None
        self._frame_capture_queue = None  # Will be set in start()

        # Idle frame generation
        self._idle_frame_task = None
        self._last_frame = None  # Cache last generated frame for idle state
        self._is_speaking = False  # Track if currently generating speech frames
        self._last_speech_frame_time = 0  # Track when last speech frame was generated

        # Audio buffering for synchronized playback with video
        self._pending_audio_buffer = bytearray()  # Buffer of raw audio bytes waiting to be pushed
        self._pending_audio_sample_rate = 16000  # Sample rate of pending audio
        self._pending_audio_num_channels = 1  # Number of channels
        self._audio_push_lock = asyncio.Lock()  # Lock for audio pushing

        # Timestamp tracking for audio-video sync
        self._base_timestamp = None  # Base timestamp when TTS starts
        self._audio_samples_pushed = 0  # Total audio samples pushed for timestamp calculation
        self._video_frames_pushed = 0  # Total video frames pushed for timestamp calculation

        # Buffering configuration
        self._audio_buffer_duration_ms = 200  # Buffer 200ms of audio before starting to push
        self._audio_buffer_threshold_bytes = int((16000 * self._audio_buffer_duration_ms / 1000) * 2)  # 200ms at 16kHz = 3200 samples = 6400 bytes

        # SDK access lock to prevent concurrent run_chunk calls
        self._sdk_lock = asyncio.Lock()

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

            # Start idle frame generation task
            self._idle_frame_task = self.create_task(self._idle_frame_generator())
            logger.info(f"{self}: Started idle frame generator at {self._target_fps} fps")

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

        if self._idle_frame_task:
            self._idle_frame_task.cancel()
            try:
                await self._idle_frame_task
            except asyncio.CancelledError:
                pass
            self._idle_frame_task = None

        self._audio_buffer = bytearray()
        self._audio_accumulator.clear()
        self._pending_audio_buffer.clear()
        # Clear synced audio queue
        while not self._synced_audio_queue.empty():
            try:
                self._synced_audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
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

                        await self._process_single_chunk(chunk_float)

                self._audio_queue.task_done()

            except asyncio.TimeoutError:
                # No audio received for VAD_STOP_SECS - finalize
                if self._event_id is not None:
                    logger.info(f"{self}: Timeout detected, finalizing utterance {self._event_id}")
                    await self._finalize_audio()
                    self._event_id = None
                    self._audio_buffer.clear()

                    # Simple approach: Just wait a fixed time for Ditto to finish
                    # Ditto processes asynchronously in background threads, and timing varies
                    # based on utterance length and GPU load
                    logger.info(f"{self}: Waiting 3 seconds for Ditto to complete video generation...")
                    await asyncio.sleep(3.0)

                    logger.info(f"{self}: ===== SPEECH FINALIZED - Setting _is_speaking = False =====")
                    self._is_speaking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames from the pipeline."""
        # Log ALL frames to debug what's flowing through
        frame_type = type(frame).__name__
        logger.warning(f"{self}: 🔍 process_frame received: {frame_type} (direction={direction})")

        if isinstance(frame, (TTSStartedFrame, TTSStoppedFrame, TTSAudioRawFrame)):
            logger.error(f"{self}: ❗❗❗ TTS FRAME DETECTED: {frame_type} ❗❗❗")

        # Handle our frames BEFORE calling super() - this ensures we intercept them!
        if isinstance(frame, StartFrame):
            await self.start(frame)

        elif isinstance(frame, TTSStartedFrame):
            logger.warning(f"{self}: ===== TTS STARTED - Setting _is_speaking = True =====")
            self._is_interrupting = False
            self._is_speaking = True  # Set speaking flag immediately when TTS starts
            self._audio_buffer.clear()
            self._audio_accumulator.clear()

            # Clear pending audio buffer for synchronized playback
            async with self._audio_push_lock:
                self._pending_audio_buffer.clear()

            # Reset timestamp tracking for new utterance
            self._base_timestamp = None
            self._audio_samples_pushed = 0
            self._video_frames_pushed = 0

            # Clear synced audio queue
            while not self._synced_audio_queue.empty():
                try:
                    self._synced_audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

            # Clear any queued idle frames to prevent them from playing during speech
            cleared_count = 0
            while not self._video_frame_queue.empty():
                try:
                    self._video_frame_queue.get_nowait()
                    cleared_count += 1
                except asyncio.QueueEmpty:
                    break
            if cleared_count > 0:
                logger.info(f"{self}: Cleared {cleared_count} queued idle frames from video queue")

            # Note: event_id will be set when first audio frame arrives

        elif isinstance(frame, TTSAudioRawFrame):
            # Defensive: Set speaking flag if not already set (in case TTSStartedFrame was missed)
            if not self._is_speaking:
                logger.warning(f"{self}: ===== TTS AUDIO RECEIVED - Setting _is_speaking = True (fallback) =====")
                self._is_speaking = True
                self._is_interrupting = False
                self._audio_accumulator.clear()

                # Clear pending audio buffer
                async with self._audio_push_lock:
                    self._pending_audio_buffer.clear()

                # Reset timestamp tracking
                self._base_timestamp = None
                self._audio_samples_pushed = 0
                self._video_frames_pushed = 0

                # Clear synced audio queue
                while not self._synced_audio_queue.empty():
                    try:
                        self._synced_audio_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

                # Clear idle frames
                cleared_count = 0
                while not self._video_frame_queue.empty():
                    try:
                        self._video_frame_queue.get_nowait()
                        cleared_count += 1
                    except asyncio.QueueEmpty:
                        break
                if cleared_count > 0:
                    logger.info(f"{self}: Cleared {cleared_count} queued idle frames (fallback)")

            if not self._initialized or self._is_interrupting:
                pass
            else:
                # Queue audio for processing by Ditto (generates video)
                await self._audio_queue.put(frame)

                # Push audio immediately with timestamp
                # WebRTC will use timestamps to synchronize with video
                # Set base timestamp on first audio frame
                if self._base_timestamp is None:
                    self._base_timestamp = asyncio.get_event_loop().time()
                    logger.info(f"{self}: Base timestamp set: {self._base_timestamp}")

                # Calculate timestamp for this audio chunk
                audio_timestamp = self._base_timestamp + (self._audio_samples_pushed / frame.sample_rate)

                # Add timestamp metadata if the frame supports it
                if hasattr(frame, 'pts'):
                    frame.pts = audio_timestamp

                # Update samples counter
                num_samples = len(frame.audio) // 2  # 16-bit = 2 bytes per sample
                self._audio_samples_pushed += num_samples

                logger.info(f"{self}: Pushing TTS audio ({len(frame.audio)} bytes, {num_samples} samples, timestamp: {audio_timestamp:.3f}s)")
                await self.push_frame(frame, direction)
                return  # Don't push again at the end

        elif isinstance(frame, TTSStoppedFrame):
            # The timeout in _audio_task_handler will handle finalization
            # Note: We don't set _is_speaking = False here because there may still be
            # audio in the buffer being processed. The timeout handler will set it to False.
            logger.debug(f"{self}: TTS stopped, waiting for audio buffer to drain")

        elif isinstance(frame, (InterruptionFrame, UserStartedSpeakingFrame)):
            logger.info(f"{self}: Interruption detected, clearing state")
            await self._handle_interruption()

        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self.stop(frame)

        # Call parent class to handle system frames and observers
        await super().process_frame(frame, direction)

        # Push frame downstream (TTSAudioRawFrame already handled above with early return)
        await self.push_frame(frame, direction)

    async def _handle_interruption(self):
        """Handle user interruption by stopping audio processing and restarting."""
        self._is_interrupting = True
        self._audio_buffer.clear()
        self._audio_accumulator.clear()

        # Clear pending audio buffer
        async with self._audio_push_lock:
            self._pending_audio_buffer.clear()

        # Reset timestamp tracking
        self._base_timestamp = None
        self._audio_samples_pushed = 0
        self._video_frames_pushed = 0

        # Clear synced audio queue
        while not self._synced_audio_queue.empty():
            try:
                self._synced_audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        self._event_id = None

        # Reset speaking flag so idle frames can resume
        logger.info(f"{self}: ===== INTERRUPTION - Setting _is_speaking = False =====")
        self._is_speaking = False

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

    async def _idle_frame_generator(self):
        """Generate idle/neutral frames at target FPS when not speaking.

        Continuously feeds silent audio to Ditto SDK to generate neutral frames
        during idle periods, ensuring smooth video playback even during silence.
        """
        logger.info(f"{self}: Idle frame generator started (target: {self._target_fps} fps)")

        frame_count = 0
        chunk_count = 0
        was_speaking = False  # Track state changes for logging

        try:
            # Feed chunks more frequently for smoother, higher FPS idle video
            # We'll feed smaller "virtual" chunks by sleeping less between SDK calls
            # Target: ~30fps output, so feed every ~0.1 seconds
            chunk_interval = 0.15  # Feed chunks every 150ms for smoother playback

            while True:
                start_time = asyncio.get_event_loop().time()

                # Log current state every 100 iterations for debugging
                if chunk_count % 100 == 0:
                    logger.debug(f"{self}: Idle generator loop - _is_speaking={self._is_speaking}, chunk_count={chunk_count}")

                # Log state for debugging (every iteration during critical transitions)
                if chunk_count < 10 or chunk_count % 10 == 0:
                    logger.info(f"{self}: Idle loop check - _is_speaking={self._is_speaking}, will_generate={not self._is_speaking and self._sdk is not None and self._initialized}")

                # Only generate idle frames when not speaking
                # This prevents interference with speech-driven frame generation
                if not self._is_speaking and self._sdk is not None and self._initialized:
                    # Log when transitioning from speaking to idle
                    if was_speaking:
                        logger.info(f"{self}: ===== Transitioning to IDLE mode - starting silent audio generation =====")
                        was_speaking = False

                    # Don't generate more idle frames if queue is getting too full
                    # This prevents idle frames from building up and playing during speech
                    queue_size = self._video_frame_queue.qsize()
                    if queue_size > 20:  # Don't let queue get too full
                        logger.debug(f"{self}: Video queue has {queue_size} frames, skipping idle generation")
                        # Sleep to maintain timing
                        elapsed = asyncio.get_event_loop().time() - start_time
                        sleep_time = max(0, chunk_interval - elapsed)
                        await asyncio.sleep(sleep_time)
                        continue

                    # Generate neutral frame by feeding silent audio to Ditto
                    # This ensures continuous frame generation during silence
                    silent_audio = np.zeros(6480, dtype=np.float32)

                    # Add history padding for proper SDK processing
                    if len(self._audio_history) >= 1920:
                        padded_audio = np.concatenate([self._audio_history[-1920:], silent_audio])
                    else:
                        padding = np.zeros(1920, dtype=np.float32)
                        padded_audio = np.concatenate([padding, silent_audio])

                    try:
                        # Run silent chunk through Ditto to generate neutral frames
                        # Use lock to prevent interference with speech audio processing
                        logger.info(f"{self}: IDLE generator about to acquire SDK lock (chunk {chunk_count})")
                        async with self._sdk_lock:
                            logger.info(f"{self}: ⚠️  IDLE generator acquired SDK lock, feeding SILENT audio (chunk {chunk_count})")
                            await asyncio.get_event_loop().run_in_executor(
                                None,
                                self._sdk.run_chunk,
                                padded_audio,
                                self._chunk_size
                            )
                            logger.info(f"{self}: IDLE generator released SDK lock (chunk {chunk_count})")

                        # Update history for next chunk
                        self._audio_history = silent_audio

                        chunk_count += 1

                        if chunk_count == 1:
                            logger.info(f"{self}: Started generating idle frames from silent audio")
                        elif chunk_count % 50 == 0:
                            logger.debug(f"{self}: Generated {chunk_count} idle audio chunks")

                    except Exception as e:
                        logger.error(f"{self}: Error generating idle frame: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    # Currently speaking - track state for transition logging
                    if not was_speaking:
                        logger.info(f"{self}: ===== IDLE mode paused - bot is speaking (_is_speaking = {self._is_speaking}) =====")
                        was_speaking = True

                # Sleep for the chunk interval to maintain smooth frame generation
                elapsed = asyncio.get_event_loop().time() - start_time
                sleep_time = max(0, chunk_interval - elapsed)
                await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            logger.info(f"{self}: Idle frame generator stopped (generated {chunk_count} idle chunks)")
        except Exception as e:
            logger.error(f"{self}: Error in idle frame generator: {e}")
            import traceback
            traceback.print_exc()

    async def _process_single_chunk(self, audio_chunk: np.ndarray):
        """Process a single 6480-sample chunk immediately."""
        # Add history padding (1920 samples from previous chunk)
        if len(self._audio_history) >= 1920:
            padded_audio = np.concatenate([self._audio_history[-1920:], audio_chunk])
        else:
            # First chunk - pad with zeros
            padding = np.zeros(1920, dtype=np.float32)
            padded_audio = np.concatenate([padding, audio_chunk])

        logger.debug(f"{self}: SPEECH processor waiting for SDK lock...")

        # Run SDK processing in executor (non-blocking)
        # Use lock to prevent interference with idle frame generator
        async with self._sdk_lock:
            logger.debug(f"{self}: SPEECH processor acquired SDK lock, feeding speech audio")
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._sdk.run_chunk,
                padded_audio,
                self._chunk_size
            )

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
        """Read frames from SDK's capture queue and add to video playback queue."""
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
                    
                    frames_read += 1

                    if frames_read == 1:
                        logger.info(f"{self}: 🎉 FIRST FRAME CAPTURED!")
                    elif frames_read % 10 == 0:
                        logger.info(f"{self}: Read {frames_read} frames from SDK")

                    # Track when speech frames are being generated
                    if self._is_speaking:
                        self._last_speech_frame_time = asyncio.get_event_loop().time()
                        logger.debug(f"{self}: Speech frame generated at {self._last_speech_frame_time}")

                    # Cache the last frame for idle state
                    self._last_frame = frame.copy()

                    # Add to video playback queue
                    await self._video_frame_queue.put(frame)
                    logger.debug(f"{self}: Added frame to video queue (qsize={self._video_frame_queue.qsize()})")
                    
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
        """Consume video frames from queue and push to Daily with timestamps.

        Video frames are pushed with timestamps for WebRTC synchronization.
        Audio is pushed separately (in TTSAudioRawFrame processing) with matching timestamps.
        WebRTC transport uses timestamps to maintain perfect audio-video sync.
        """
        logger.info(f"{self}: Video playback task started")

        frames_pushed = 0

        try:
            while True:
                try:
                    # Get frame from the queue (with timeout to check for cancellation)
                    frame = await asyncio.wait_for(
                        self._video_frame_queue.get(),
                        timeout=0.1
                    )

                    if frame is None:
                        logger.info(f"{self}: Received None, stopping video playback")
                        break

                    frames_pushed += 1

                    if frames_pushed == 1:
                        logger.info(f"{self}: 🎥 FIRST FRAME PUSHED TO DAILY!")
                    elif frames_pushed % 10 == 0:
                        logger.info(f"{self}: Pushed {frames_pushed} video frames to Daily")

                    # Calculate timestamp for this video frame
                    # timestamp = base + (frames_pushed / fps)
                    # Note: base_timestamp set when first audio arrives
                    if self._base_timestamp is not None:
                        video_timestamp = self._base_timestamp + (self._video_frames_pushed / 20.0)  # 20fps
                    else:
                        # Fallback if no audio yet
                        video_timestamp = asyncio.get_event_loop().time()

                    # Create OutputImageRawFrame for Daily with timestamp
                    output_frame = OutputImageRawFrame(
                        image=frame,
                        size=(frame.shape[1], frame.shape[0]),  # (width, height)
                        format="RGB"
                    )

                    # Add timestamp metadata if the frame supports it
                    if hasattr(output_frame, 'pts'):
                        output_frame.pts = video_timestamp

                    # Push video frame to pipeline
                    await self.push_frame(output_frame)

                    # Update video frames counter
                    self._video_frames_pushed += 1

                    logger.debug(f"{self}: Pushed video frame {frames_pushed} (timestamp: {video_timestamp:.3f}s)")

                except asyncio.TimeoutError:
                    # No frame available, continue waiting
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