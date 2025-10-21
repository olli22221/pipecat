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
    Ditto Talking Head Service for Pipecat

    This service uses the Ditto talking head model to generate realistic
    talking head videos from an input image and audio using real-time streaming.

    Streaming Implementation Details:
    --------------------------------
    The online implementation uses overlapping audio segments for real-time inference:

    1. **Overlapping Segments**: Audio is split into overlapping windows (e.g., 1-2s windows)
       - Window size = sum(chunk_size) * 640 samples (e.g., 6400 for (3,5,2))
       - Stride = chunk_size[1] * 640 samples (e.g., 3200 for (3,5,2))
       - This creates 50% overlap between consecutive chunks

    2. **State Cache Management**: SDK maintains hidden state caches (transformer KV caches)
       across segments to ensure temporal consistency and smooth motion

    3. **Overlap-Add Blending**: SDK blends output segments to avoid boundary artifacts,
       using overlap ratio that matches the training configuration

    4. **Incremental Processing**: Each stage (audio→motion→rendering) runs incrementally,
       reusing previous context for efficiency

    Critical Considerations:
    -----------------------
    ⚠️ Segment timing: chunk_size must match model config or lip-sync will drift
    ⚠️ Cache management: SDK handles state internally; interruptions reset state
    ⚠️ Overlap blending: Built into SDK, matches training overlap ratio (typically 50%)
    ⚠️ Audio buffering: Ensure no gaps or misalignment between chunks
    ⚠️ Latency-quality tradeoff: Smaller chunks = lower latency but less stable motion

    Requirements:
    - CUDA-capable GPU (A100 or similar recommended)
    - Ditto installed from https://github.com/antgroup/ditto-talkinghead
    - TensorRT backend for best performance
    - opencv-python, librosa for processing

    Args:
        ditto_path: Path to Ditto installation directory
        data_root: Path to Ditto model checkpoints
        cfg_pkl: Path to Ditto config file (use online config like v0.4_hubert_cfg_trt_online.pkl)
        source_image_path: Path to the source image (the avatar face to animate)
        chunk_size: Audio chunk size tuple (history, current, future) frames
                   - Default (3, 5, 2) = ~200ms latency, 50% overlap
                   - Smaller 'current' = lower latency but less stable
                   - Must match model training configuration
        **kwargs: Additional arguments passed to FrameProcessor

    Note:
        - source_image_path is the avatar face that will be animated
        - Audio will be automatically resampled to 16kHz (Ditto requirement)
        - Use an "online" config file for real-time streaming
        - chunk_size must match the model's expected configuration
    """

    def __init__(
        self,
        *,
        ditto_path: str,
        data_root: str,
        cfg_pkl: str,
        source_image_path: str,
        chunk_size: tuple = (3, 5, 2),  # (history, current, future) frames
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._ditto_path = ditto_path
        self._data_root = data_root
        self._cfg_pkl = cfg_pkl
        self._source_image_path = source_image_path
        self._chunk_size = chunk_size

        self._initialized = False
        self._audio_buffer = []  # Buffer for 16kHz audio samples
        self._sdk = None

        # Background tasks and queues
        self._video_playback_task: Optional[asyncio.Task] = None
        self._frame_reader_task: Optional[asyncio.Task] = None
        self._audio_processing_task: Optional[asyncio.Task] = None
        self._video_queue = asyncio.Queue()
        self._frame_capture_queue = queue_module.Queue()  # Thread-safe queue for SDK's writer callback
        self._frame_reader_running = False

        # State tracking
        self._processing_lock = asyncio.Lock()
        self._interrupted = False
        self._current_num_frames = 0
        self._sdk_initialized_for_utterance = False

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
        """
        Initialize the Ditto SDK and start background tasks.

        This is where the source image is loaded:
        - SDK.setup() loads the source_image_path (the avatar face)
        - The SDK processes and prepares the image for animation
        - This image will be used for all subsequent run_chunk() calls
        - The same avatar face is animated throughout the session

        The source image must be:
        - A local file path (not URL or bytes)
        - A clear front-facing photo of a person's face
        - PNG or JPG format, 512x512+ resolution recommended
        """
        if self._initialized:
            return

        logger.info(f"{self}: Initializing Ditto Talking Head service")
        logger.info(f"{self}: Source image (avatar): {self._source_image_path}")
        logger.info(f"{self}: Data root: {self._data_root}")
        logger.info(f"{self}: Config: {self._cfg_pkl}")

        try:
            # Import Ditto's StreamSDK from stream_pipeline_online for real-time processing
            import sys
            sys.path.insert(0, self._ditto_path)
            from stream_pipeline_online import StreamSDK

            # Initialize SDK
            logger.info(f"{self}: Initializing Ditto StreamSDK...")
            self._sdk = StreamSDK(self._cfg_pkl, self._data_root)

            # Log online_mode if available
            if hasattr(self._sdk, 'online_mode'):
                logger.info(f"{self}: SDK online_mode: {self._sdk.online_mode}")
            else:
                logger.info(f"{self}: SDK initialized (online_mode attribute not available)")

            # Setup SDK with source image and temporary output path
            import tempfile
            temp_output = os.path.join(tempfile.gettempdir(), f"ditto_output_{id(self)}.mp4")

            # CRITICAL: This is where the source image is loaded and processed
            # The SDK will use this image for all video generation in this session
            # This also starts the worker threads
            logger.info(f"{self}: Loading avatar image and setting up SDK...")
            self._sdk.setup(
                source_path=self._source_image_path,  # Avatar face image loaded here
                output_path=temp_output,
            )
            logger.info(f"{self}: SDK setup completed, worker threads started")
            logger.info(f"{self}: Original writer type: {type(self._sdk.writer)}")

            # Wrap the writer object with a custom callable
            # The worker thread calls self.writer(frame, fmt="rgb")
            frame_count = [0]  # Use list to allow mutation in nested function
            original_writer = self._sdk.writer

            class WriterWrapper:
                """Wrapper that captures frames before passing to original writer"""
                def __init__(self, original, capture_queue, service_self):
                    self.original = original
                    self.capture_queue = capture_queue
                    self.service_self = service_self
                    self.frame_count = 0

                def __call__(self, frame_rgb, fmt="rgb"):
                    try:
                        # Capture frame for streaming
                        if isinstance(frame_rgb, np.ndarray):
                            self.capture_queue.put(frame_rgb)
                            self.frame_count += 1
                            if self.frame_count % 25 == 0:  # Log every second
                                logger.info(f"{self.service_self}: Custom writer captured {self.frame_count} frames")

                        # Call original writer
                        return self.original(frame_rgb, fmt=fmt)
                    except Exception as e:
                        logger.error(f"{self.service_self}: Error in writer wrapper: {e}")
                        import traceback
                        traceback.print_exc()

                def __getattr__(self, name):
                    # Delegate attribute access to original writer
                    return getattr(self.original, name)

            # Replace the writer with our wrapper
            self._sdk.writer = WriterWrapper(original_writer, self._frame_capture_queue, self)
            logger.info(f"{self}: Wrapped VideoWriterByImageIO with frame capturer")
            logger.debug(f"{self}: Original writer type: {type(original_writer)}")

            # Start background task to read frames from SDK's writer_queue
            self._frame_reader_running = True
            self._frame_reader_task = self.create_task(self._read_frames_from_sdk())

            # Start background task to push video frames downstream
            self._video_playback_task = self.create_task(self._consume_and_push_video())

            self._initialized = True
            logger.info(f"{self}: Ditto service initialized successfully")

        except Exception as e:
            logger.error(f"{self}: Failed to initialize Ditto: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to initialize Ditto: {e}")

    async def stop(self, frame: EndFrame):
        """Clean up resources"""
        self._frame_reader_running = False

        # Cancel tasks
        if self._audio_processing_task:
            self._audio_processing_task.cancel()
            try:
                await self._audio_processing_task
            except asyncio.CancelledError:
                pass
            self._audio_processing_task = None

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

        self._audio_buffer = []
        self._sdk = None
        self._initialized = False

    async def cancel(self, frame: CancelFrame):
        """Cancel any ongoing operations"""
        self._interrupted = True

        if self._audio_processing_task:
            self._audio_processing_task.cancel()
            try:
                await self._audio_processing_task
            except asyncio.CancelledError:
                pass
            self._audio_processing_task = None

        self._audio_buffer = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """
        Process incoming frames from the pipeline.

        Frame-based Accumulation Strategy (following Simli's pattern):
        -------------------------------------------------------------
        Audio arrives as individual TTSAudioRawFrame objects (typically small chunks,
        e.g., 20-50ms each). We accumulate multiple frames before processing:

        1. Each TTSAudioRawFrame adds ~20-50ms of audio to buffer
        2. We accumulate until buffer >= split_len (~400ms window)
        3. Then trigger background processing (non-blocking, like Simli)
        4. Background task processes all available chunks with overlapping windows
        5. Only one processing task runs at a time (prevents concurrent tasks)

        Example timeline:
        - Frame 1 (20ms): buffer = 320 samples → wait
        - Frame 2 (20ms): buffer = 640 samples → wait
        - ...
        - Frame 10 (20ms): buffer = 6480+ samples → trigger background task
        - Frame 11+: accumulate while background task processes

        Flow (similar to Simli's architecture):
        1. TTSAudioRawFrame → Resample to 16kHz, accumulate in buffer
        2. When buffer >= window size → spawn background task (non-blocking)
        3. Background task processes chunks with overlap via SDK
        4. Audio frame passes through immediately (plays in real-time)
        5. SDK generates video frames in background threads
        6. _read_frames_from_sdk() pulls frames from SDK's writer_queue
        7. _consume_and_push_video() pushes frames downstream at 25fps
        8. TTSStoppedFrame → process remaining buffered audio
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self.start(frame)

        elif isinstance(frame, TTSStartedFrame):
            # Reset state for new utterance
            logger.info(f"{self}: Starting new TTS utterance")
            self._interrupted = False
            self._audio_buffer = []
            self._current_num_frames = 0
            self._sdk_initialized_for_utterance = False

        elif isinstance(frame, TTSAudioRawFrame):
            if not self._initialized or self._interrupted:
                pass  # Skip if not ready or interrupted
            else:
                # Resample audio to 16kHz (Ditto requirement)
                audio_array = np.frombuffer(frame.audio, dtype=np.int16)

                if frame.sample_rate != 16000:
                    # Convert to float and resample
                    audio_float = audio_array.astype(np.float32) / 32768.0
                    audio_resampled = librosa.resample(
                        audio_float,
                        orig_sr=frame.sample_rate,
                        target_sr=16000
                    )
                else:
                    audio_resampled = audio_array.astype(np.float32) / 32768.0

                # Add to buffer (accumulate frames)
                self._audio_buffer.extend(audio_resampled.tolist())

                # Trigger processing in background if we have enough audio
                # Similar to Simli: don't block the pipeline
                split_len = int(sum(self._chunk_size) * 0.04 * 16000) + 80
                if len(self._audio_buffer) >= split_len:
                    # Create background task to process chunks (non-blocking)
                    # Only create if not already processing
                    if self._audio_processing_task is None or self._audio_processing_task.done():
                        self._audio_processing_task = self.create_task(self._process_audio_chunks())

        elif isinstance(frame, TTSStoppedFrame):
            logger.info(f"{self}: TTS stopped, processing remaining audio")
            # Process any remaining audio in buffer
            await self._finalize_audio()

        elif isinstance(frame, (InterruptionFrame, UserStartedSpeakingFrame)):
            # User interrupted
            logger.info(f"{self}: Interruption detected, clearing state")
            self._interrupted = True
            self._audio_buffer = []
            self._sdk_initialized_for_utterance = False
            # Clear video queue
            while not self._video_queue.empty():
                try:
                    self._video_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self.stop(frame)

        # Always pass frame downstream (audio plays immediately)
        await self.push_frame(frame, direction)

    async def _process_audio_chunks(self):
        """
        Process accumulated audio chunks using Ditto's run_chunk() method with overlapping windows.

        This function is called when:
        1. We've accumulated enough audio frames for a full window
        2. TTS has stopped and we need to process remaining audio

        Processing strategy:
        - Accumulates multiple TTSAudioRawFrame inputs into buffer
        - Processes when buffer >= split_len (full window size)
        - Uses overlapping windows: takes split_len but only advances by stride
        - May process multiple chunks if buffer has grown large

        Follows the pattern from inference.py online mode:
        - chunk_size = (history, current, future) e.g., (3, 5, 2) frames
        - split_len = total window size (history+current+future) = ~6480 samples
        - stride = current frame count = 3200 samples (50% overlap)
        """
        if not self._sdk or self._interrupted:
            return

        async with self._processing_lock:
            # Calculate sizes
            # chunk_size = (history, current, future) in 25fps frames
            # At 16kHz, one frame = 640 samples (16000/25)
            stride_samples = self._chunk_size[1] * 640  # How much to advance (current)
            split_len = int(sum(self._chunk_size) * 0.04 * 16000) + 80  # Total window size

            # Process all available chunks (may be multiple if buffer has grown)
            chunks_processed = 0
            while len(self._audio_buffer) >= split_len and not self._interrupted:
                try:
                    # Initialize SDK for this utterance if not done yet (first chunk only)
                    if not self._sdk_initialized_for_utterance:
                        # Add history padding at the very beginning (only once)
                        history_samples = self._chunk_size[0] * 640
                        if history_samples > 0:
                            padding = np.zeros((history_samples,), dtype=np.float32)
                            self._audio_buffer = padding.tolist() + self._audio_buffer
                            logger.debug(f"{self}: Added {history_samples} samples of history padding")

                        # Estimate total number of frames
                        estimated_audio_length = len(self._audio_buffer) / 16000  # seconds
                        num_frames = math.ceil(estimated_audio_length * 25)  # 25fps

                        logger.info(f"{self}: Setting up SDK for ~{num_frames} frames")

                        # Setup number of frames (SDK requirement)
                        self._sdk.setup_Nd(
                            N_d=num_frames,
                            fade_in=-1,  # No fade in
                            fade_out=-1,  # No fade out
                            ctrl_info={}  # No special control
                        )
                        self._sdk_initialized_for_utterance = True
                        self._current_num_frames = num_frames

                    # Extract overlapping window
                    # Take full split_len but only advance by stride_samples
                    audio_chunk = np.array(self._audio_buffer[:split_len], dtype=np.float32)

                    # Pad if necessary (for final chunk)
                    if len(audio_chunk) < split_len:
                        audio_chunk = np.pad(
                            audio_chunk,
                            (0, split_len - len(audio_chunk)),
                            mode="constant"
                        )

                    # Process chunk in executor (GPU operation)
                    # SDK handles state caches, overlap-add blending, etc. internally
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None,
                        self._run_chunk,
                        audio_chunk
                    )

                    # CRITICAL: Only advance by stride (current frames), not full window
                    # This creates the overlap between consecutive chunks
                    self._audio_buffer = self._audio_buffer[stride_samples:]
                    chunks_processed += 1

                    logger.debug(
                        f"{self}: Processed chunk #{chunks_processed} (window={split_len}, stride={stride_samples}), "
                        f"{len(self._audio_buffer)} samples remaining"
                    )

                except Exception as e:
                    logger.error(f"{self}: Error processing audio chunk: {e}")
                    import traceback
                    traceback.print_exc()
                    break  # Stop processing on error

            if chunks_processed > 0:
                logger.debug(f"{self}: Batch processed {chunks_processed} chunk(s)")

    def _run_chunk(self, audio_chunk: np.ndarray):
        """
        Run Ditto's chunk processing (called in executor).
        This is where the actual inference happens for the audio chunk.
        """
        logger.debug(f"{self}: Calling SDK.run_chunk with {len(audio_chunk)} samples")
        self._sdk.run_chunk(audio_chunk, self._chunk_size)
        logger.debug(f"{self}: SDK.run_chunk completed")

        # Check if any worker thread encountered an exception
        if hasattr(self._sdk, 'worker_exception') and self._sdk.worker_exception is not None:
            logger.error(f"{self}: Worker thread exception: {self._sdk.worker_exception}")
            raise self._sdk.worker_exception

        # Log ALL queue sizes to find where frames are stuck
        import time
        time.sleep(0.5)  # Give worker threads a moment to process

        queue_names = ['audio2motion_queue', 'motion_stitch_queue', 'warp_f3d_queue',
                      'decode_f3d_queue', 'putback_queue', 'writer_queue']
        queue_sizes = {}
        for qname in queue_names:
            if hasattr(self._sdk, qname):
                queue_sizes[qname] = getattr(self._sdk, qname).qsize()

        logger.info(f"{self}: SDK pipeline queues: {queue_sizes}")
        logger.info(f"{self}: frame_capture_queue size: {self._frame_capture_queue.qsize()}")

    async def _finalize_audio(self):
        """Process any remaining audio when TTS completes"""
        logger.info(f"{self}: Finalizing audio processing, buffer has {len(self._audio_buffer)} samples")

        # Add future padding to flush the pipeline
        # The SDK needs 'future' frames worth of audio to output the last frames
        future_samples = self._chunk_size[2] * 640  # e.g., 2 * 640 = 1280 for (3,5,2)
        if future_samples > 0 and not self._interrupted:
            logger.info(f"{self}: Adding {future_samples} samples of future padding to flush pipeline")
            future_padding = np.zeros((future_samples,), dtype=np.float32)
            self._audio_buffer.extend(future_padding.tolist())
            logger.info(f"{self}: After padding, buffer has {len(self._audio_buffer)} samples")

        # Process all remaining chunks
        # Note: _process_audio_chunks has its own while loop that processes chunks >= split_len
        stride_samples = self._chunk_size[1] * 640
        split_len = int(sum(self._chunk_size) * 0.04 * 16000) + 80

        logger.info(f"{self}: Processing remaining chunks (stride={stride_samples}, split_len={split_len})")

        # Keep processing until buffer is too small for even a stride
        iteration = 0
        while len(self._audio_buffer) >= stride_samples and not self._interrupted:
            iteration += 1
            buffer_size_before = len(self._audio_buffer)
            logger.debug(f"{self}: Iteration {iteration}: buffer size = {buffer_size_before}")

            await self._process_audio_chunks()

            buffer_size_after = len(self._audio_buffer)

            # If buffer didn't change, force process the final partial chunk
            if buffer_size_before == buffer_size_after:
                logger.info(f"{self}: Buffer unchanged at {buffer_size_after} samples, processing final partial chunk")
                # Manually process final chunk by temporarily lowering requirement
                if buffer_size_after > 0:
                    async with self._processing_lock:
                        # Pad buffer to split_len
                        audio_chunk = np.array(self._audio_buffer, dtype=np.float32)
                        if len(audio_chunk) < split_len:
                            audio_chunk = np.pad(
                                audio_chunk,
                                (0, split_len - len(audio_chunk)),
                                mode="constant"
                            )

                        logger.info(f"{self}: Processing final padded chunk of {len(audio_chunk)} samples")
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, self._run_chunk, audio_chunk)

                        # Clear buffer
                        self._audio_buffer = []
                        logger.info(f"{self}: Final chunk processed, buffer cleared")
                break

        logger.info(f"{self}: Audio processing finalized")

        # Wait for frames to propagate through the worker thread pipeline
        # The SDK processes frames asynchronously through multiple stages
        logger.info(f"{self}: Waiting for frames to propagate through pipeline...")
        for i in range(6):  # Check every second for 6 seconds
            await asyncio.sleep(1.0)

            # Check if we have any frames yet
            frames_captured = self._frame_capture_queue.qsize()
            if frames_captured > 0:
                logger.info(f"{self}: Frames started arriving! {frames_captured} frames captured so far")
                break

            # Log queue status
            if i % 2 == 0:  # Every 2 seconds
                queue_sizes = {}
                for qname in ['audio2motion_queue', 'motion_stitch_queue', 'warp_f3d_queue',
                             'decode_f3d_queue', 'putback_queue', 'writer_queue']:
                    if hasattr(self._sdk, qname):
                        queue_sizes[qname] = getattr(self._sdk, qname).qsize()
                logger.debug(f"{self}: After {i+1}s - Pipeline: {queue_sizes}, Captured: {frames_captured}")

        # Final status
        logger.info(f"{self}: Final frame_capture_queue size: {self._frame_capture_queue.qsize()}")

    async def _read_frames_from_sdk(self):
        """
        Background task that reads generated frames from our custom frame capture queue.
        The SDK's writer callback puts frames into this queue as they're generated.
        """
        logger.info(f"{self}: Frame reader task started")

        frame_counter = 0
        fps = 25

        try:
            while self._frame_reader_running:
                try:
                    # Try to get frame from our capture queue
                    # Run in executor since queue.get() is blocking
                    loop = asyncio.get_event_loop()
                    frame_rgb = await loop.run_in_executor(
                        None,
                        lambda: self._frame_capture_queue.get(timeout=0.1)
                    )

                    if frame_rgb is None:
                        # Termination signal
                        continue

                    # Convert frame to OutputImageRawFrame
                    if isinstance(frame_rgb, np.ndarray):
                        height, width = frame_rgb.shape[:2]

                        # Resize to 512x512 to match Daily transport configuration
                        # Ditto generates frames at 1440x1920, but we need to match the transport output size
                        target_width, target_height = 512, 512
                        if width != target_width or height != target_height:
                            logger.debug(f"{self}: Resizing frame from {width}x{height} to {target_width}x{target_height}")
                            frame_rgb = cv2.resize(frame_rgb, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
                            width, height = target_width, target_height

                        frame_bytes = frame_rgb.tobytes()

                        output_frame = OutputImageRawFrame(
                            image=frame_bytes,
                            size=(width, height),
                            format="RGB"
                        )

                        # Set PTS (presentation timestamp) for video synchronization
                        # Following Simli's pattern for proper timing
                        output_frame.pts = frame_counter
                        frame_counter += 1

                        # Queue for playback at correct frame rate (25fps)
                        await self._video_queue.put((output_frame, 1.0 / fps))
                        logger.debug(f"{self}: Queued video frame {width}x{height} (pts={output_frame.pts})")

                except queue_module.Empty:
                    # No frame available yet
                    await asyncio.sleep(0.01)
                except Exception as e:
                    if self._frame_reader_running:
                        logger.error(f"{self}: Error reading frame: {e}")
                    break

        except asyncio.CancelledError:
            logger.info(f"{self}: Frame reader task cancelled")
            raise
        except Exception as e:
            logger.error(f"{self}: Error in frame reader: {e}")
            import traceback
            traceback.print_exc()

    async def _consume_and_push_video(self):
        """
        Background task that pushes video frames downstream at the correct frame rate.
        Similar to Simli's approach.
        """
        logger.info(f"{self}: Video playback task started")

        try:
            frame_count = 0
            while True:
                # Get next frame from queue (blocks until available)
                frame, frame_duration = await self._video_queue.get()

                # Push frame downstream
                await self.push_frame(frame)
                frame_count += 1

                if frame_count % 25 == 0:  # Log every second (25fps)
                    logger.debug(f"{self}: Pushed {frame_count} video frames downstream")

                # Wait for correct frame rate
                await asyncio.sleep(frame_duration)

        except asyncio.CancelledError:
            logger.info(f"{self}: Video playback task cancelled")
            raise
        except Exception as e:
            logger.error(f"{self}: Error in video playback: {e}")
            import traceback
            traceback.print_exc()
