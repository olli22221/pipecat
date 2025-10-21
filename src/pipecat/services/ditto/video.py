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
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._ditto_path = ditto_path
        self._data_root = data_root
        self._cfg_pkl = cfg_pkl
        self._source_image_path = source_image_path
        self._chunk_size = chunk_size
        self._save_frames_dir = save_frames_dir

        self._initialized = False
        self._audio_buffer = []
        self._sdk = None

        # Background tasks and queues
        self._video_playback_task: Optional[asyncio.Task] = None
        self._frame_reader_task: Optional[asyncio.Task] = None
        self._audio_processing_task: Optional[asyncio.Task] = None
        self._video_queue = asyncio.Queue()
        self._frame_reader_running = False

        # State tracking
        self._processing_lock = asyncio.Lock()
        self._interrupted = False
        self._current_num_frames = 0
        self._sdk_initialized_for_utterance = False

        # Frame saving
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
        """
        Initialize the Ditto SDK and start background tasks for real-time streaming.
        
        This is where the source image is loaded:
        - SDK.setup() loads the source_image_path (the avatar face)
        - The SDK processes and prepares the image for animation
        - This image will be used for all subsequent run_chunk() calls
        - The same avatar face is animated throughout the session
        """
        if self._initialized:
            return

        logger.info(f"{self}: Initializing Ditto Talking Head service (ONLINE MODE)")
        logger.info(f"{self}: Source image (avatar): {self._source_image_path}")
        logger.info(f"{self}: Data root: {self._data_root}")
        logger.info(f"{self}: Config: {self._cfg_pkl}")

        try:
            # Import Ditto's StreamSDK from stream_pipeline_online for real-time processing
            import sys
            sys.path.insert(0, self._ditto_path)
            from stream_pipeline_online import StreamSDK

            # Initialize SDK with online_mode=True for real-time streaming
            logger.info(f"{self}: Initializing Ditto StreamSDK in online mode...")
            self._sdk = StreamSDK(self._cfg_pkl, self._data_root, online_mode=True)

            # Log initial online_mode
            if hasattr(self._sdk, 'online_mode'):
                logger.info(f"{self}: SDK online_mode after init: {self._sdk.online_mode}")

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
            logger.info(f"{self}: SDK setup completed")

            # CRITICAL: Force online_mode=True AFTER setup (config file overrides it during setup)
            # The SDK object must be in online mode for real-time chunk processing
            self._sdk.online_mode = True

            # Also force online_mode on audio2motion component if it exists
            if hasattr(self._sdk, 'audio2motion') and hasattr(self._sdk.audio2motion, 'online_mode'):
                self._sdk.audio2motion.online_mode = True
                logger.info(f"{self}: Forced audio2motion.online_mode to True")

            logger.info(f"{self}: Forced SDK online_mode to True for real-time streaming")
            logger.info(f"{self}: Final SDK online_mode: {self._sdk.online_mode}")

            # CHECK IF SDK NEEDS EXPLICIT START/BEGIN CALL
            logger.info(f"{self}: Checking if SDK needs explicit start...")
            
            if hasattr(self._sdk, 'start'):
                logger.info(f"{self}: Found SDK.start() method, calling it...")
                try:
                    self._sdk.start()
                    logger.info(f"{self}: SDK.start() completed successfully")
                except Exception as e:
                    logger.warning(f"{self}: SDK.start() raised exception: {e}")
            
            if hasattr(self._sdk, 'start_workers'):
                logger.info(f"{self}: Found SDK.start_workers() method, calling it...")
                try:
                    self._sdk.start_workers()
                    logger.info(f"{self}: SDK.start_workers() completed successfully")
                except Exception as e:
                    logger.warning(f"{self}: SDK.start_workers() raised exception: {e}")
            
            if hasattr(self._sdk, 'begin'):
                logger.info(f"{self}: Found SDK.begin() method, calling it...")
                try:
                    self._sdk.begin()
                    logger.info(f"{self}: SDK.begin() completed successfully")
                except Exception as e:
                    logger.warning(f"{self}: SDK.begin() raised exception: {e}")

            # CHECK WORKER THREAD STATUS AFTER SETUP
            logger.info(f"{self}: ===== POST-SETUP WORKER THREAD STATUS =====")
            
            if hasattr(self._sdk, 'workers'):
                logger.info(f"{self}: Found {len(self._sdk.workers)} worker threads")
                for i, worker in enumerate(self._sdk.workers):
                    worker_info = f"Worker {i}"
                    if hasattr(worker, 'name'):
                        worker_info += f" ({worker.name})"
                    
                    if hasattr(worker, 'is_alive'):
                        is_alive = worker.is_alive()
                        worker_info += f": alive={is_alive}"
                        if not is_alive:
                            logger.error(f"{self}: ⚠️ {worker_info} - THREAD IS DEAD!")
                        else:
                            logger.info(f"{self}: ✅ {worker_info} - thread is running")
                    else:
                        worker_info += f": type={type(worker)}"
                        logger.info(f"{self}: {worker_info}")
                    
                    if hasattr(worker, 'daemon'):
                        logger.info(f"{self}:   - daemon={worker.daemon}")
                    
                    if hasattr(worker, '_target'):
                        logger.info(f"{self}:   - target={worker._target.__name__ if hasattr(worker._target, '__name__') else worker._target}")
            else:
                logger.warning(f"{self}: ⚠️ SDK has no 'workers' attribute!")
                # Try alternative worker management
                worker_attrs = [attr for attr in dir(self._sdk) if 'worker' in attr.lower() or 'thread' in attr.lower()]
                if worker_attrs:
                    logger.info(f"{self}: Found alternative worker-related attributes: {worker_attrs}")
            
            # Check if SDK has process/daemon/thread manager
            if hasattr(self._sdk, 'pipeline'):
                logger.info(f"{self}: SDK has 'pipeline' attribute: {type(self._sdk.pipeline)}")
            if hasattr(self._sdk, 'started'):
                logger.info(f"{self}: SDK.started = {self._sdk.started}")
            if hasattr(self._sdk, 'running'):
                logger.info(f"{self}: SDK.running = {self._sdk.running}")
            if hasattr(self._sdk, '_started'):
                logger.info(f"{self}: SDK._started = {self._sdk._started}")
            
            logger.info(f"{self}: ===== END POST-SETUP WORKER THREAD STATUS =====")

            # Diagnose SDK configuration and queues
            self._diagnose_sdk_queues()

            # Check writer configuration
            if hasattr(self._sdk, 'writer'):
                logger.info(f"{self}: ===== WRITER CONFIGURATION =====")
                writer = self._sdk.writer
                logger.info(f"{self}: Writer type: {type(writer)}")
                logger.info(f"{self}: Writer class: {writer.__class__.__name__}")
                
                # Check writer attributes
                writer_attrs = [attr for attr in dir(writer) if not attr.startswith('_')]
                logger.info(f"{self}: Writer public attributes/methods: {writer_attrs}")
                
                if hasattr(writer, 'output_path'):
                    logger.info(f"{self}: Writer output_path: {writer.output_path}")
                if hasattr(writer, 'fps'):
                    logger.info(f"{self}: Writer fps: {writer.fps}")
                if hasattr(writer, 'started'):
                    logger.info(f"{self}: Writer started: {writer.started}")
                
                logger.info(f"{self}: ===== END WRITER CONFIGURATION =====")

            # Start background task to read frames from SDK's internal queues
            self._frame_reader_running = True
            self._frame_reader_task = self.create_task(self._read_frames_from_sdk())
            logger.info(f"{self}: Started frame reader task")

            # Start background task to push video frames downstream
            self._video_playback_task = self.create_task(self._consume_and_push_video())
            logger.info(f"{self}: Started video playback task")

            self._initialized = True
            logger.info(f"{self}: ✅ Ditto service initialized successfully")
            logger.info(f"{self}: Ready to process audio and generate talking head video")

        except Exception as e:
            logger.error(f"{self}: ❌ Failed to initialize Ditto: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to initialize Ditto: {e}")

    def _diagnose_sdk_queues(self):
        """Diagnose SDK configuration to find where frames are being output."""
        logger.info(f"{self}: ===== SDK DIAGNOSTICS =====")
        
        # Check online mode
        logger.info(f"{self}: SDK online_mode: {getattr(self._sdk, 'online_mode', 'N/A')}")
        
        # Check all queue attributes
        queue_attrs = [attr for attr in dir(self._sdk) if 'queue' in attr.lower()]
        logger.info(f"{self}: Queue attributes found: {queue_attrs}")
        
        for attr_name in queue_attrs:
            try:
                attr = getattr(self._sdk, attr_name)
                if hasattr(attr, 'qsize'):
                    logger.info(f"{self}:   - {attr_name}: qsize={attr.qsize()}, empty={attr.empty()}")
            except Exception as e:
                logger.warning(f"{self}:   - {attr_name}: Error accessing - {e}")
        
        # Check writer
        if hasattr(self._sdk, 'writer'):
            writer = self._sdk.writer
            logger.info(f"{self}: Writer type: {type(writer)}")
            logger.info(f"{self}: Writer class: {writer.__class__.__name__}")
            writer_methods = [m for m in dir(writer) if not m.startswith('_')]
            logger.info(f"{self}: Writer methods: {writer_methods}")
        
        # Check for frame retrieval methods
        frame_methods = [m for m in dir(self._sdk) if 'frame' in m.lower() and not m.startswith('_')]
        logger.info(f"{self}: Frame-related methods: {frame_methods}")
        
        # Check for buffer attributes
        buffer_attrs = [attr for attr in dir(self._sdk) if 'buffer' in attr.lower()]
        logger.info(f"{self}: Buffer attributes: {buffer_attrs}")
        
        logger.info(f"{self}: ===== END DIAGNOSTICS =====")

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
        """Process incoming frames from the pipeline."""
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self.start(frame)

        elif isinstance(frame, TTSStartedFrame):
            logger.info(f"{self}: Starting new TTS utterance")
            self._interrupted = False
            self._audio_buffer = []
            self._current_num_frames = 0
            self._sdk_initialized_for_utterance = False

        elif isinstance(frame, TTSAudioRawFrame):
            if not self._initialized or self._interrupted:
                pass
            else:
                # Resample audio to 16kHz
                audio_array = np.frombuffer(frame.audio, dtype=np.int16)

                if frame.sample_rate != 16000:
                    audio_float = audio_array.astype(np.float32) / 32768.0
                    audio_resampled = librosa.resample(
                        audio_float,
                        orig_sr=frame.sample_rate,
                        target_sr=16000
                    )
                else:
                    audio_resampled = audio_array.astype(np.float32) / 32768.0

                self._audio_buffer.extend(audio_resampled.tolist())

                # Trigger processing if we have enough audio
                split_len = int(sum(self._chunk_size) * 0.04 * 16000) + 80
                if len(self._audio_buffer) >= split_len:
                    if self._audio_processing_task is None or self._audio_processing_task.done():
                        self._audio_processing_task = self.create_task(self._process_audio_chunks())

        elif isinstance(frame, TTSStoppedFrame):
            logger.info(f"{self}: TTS stopped, processing remaining audio")
            await self._finalize_audio()

        elif isinstance(frame, (InterruptionFrame, UserStartedSpeakingFrame)):
            logger.info(f"{self}: Interruption detected, clearing state")
            self._interrupted = True
            self._audio_buffer = []
            self._sdk_initialized_for_utterance = False
            while not self._video_queue.empty():
                try:
                    self._video_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self.stop(frame)

        await self.push_frame(frame, direction)

    async def _process_audio_chunks(self):
        """Process accumulated audio chunks using Ditto's run_chunk() method."""
        if not self._sdk or self._interrupted:
            return

        async with self._processing_lock:
            stride_samples = self._chunk_size[1] * 640
            split_len = int(sum(self._chunk_size) * 0.04 * 16000) + 80

            chunks_processed = 0
            while len(self._audio_buffer) >= split_len and not self._interrupted:
                try:
                    if not self._sdk_initialized_for_utterance:
                        history_samples = self._chunk_size[0] * 640
                        if history_samples > 0:
                            padding = np.zeros((history_samples,), dtype=np.float32)
                            self._audio_buffer = padding.tolist() + self._audio_buffer
                            logger.debug(f"{self}: Added {history_samples} samples of history padding")

                        estimated_audio_length = len(self._audio_buffer) / 16000
                        num_frames = math.ceil(estimated_audio_length * 25)

                        logger.info(f"{self}: Setting up SDK for ~{num_frames} frames")

                        self._sdk.setup_Nd(
                            N_d=num_frames,
                            fade_in=-1,
                            fade_out=-1,
                            ctrl_info={}
                        )
                        self._sdk_initialized_for_utterance = True
                        self._current_num_frames = num_frames

                    audio_chunk = np.array(self._audio_buffer[:split_len], dtype=np.float32)

                    if len(audio_chunk) < split_len:
                        audio_chunk = np.pad(
                            audio_chunk,
                            (0, split_len - len(audio_chunk)),
                            mode="constant"
                        )

                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None,
                        self._run_chunk,
                        audio_chunk
                    )

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
                    break

            if chunks_processed > 0:
                logger.debug(f"{self}: Batch processed {chunks_processed} chunk(s)")

    def _run_chunk(self, audio_chunk: np.ndarray):
    """Run Ditto's chunk processing in executor."""
    logger.debug(f"{self}: Calling SDK.run_chunk with {len(audio_chunk)} samples")
    logger.info(f"{self}: Audio shape: {audio_chunk.shape}, chunk_size: {self._chunk_size}")

    # CHECK WORKER THREADS STATUS BEFORE PROCESSING
    logger.info(f"{self}: ===== WORKER THREAD STATUS =====")
    if hasattr(self._sdk, 'workers'):
        for i, worker in enumerate(self._sdk.workers):
            if hasattr(worker, 'is_alive'):
                is_alive = worker.is_alive()
                logger.info(f"{self}: Worker {i} ({worker.name if hasattr(worker, 'name') else 'unnamed'}): alive={is_alive}")
                if not is_alive:
                    logger.error(f"{self}: ⚠️ Worker {i} is DEAD!")
            else:
                logger.info(f"{self}: Worker {i}: {type(worker)}")
    else:
        logger.warning(f"{self}: SDK has no 'workers' attribute!")
    
    # Check if SDK has a process/daemon/thread manager
    if hasattr(self._sdk, 'pipeline'):
        logger.info(f"{self}: SDK has pipeline attribute")
    if hasattr(self._sdk, 'started'):
        logger.info(f"{self}: SDK started: {self._sdk.started}")
    if hasattr(self._sdk, 'running'):
        logger.info(f"{self}: SDK running: {self._sdk.running}")
    
    logger.info(f"{self}: ===== END WORKER THREAD STATUS =====")

    # Log what's about to be processed
    logger.info(f"{self}: Calling run_chunk with:")
    logger.info(f"{self}:   - audio_chunk.shape: {audio_chunk.shape}")
    logger.info(f"{self}:   - audio_chunk.dtype: {audio_chunk.dtype}")
    logger.info(f"{self}:   - audio_chunk min/max: {audio_chunk.min():.4f} / {audio_chunk.max():.4f}")
    logger.info(f"{self}:   - chunk_size: {self._chunk_size}")
    logger.info(f"{self}:   - SDK online_mode: {getattr(self._sdk, 'online_mode', 'N/A')}")

    # Call run_chunk
    try:
        self._sdk.run_chunk(audio_chunk, self._chunk_size)
        logger.debug(f"{self}: SDK.run_chunk completed successfully")
    except Exception as e:
        logger.error(f"{self}: SDK.run_chunk raised exception: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Check for worker exceptions
    if hasattr(self._sdk, 'worker_exception') and self._sdk.worker_exception is not None:
        logger.error(f"{self}: ⚠️ Worker thread exception detected: {self._sdk.worker_exception}")
        raise self._sdk.worker_exception

    # Wait longer for processing
    import time
    logger.info(f"{self}: Waiting 2 seconds for worker threads to process...")
    time.sleep(2.0)

    # Check ALL queues in detail
    logger.info(f"{self}: ===== QUEUE STATUS AFTER run_chunk =====")
    for attr_name in dir(self._sdk):
        try:
            attr = getattr(self._sdk, attr_name)
            if hasattr(attr, 'qsize'):
                qsize = attr.qsize()
                if qsize > 0:
                    logger.warning(f"{self}: ⚠️ {attr_name} HAS DATA! qsize={qsize}")
                    # Try to peek at what's in the queue
                    if hasattr(attr, 'queue'):
                        logger.info(f"{self}:   Queue contents preview: {list(attr.queue)[:3]}")
                else:
                    logger.info(f"{self}: {attr_name}: qsize=0 (empty)")
        except Exception as e:
            pass
    logger.info(f"{self}: ===== END QUEUE STATUS =====")

    # Check if run_chunk actually did anything
    if hasattr(self._sdk, 'n_generated_frames'):
        logger.info(f"{self}: n_generated_frames: {self._sdk.n_generated_frames}")
    
    # Check internal counters
    for attr_name in ['n_chunks_processed', 'frame_count', 'total_frames']:
        if hasattr(self._sdk, attr_name):
            logger.info(f"{self}: {attr_name}: {getattr(self._sdk, attr_name)}")

    async def _finalize_audio(self):
        """Process any remaining audio when TTS completes."""
        logger.info(f"{self}: Finalizing audio processing, buffer has {len(self._audio_buffer)} samples")

        future_samples = self._chunk_size[2] * 640
        if future_samples > 0 and not self._interrupted:
            logger.info(f"{self}: Adding {future_samples} samples of future padding")
            future_padding = np.zeros((future_samples,), dtype=np.float32)
            self._audio_buffer.extend(future_padding.tolist())
            logger.info(f"{self}: After padding, buffer has {len(self._audio_buffer)} samples")

        stride_samples = self._chunk_size[1] * 640
        split_len = int(sum(self._chunk_size) * 0.04 * 16000) + 80

        logger.info(f"{self}: Processing remaining chunks (stride={stride_samples}, split_len={split_len})")

        iteration = 0
        while len(self._audio_buffer) >= stride_samples and not self._interrupted:
            iteration += 1
            buffer_size_before = len(self._audio_buffer)
            logger.debug(f"{self}: Iteration {iteration}: buffer size = {buffer_size_before}")

            await self._process_audio_chunks()

            buffer_size_after = len(self._audio_buffer)

            if buffer_size_before == buffer_size_after:
                logger.info(f"{self}: Buffer unchanged at {buffer_size_after} samples, processing final partial chunk")
                if buffer_size_after > 0:
                    async with self._processing_lock:
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

                        self._audio_buffer = []
                        logger.info(f"{self}: Final chunk processed, buffer cleared")
                break

        logger.info(f"{self}: Audio processing finalized")

        # Wait for frames to propagate
        logger.info(f"{self}: Waiting for frames to propagate through pipeline...")
        for i in range(6):
            await asyncio.sleep(1.0)

            # Check all queues for frames
            found_frames = False
            for attr_name in dir(self._sdk):
                try:
                    attr = getattr(self._sdk, attr_name)
                    if hasattr(attr, 'qsize') and attr.qsize() > 0:
                        logger.info(f"{self}: Found {attr.qsize()} items in {attr_name}")
                        found_frames = True
                except:
                    pass

            if found_frames:
                logger.info(f"{self}: Frames detected in SDK queues!")
                break

    async def _read_frames_from_sdk(self):
        """
        Background task that reads generated frames from SDK's internal queues.
        THIS IS THE KEY METHOD - reads directly from SDK queues in online mode.
        """
        logger.info(f"{self}: Frame reader task started - READING FROM SDK QUEUES")

        frame_counter = 0
        fps = 25
        frame_save_counter = 0

        try:
            while self._frame_reader_running:
                try:
                    # Try multiple possible queue locations in SDK
                    frame_rgb = None
                    queue_source = None

                    # Strategy 1: Check writer_queue
                    if hasattr(self._sdk, 'writer_queue') and not self._sdk.writer_queue.empty():
                        try:
                            loop = asyncio.get_event_loop()
                            frame_data = await loop.run_in_executor(
                                None,
                                lambda: self._sdk.writer_queue.get(timeout=0.05)
                            )
                            
                            if isinstance(frame_data, tuple):
                                frame_rgb = frame_data[0]
                            else:
                                frame_rgb = frame_data
                            queue_source = "writer_queue"
                            
                        except queue_module.Empty:
                            pass

                    # Strategy 2: Check putback_queue (common in Ditto)
                    if frame_rgb is None and hasattr(self._sdk, 'putback_queue') and not self._sdk.putback_queue.empty():
                        try:
                            loop = asyncio.get_event_loop()
                            frame_data = await loop.run_in_executor(
                                None,
                                lambda: self._sdk.putback_queue.get(timeout=0.05)
                            )
                            
                            # putback_queue might contain dict with 'out' key
                            if isinstance(frame_data, dict) and 'out' in frame_data:
                                frame_rgb = frame_data['out']
                            elif isinstance(frame_data, np.ndarray):
                                frame_rgb = frame_data
                            queue_source = "putback_queue"
                            
                        except queue_module.Empty:
                            pass

                    # Strategy 3: Check decode_f3d_queue
                    if frame_rgb is None and hasattr(self._sdk, 'decode_f3d_queue') and not self._sdk.decode_f3d_queue.empty():
                        try:
                            loop = asyncio.get_event_loop()
                            frame_data = await loop.run_in_executor(
                                None,
                                lambda: self._sdk.decode_f3d_queue.get(timeout=0.05)
                            )
                            if isinstance(frame_data, dict) and 'out' in frame_data:
                                frame_rgb = frame_data['out']
                            elif isinstance(frame_data, np.ndarray):
                                frame_rgb = frame_data
                            queue_source = "decode_f3d_queue"
                        except queue_module.Empty:
                            pass

                    if frame_rgb is not None and isinstance(frame_rgb, np.ndarray):
                        if frame_counter == 0:
                            logger.info(f"{self}: 🎉 FOUND FRAMES in {queue_source}!")
                        
                        # Save frame if directory specified
                        if self._save_frames_dir:
                            frame_save_counter += 1
                            frame_path = os.path.join(
                                self._save_frames_dir,
                                f"frame_{frame_save_counter:06d}.png"
                            )
                            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(frame_path, frame_bgr)
                            if frame_save_counter % 25 == 0:
                                logger.info(f"{self}: Saved {frame_save_counter} frames")

                        height, width = frame_rgb.shape[:2]
                        target_width, target_height = 512, 512

                        if width != target_width or height != target_height:
                            frame_rgb = cv2.resize(
                                frame_rgb,
                                (target_width, target_height),
                                interpolation=cv2.INTER_LINEAR
                            )
                            width, height = target_width, target_height

                        frame_bytes = frame_rgb.tobytes()

                        output_frame = OutputImageRawFrame(
                            image=frame_bytes,
                            size=(width, height),
                            format="RGB"
                        )

                        output_frame.pts = frame_counter
                        frame_counter += 1

                        await self._video_queue.put((output_frame, 1.0 / fps))

                        if frame_counter % 25 == 0:
                            logger.info(f"{self}: Queued {frame_counter} video frames from {queue_source}")
                    else:
                        # No frames available yet
                        await asyncio.sleep(0.01)

                except Exception as e:
                    if self._frame_reader_running:
                        logger.error(f"{self}: Error reading frame: {e}")
                        import traceback
                        traceback.print_exc()
                    await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            logger.info(f"{self}: Frame reader task cancelled (read {frame_counter} frames)")
            raise
        except Exception as e:
            logger.error(f"{self}: Error in frame reader: {e}")
            import traceback
            traceback.print_exc()

    async def _consume_and_push_video(self):
        """Push video frames downstream at the correct frame rate."""
        logger.info(f"{self}: Video playback task started")

        try:
            frame_count = 0
            while True:
                frame, frame_duration = await self._video_queue.get()

                await self.push_frame(frame)
                frame_count += 1

                if frame_count % 25 == 0:
                    logger.debug(f"{self}: Pushed {frame_count} video frames downstream")

                await asyncio.sleep(frame_duration)

        except asyncio.CancelledError:
            logger.info(f"{self}: Video playback task cancelled")
            raise
        except Exception as e:
            logger.error(f"{self}: Error in video playback: {e}")
            import traceback
            traceback.print_exc()