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
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self._ditto_path = ditto_path
        self._data_root = data_root
        self._cfg_pkl = cfg_pkl
        self._source_image_path = source_image_path
        self._chunk_size = chunk_size
        self._save_frames_dir = save_frames_dir
        
        # Initialize ALL queues and state variables
        self._video_frame_queue = asyncio.Queue()  # For transferring frames from capture to playback
        self._video_queue = asyncio.Queue()        # For video frames with timestamps (used in process_frame)
        
        # Locks for thread safety
        self._processing_lock = asyncio.Lock()     # Prevents concurrent audio processing
        
        # SDK and initialization state
        self._sdk = None
        self._initialized = False
        
        # Audio processing state
        self._audio_buffer = np.array([], dtype=np.float32)
        self._audio_history = np.array([], dtype=np.float32)
        self._audio_processing_task = None  # For tracking async audio processing
        
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

            # Check worker threads
            logger.info(f"{self}: ===== POST-SETUP WORKER THREAD STATUS =====")
            if hasattr(self._sdk, 'thread_list'):
                logger.info(f"{self}: Found {len(self._sdk.thread_list)} worker threads")
                for i, thread in enumerate(self._sdk.thread_list):
                    is_alive = thread.is_alive()
                    thread_name = thread.name if hasattr(thread, 'name') else f"Thread-{i}"
                    logger.info(f"{self}: {'✅' if is_alive else '⚠️'} Worker {i} ({thread_name}): {'ALIVE' if is_alive else 'DEAD'}")
            
            logger.info(f"{self}: ===== END POST-SETUP WORKER THREAD STATUS =====")

            # Diagnose queues
            self._diagnose_sdk_queues()

            # Start background tasks
            self._frame_reader_running = True
            self._frame_reader_task = self.create_task(self._read_frames_from_sdk())
            logger.info(f"{self}: Started frame reader task")

            self._video_playback_task = self.create_task(self._consume_and_push_video())
            logger.info(f"{self}: Started video playback task")

            self._initialized = True
            logger.info(f"{self}: ✅ Ditto service initialized successfully")

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
        
        # Check thread_list (worker threads)
        if hasattr(self._sdk, 'thread_list'):
            logger.info(f"{self}: Number of worker threads: {len(self._sdk.thread_list)}")
        
        # Check all queue attributes
        queue_attrs = ['audio2motion_queue', 'motion_stitch_queue', 'warp_f3d_queue',
                    'decode_f3d_queue', 'putback_queue', 'writer_queue']
        logger.info(f"{self}: Checking pipeline queues...")
        
        for attr_name in queue_attrs:
            if hasattr(self._sdk, attr_name):
                attr = getattr(self._sdk, attr_name)
                logger.info(f"{self}:   - {attr_name}: qsize={attr.qsize()}, empty={attr.empty()}")
            else:
                logger.warning(f"{self}:   - {attr_name}: NOT FOUND")
        
        # Check writer
        if hasattr(self._sdk, 'writer'):
            writer = self._sdk.writer
            logger.info(f"{self}: Writer type: {type(writer)}")
            logger.info(f"{self}: Writer class: {writer.__class__.__name__}")
        
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
        """Run Ditto's chunk processing (called in executor)."""
        logger.debug(f"{self}: Calling SDK.run_chunk with {len(audio_chunk)} samples")
        logger.info(f"{self}: Audio shape: {audio_chunk.shape}, chunk_size: {self._chunk_size}")

        # CHECK WORKER THREAD STATUS BEFORE PROCESSING
        logger.info(f"{self}: ===== WORKER THREAD STATUS =====")
        if hasattr(self._sdk, 'thread_list'):
            logger.info(f"{self}: Found {len(self._sdk.thread_list)} threads in thread_list")
            for i, thread in enumerate(self._sdk.thread_list):
                if hasattr(thread, 'is_alive'):
                    is_alive = thread.is_alive()
                    thread_name = thread.name if hasattr(thread, 'name') else f"Thread-{i}"
                    if is_alive:
                        logger.info(f"{self}: ✅ Thread {i} ({thread_name}): ALIVE")
                    else:
                        logger.error(f"{self}: ⚠️ Thread {i} ({thread_name}): DEAD!")
            
            # Check if stop_event is set
            if hasattr(self._sdk, 'stop_event'):
                logger.info(f"{self}: SDK stop_event.is_set(): {self._sdk.stop_event.is_set()}")
        else:
            logger.warning(f"{self}: SDK has no 'thread_list' attribute!")
        
        logger.info(f"{self}: ===== END WORKER THREAD STATUS =====")

        # Check queue sizes BEFORE run_chunk
        logger.info(f"{self}: Queue sizes BEFORE run_chunk:")
        for qname in ['audio2motion_queue', 'motion_stitch_queue', 'warp_f3d_queue',
                    'decode_f3d_queue', 'putback_queue', 'writer_queue']:
            if hasattr(self._sdk, qname):
                q = getattr(self._sdk, qname)
                logger.info(f"{self}:   {qname}: {q.qsize()}")

        logger.info(f"{self}: Calling run_chunk...")
        
        # Call run_chunk
        try:
            self._sdk.run_chunk(audio_chunk, self._chunk_size)
            logger.debug(f"{self}: SDK.run_chunk completed successfully")
        except Exception as e:
            logger.error(f"{self}: SDK.run_chunk raised exception: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Check for worker exceptions IMMEDIATELY
        if hasattr(self._sdk, 'worker_exception') and self._sdk.worker_exception is not None:
            logger.error(f"{self}: ⚠️ WORKER EXCEPTION DETECTED: {self._sdk.worker_exception}")
            import traceback
            logger.error(f"{self}: Worker exception traceback:")
            traceback.print_exception(
                type(self._sdk.worker_exception), 
                self._sdk.worker_exception, 
                self._sdk.worker_exception.__traceback__
            )
            # Don't raise here, let it continue but log it

        # Check queue sizes IMMEDIATELY after run_chunk (before sleep)
        logger.info(f"{self}: Queue sizes IMMEDIATELY after run_chunk:")
        for qname in ['audio2motion_queue', 'motion_stitch_queue', 'warp_f3d_queue',
                    'decode_f3d_queue', 'putback_queue', 'writer_queue']:
            if hasattr(self._sdk, qname):
                q = getattr(self._sdk, qname)
                size = q.qsize()
                if size > 0:
                    logger.warning(f"{self}:   ⚠️ {qname}: {size} (HAS DATA!)")
                else:
                    logger.info(f"{self}:   {qname}: {size}")

        # NEW: Wait LONGER and check multiple times
        import time
        logger.info(f"{self}: Monitoring pipeline for 3 seconds...")
        for check_num in range(6):  # Check every 0.5s for 3 seconds total
            time.sleep(0.5)
            
            # Check all queues
            queue_status = {}
            for qname in ['audio2motion_queue', 'motion_stitch_queue', 'warp_f3d_queue',
                        'decode_f3d_queue', 'putback_queue', 'writer_queue']:
                if hasattr(self._sdk, qname):
                    queue_status[qname] = getattr(self._sdk, qname).qsize()
            
            # Log if any queue has data
            has_data = any(size > 0 for size in queue_status.values())
            if has_data:
                logger.warning(f"{self}: Check #{check_num+1}: Pipeline active! {queue_status}")
            
            # Check if frames reached writer_queue
            if queue_status.get('writer_queue', 0) > 0:
                logger.info(f"{self}: 🎉 FRAMES REACHED WRITER QUEUE!")
                break
            
            # Check for worker exceptions
            if hasattr(self._sdk, 'worker_exception') and self._sdk.worker_exception is not None:
                logger.error(f"{self}: Worker exception during monitoring: {self._sdk.worker_exception}")
                break
        
        # Final status
        logger.info(f"{self}: Final queue status after monitoring: {queue_status}")

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