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

            # ==================== GPU DIAGNOSTICS ====================
            import torch
            logger.info(f"{self}: ===== GPU DIAGNOSTICS =====")
            logger.info(f"{self}: PyTorch CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"{self}: GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"{self}: Initial VRAM: {torch.cuda.memory_allocated(0) / 1e9:.2f}GB")
            
            # Check Ditto model device
            if hasattr(self._sdk, 'audio2motion'):
                a2m = self._sdk.audio2motion
                
                # Find the actual model
                if hasattr(a2m, 'audio2motion_model'):
                    model = a2m.audio2motion_model
                    device = next(model.parameters()).device
                    logger.info(f"{self}: ⚠️ Audio2Motion model is on: {device}")
                    
                    # Force to GPU if not already
                    if device.type == 'cpu':
                        logger.warning(f"{self}: Model is on CPU! Moving to GPU...")
                        a2m.audio2motion_model = a2m.audio2motion_model.cuda()
                        logger.info(f"{self}: ✅ Model moved to GPU")
                        
                        # Verify
                        device = next(a2m.audio2motion_model.parameters()).device
                        logger.info(f"{self}: Verified device: {device}")
                
                # Check other models if they exist
                if hasattr(a2m, 'hubert_model'):
                    device = next(a2m.hubert_model.parameters()).device
                    logger.info(f"{self}: Hubert model on: {device}")
                    if device.type == 'cpu':
                        a2m.hubert_model = a2m.hubert_model.cuda()
                        logger.info(f"{self}: Moved Hubert to GPU")
            
            # Check other pipeline components
            for attr_name in ['motion_stitch', 'f3d_warper', 'f3d_decoder']:
                if hasattr(self._sdk, attr_name):
                    component = getattr(self._sdk, attr_name)
                    if hasattr(component, 'model'):
                        device = next(component.model.parameters()).device
                        logger.info(f"{self}: {attr_name} model on: {device}")
                        if device.type == 'cpu':
                            component.model = component.model.cuda()
                            logger.info(f"{self}: Moved {attr_name} to GPU")
            
            if torch.cuda.is_available():
                logger.info(f"{self}: After loading VRAM: {torch.cuda.memory_allocated(0) / 1e9:.2f}GB")
            logger.info(f"{self}: ===== END GPU DIAGNOSTICS =====")
            # ==================== END GPU DIAGNOSTICS ====================

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
            self._audio_buffer = np.array([], dtype=np.float32)  # Use numpy array!
            self._current_num_frames = 0

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

                # Add to buffer (numpy array, not list!)
                self._audio_buffer = np.concatenate([self._audio_buffer, audio_resampled])

                # Process chunks continuously as they arrive
                while len(self._audio_buffer) >= 6480:
                    chunk = self._audio_buffer[:6480]
                    self._audio_buffer = self._audio_buffer[6480:]
                    
                    # Process chunk immediately (don't wait!)
                    logger.debug(f"{self}: Processing chunk of {len(chunk)} samples immediately")
                    await self._process_single_chunk(chunk)

        elif isinstance(frame, TTSStoppedFrame):
            logger.info(f"{self}: TTS stopped, processing remaining audio")
            # Process any remaining audio in buffer
            if len(self._audio_buffer) > 0:
                await self._finalize_audio()

        elif isinstance(frame, (InterruptionFrame, UserStartedSpeakingFrame)):
            logger.info(f"{self}: Interruption detected, clearing state")
            self._interrupted = True
            self._audio_buffer = np.array([], dtype=np.float32)
            while not self._video_queue.empty():
                try:
                    self._video_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self.stop(frame)

        await self.push_frame(frame, direction)

    async def _process_single_chunk(self, audio_chunk: np.ndarray):
        """Process a single 6480-sample chunk immediately."""
        async with self._processing_lock:
            # Add history padding (1920 samples from previous chunk)
            if len(self._audio_history) >= 1920:
                padded_audio = np.concatenate([self._audio_history[-1920:], audio_chunk])
            else:
                # First chunk - pad with zeros
                padding = np.zeros(1920, dtype=np.float32)
                padded_audio = np.concatenate([padding, audio_chunk])
            
            logger.debug(f"{self}: Running SDK chunk (total: {len(padded_audio)} samples)")
            
            # Run SDK processing in executor (non-blocking)
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._sdk.run_chunk,
                padded_audio,
                self._chunk_size
            )
            
            # Update history for next chunk
            self._audio_history = audio_chunk
            
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
        """Process any remaining audio in the buffer after TTS stops."""
        if len(self._audio_buffer) == 0:
            return
            
        logger.info(f"{self}: Finalizing audio processing, buffer has {len(self._audio_buffer)} samples")
        
        # Add future context padding
        padding_samples = 1280  # 2 frames * 640
        logger.info(f"{self}: Adding {padding_samples} samples of future padding")
        
        # Pad with last sample repeated (or zeros if buffer empty)
        if len(self._audio_buffer) > 0:
            last_sample = self._audio_buffer[-1]
            future_padding = np.full(padding_samples, last_sample, dtype=np.float32)
            self._audio_buffer = np.concatenate([self._audio_buffer, future_padding])  # ← Changed from extend
        
        logger.info(f"{self}: After padding, buffer has {len(self._audio_buffer)} samples")
        
        # Process remaining chunks
        logger.info(f"{self}: Processing remaining chunks (chunk_size=6480)")
        
        iteration = 0
        while len(self._audio_buffer) >= 6480:
            iteration += 1
            logger.debug(f"{self}: Iteration {iteration}: buffer size = {len(self._audio_buffer)}")
            
            chunk = self._audio_buffer[:6480]
            self._audio_buffer = self._audio_buffer[6480:]  # ← Changed from list slicing (still works)
            
            # Process this chunk
            await self._process_single_chunk(chunk)
        
        # Process final partial chunk if any remains
        if len(self._audio_buffer) > 0:
            logger.info(f"{self}: Processing final partial chunk ({len(self._audio_buffer)} samples)")
            # Pad to minimum size if needed
            if len(self._audio_buffer) < 6480:
                padding_needed = 6480 - len(self._audio_buffer)
                final_padding = np.full(padding_needed, self._audio_buffer[-1], dtype=np.float32)
                final_chunk = np.concatenate([self._audio_buffer, final_padding])
            else:
                final_chunk = self._audio_buffer
            
            await self._process_single_chunk(final_chunk)
        
        # Clear buffer
        self._audio_buffer = np.array([], dtype=np.float32)
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
        """Consume video frames from queue and push to Daily."""
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
                        logger.info(f"{self}: Pushed {frames_pushed} frames to Daily")
                    
                    # Create OutputImageRawFrame for Daily
                    output_frame = OutputImageRawFrame(
                        image=frame,
                        size=(frame.shape[1], frame.shape[0]),  # (width, height)
                        format="RGB"
                    )
                    
                    # Push to pipeline (this sends to Daily)
                    await self.push_frame(output_frame)
                    logger.debug(f"{self}: Pushed frame {frames_pushed} to Daily")
                    
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