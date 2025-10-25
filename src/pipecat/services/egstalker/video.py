#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""EGSTalker video service for real-time talking head generation.

This module provides integration with EGSTalker, a 3D Gaussian-based audio-driven
talking head synthesis framework, for real-time video generation in Pipecat pipelines.

IMPORTANT: EGSTalker requires significant preprocessing before use:
1. Train the model on your target face/avatar (see EGSTalker training docs)
2. Prepare a dataset with camera configurations and point clouds
3. Ensure model checkpoints are saved in the correct format
4. This implementation adapts EGSTalker's batch rendering to streaming mode

For more details: https://github.com/ZhuTianheng/EGSTalker
"""

import asyncio
import os
import tempfile
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
    import torch
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use EGSTalker, you need to `pip install opencv-python librosa torch`."
    )
    raise Exception(f"Missing module: {e}")


class EGSTalkerVideoService(FrameProcessor):
    """EGSTalker video service for real-time talking head generation.

    This service uses the EGSTalker model to generate realistic talking head videos
    using 3D Gaussian Splatting deformation driven by audio input.

    PERFORMANCE: EGSTalker achieves ~1000 FPS rendering speed on modern GPUs,
    making it one of the fastest local talking head systems. The main latency
    comes from audio accumulation and scene loading, not rendering itself.

    SETUP REQUIREMENTS:
    1. Train EGSTalker model on your target face/avatar:
       - Prepare ER-NeRF format dataset with videos and audio
       - Run EGSTalker training (typically 30,000 iterations)
       - Ensure checkpoints are saved in model_path

    2. Dataset structure should include:
       - Camera configuration files (transforms.json)
       - Point cloud data
       - Training iteration checkpoints

    3. Hardware requirements:
       - CUDA-capable GPU (RTX 3090 or better recommended)
       - EGSTalker installed from https://github.com/ZhuTianheng/EGSTalker
       - opencv-python, librosa, torch, scipy

    Args:
        egstalker_path: Path to EGSTalker installation directory (root of git repo)
        source_path: Path to prepared dataset directory (contains transforms.json, point_cloud, etc.)
        model_path: Path to trained model checkpoint directory (contains iterations/)
        iteration: Training iteration to load (e.g., 30000 for final model)
        sh_degree: Spherical harmonics degree (default: 3)
        audio_sample_rate: Target audio sample rate (default: 16000 Hz)
        target_fps: Target frame rate for video generation (default: 25 fps)
        save_frames_dir: Optional directory path to save generated frames as PNG files
        batch_size: Batch size for rendering (default: 1 for real-time)
        **kwargs: Additional arguments passed to FrameProcessor

    Example:
        service = EGSTalkerVideoService(
            egstalker_path="/path/to/EGSTalker",
            source_path="/path/to/prepared_dataset",
            model_path="/path/to/output/trained_model",
            iteration=30000,
            sh_degree=3,
            audio_sample_rate=16000,
            target_fps=25
        )
    """

    def __init__(
        self,
        *,
        egstalker_path: str,
        source_path: str,
        model_path: str,
        iteration: int = 30000,
        sh_degree: int = 3,
        audio_sample_rate: int = 16000,
        target_fps: int = 25,
        save_frames_dir: Optional[str] = None,
        batch_size: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._egstalker_path = egstalker_path
        self._source_path = source_path
        self._model_path = model_path
        self._iteration = iteration
        self._sh_degree = sh_degree
        self._audio_sample_rate = audio_sample_rate
        self._target_fps = target_fps
        self._save_frames_dir = save_frames_dir
        self._batch_size = batch_size

        # State management
        self._initialized = False
        self._model = None
        self._is_speaking = False
        self._is_interrupting = False

        # Audio processing
        self._audio_buffer = bytearray()
        self._audio_queue: Optional[asyncio.Queue] = None
        self._audio_task: Optional[asyncio.Task] = None
        self._event_id: Optional[str] = None

        # Video frame management
        self._video_frame_queue: asyncio.Queue = asyncio.Queue()
        self._video_playback_task: Optional[asyncio.Task] = None
        self._idle_frame_task: Optional[asyncio.Task] = None
        self._last_frame: Optional[np.ndarray] = None

        # Frame generation control
        self._frame_interval = 1.0 / self._target_fps
        self._frame_count = 0

        # Rendering lock to prevent concurrent rendering (speech vs idle)
        self._render_lock = asyncio.Lock()

        # Audio-video synchronization tracking
        self._base_timestamp = None  # Base timestamp when TTS starts
        self._audio_samples_pushed = 0  # Total audio samples pushed for timestamp calculation
        self._video_frames_pushed = 0  # Total video frames pushed
        self._current_audio_chunk_timestamp = 0  # Timestamp of current audio chunk being processed

        # Temporary files for audio processing
        self._temp_dir = tempfile.mkdtemp(prefix="egstalker_")

        # EGSTalker components (loaded during initialization)
        self._gaussians = None
        self._scene = None
        self._pipeline = None
        self._dataset_params = None
        self._hyperparam = None

        # Create save directory if specified
        if self._save_frames_dir:
            os.makedirs(self._save_frames_dir, exist_ok=True)
            logger.info(f"Frames will be saved to: {self._save_frames_dir}")

        # Validate paths
        if not os.path.exists(egstalker_path):
            raise ValueError(f"EGSTalker path not found: {egstalker_path}")
        if not os.path.exists(source_path):
            raise ValueError(f"Dataset source path not found: {source_path}")
        if not os.path.exists(model_path):
            raise ValueError(f"Model checkpoint not found: {model_path}")

    async def start(self, frame: StartFrame):
        """Initialize the EGSTalker model and start background tasks."""
        if self._initialized:
            return

        logger.info(f"{self}: Initializing EGSTalker Talking Head service")
        logger.info(f"{self}: Source dataset: {self._source_path}")
        logger.info(f"{self}: Model path: {self._model_path}")
        logger.info(f"{self}: Iteration: {self._iteration}")
        logger.info(f"{self}: SH Degree: {self._sh_degree}")

        try:
            # Import EGSTalker modules
            import sys

            sys.path.insert(0, self._egstalker_path)

            # Import required EGSTalker components
            try:
                from scene import Scene, GaussianModel
                from arguments import ModelParams, PipelineParams, ModelHiddenParams
                from gaussian_renderer import render_from_batch
                from utils.general_utils import safe_state
                import sys

                # Store render function for later use
                self._render_from_batch = render_from_batch

                logger.info(f"{self}: Successfully imported EGSTalker modules")
            except ImportError as e:
                logger.error(f"Failed to import EGSTalker modules: {e}")
                logger.error(
                    "Please ensure EGSTalker is properly installed with all required modules"
                )
                raise RuntimeError(f"Failed to import EGSTalker: {e}")

            # Initialize parameters in executor to avoid blocking
            def init_egstalker():
                """Initialize EGSTalker components (runs in thread pool)."""
                # Set up dataset parameters
                dataset_params = ModelParams()
                dataset_params.sh_degree = self._sh_degree
                dataset_params.source_path = self._source_path
                dataset_params.model_path = self._model_path
                dataset_params.eval = True

                # Set up hidden hyperparameters (neural network config)
                hyperparam = ModelHiddenParams()

                # Set up rendering pipeline parameters
                pipeline_params = PipelineParams()

                # Initialize Gaussian model
                gaussians = GaussianModel(dataset_params.sh_degree, hyperparam)

                # Load the scene with the trained model
                logger.info(f"{self}: Loading scene from {self._source_path}")
                scene = Scene(
                    dataset_params,
                    gaussians,
                    load_iteration=self._iteration,
                    shuffle=False
                )

                # Load trained model weights
                logger.info(f"{self}: Loading model checkpoint from iteration {self._iteration}")
                gaussians.load_ply(
                    os.path.join(
                        self._model_path,
                        f"point_cloud/iteration_{self._iteration}/point_cloud.ply"
                    )
                )

                return dataset_params, hyperparam, pipeline_params, gaussians, scene

            # Initialize EGSTalker in executor (may take time to load)
            logger.info(f"{self}: Loading EGSTalker model (this may take a moment)...")
            (
                self._dataset_params,
                self._hyperparam,
                self._pipeline,
                self._gaussians,
                self._scene,
            ) = await asyncio.get_event_loop().run_in_executor(None, init_egstalker)

            logger.info(f"{self}: EGSTalker model loaded successfully")

            # Start background tasks
            await self._create_audio_task()
            self._video_playback_task = self.create_task(self._consume_and_push_video())
            logger.info(f"{self}: Started video playback task")

            self._idle_frame_task = self.create_task(self._idle_frame_generator())
            logger.info(f"{self}: Started idle frame generator at {self._target_fps} fps")

            self._initialized = True
            logger.info(f"{self}: ✅ EGSTalker service initialized successfully")

        except Exception as e:
            logger.error(f"{self}: ❌ Failed to initialize EGSTalker: {e}")
            import traceback

            traceback.print_exc()
            raise RuntimeError(f"Failed to initialize EGSTalker: {e}")

    async def stop(self, frame: EndFrame):
        """Clean up resources."""
        logger.info(f"{self}: Stopping EGSTalker service")

        # Cancel tasks
        await self._cancel_audio_task()

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

        # Clear buffers
        self._audio_buffer.clear()
        while not self._video_frame_queue.empty():
            try:
                self._video_frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Clean up temporary directory
        import shutil

        if os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)

        self._model = None
        self._initialized = False
        logger.info(f"{self}: EGSTalker service stopped")

    async def cancel(self, frame: CancelFrame):
        """Cancel any ongoing operations."""
        self._is_interrupting = True
        await self._cancel_audio_task()
        self._audio_buffer.clear()

    async def _create_audio_task(self):
        """Create the audio processing task."""
        if not self._audio_task:
            self._audio_queue = asyncio.Queue()
            self._audio_task = self.create_task(self._audio_task_handler())
            logger.info(f"{self}: Audio processing task created")

    async def _cancel_audio_task(self):
        """Cancel the audio processing task."""
        if self._audio_task:
            self._audio_task.cancel()
            try:
                await self._audio_task
            except asyncio.CancelledError:
                pass
            self._audio_task = None
            logger.debug(f"{self}: Audio task cancelled")

    async def _audio_task_handler(self):
        """Handle processing audio frames and generating video.

        Accumulates audio frames and processes them in chunks to generate
        video frames using the EGSTalker model.
        """
        VAD_STOP_SECS = 0.5

        while True:
            try:
                # Wait for audio frames with timeout
                frame = await asyncio.wait_for(
                    self._audio_queue.get(), timeout=VAD_STOP_SECS
                )

                if self._is_interrupting:
                    logger.debug(f"{self}: Interrupting, breaking audio task")
                    break

                if isinstance(frame, TTSAudioRawFrame):
                    # Starting new inference
                    if self._event_id is None:
                        self._event_id = str(frame.id)
                        logger.info(f"{self}: Starting new utterance {self._event_id}")

                    # Resample audio to target sample rate if needed
                    if frame.sample_rate != self._audio_sample_rate:
                        audio_array = np.frombuffer(frame.audio, dtype=np.int16)
                        audio_float = audio_array.astype(np.float32) / 32768.0
                        audio_resampled = librosa.resample(
                            audio_float,
                            orig_sr=frame.sample_rate,
                            target_sr=self._audio_sample_rate,
                        )
                        audio_int16 = (audio_resampled * 32768.0).astype(np.int16)
                        self._audio_buffer.extend(audio_int16.tobytes())
                    else:
                        self._audio_buffer.extend(frame.audio)

                    # Process accumulated audio when we have enough
                    # EGSTalker renders at ~1000fps, so rendering is very fast
                    # Main latency comes from audio accumulation, not rendering
                    # Use smaller chunks (0.3s) for lower latency
                    min_chunk_size = int(self._audio_sample_rate * 0.3)  # 0.3 seconds for faster response
                    if len(self._audio_buffer) >= min_chunk_size * 2:
                        await self._process_audio_chunk()

                self._audio_queue.task_done()

            except asyncio.TimeoutError:
                # No audio received for VAD_STOP_SECS - finalize
                if self._event_id is not None:
                    logger.info(f"{self}: Timeout detected, finalizing utterance {self._event_id}")
                    await self._finalize_audio()
                    self._event_id = None

                    # Wait briefly for video generation to complete
                    # With 1000fps rendering, frames generate very quickly
                    # Main delay is scene loading (~50-100ms), so 0.5s is more than enough
                    await asyncio.sleep(0.5)

                    logger.info(f"{self}: ===== SPEECH FINALIZED - Setting _is_speaking = False =====")
                    self._is_speaking = False

    async def _process_audio_chunk(self):
        """Process accumulated audio and generate video frames.

        With EGSTalker's 1000fps capability, rendering is extremely fast.
        The latency comes mainly from audio I/O and scene loading.
        """
        if len(self._audio_buffer) == 0:
            return

        try:
            # Convert audio buffer to numpy array
            audio_array = np.frombuffer(bytes(self._audio_buffer), dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0

            duration_sec = len(audio_array) / self._audio_sample_rate

            # Calculate timestamp for this audio chunk (for video frame sync)
            # This chunk represents audio from _current_audio_chunk_timestamp
            if self._base_timestamp is not None:
                # Track where we are in the audio timeline
                # This will be used to timestamp video frames generated from this chunk
                # Note: We use audio_samples_pushed BEFORE this chunk to get the start time
                audio_samples_before_chunk = self._audio_samples_pushed - len(audio_array)
                self._current_audio_chunk_timestamp = self._base_timestamp + (
                    audio_samples_before_chunk / self._audio_sample_rate
                )
                logger.debug(
                    f"{self}: Processing audio chunk at timestamp {self._current_audio_chunk_timestamp:.3f}s"
                )

            # Save audio to temporary file for processing
            audio_file = os.path.join(self._temp_dir, f"audio_{self._event_id}.wav")
            import scipy.io.wavfile as wavfile

            wavfile.write(audio_file, self._audio_sample_rate, audio_array)

            # Generate video frames using EGSTalker
            # At 1000fps render speed, expect this to be very fast (~1ms per frame)
            # Use lock to prevent concurrent rendering with idle frames
            logger.info(f"{self}: Generating video for {duration_sec:.2f}s audio (render: ~1000fps)")
            logger.debug(f"{self}: SPEECH renderer waiting for render lock...")
            async with self._render_lock:
                logger.debug(f"{self}: SPEECH renderer acquired render lock")
                video_frames = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._generate_frames_from_audio,
                    audio_file,
                )
            logger.debug(f"{self}: SPEECH renderer released render lock")

            # Queue generated frames for playback with timestamps
            if video_frames is not None:
                for frame in video_frames:
                    # Tag frame with current audio chunk timestamp for A/V sync
                    # All frames from this chunk get the same base timestamp
                    frame_with_timestamp = (frame, self._current_audio_chunk_timestamp)
                    await self._video_frame_queue.put(frame_with_timestamp)
                    self._last_frame = frame
                logger.info(f"{self}: Queued {len(video_frames)} video frames with timestamp {self._current_audio_chunk_timestamp:.3f}s")

            # Clear the audio buffer
            self._audio_buffer.clear()

            # Clean up temporary audio file
            if os.path.exists(audio_file):
                os.remove(audio_file)

        except Exception as e:
            logger.error(f"{self}: Error processing audio chunk: {e}")
            import traceback

            traceback.print_exc()

    def _generate_frames_from_audio(self, audio_file: str) -> Optional[list]:
        """Generate video frames from audio file using EGSTalker model.

        Uses EGSTalker's render_from_batch to generate frames driven by audio.

        PERFORMANCE: EGSTalker achieves ~1000 FPS rendering speed on RTX 3090+,
        making it one of the fastest talking head rendering systems available.
        For 25fps output, this means rendering is ~40x real-time!

        Args:
            audio_file: Path to audio file

        Returns:
            List of video frames (numpy arrays in RGB format) or None on error
        """
        try:
            import torch
            from tqdm import tqdm

            logger.info(f"{self}: Rendering frames for audio: {audio_file} (expect ~1000fps render speed)")

            # Get custom camera views for this audio
            # EGSTalker requires reloading scene with custom audio
            custom_scene = self._scene.__class__(
                self._dataset_params,
                self._gaussians,
                load_iteration=self._iteration,
                shuffle=False,
                custom_aud=audio_file  # This tells Scene to load custom audio
            )

            # Get camera views (typically uses "custom" dataset with the new audio)
            views = custom_scene.getCustomCameras()

            if not views:
                logger.error(f"{self}: No camera views found in scene")
                return None

            frames = []

            # Render frames batch by batch
            with torch.no_grad():
                for idx in tqdm(range(0, len(views), self._batch_size), desc="Rendering"):
                    batch_views = views[idx : idx + self._batch_size]

                    for view in batch_views:
                        # Render this view with audio-driven deformation
                        rendering = self._render_from_batch(
                            view,
                            self._gaussians,
                            self._pipeline,
                            background=torch.tensor([1, 1, 1], dtype=torch.float32).cuda(),
                        )

                        # Extract RGB image from rendering
                        image_tensor = rendering["render"]  # Shape: (3, H, W)

                        # Convert tensor to numpy array in RGB format
                        image_np = self._tensor_to_image(image_tensor)

                        frames.append(image_np)

            logger.info(f"{self}: Generated {len(frames)} frames")
            return frames

        except Exception as e:
            logger.error(f"{self}: Error generating frames: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _tensor_to_image(self, tensor, normalize=True):
        """Convert PyTorch tensor to numpy image array.

        Args:
            tensor: PyTorch tensor of shape (3, H, W) or (H, W, 3)
            normalize: Whether to normalize to 0-255 range

        Returns:
            Numpy array in RGB format (H, W, 3) with uint8 dtype
        """
        # Convert to numpy
        if tensor.dim() == 3:
            if tensor.shape[0] == 3:  # (3, H, W) -> (H, W, 3)
                image = tensor.permute(1, 2, 0).cpu().numpy()
            else:  # (H, W, 3)
                image = tensor.cpu().numpy()
        else:
            image = tensor.cpu().numpy()

        if normalize:
            # Normalize to 0-255 range
            image = (image * 255).clip(0, 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

        return image

    async def _finalize_audio(self):
        """Process any remaining audio in the buffer."""
        if len(self._audio_buffer) > 0:
            logger.info(
                f"{self}: Finalizing audio processing, buffer has {len(self._audio_buffer)} bytes"
            )
            await self._process_audio_chunk()

    async def _idle_frame_generator(self):
        """Generate idle/neutral frames when not speaking.

        Similar to Ditto's approach, this generates actual frames driven by silent audio
        rather than just repeating the last frame. This creates natural idle animations
        with subtle breathing and micro-movements.
        """
        logger.info(f"{self}: Idle frame generator started (target: {self._target_fps} fps)")

        chunk_count = 0
        was_speaking = False

        try:
            # Generate idle frames using silent audio chunks
            # Each chunk is ~1 second of silence at target sample rate
            chunk_duration = 1.0  # 1 second chunks
            chunk_samples = int(self._audio_sample_rate * chunk_duration)

            while True:
                # Log state transitions for debugging
                if chunk_count % 10 == 0:
                    logger.debug(
                        f"{self}: Idle generator check - _is_speaking={self._is_speaking}, chunk={chunk_count}"
                    )

                # Only generate idle frames when not speaking
                if not self._is_speaking and self._initialized:
                    # Log transition from speaking to idle
                    if was_speaking:
                        logger.info(
                            f"{self}: ===== Transitioning to IDLE mode - generating silent frames ====="
                        )
                        was_speaking = False

                    # Don't generate more idle frames if queue is getting too full
                    # This prevents idle frames from building up during speech transitions
                    queue_size = self._video_frame_queue.qsize()
                    if queue_size > 20:
                        logger.debug(
                            f"{self}: Video queue has {queue_size} frames, skipping idle generation"
                        )
                        await asyncio.sleep(chunk_duration)
                        continue

                    try:
                        # Generate silent audio chunk
                        silent_audio = np.zeros(chunk_samples, dtype=np.int16)

                        # Save to temporary file
                        idle_audio_file = os.path.join(
                            self._temp_dir, f"idle_audio_{chunk_count}.wav"
                        )
                        import scipy.io.wavfile as wavfile

                        wavfile.write(idle_audio_file, self._audio_sample_rate, silent_audio)

                        # Generate idle frames using EGSTalker with silent audio
                        # This creates natural neutral/breathing animations
                        # Use lock to prevent concurrent rendering with speech frames
                        logger.debug(f"{self}: IDLE generator waiting for render lock (chunk {chunk_count})")
                        async with self._render_lock:
                            logger.debug(f"{self}: IDLE generator acquired render lock")
                            idle_frames = await asyncio.get_event_loop().run_in_executor(
                                None, self._generate_idle_frames, idle_audio_file
                            )
                        logger.debug(f"{self}: IDLE generator released render lock")

                        # Queue generated idle frames with timestamps
                        if idle_frames is not None and len(idle_frames) > 0:
                            # Idle frames get current time as timestamp
                            current_time = asyncio.get_event_loop().time()
                            for frame in idle_frames:
                                frame_with_timestamp = (frame, current_time)
                                await self._video_frame_queue.put(frame_with_timestamp)
                                self._last_frame = frame

                            if chunk_count == 0:
                                logger.info(
                                    f"{self}: Started generating idle frames ({len(idle_frames)} frames per chunk)"
                                )
                            elif chunk_count % 10 == 0:
                                logger.debug(
                                    f"{self}: Generated {chunk_count} idle chunks ({chunk_count * len(idle_frames)} frames)"
                                )

                        # Clean up temporary file
                        if os.path.exists(idle_audio_file):
                            os.remove(idle_audio_file)

                        chunk_count += 1

                    except Exception as e:
                        logger.error(f"{self}: Error generating idle frame: {e}")
                        import traceback

                        traceback.print_exc()
                        # On error, fall back to repeating last frame with current timestamp
                        if self._last_frame is not None and queue_size < 10:
                            current_time = asyncio.get_event_loop().time()
                            frame_with_timestamp = (self._last_frame.copy(), current_time)
                            await self._video_frame_queue.put(frame_with_timestamp)
                else:
                    # Currently speaking - track state for transition logging
                    if not was_speaking:
                        logger.info(
                            f"{self}: ===== IDLE mode paused - bot is speaking (_is_speaking = {self._is_speaking}) ====="
                        )
                        was_speaking = True

                # Sleep for the chunk duration
                await asyncio.sleep(chunk_duration)

        except asyncio.CancelledError:
            logger.info(f"{self}: Idle frame generator stopped (generated {chunk_count} idle chunks)")
        except Exception as e:
            logger.error(f"{self}: Error in idle frame generator: {e}")
            import traceback

            traceback.print_exc()

    def _generate_idle_frames(self, audio_file: str) -> Optional[list]:
        """Generate idle frames from silent audio (for breathing/neutral animations).

        Uses the same rendering pipeline as speech but with silent audio,
        creating natural idle animations instead of static frames.

        Args:
            audio_file: Path to silent audio file

        Returns:
            List of video frames (numpy arrays in RGB format) or None on error
        """
        try:
            import torch

            logger.debug(f"{self}: Rendering idle frames for: {audio_file}")

            # Get custom camera views for silent audio
            custom_scene = self._scene.__class__(
                self._dataset_params,
                self._gaussians,
                load_iteration=self._iteration,
                shuffle=False,
                custom_aud=audio_file,
            )

            # Get camera views
            views = custom_scene.getCustomCameras()

            if not views:
                logger.warning(f"{self}: No camera views found for idle frames")
                return None

            frames = []

            # Render frames without progress bar (to avoid cluttering logs)
            with torch.no_grad():
                for idx in range(0, len(views), self._batch_size):
                    batch_views = views[idx : idx + self._batch_size]

                    for view in batch_views:
                        # Render idle frame
                        rendering = self._render_from_batch(
                            view,
                            self._gaussians,
                            self._pipeline,
                            background=torch.tensor([1, 1, 1], dtype=torch.float32).cuda(),
                        )

                        # Extract RGB image
                        image_tensor = rendering["render"]
                        image_np = self._tensor_to_image(image_tensor)
                        frames.append(image_np)

            return frames

        except Exception as e:
            logger.error(f"{self}: Error generating idle frames: {e}")
            return None

    async def _consume_and_push_video(self):
        """Consume video frames from queue and push downstream with timestamps.

        Pushes video frames with audio-derived timestamps for perfect A/V sync.
        The transport layer (Daily) uses these timestamps to synchronize playback.
        """
        logger.info(f"{self}: Video playback task started (timestamp-based A/V sync)")

        playback_start_time = None
        frame_interval = 1.0 / self._target_fps
        last_logged_drift = 0

        try:
            while True:
                try:
                    frame_data = await asyncio.wait_for(
                        self._video_frame_queue.get(), timeout=0.1
                    )

                    if frame_data is None:
                        break

                    # Unpack frame and its audio-derived timestamp
                    try:
                        frame, audio_timestamp = frame_data
                        logger.debug(f"{self}: Unpacked frame with timestamp {audio_timestamp:.3f}s")
                    except (TypeError, ValueError) as e:
                        logger.error(f"{self}: Failed to unpack frame_data: {e}, type: {type(frame_data)}")
                        # Fallback: if unpacking fails, treat as raw frame
                        if isinstance(frame_data, np.ndarray):
                            frame = frame_data
                            audio_timestamp = asyncio.get_event_loop().time()
                            logger.warning(f"{self}: Got raw frame instead of tuple, using current time")
                        else:
                            logger.error(f"{self}: Unexpected frame_data format, skipping")
                            continue

                    # Set initial timing on first frame for drift tracking
                    if playback_start_time is None:
                        playback_start_time = asyncio.get_event_loop().time()
                        logger.info(f"{self}: Starting video playback at {playback_start_time:.3f}s")

                    self._frame_count += 1

                    # Save frame if directory specified
                    if self._save_frames_dir and self._frame_count % 10 == 0:
                        frame_path = os.path.join(
                            self._save_frames_dir, f"frame_{self._frame_count:06d}.png"
                        )
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(frame_path, frame_bgr)

                    # Drift correction: Check if we're drifting from audio timeline
                    if self._base_timestamp is not None:
                        expected_playback_time = self._frame_count * frame_interval
                        actual_audio_time = audio_timestamp - self._base_timestamp
                        drift = expected_playback_time - actual_audio_time

                        # Log significant drift (every 1 second of drift change)
                        if abs(drift - last_logged_drift) > 1.0:
                            logger.info(
                                f"{self}: A/V drift: {drift*1000:.0f}ms (expected: {expected_playback_time:.2f}s, audio: {actual_audio_time:.2f}s)"
                            )
                            last_logged_drift = drift

                    # Create output frame with PTS timestamp for Daily sync
                    output_frame = OutputImageRawFrame(
                        image=frame.tobytes(),
                        size=(frame.shape[1], frame.shape[0]),
                        format="RGB",
                    )

                    # Add audio-derived timestamp (PTS) for transport-level sync
                    if hasattr(output_frame, "pts"):
                        output_frame.pts = audio_timestamp

                    # Push video frame with timestamp for Daily synchronization
                    await self.push_frame(output_frame)

                    if self._frame_count == 1:
                        logger.info(f"{self}: 🎥 First frame pushed with timestamp {audio_timestamp:.3f}s!")
                    elif self._frame_count % 50 == 0:
                        logger.info(f"{self}: Pushed {self._frame_count} video frames")

                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"{self}: Error pushing video frame: {e}")
                    break

        except asyncio.CancelledError:
            logger.info(f"{self}: Video playback task cancelled")
        except Exception as e:
            logger.error(f"{self}: Error in video playback: {e}")
        finally:
            logger.info(f"{self}: Video playback finished (pushed {self._frame_count} frames)")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames from the pipeline."""
        if isinstance(frame, StartFrame):
            await self.start(frame)

        elif isinstance(frame, TTSStartedFrame):
            logger.info(f"{self}: ===== TTS STARTED - Setting _is_speaking = True =====")
            self._is_interrupting = False
            self._is_speaking = True
            self._audio_buffer.clear()

            # Reset timestamp tracking for new utterance
            self._base_timestamp = None
            self._audio_samples_pushed = 0
            self._video_frames_pushed = 0
            self._current_audio_chunk_timestamp = 0

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

        elif isinstance(frame, TTSAudioRawFrame):
            # Defensive: Set speaking flag if not already set (in case TTSStartedFrame was missed)
            if not self._is_speaking:
                logger.warning(
                    f"{self}: ===== TTS AUDIO RECEIVED - Setting _is_speaking = True (fallback) ====="
                )
                self._is_speaking = True
                self._is_interrupting = False

                # Reset timestamp tracking (fallback)
                self._base_timestamp = None
                self._audio_samples_pushed = 0
                self._video_frames_pushed = 0
                self._current_audio_chunk_timestamp = 0

                # Clear idle frames (fallback)
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
                # Track timestamp for audio-video sync
                if self._base_timestamp is None:
                    self._base_timestamp = asyncio.get_event_loop().time()
                    logger.info(f"{self}: Base timestamp set: {self._base_timestamp}")

                # Queue audio for processing by EGSTalker (generates video)
                await self._audio_queue.put(frame)

                # Push audio downstream to Daily for synchronized playback
                logger.debug(f"{self}: Pushing TTS audio to Daily ({len(frame.audio)} bytes)")
                await self.push_frame(frame, direction)

                # Update audio sample counter for timestamp tracking
                num_samples = len(frame.audio) // 2  # 16-bit = 2 bytes per sample
                self._audio_samples_pushed += num_samples

                return  # Don't push again at the end

        elif isinstance(frame, TTSStoppedFrame):
            logger.debug(f"{self}: TTS stopped, waiting for audio buffer to drain")

        elif isinstance(frame, (InterruptionFrame, UserStartedSpeakingFrame)):
            logger.info(f"{self}: Interruption detected, clearing state")
            await self._handle_interruption()

        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self.stop(frame)

        # Call parent class
        await super().process_frame(frame, direction)

        # Push frame downstream
        await self.push_frame(frame, direction)

    async def _handle_interruption(self):
        """Handle user interruption."""
        self._is_interrupting = True
        self._audio_buffer.clear()
        self._event_id = None

        # Reset speaking flag so idle frames can resume
        logger.info(f"{self}: ===== INTERRUPTION - Setting _is_speaking = False =====")
        self._is_speaking = False

        # Reset timestamp tracking
        self._base_timestamp = None
        self._audio_samples_pushed = 0
        self._video_frames_pushed = 0
        self._current_audio_chunk_timestamp = 0

        # Cancel and restart audio task
        await self._cancel_audio_task()
        self._is_interrupting = False
        await self._create_audio_task()

        # Clear video queue
        cleared_count = 0
        while not self._video_frame_queue.empty():
            try:
                self._video_frame_queue.get_nowait()
                cleared_count += 1
            except asyncio.QueueEmpty:
                break
        if cleared_count > 0:
            logger.info(f"{self}: Cleared {cleared_count} frames from video queue on interruption")
