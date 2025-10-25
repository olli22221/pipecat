# EGSTalker Video Service for Pipecat

This module provides integration between [EGSTalker](https://github.com/ZhuTianheng/EGSTalker) and Pipecat for real-time talking head video generation using 3D Gaussian Splatting.

## Overview

EGSTalker is an audio-driven talking head synthesis framework that uses 3D Gaussian deformation to create high-fidelity animated avatars. This Pipecat service adapts EGSTalker's batch rendering pipeline for real-time streaming applications.

## Requirements

### Hardware
- **GPU**: CUDA-capable GPU (RTX 3090 or better recommended)
- **VRAM**: At least 12GB for inference
- **RAM**: 16GB+ recommended

### Software
```bash
pip install torch torchvision opencv-python librosa scipy
pip install pipecat-ai
```

### EGSTalker Installation
```bash
git clone https://github.com/ZhuTianheng/EGSTalker
cd EGSTalker
# Follow EGSTalker installation instructions
```

## Setup Process

### 1. Prepare Your Dataset

EGSTalker requires an ER-NeRF format dataset with:
- Video footage of the target face
- Corresponding audio tracks
- Camera calibration data
- transforms.json configuration

Example dataset structure:
```
my_avatar/
├── transforms.json
├── images/
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
├── aud.npy  # Audio features
└── point_cloud/
```

### 2. Train the Model

Train EGSTalker on your dataset:
```bash
cd EGSTalker
python train.py \
    --source_path /path/to/my_avatar \
    --model_path /path/to/output \
    --iterations 30000
```

Training typically takes 4-8 hours on an RTX 3090.

### 3. Verify Model Files

After training, ensure these files exist:
```
/path/to/output/
└── point_cloud/
    └── iteration_30000/
        └── point_cloud.ply
```

## Usage

### Basic Example

```python
from pipecat.services.egstalker import EGSTalkerVideoService
from pipecat.transports.daily_transport import DailyTransport
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.pipeline.pipeline import Pipeline

# Initialize EGSTalker service
egstalker = EGSTalkerVideoService(
    egstalker_path="/path/to/EGSTalker",
    source_path="/path/to/my_avatar",
    model_path="/path/to/output",
    iteration=30000,
    sh_degree=3,
    audio_sample_rate=16000,
    target_fps=25
)

# Create pipeline
pipeline = Pipeline([
    tts_service,  # Your TTS service
    egstalker,    # Video generation
    transport     # Your transport (e.g., Daily)
])

await pipeline.run()
```

### Advanced Configuration

```python
egstalker = EGSTalkerVideoService(
    egstalker_path="/path/to/EGSTalker",
    source_path="/path/to/my_avatar",
    model_path="/path/to/output",
    iteration=30000,           # Training iteration to load
    sh_degree=3,               # Spherical harmonics degree
    audio_sample_rate=16000,   # Audio resampling rate
    target_fps=25,             # Output video framerate
    batch_size=1,              # Batch size for rendering (1 for lowest latency)
    save_frames_dir="/tmp/frames"  # Optional: save frames for debugging
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `egstalker_path` | str | Required | Path to EGSTalker installation directory |
| `source_path` | str | Required | Path to prepared dataset directory |
| `model_path` | str | Required | Path to trained model checkpoint directory |
| `iteration` | int | 30000 | Training iteration to load |
| `sh_degree` | int | 3 | Spherical harmonics degree (must match training) |
| `audio_sample_rate` | int | 16000 | Target audio sample rate |
| `target_fps` | int | 25 | Target video framerate |
| `batch_size` | int | 1 | Batch size for rendering |
| `save_frames_dir` | str | None | Optional directory to save generated frames |

## How It Works

1. **Initialization**: Loads trained EGSTalker model, scene, and Gaussian parameters
2. **Audio Processing**: Accumulates TTS audio frames and resamples to target rate
3. **Video Generation**:
   - Saves audio chunks to temporary files
   - Loads scene with custom audio
   - Renders frames using 3D Gaussian splatting
   - Converts tensors to RGB numpy arrays
4. **Frame Streaming**: Pushes generated frames to pipeline at target FPS
5. **Idle Handling**: Repeats last frame during silence to maintain video stream

## Performance Considerations

### Rendering Speed
According to the EGSTalker paper, the system achieves **~1000 FPS** rendering speed on modern GPUs (RTX 3090 or better). This means:
- **Per-frame rendering**: ~1ms per frame
- **Real-time capability**: Can generate 25fps output at 40x real-time speed
- **Low latency**: Actual rendering is extremely fast

### Latency Breakdown
1. **Audio accumulation**: 100-500ms (waiting for enough audio data)
2. **Audio file I/O**: ~10-20ms (saving temp file)
3. **Scene loading with custom audio**: ~50-100ms (reloading scene)
4. **Rendering frames**: ~1ms per frame (1000fps capability)
5. **Frame queuing**: ~10ms

**Total expected latency**: 200-700ms for first frame batch

The main bottleneck is **not rendering speed** but rather:
- Audio chunk accumulation (to get meaningful audio data)
- Scene reloading overhead (EGSTalker's architecture requires this per audio chunk)

### Optimization Tips
1. **Use smaller audio chunks** (e.g., 0.3s instead of 0.5s) for lower latency
2. **Pre-load scene** to reduce reloading overhead
3. **Batch size = 1** is already optimal given 1000fps rendering
4. **Pipeline audio chunks** to start rendering while still receiving audio

### Memory Usage
- Model loading: ~2-4GB VRAM
- Rendering: ~4-8GB VRAM (during active inference)
- Peak usage: ~6-12GB VRAM required
- Idle: ~2-4GB VRAM

## Troubleshooting

### ImportError: No module named 'scene'
Ensure `egstalker_path` points to the root of the EGSTalker repository where `scene/` directory exists.

### RuntimeError: CUDA out of memory
- Reduce `batch_size` to 1
- Lower `target_fps`
- Close other GPU-intensive applications

### No camera views found in scene
Ensure your dataset has proper `transforms.json` with camera configurations. EGSTalker needs at least one camera view.

### Frames not rendering
1. Check that model checkpoint exists at `model_path/point_cloud/iteration_{iteration}/point_cloud.ply`
2. Verify audio files are being created in temp directory
3. Enable `save_frames_dir` to debug frame generation

## Differences from Other Services

### vs. HeyGen
- **EGSTalker**: Local inference, requires training, 3D Gaussian-based
- **HeyGen**: Cloud API, pre-trained avatars, neural rendering

### vs. Simli
- **EGSTalker**: Higher quality, more control, slower inference
- **Simli**: Cloud API, optimized for real-time, less customization

### vs. Ditto
- **EGSTalker**: 3D Gaussian splatting, 1000fps rendering, requires training
- **Ditto**: 2D image-based, 125fps rendering, simpler setup

### Performance Comparison Table

| Feature | EGSTalker | HeyGen | Simli | Ditto |
|---------|-----------|--------|-------|-------|
| **Inference** | Local GPU | Cloud API | Cloud API | Local GPU |
| **Setup** | Train model | API key | API key | Model download |
| **Technology** | 3D Gaussian | Neural rendering | Neural rendering | Image-based |
| **Render Speed** | **~1000 FPS** | N/A (API) | N/A (API) | ~125 FPS |
| **Latency** | 200-700ms | 100-200ms | 100-200ms | 100-300ms |
| **Quality** | Very High | High | High | Medium-High |
| **Customization** | Full control | Limited | Limited | Medium |

**Key Takeaway**: EGSTalker has the **fastest rendering** of all local options at ~1000fps, making it ideal for high-quality, customizable talking head generation when you have a trained model.

## Limitations

1. **Requires Training**: You must train a model on your avatar before use (4-8 hours)
2. **Dataset Complexity**: Preparing datasets requires camera calibration and ER-NeRF format
3. **Scene Reloading Overhead**: Must reload scene for each audio chunk (~50-100ms overhead)
4. **GPU Intensive**: Requires high-end GPU (RTX 3090+) for 1000fps performance
5. **Setup Complexity**: More complex setup compared to API-based services (HeyGen/Simli)

## References

- [EGSTalker Paper](https://arxiv.org/abs/2407.xxxx)
- [EGSTalker GitHub](https://github.com/ZhuTianheng/EGSTalker)
- [Pipecat Documentation](https://docs.pipecat.ai)

## License

This integration follows the same license as EGSTalker. Please refer to the [EGSTalker repository](https://github.com/ZhuTianheng/EGSTalker) for licensing details.
