# Ditto Talking Head Service

A pipecat service for generating realistic talking head videos using the [Ditto](https://github.com/antgroup/ditto-talkinghead) model.

## Overview

The Ditto Talking Head Service processes audio frames and generates synchronized video output showing a realistic talking head avatar. It's ideal for creating engaging video conversations with AI assistants.

## Features

- Real-time talking head video generation
- Synchronizes video with TTS audio output
- Supports local inference (no API calls required)
- Integrates seamlessly with pipecat pipelines

## Requirements

### Hardware
- CUDA-capable GPU (A100 or similar recommended)
- Minimum 24GB VRAM

### Software
```bash
# Install Ditto
git clone https://github.com/antgroup/ditto-talkinghead
cd ditto-talkinghead
conda env create -f environment.yaml
conda activate ditto

# Download model checkpoints from HuggingFace
# Follow instructions in the Ditto repository

# Install additional dependencies for pipecat
pip install opencv-python librosa
```

## Installation

1. Clone and install Ditto following the [official instructions](https://github.com/antgroup/ditto-talkinghead)
2. Download the model checkpoints
3. **Prepare a source image for your talking head**:
   - Format: PNG or JPG
   - Content: A clear front-facing photo of a person's face
   - Recommended: 512x512 or higher resolution
   - This is the face that will be animated to create the talking head
   - Example: `./my_avatar.png`

## Usage

```python
from pipecat.services.ditto import DittoTalkingHeadService

# Initialize the service
ditto = DittoTalkingHeadService(
    ditto_path="./ditto-talkinghead",
    data_root="./checkpoints/ditto_trt_Ampere_Plus",
    cfg_pkl="./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl",
    source_image_path="./my_avatar.png",  # ← Your avatar face image
    chunk_size=(3, 5, 2),  # Latency/quality tradeoff
)

# Use in a pipeline
pipeline = Pipeline([
    transport.input(),
    stt,
    llm,
    tts,  # Generate audio
    ditto,  # Generate video from audio + source image
    transport.output(),
])
```

### How the Source Image Works

**Image Input Flow**:
```
1. Service initialization:
   source_image_path="./my_avatar.png" → stored in service

2. Pipeline starts (StartFrame):
   → SDK.setup(source_path="./my_avatar.png")
   → SDK loads and processes the image
   → Avatar is ready for animation

3. Audio arrives (TTSAudioRawFrame):
   → SDK.run_chunk(audio) uses the loaded avatar
   → Animates the same face for entire session

4. Video output (OutputImageRawFrame):
   → Talking head video with your avatar's face
```

**Key Points**:
- **Static per session**: One avatar image per service instance
- **Loaded once**: Image is loaded during SDK setup, not per frame
- **File path**: Must be a valid local file path (not URL or bytes)
- **Reusable**: Same image used for all audio chunks in the session
- **Can't change mid-session**: To use different avatar, create new service instance

## Configuration

### Environment Variables

You can configure the service using environment variables:

```bash
export DITTO_PATH="./ditto-talkinghead"
export DITTO_DATA_ROOT="./checkpoints/ditto_trt_Ampere_Plus"
export DITTO_CFG_PKL="./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl"
export DITTO_SOURCE_IMAGE="./avatar.png"
```

### Parameters

- `ditto_path`: Path to Ditto installation directory
- `data_root`: Path to model checkpoints directory
- `cfg_pkl`: Path to config file
- `source_image_path`: Path to source image (the face for the talking head)
- `output_fps`: Video frame rate (default: 25)
- `sample_rate`: Audio sample rate (default: 24000)

## Example

See `examples/simple/ditto_example.py` for a complete working example that combines:
- Deepgram STT for speech recognition
- OpenAI LLM for conversation
- Higgs TTS for speech synthesis
- Ditto for talking head video generation
- Daily transport for real-time communication

## How It Works

### Streaming Architecture

**Frame-Based Accumulation**:

Audio arrives in the pipeline as individual `TTSAudioRawFrame` objects (typically 20-50ms chunks from TTS). The service accumulates multiple frames before processing:

```
Frame 1 (20ms) → buffer += 320 samples  → wait (not enough yet)
Frame 2 (20ms) → buffer += 320 samples  → wait
...
Frame 10+ → buffer >= 6480 samples → PROCESS chunk(s)
```

**Processing Flow**:

1. **Audio Input**: Service receives `TTSAudioRawFrame` frames (audio passes through immediately for playback)
2. **Accumulation**: Each frame adds samples to buffer; process when buffer >= window size
3. **Resampling**: Audio is automatically resampled to 16kHz (Ditto requirement)
4. **Overlapping Windows**: Accumulated audio is processed in overlapping chunks:
   - Window size: `sum(chunk_size) * 640` samples (e.g., 6400 for (3,5,2))
   - Stride: `chunk_size[1] * 640` samples (e.g., 3200 for (3,5,2))
   - Creates 50% overlap for smooth motion continuity
4. **Inference**: Ditto's `StreamSDK.run_chunk()` generates video frames in real-time
   - Maintains transformer KV caches across chunks for temporal consistency
   - Applies overlap-add blending to avoid boundary artifacts
   - Processes incrementally: audio→motion→rendering stages
5. **Frame Extraction**: Background task extracts frames from SDK's internal `writer_queue`
6. **Playback**: Video frames pushed as `OutputImageRawFrame` at 25fps
7. **Display**: Transport shows synchronized talking head video

### Key Implementation Details

**Overlapping Segments**:
- Prevents discontinuities between chunks
- SDK maintains hidden state (KV caches) for smooth transitions
- Overlap ratio matches model training (typically 50%)

**State Management**:
- SDK automatically manages transformer caches across segments
- Interruptions trigger state reset
- Long sessions maintain continuity without drift

**Timing Alignment**:
- `chunk_size` must match model configuration
- Mismatched timing causes lip-sync drift
- Default (3,5,2) = ~200ms latency with stable motion

## Performance Notes

- Video generation happens in real-time as audio chunks arrive (streaming mode)
- Audio plays immediately while video generates in background
- Processing latency depends on chunk size and GPU performance
- **Use TensorRT backend for best real-time performance** (recommended)
- Use an "online" config file (e.g., `v0.4_hubert_cfg_trt_online.pkl`) for streaming mode
- Video frames are pushed at 25fps to maintain smooth playback
- Chunk size `(3, 5, 2)` = ~200ms latency (tunable)

## Troubleshooting

### ⚠️ Critical Issues

**Lip-sync drift / timing issues**:
- **Cause**: `chunk_size` parameter doesn't match model configuration
- **Solution**: Use the exact `chunk_size` from your model's training config
- **Default**: (3, 5, 2) works for most Ditto models
- **Symptom**: Video mouth movements don't align with audio over time

**Motion discontinuities / jitter**:
- **Cause**: State cache not being maintained properly
- **Solution**:
  - Ensure you're using an "online" config file (e.g., `v0.4_hubert_cfg_trt_online.pkl`)
  - Avoid manual SDK resets between chunks
  - Let interruptions complete cleanly
- **Symptom**: Sudden jumps or stutters in head/lip motion

**Audio buffering gaps**:
- **Cause**: Gaps in audio stream or incorrect resampling
- **Solution**:
  - Ensure continuous audio flow from TTS
  - Verify 16kHz resampling is working correctly
  - Check `librosa` is installed
- **Symptom**: Video freezes or desync with audio

**Poor quality with low latency**:
- **Cause**: Chunk size too small (current frames < 5)
- **Solution**: Increase `chunk_size[1]` (current frames)
- **Tradeoff**: Larger chunks = higher latency but more stable motion
- **Recommended**: chunk_size=(3, 5, 2) balances latency (~200ms) and quality

**Overlap artifacts / blending issues**:
- **Cause**: Overlap ratio mismatch with training
- **Solution**: SDK handles this automatically - ensure using correct config file
- **Note**: Don't modify `chunk_size` ratio without model retraining

### Common Issues

**"Ditto inference.py not found"**:
Make sure the `ditto_path` points to the correct Ditto installation directory.

**"Source image not found"**:
Verify that the `source_image_path` points to a valid image file (PNG or JPG).

**"CUDA out of memory"**:
Reduce the batch size or use a smaller model variant. Ensure you have at least 24GB VRAM.

**Slow performance**:
- Use TensorRT backend instead of PyTorch for faster inference
- Ensure GPU drivers and CUDA are properly installed
- Check that the model is loading on GPU, not CPU
- Use an "online" config file optimized for streaming

**Video lags behind audio**:
- This is expected with chunk-based processing (~200ms latency)
- Reduce `chunk_size[1]` for lower latency (but less stable motion)
- Optimize GPU performance with TensorRT backend

## License

This service is provided under the BSD 2-Clause License (matching pipecat).
The Ditto model itself is licensed under Apache-2.0 (see the [Ditto repository](https://github.com/antgroup/ditto-talkinghead)).

## Credits

- Ditto model by Ant Group
- Paper: "Motion-Space Diffusion for Controllable Realtime Talking Head Synthesis" (ACM MM 2025)
