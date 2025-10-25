# Daily Transport with PTS Support

This is a modified version of the Daily transport that includes PTS (Presentation Timestamp) support for both audio and video frames. It provides improved audio-video synchronization for use cases like talking head generation.

## What's Different?

The standard Daily transport writes audio and video frames immediately without considering their timestamps. This can cause audio-video desynchronization when frames are generated with specific timing requirements.

This PTS-enabled version:
1. Checks if audio/video frames have a `pts` (Presentation Timestamp) attribute
2. Paces frame delivery using `asyncio.sleep()` to write frames at the correct wall-clock time
3. Maintains proper audio-video synchronization even when Daily's API doesn't natively support PTS
4. Independently paces audio and video based on their own PTS timelines

## When to Use

Use this transport when:
- You're generating audio and video frames synchronized together (e.g., DittoTalkingHeadService)
- Your audio and/or video frames have PTS timestamps
- You need precise audio-video synchronization
- You're experiencing audio-video desynchronization with the standard Daily transport
- You want frames pushed at exact timestamps rather than immediately

## Usage

Replace the standard Daily transport import:

```python
# Before
from pipecat.transports.daily import DailyTransport, DailyParams

# After - use the PTS-enabled version
from pipecat.transports.daily_wpts.transport import DailyTransport, DailyParams
from pipecat.transports.daily_wpts.utils import DailyRESTHelper, DailyRoomParams
```

The API is identical to the standard Daily transport. The PTS pacing happens automatically when your video frames have a `pts` attribute.

## Example with Ditto

```python
from pipecat.services.ditto import DittoTalkingHeadService
from pipecat.transports.daily_wpts.transport import DailyTransport, DailyParams

# Initialize Ditto service (generates timestamped video frames)
ditto = DittoTalkingHeadService(
    ditto_path="./ditto-talkinghead",
    data_root="./checkpoints/ditto_trt_Ampere_Plus",
    cfg_pkl="./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch_online.pkl",
    source_image_path="./avatar.png",
)

# Use PTS-enabled Daily transport
transport = DailyTransport(
    room_url=room_url,
    token=token,
    bot_name="Ditto Bot",
    params=DailyParams(
        audio_out_enabled=True,
        audio_out_sample_rate=24000,
        video_out_enabled=True,
        video_out_width=1440,
        video_out_height=1920,
        video_out_framerate=30,
    ),
)

# Build pipeline: TTS -> Ditto -> Transport
# Ditto adds PTS timestamps to video frames
# PTS-enabled transport paces frames for proper A/V sync
pipeline = Pipeline([
    transport.input(),
    stt,
    llm,
    tts,
    ditto,  # Generates video frames with PTS timestamps
    transport.output(),  # PTS-based pacing ensures proper timing
])
```

## How It Works

### Without PTS Support (Standard Daily Transport)
```
Audio → Push immediately
Video → Push immediately
Result: Frames may arrive out of sync if generated at different times
```

### With PTS Support (This Transport)
```
Audio → Generate with PTS → Calculate target time → Wait → Push at correct time
Video → Generate with PTS → Calculate target time → Wait → Push at correct time
Result: Both pushed at exact PTS timestamps → Perfect synchronization
```

### Implementation Details

The transport tracks PTS for both audio and video independently:

**For Video:**
- `_video_pts_base_time`: Wall-clock time when the first video frame with PTS arrives
- `_video_pts_base`: PTS value of the first video frame
- `_video_frame_write_lock`: Ensures serial video frame writes with proper pacing

**For Audio:**
- `_audio_pts_base_time`: Wall-clock time when the first audio frame with PTS arrives
- `_audio_pts_base`: PTS value of the first audio frame
- `_audio_frame_write_lock`: Ensures serial audio frame writes with proper pacing

For each frame (audio or video) with a PTS:
1. Calculate PTS offset: `pts_offset = frame.pts - base_pts`
2. Calculate target write time: `target_time = base_time + pts_offset`
3. Calculate wait time: `wait = target_time - current_time`
4. If early, sleep until target time: `await asyncio.sleep(wait)`
5. Write frame to Daily

This ensures both audio and video frames are delivered at the exact wall-clock time corresponding to their PTS timestamps, maintaining perfect synchronization.

## Performance

PTS-based pacing adds minimal overhead:
- No additional processing of frame data
- Only timing calculations and async sleep calls
- Frames without PTS are written immediately (backwards compatible)
- Separate locks for audio and video prevent race conditions without blocking each other
- Audio and video can be paced independently in parallel

## Compatibility

- Fully compatible with the standard Daily transport API
- Works with all Daily features (dial-in/out, recording, transcription, etc.)
- Backwards compatible: frames without PTS are written immediately
- Can be used as a drop-in replacement for the standard transport

## Limitations

- PTS pacing only works when frames have a `pts` attribute
- Daily's audio/video APIs don't natively support PTS, so we simulate it with timing
- Very high frame rates (>60fps video or >100Hz audio) may experience slight timing drift
- System clock accuracy affects synchronization precision
- Audio and video use independent PTS timelines (each starts from their first frame with PTS)
