# Simli Video + Higgs Audio TTS Example

This example demonstrates how to create a video AI bot using:
- **Simli** for AI-generated video avatars
- **Higgs Audio** for natural text-to-speech
- **Deepgram** for speech-to-text
- **OpenAI** for conversational AI

## Prerequisites

### 1. Install Dependencies

```bash
pip install pipecat-ai[simli,higgs,deepgram,openai,daily]
```

### 2. Required API Keys

Create a `.env` file with the following keys:

```env
# Simli Configuration
SIMLI_API_KEY=your_simli_api_key
SIMLI_FACE_ID=your_simli_face_id

# STT Service
DEEPGRAM_API_KEY=your_deepgram_api_key

# LLM Service
OPENAI_API_KEY=your_openai_api_key

# Transport (Daily or WebRTC)
DAILY_API_KEY=your_daily_api_key  # If using Daily transport
```

### 3. Get Your API Keys

- **Simli**: Sign up at [simli.com](https://simli.com) to get your API key and Face ID
- **Deepgram**: Get your API key from [deepgram.com](https://deepgram.com)
- **OpenAI**: Get your API key from [platform.openai.com](https://platform.openai.com)
- **Daily**: Get your API key from [daily.co](https://daily.co) (if using Daily transport)

## Running the Example

### Option 1: Using Daily Transport

```bash
python simli-higgs-example.py --transport daily
```

This will create a Daily room and provide you with a URL to join.

### Option 2: Using WebRTC Transport

```bash
python simli-higgs-example.py --transport webrtc
```

This uses WebRTC for local connections.

## How It Works

The pipeline flows as follows:

1. **User speaks** → Audio captured via transport
2. **Deepgram STT** → Converts speech to text
3. **OpenAI LLM** → Generates intelligent response
4. **Higgs Audio TTS** → Converts text to natural speech (24kHz audio)
5. **Simli Video** → Generates video avatar from audio
6. **Transport Output** → Sends video + audio to user

## Key Features

- **Natural Speech**: Higgs Audio provides high-quality, natural-sounding TTS
- **Video Avatar**: Simli creates realistic video from the generated audio
- **Smart Turn-Taking**: Local smart turn analyzer for natural conversations
- **VAD**: Silero VAD for detecting when user starts/stops speaking
- **Metrics**: Enabled for monitoring performance

## Configuration Options

### Higgs Audio TTS

```python
tts = HiggsAudioTTSService(
    model_path="bosonai/higgs-audio-v2-generation-3B-base",
    audio_tokenizer_path="bosonai/higgs-audio-v2-tokenizer",
    temperature=0.3,  # Lower = more consistent, Higher = more expressive
)
```

### Simli Video

```python
simli_ai = SimliVideoService(
    SimliConfig(
        api_key=os.getenv("SIMLI_API_KEY"),
        face_id=os.getenv("SIMLI_FACE_ID")
    ),
)
```

### Video Output Settings

The example is configured for 512x512 video output, which works well with Simli:

```python
video_out_width=512,
video_out_height=512,
```

## Troubleshooting

### Issue: Higgs model not found
Higgs Audio will automatically download models on first run. Ensure you have:
- Sufficient disk space (~3GB for models)
- Internet connection for initial download
- Proper permissions to write to cache directory

### Issue: Simli video not appearing
Check that:
- Your SIMLI_API_KEY and SIMLI_FACE_ID are correct
- Video output is enabled in transport params
- Video dimensions are set to 512x512

### Issue: Audio quality issues
- Higgs Audio works best with 24kHz sample rate (already configured)
- Check your network connection quality
- Ensure sufficient system resources (GPU recommended for Higgs)

## Performance Notes

- **GPU**: Higgs Audio TTS runs much faster with CUDA-enabled GPU
- **Memory**: Expect ~4GB RAM usage for Higgs models
- **Latency**: First request may be slower due to model loading

## Customization

### Change the Avatar Voice/Style
Modify the Higgs temperature parameter (0.0-1.0):
- Lower (0.2-0.4): More consistent, professional
- Higher (0.6-0.8): More expressive, varied

### Change the System Prompt
Edit the system message in the `messages` list to change the bot's personality.

### Use Different LLM
Replace OpenAI with any supported LLM service (Anthropic, etc.)
