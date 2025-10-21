"""
Example: Using Ditto Talking Head Service

This example shows how to use the DittoTalkingHeadService in a voice bot.
The service generates realistic talking head videos synchronized with the audio.

SETUP REQUIRED:
--------------
1. Install Ditto from https://github.com/antgroup/ditto-talkinghead
2. Download model checkpoints to ./checkpoints/
3. **IMPORTANT: Prepare your avatar image**:

   Place a clear front-facing photo of a person's face at one of:
   - ./example/image.png (default location)
   - Or any other path and set DITTO_SOURCE_IMAGE env var

   Image requirements:
   - Format: PNG or JPG
   - Resolution: 512x512 or higher recommended
   - Content: Front-facing portrait with visible face
   - This face will be animated to speak in sync with the TTS audio

4. Set environment variables:
   export DITTO_PATH="./ditto-talkinghead"
   export DITTO_SOURCE_IMAGE="./my_avatar.png"  # Your avatar image path

Example structure:
  your_project/
  ├── ditto-talkinghead/          # Ditto installation
  │   ├── inference.py
  │   └── checkpoints/
  ├── examples/
  │   └── simple/
  │       └── ditto_example.py    # This file
  └── my_avatar.png               # Your avatar image here!
"""

import asyncio
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantContextAggregator,
    LLMUserContextAggregator,
)
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.higgs import HiggsAudioTTSService
from pipecat.services.ditto import DittoTalkingHeadService
from pipecat.transports.daily.transport import DailyParams, DailyTransport
from pipecat.transports.daily.utils import DailyRESTHelper, DailyRoomParams

import aiohttp

load_dotenv(override=True)


async def create_daily_room(daily_api_key: str):
    """Create a Daily room"""
    async with aiohttp.ClientSession() as session:
        daily_helper = DailyRESTHelper(
            daily_api_key=daily_api_key,
            aiohttp_session=session,
        )

        room_params = DailyRoomParams(
            privacy="public",
            properties={
                "exp": int((datetime.now() + timedelta(minutes=30)).timestamp()),
            }
        )

        room = await daily_helper.create_room(room_params)
        token = await daily_helper.get_token(room.url, expiry_time=1800)

        return room.url, token


async def main():
    logger.info("=" * 70)
    logger.info("🎬 Ditto Talking Head Voice Bot Example")
    logger.info("=" * 70)
    logger.info("")
    logger.info("This example generates a talking head video using your avatar image.")
    logger.info("Make sure you have set up your avatar image (see file header for details).")
    logger.info("")

    # Get API keys and paths
    daily_api_key = os.getenv("DAILY_API_KEY")
    deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Ditto configuration
    # IMPORTANT: Update these paths to match your Ditto installation
    # NOTE: Use an "online" config file for real-time streaming (e.g., v0.4_hubert_cfg_trt_online.pkl)
    ditto_path = os.getenv("DITTO_PATH", "./ditto-talkinghead")
    ditto_data_root = os.getenv("DITTO_DATA_ROOT", "./checkpoints/ditto_trt_Ampere_Plus")
    ditto_cfg_pkl = os.getenv("DITTO_CFG_PKL", "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl")
    source_image = os.getenv("DITTO_SOURCE_IMAGE", "./example/image.png")

    # Validate Ditto paths
    if not os.path.exists(ditto_path):
        logger.error(f"Ditto path not found: {ditto_path}")
        logger.error("Please set DITTO_PATH environment variable or install Ditto from:")
        logger.error("https://github.com/antgroup/ditto-talkinghead")
        return

    if not os.path.exists(source_image):
        logger.error(f"Source image not found: {source_image}")
        logger.error("Please provide a source image path via DITTO_SOURCE_IMAGE environment variable")
        logger.error("")
        logger.error("The source image should be:")
        logger.error("  - A clear front-facing photo of a person's face")
        logger.error("  - PNG or JPG format")
        logger.error("  - 512x512 or higher resolution recommended")
        logger.error("  - This face will be animated to create the talking head")
        return

    # Log the avatar image being used
    logger.info(f"✅ Using avatar image: {source_image}")
    logger.info(f"   This face will be animated to create the talking head video")
    logger.info("")

    # Create Daily room
    logger.info("📞 Creating Daily room...")
    room_url, token = await create_daily_room(daily_api_key)
    logger.info(f"🔗 JOIN ROOM: {room_url}")

    # Initialize transport with video enabled
    transport = DailyTransport(
        room_url=room_url,
        token=token,
        bot_name="Ditto Bot",
        params=DailyParams(
            audio_out_enabled=True,
            audio_out_sample_rate=24000,
            camera_out_enabled=True,  # Enable video output
            camera_out_width=512,
            camera_out_height=512,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            join_timeout=30,  # Increase timeout for Ditto initialization (default: 10s)
        ),
    )

    # Initialize services
    stt = DeepgramSTTService(api_key=deepgram_api_key)
    llm = OpenAILLMService(api_key=openai_api_key, model="gpt-4o-mini")

    # Initialize Higgs Audio TTS
    tts = HiggsAudioTTSService(
        model_path="bosonai/higgs-audio-v2-generation-3B-base",
        audio_tokenizer_path="bosonai/higgs-audio-v2-tokenizer",
        temperature=0.3,
    )

    # Initialize Ditto Talking Head Service
    # This service:
    # 1. Uses Ditto's StreamSDK for real-time video generation
    # 2. Processes audio chunks as they arrive from TTS
    # 3. Generates video frames that are synchronized with the audio
    # 4. The source_image is the avatar face that gets animated
    ditto = DittoTalkingHeadService(
        ditto_path=ditto_path,
        data_root=ditto_data_root,
        cfg_pkl=ditto_cfg_pkl,
        source_image_path=source_image,  # Your avatar face image
        chunk_size=(3, 5, 2),  # (history, current, future) frames - must match model training
    )

    # Create context
    context = OpenAILLMContext(
        messages=[{
            "role": "system",
            "content": "You are a helpful AI assistant with a friendly personality. Keep responses brief and conversational."
        }]
    )

    # Build pipeline
    # The flow is: STT -> LLM -> TTS -> Ditto (generates video from audio) -> Transport
    pipeline = Pipeline([
        transport.input(),
        stt,
        LLMUserContextAggregator(context),
        llm,
        tts,  # Generate audio
        ditto,  # Generate video from audio
        transport.output(),
        LLMAssistantContextAggregator(context),
    ])

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    @transport.event_handler("on_first_participant_joined")
    async def on_participant_joined(transport, participant):
        logger.info(f"👋 Participant joined!")
        await task.queue_frames([TextFrame("Hello! I'm your AI assistant with a talking head. How can I help you today?")])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        await task.cancel()

    logger.info("🚀 Bot ready!")
    logger.info("💡 The bot will generate talking head videos using Ditto")

    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
