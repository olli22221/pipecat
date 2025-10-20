"""
Example: Using Higgs Audio TTS Service

This example shows how to use the HiggsAudioTTSService in a voice bot.
The service automatically buffers text into complete sentences for natural speech.
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
    logger.info("🎤 Higgs Audio Voice Bot Example")
    logger.info("=" * 70)
    
    # Get API keys
    daily_api_key = os.getenv("DAILY_API_KEY")
    deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # Create Daily room
    logger.info("📞 Creating Daily room...")
    room_url, token = await create_daily_room(daily_api_key)
    logger.info(f"🔗 JOIN ROOM: {room_url}")
    
    # Initialize transport
    transport = DailyTransport(
        room_url=room_url,
        token=token,
        bot_name="Higgs Bot",
        params=DailyParams(
            audio_out_enabled=True,
            audio_out_sample_rate=24000,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )
    
    # Initialize services
    stt = DeepgramSTTService(api_key=deepgram_api_key)
    llm = OpenAILLMService(api_key=openai_api_key, model="gpt-4o-mini")
    
    # Initialize Higgs Audio TTS - Simple!
    tts = HiggsAudioTTSService(
        model_path="bosonai/higgs-audio-v2-generation-3B-base",
        audio_tokenizer_path="bosonai/higgs-audio-v2-tokenizer",
        temperature=0.3,
    )
    
    # Create context
    context = OpenAILLMContext(
        messages=[{
            "role": "system",
            "content": "You are a helpful AI assistant. Keep responses brief."
        }]
    )
    
    # Build pipeline
    pipeline = Pipeline([
        transport.input(),
        stt,
        LLMUserContextAggregator(context),
        llm,
        tts,  # Just plug in the Higgs Audio service!
        transport.output(),
        LLMAssistantContextAggregator(context),
    ])
    
    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))
    
    @transport.event_handler("on_first_participant_joined")
    async def on_participant_joined(transport, participant):
        logger.info(f"👋 Participant joined!")
        await task.queue_frames([TextFrame("Hello! How can I help you?")])
    
    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        await task.cancel()
    
    logger.info("🚀 Bot ready!")
    
    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())