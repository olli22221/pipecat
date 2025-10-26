#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""
Example: Simli Video with Higgs Audio TTS (Daily Only)

This example demonstrates a video AI bot using:
- Simli for AI-generated video avatars
- Higgs Audio for natural text-to-speech
- Deepgram for speech-to-text
- OpenAI for conversational AI
- Daily for video calling

Setup:
------
1. Install dependencies:
   pip install pipecat-ai[simli,higgs,deepgram,openai,daily]

2. Set environment variables in .env:
   SIMLI_API_KEY=your_simli_api_key
   SIMLI_FACE_ID=your_simli_face_id
   DEEPGRAM_API_KEY=your_deepgram_api_key
   OPENAI_API_KEY=your_openai_api_key
   DAILY_API_KEY=your_daily_api_key

3. Run:
   python simli-higgs-daily.py

The script will create a Daily room and print the URL to join.
"""

import asyncio
import os
from datetime import datetime, timedelta

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from simli import SimliConfig

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.higgs import HiggsAudioTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.simli.video import SimliVideoService
from pipecat.transports.daily.transport import DailyParams, DailyTransport
from pipecat.transports.daily.utils import DailyRESTHelper, DailyRoomParams

load_dotenv(override=True)


async def create_daily_room(daily_api_key: str):
    """Create a Daily room for the video call."""
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
    logger.info("🎬 Simli Video + Higgs Audio TTS Bot (Daily)")
    logger.info("=" * 70)

    # Get API keys
    daily_api_key = os.getenv("DAILY_API_KEY")
    deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    simli_api_key = os.getenv("SIMLI_API_KEY")
    simli_face_id = os.getenv("SIMLI_FACE_ID")

    # Validate required keys
    if not all([daily_api_key, deepgram_api_key, openai_api_key, simli_api_key, simli_face_id]):
        logger.error("❌ Missing required API keys. Please check your .env file.")
        logger.error("Required: DAILY_API_KEY, DEEPGRAM_API_KEY, OPENAI_API_KEY, SIMLI_API_KEY, SIMLI_FACE_ID")
        return

    # Create Daily room
    logger.info("📞 Creating Daily room...")
    room_url, token = await create_daily_room(daily_api_key)
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"🔗 JOIN ROOM: {room_url}")
    logger.info("=" * 70)
    logger.info("")

    # Initialize transport with optimized settings for video
    transport = DailyTransport(
        room_url=room_url,
        token=token,
        bot_name="Simli Avatar",
        params=DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_out_sample_rate=24000,  # Higgs Audio works best at 24kHz
            video_out_enabled=True,
            video_out_is_live=True,
            video_out_width=512,  # Simli recommended resolution
            video_out_height=512,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
        ),
    )

    # Initialize services
    logger.info("🔧 Initializing services...")

    # Speech-to-text
    stt = DeepgramSTTService(api_key=deepgram_api_key)

    # LLM for conversation
    llm = OpenAILLMService(api_key=openai_api_key, model="gpt-4o-mini")

    # Higgs Audio TTS - natural sounding text-to-speech
    logger.info("🎵 Loading Higgs Audio TTS (this may take a moment on first run)...")
    tts = HiggsAudioTTSService(
        model_path="bosonai/higgs-audio-v2-generation-3B-base",
        audio_tokenizer_path="bosonai/higgs-audio-v2-tokenizer",
        temperature=0.3,  # Lower temperature for more consistent speech
    )

    # Simli AI video service - generates video from audio
    logger.info("🎬 Initializing Simli video service...")
    simli_ai = SimliVideoService(
        SimliConfig(simli_api_key, simli_face_id),
    )

    # Create conversation context
    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant in a video call. Keep your responses conversational and concise. Your output will be converted to audio and video, so avoid using special characters or formatting. Be friendly and engaging.",
        },
    ]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

    # Build pipeline: STT -> LLM -> TTS -> Video -> Output
    logger.info("🔨 Building pipeline...")
    pipeline = Pipeline(
        [
            transport.input(),  # User audio/video input
            stt,  # Speech to text (Deepgram)
            context_aggregator.user(),  # Add to conversation context
            llm,  # Generate response (OpenAI)
            tts,  # Text to speech (Higgs Audio)
            simli_ai,  # Speech to video (Simli)
            transport.output(),  # Send to Daily
            context_aggregator.assistant(),  # Save assistant response
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            allow_interruptions=True,
        ),
    )

    # Event handlers
    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        logger.info(f"👋 First participant joined: {participant.get('id', 'unknown')}")
        # Start conversation - empty LLMRunFrame lets LLM follow system instructions
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        logger.info(f"👋 Participant left: {reason}")
        await task.cancel()

    @transport.event_handler("on_call_state_updated")
    async def on_call_state_updated(transport, state):
        if state == "joined":
            logger.info("✅ Bot joined the room successfully")

    logger.info("🚀 Bot is ready! Waiting for participants...")
    logger.info("")
    logger.info("💡 Tips:")
    logger.info("   - The bot will greet you when you join")
    logger.info("   - Speak naturally and wait for the response")
    logger.info("   - The avatar video is generated in real-time from the speech")
    logger.info("")

    # Run the bot
    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
