"""
Example: Using Ditto Talking Head Service with SmallWebRTC

This example shows how to use the DittoTalkingHeadService with SmallWebRTC transport.
SmallWebRTC properly uses PTS timestamps for video synchronization, providing better
audio-video sync compared to Daily transport.

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
   export DITTO_SAVE_FRAMES_DIR="./ditto_frames"  # Optional: save frames to disk

5. Install SmallWebRTC dependencies:
   pip install pipecat-ai[webrtc]
   pip install pipecat-ai-small-webrtc-prebuilt

Example structure:
  your_project/
  ├── ditto-talkinghead/          # Ditto installation
  │   ├── inference.py
  │   └── checkpoints/
  ├── examples/
  │   └── simple/
  │       └── ditto_webrtc_example.py    # This file
  └── my_avatar.png               # Your avatar image here!

Usage:
  python ditto_webrtc_example.py --host 0.0.0.0 --port 7860

  Then open http://localhost:7860 in your browser to connect to the bot.
"""

import argparse
import asyncio
import os
from contextlib import asynccontextmanager
from typing import Dict

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import RedirectResponse
from loguru import logger
from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import TextFrame, LLMRunFrame
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
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import IceServer, SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

load_dotenv(override=True)

app = FastAPI()

# Store connections by pc_id
pcs_map: Dict[str, SmallWebRTCConnection] = {}

# ICE servers for WebRTC connection
ice_servers = [
    IceServer(
        urls="stun:stun.l.google.com:19302",
    )
]

# Mount the frontend at /
app.mount("/client", SmallWebRTCPrebuiltUI)


async def run_ditto_bot(webrtc_connection: SmallWebRTCConnection):
    """Run the Ditto talking head bot with SmallWebRTC transport."""
    logger.info("=" * 70)
    logger.info("🎬 Ditto Talking Head Voice Bot (SmallWebRTC)")
    logger.info("=" * 70)
    logger.info("")
    logger.info("This bot uses SmallWebRTC which properly handles PTS timestamps")
    logger.info("for improved audio-video synchronization.")
    logger.info("")

    # Ditto configuration
    ditto_path = os.getenv("DITTO_PATH", "./ditto-talkinghead")
    ditto_data_root = os.getenv("DITTO_DATA_ROOT", "./checkpoints/ditto_trt_Ampere_Plus")
    ditto_cfg_pkl = os.getenv("DITTO_CFG_PKL", "./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch_online.pkl")
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

    # Create SmallWebRTC transport with video enabled
    # SmallWebRTC properly uses PTS timestamps for video synchronization
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_out_sample_rate=24000,
            video_out_enabled=True,  # Enable video output
            video_out_width=1440,  # Match Ditto's output size
            video_out_height=1920,  # Match Ditto's output size
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    # Initialize services
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

    # Initialize Higgs Audio TTS
    tts = HiggsAudioTTSService(
        model_path="bosonai/higgs-audio-v2-generation-3B-base",
        audio_tokenizer_path="bosonai/higgs-audio-v2-tokenizer",
        temperature=0.3,
    )

    # Initialize Ditto Talking Head Service
    # This service generates video frames synchronized with audio
    # SmallWebRTC will use the PTS timestamps for proper A/V sync
    save_frames_dir = os.getenv("DITTO_SAVE_FRAMES_DIR", "./ditto_frames")

    ditto = DittoTalkingHeadService(
        ditto_path=ditto_path,
        data_root=ditto_data_root,
        cfg_pkl=ditto_cfg_pkl,
        source_image_path=source_image,  # Your avatar face image
        chunk_size=(3, 5, 2),  # (history, current, future) frames
        save_frames_dir=save_frames_dir,  # Optional: save frames to disk
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
        ditto,  # Generate video from audio with proper PTS timestamps
        transport.output(),
        LLMAssistantContextAggregator(context),
    ])

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"👋 Client connected!")
        await task.queue_frames([
            TextFrame("Hello! I'm your AI assistant with a talking head powered by Ditto. How can I help you today?")
        ])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    logger.info("🚀 Bot ready!")
    logger.info("💡 The bot will generate talking head videos with improved A/V sync")

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


@app.get("/", include_in_schema=False)
async def root_redirect():
    """Redirect root to the client UI."""
    return RedirectResponse(url="/client/")


@app.post("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks):
    """Handle WebRTC offer from the client."""
    pc_id = request.get("pc_id")

    if pc_id and pc_id in pcs_map:
        pipecat_connection = pcs_map[pc_id]
        logger.info(f"Reusing existing connection for pc_id: {pc_id}")
        await pipecat_connection.renegotiate(
            sdp=request["sdp"],
            type=request["type"],
            restart_pc=request.get("restart_pc", False),
        )
    else:
        pipecat_connection = SmallWebRTCConnection(ice_servers)
        await pipecat_connection.initialize(sdp=request["sdp"], type=request["type"])

        @pipecat_connection.event_handler("closed")
        async def handle_disconnected(webrtc_connection: SmallWebRTCConnection):
            logger.info(f"Discarding peer connection for pc_id: {webrtc_connection.pc_id}")
            pcs_map.pop(webrtc_connection.pc_id, None)

        # Run Ditto bot with SmallWebRTC transport
        background_tasks.add_task(run_ditto_bot, pipecat_connection)

    answer = pipecat_connection.get_answer()
    # Update the peer connection in the map
    pcs_map[answer["pc_id"]] = pipecat_connection

    return answer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    yield  # Run app
    # Cleanup all connections on shutdown
    coros = [pc.disconnect() for pc in pcs_map.values()]
    await asyncio.gather(*coros)
    pcs_map.clear()


app.router.lifespan_context = lifespan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ditto Talking Head Bot with SmallWebRTC")
    parser.add_argument(
        "--host", default="localhost", help="Host for HTTP server (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port for HTTP server (default: 7860)"
    )
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Starting Ditto WebRTC server...")
    logger.info(f"Server will be available at: http://{args.host}:{args.port}")
    logger.info("=" * 70)

    uvicorn.run(app, host=args.host, port=args.port)
