"""
Example: Using Ditto Talking Head Service with WebSocket Streaming

This example shows how to use the DittoTalkingHeadService with WebSocket
streaming instead of WebRTC. Much simpler for local testing!

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

5. Install dependencies:
   pip install websockets opencv-python

RUNNING THE EXAMPLE:
-------------------
1. Run this script: python websocket_ditto_example.py
2. Open your browser to: http://localhost:8080/
3. Click "Connect" to start streaming
4. Speak and watch the synchronized talking head!

WHY WEBSOCKETS?
---------------
- Much simpler than WebRTC (no ICE, STUN, TURN complexity)
- Perfect for local development and testing
- Direct audio/video streaming from server to browser
- Easy to debug and understand
"""

import argparse
import asyncio
import base64
import json
import os
from typing import Set

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    OutputAudioRawFrame,
    OutputImageRawFrame,
    TextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantContextAggregator,
    LLMUserContextAggregator,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.higgs import HiggsAudioTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.ditto import DittoTalkingHeadService

try:
    import cv2
    import numpy as np
except ImportError:
    logger.error("Please install: pip install opencv-python numpy")
    raise

load_dotenv(override=True)

# Store active WebSocket connections
active_connections: Set[WebSocket] = set()


class WebSocketOutputProcessor(FrameProcessor):
    """Processor that sends audio and video frames to WebSocket clients."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._frame_count = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Send video frames to all connected clients
        if isinstance(frame, OutputImageRawFrame):
            self._frame_count += 1

            # Convert RGB frame to JPEG for efficient streaming
            # frame.image is already RGB numpy array
            height, width = frame.size[1], frame.size[0]
            frame_array = np.frombuffer(frame.image, dtype=np.uint8).reshape((height, width, 3))

            # Encode as JPEG
            _, jpeg_data = cv2.imencode('.jpg', cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
            jpeg_base64 = base64.b64encode(jpeg_data.tobytes()).decode('utf-8')

            # Send to all connected clients
            message = {
                "type": "video",
                "data": jpeg_base64,
                "width": width,
                "height": height,
                "format": "jpeg"
            }

            if self._frame_count % 30 == 0:
                logger.debug(f"Sent video frame {self._frame_count} to {len(active_connections)} clients")

            await self._broadcast(message)

        # Send audio frames to all connected clients
        elif isinstance(frame, OutputAudioRawFrame):
            # Convert audio to base64
            audio_base64 = base64.b64encode(frame.audio).decode('utf-8')

            message = {
                "type": "audio",
                "data": audio_base64,
                "sample_rate": frame.sample_rate,
                "num_channels": frame.num_channels
            }

            await self._broadcast(message)

        await self.push_frame(frame, direction)

    async def _broadcast(self, message: dict):
        """Broadcast message to all connected WebSocket clients."""
        if not active_connections:
            return

        message_json = json.dumps(message)
        disconnected = set()

        for websocket in active_connections:
            try:
                await websocket.send_text(message_json)
            except Exception as e:
                logger.error(f"Error sending to WebSocket: {e}")
                disconnected.add(websocket)

        # Remove disconnected clients
        active_connections.difference_update(disconnected)


app = FastAPI()

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WS_STATIC_DIR = os.path.join(SCRIPT_DIR, "ws_static")

# Create static directory if it doesn't exist
os.makedirs(WS_STATIC_DIR, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=WS_STATIC_DIR), name="static")


@app.get("/")
async def root():
    """Serve the WebSocket client."""
    return FileResponse(os.path.join(WS_STATIC_DIR, "index.html"))


# Global pipeline task
task = None


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming audio/video."""
    global task

    await websocket.accept()
    active_connections.add(websocket)
    logger.info(f"WebSocket client connected. Total clients: {len(active_connections)}")

    try:
        # If this is the first connection, start the bot
        if task is None:
            logger.info("Starting Ditto bot...")
            asyncio.create_task(start_bot())

        # Keep connection alive and handle incoming messages
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            elif message.get("type") == "text":
                # User sent text message (for testing without speech)
                if task:
                    await task.queue_frames([TextFrame(message.get("text", ""))])

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        active_connections.discard(websocket)
        logger.info(f"Client removed. Remaining clients: {len(active_connections)}")


async def start_bot():
    """Start the Ditto bot pipeline."""
    global task

    logger.info("=" * 70)
    logger.info("🎬 Starting Ditto Talking Head Bot with WebSocket Streaming")
    logger.info("=" * 70)

    # Get API keys
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Ditto configuration
    ditto_path = os.getenv("DITTO_PATH", "./ditto-talkinghead")
    ditto_data_root = os.getenv("DITTO_DATA_ROOT", "./checkpoints/ditto_trt_Ampere_Plus")
    ditto_cfg_pkl = os.getenv("DITTO_CFG_PKL", "./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch_online.pkl")
    source_image = os.getenv("DITTO_SOURCE_IMAGE", "./example/image.png")

    # Validate Ditto paths
    if not os.path.exists(ditto_path):
        logger.error(f"Ditto path not found: {ditto_path}")
        return

    if not os.path.exists(source_image):
        logger.error(f"Source image not found: {source_image}")
        return

    logger.info(f"✅ Using avatar image: {source_image}")

    # Initialize services
    llm = OpenAILLMService(api_key=openai_api_key, model="gpt-4o-mini")

    # Initialize Higgs Audio TTS
    tts = HiggsAudioTTSService(
        model_path="bosonai/higgs-audio-v2-generation-3B-base",
        audio_tokenizer_path="bosonai/higgs-audio-v2-tokenizer",
        temperature=0.3,
    )

    # Initialize Ditto Talking Head Service
    save_frames_dir = os.getenv("DITTO_SAVE_FRAMES_DIR", "./ditto_frames")

    ditto = DittoTalkingHeadService(
        ditto_path=ditto_path,
        data_root=ditto_data_root,
        cfg_pkl=ditto_cfg_pkl,
        source_image_path=source_image,
        chunk_size=(3, 5, 2),
        save_frames_dir=save_frames_dir,
        target_fps=30,
    )

    # WebSocket output processor
    ws_output = WebSocketOutputProcessor()

    # Create context
    context = OpenAILLMContext(
        messages=[{
            "role": "system",
            "content": "You are a helpful AI assistant with a friendly personality. Keep responses brief and conversational."
        }]
    )

    # Build pipeline (no STT for now - will add later)
    pipeline = Pipeline([
        LLMUserContextAggregator(context),
        llm,
        tts,
        ditto,
        ws_output,  # Send to WebSocket clients
        LLMAssistantContextAggregator(context),
    ])

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    # Send initial greeting
    await task.queue_frames([TextFrame("Hello! I'm your AI assistant with a talking head. Type a message to chat!")])

    logger.info("🚀 Bot ready!")

    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ditto WebSocket Bot")
    parser.add_argument(
        "--host", default="localhost", help="Host for HTTP server (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    args = parser.parse_args()

    logger.info("")
    logger.info("=" * 70)
    logger.info("🚀 Starting Ditto WebSocket Server")
    logger.info("=" * 70)
    logger.info("")
    logger.info(f"📡 Server will be available at: http://{args.host}:{args.port}/")
    logger.info("💡 Open this URL in your browser to connect to the bot")
    logger.info("")
    logger.info("🎯 Why WebSockets?")
    logger.info("   WebSockets are much simpler than WebRTC - no ICE, STUN, or")
    logger.info("   complex negotiation. Perfect for local development!")
    logger.info("")

    uvicorn.run(app, host=args.host, port=args.port)
