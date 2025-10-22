#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import re
from typing import AsyncGenerator

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService

try:
    import numpy as np
    import torch
    from boson_multimodal.data_types import ChatMLSample, Message
    from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Higgs Audio, you need to `pip install boson-multimodal torch torchaudio`. "
        "Also run: git clone https://github.com/boson-ai/higgs-audio.git && cd higgs-audio && pip install -e ."
    )
    raise Exception(f"Missing module: {e}")


class HiggsAudioTTSService(TTSService):
    """
    Higgs Audio TTS Service for Pipecat
    
    This service uses Boson AI's Higgs Audio V2 model for high-quality,
    expressive text-to-speech synthesis. It buffers incoming text tokens
    into complete sentences before generating speech for natural output.
    
    Requirements:
    - CUDA-capable GPU (minimum 24GB VRAM recommended)
    - boson-multimodal package installed
    - torch and torchaudio
    
    Args:
        model_path: HuggingFace model path (default: "bosonai/higgs-audio-v2-generation-3B-base")
        audio_tokenizer_path: Audio tokenizer path (default: "bosonai/higgs-audio-v2-tokenizer")
        temperature: Generation temperature (default: 0.3)
        max_new_tokens: Maximum tokens to generate (default: 1024)
        sample_rate: Output sample rate (default: 24000)
        voice_id: Not used, kept for API compatibility
        **kwargs: Additional arguments passed to TTSService
    """

    def __init__(
        self,
        *,
        model_path: str = "bosonai/higgs-audio-v2-generation-3B-base",
        audio_tokenizer_path: str = "bosonai/higgs-audio-v2-tokenizer",
        temperature: float = 0.3,
        max_new_tokens: int = 1024,
        sample_rate: int = 24000,
        voice_id: str = "",
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._model_path = model_path
        self._audio_tokenizer_path = audio_tokenizer_path
        self._temperature = temperature
        self._max_new_tokens = max_new_tokens
        self._engine = None
        self._initialized = False
        self._text_buffer = ""

    def can_generate_metrics(self) -> bool:
        return True

    async def start(self, frame: StartFrame):
        """Initialize the Higgs Audio engine"""
        await super().start(frame)
        
        if self._initialized:
            return
            
        if not torch.cuda.is_available():
            error_msg = "CUDA GPU is required for Higgs Audio TTS"
            logger.error(f"{self}: {error_msg}")
            raise RuntimeError(error_msg)
        
        logger.info(f"{self}: Initializing Higgs Audio model from {self._model_path}")
        logger.info(f"{self}: Device: cuda ({torch.cuda.get_device_name(0)})")
        
        try:
            self._engine = HiggsAudioServeEngine(
                self._model_path,
                self._audio_tokenizer_path,
                device="cuda"
            )
            self._initialized = True
            logger.info(f"{self}: Higgs Audio model initialized successfully")
        except Exception as e:
            error_msg = f"Failed to initialize Higgs Audio: {e}"
            logger.error(f"{self}: {error_msg}")
            raise RuntimeError(error_msg)

    async def stop(self, frame: EndFrame):
        """Clean up resources"""
        await super().stop(frame)
        self._engine = None
        self._initialized = False
        self._text_buffer = ""

    async def cancel(self, frame: CancelFrame):
        """Cancel any ongoing operations"""
        await super().cancel(frame)
        self._text_buffer = ""

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """
        Generate speech for the given text.
        
        This method buffers text and generates speech for complete sentences.
        Text is accumulated until sentence-ending punctuation is detected.
        """
        if not self._initialized or self._engine is None:
            logger.error(f"{self}: Higgs Audio engine not initialized")
            yield ErrorFrame("Higgs Audio engine not initialized")
            return

        # Add text to buffer
        self._text_buffer += text
        
        # Process complete sentences from buffer
        async for frame in self._process_buffer():
            yield frame

    async def flush_audio(self) -> AsyncGenerator[Frame, None]:
        """Flush any remaining buffered text"""
        if self._text_buffer.strip():
            async for frame in self._generate_speech(self._text_buffer.strip()):
                yield frame
            self._text_buffer = ""

    async def _process_buffer(self) -> AsyncGenerator[Frame, None]:
        """Process the text buffer and generate speech for complete sentences"""
        # Pattern to match sentence endings: . ! ? followed by space or end of string
        sentence_pattern = r'([.!?]+(?:\s+|$))'
        
        # Find all complete sentences
        sentences = re.split(sentence_pattern, self._text_buffer)
        
        # Reconstruct complete sentences
        complete_sentences = []
        current = ""
        
        for part in sentences:
            current += part
            # If this part is punctuation
            if re.match(sentence_pattern, part):
                complete_sentences.append(current.strip())
                current = ""
        
        # Generate speech for each complete sentence
        for sentence in complete_sentences:
            if sentence:
                async for frame in self._generate_speech(sentence):
                    yield frame
        
        # Keep incomplete text in buffer
        self._text_buffer = current

    async def _generate_speech(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech using Higgs Audio for a complete sentence"""
        if not text.strip():
            return

        try:
            logger.debug(f"{self}: Generating TTS for: '{text[:100]}...'")
            
            await self.start_ttfb_metrics()
            
            # Signal TTS started
            logger.warning(f"🔊🔊🔊 HIGGS TTS: Yielding TTSStartedFrame 🔊🔊🔊")
            yield TTSStartedFrame()
            
            # Create ChatML format messages
            system_prompt = (
                "Generate audio following instruction.\n\n"
                "<|scene_desc_start|>\n"
                "Audio is recorded from a quiet room.\n"
                "<|scene_desc_end|>"
            )
            
            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=text)
            ]
            
            chat_ml_sample = ChatMLSample(messages=messages)
            
            # Run generation in thread pool (GPU intensive)
            loop = asyncio.get_event_loop()
            output = await loop.run_in_executor(
                None,
                lambda: self._engine.generate(
                    chat_ml_sample=chat_ml_sample,
                    max_new_tokens=self._max_new_tokens,
                    temperature=self._temperature
                )
            )
            
            await self.start_tts_usage_metrics(text)
            
            # Process audio output
            audio_array = output.audio
            
            # Handle stereo/mono
            if audio_array.ndim == 2:
                audio_array = audio_array[0]
            
            # Ensure float32 and normalize to int16
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            
            audio_array = (audio_array * 32767).astype(np.int16)
            
            # Create and yield audio frame
            frame = TTSAudioRawFrame(
                audio=audio_array.tobytes(),
                sample_rate=output.sampling_rate,
                num_channels=1
            )

            logger.warning(f"🔊🔊🔊 HIGGS TTS: Yielding TTSAudioRawFrame (size={len(audio_array.tobytes())} bytes) 🔊🔊🔊")
            yield frame
            
            # Signal TTS completed
            logger.warning(f"🔊🔊🔊 HIGGS TTS: Yielding TTSStoppedFrame 🔊🔊🔊")
            yield TTSStoppedFrame()
            
            duration_secs = len(audio_array) / output.sampling_rate
            logger.debug(
                f"{self}: Generated {len(audio_array)} samples "
                f"({duration_secs:.2f}s) at {output.sampling_rate}Hz"
            )
            
        except Exception as e:
            logger.error(f"{self}: Error generating TTS: {e}")
            import traceback
            traceback.print_exc()
            yield ErrorFrame(f"TTS generation failed: {e}")