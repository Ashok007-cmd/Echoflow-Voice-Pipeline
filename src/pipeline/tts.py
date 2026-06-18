"""Text-to-Speech (TTS) engine with multiple backends.

Supported backends:
  - edge_tts: Microsoft Edge TTS (free, high quality, requires internet)
  - pyttsx3: Offline TTS (free, lower quality, no internet needed)
  - mock: Silent mock for testing
"""

import asyncio
import io
import logging
import time
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional

import numpy as np

from src.config import TTSBackend, TTSConfig
from src.pipeline.models import LLMResponse, TTSResult

logger = logging.getLogger(__name__)


class TTSEngine(ABC):
    """Abstract base class for TTS engines."""

    def __init__(self, config: TTSConfig):
        self.config = config
        self._current_stream: Optional[asyncio.Task] = None

    @abstractmethod
    async def synthesize(self, text: str) -> TTSResult:
        """Synthesize a complete text string to audio."""
        ...

    @abstractmethod
    async def synthesize_stream(
        self, text_stream: AsyncGenerator[LLMResponse, None]
    ) -> AsyncGenerator[TTSResult, None]:
        """Streaming TTS — synthesize tokens as they arrive from the LLM."""
        ...

    @abstractmethod
    async def play(self, audio_data: bytes) -> None:
        """Play audio data through speakers."""
        ...

    async def stop(self) -> None:
        """Stop current playback."""
        if self._current_stream and not self._current_stream.done():
            self._current_stream.cancel()
            try:
                await self._current_stream
            except asyncio.CancelledError:
                pass


LANGUAGE_VOICE_MAP = {
    "en": "en-US-JennyNeural",
    "en-us": "en-US-JennyNeural",
    "en-gb": "en-GB-SoniaNeural",
    "es": "es-ES-AlvaroNeural",
    "es-es": "es-ES-AlvaroNeural",
    "es-mx": "es-MX-JorgeNeural",
    "fr": "fr-FR-DeniseNeural",
    "fr-fr": "fr-FR-DeniseNeural",
    "de": "de-DE-ConradNeural",
    "de-de": "de-DE-ConradNeural",
    "it": "it-IT-ElsaNeural",
    "it-it": "it-IT-ElsaNeural",
    "ja": "ja-JP-NanamiNeural",
    "ja-jp": "ja-JP-NanamiNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "zh-cn": "zh-CN-XiaoxiaoNeural",
}


class EdgeTTSEngine(TTSEngine):
    """Microsoft Edge TTS backend — free, high quality, streaming support."""

    def __init__(self, config: TTSConfig):
        super().__init__(config)
        self._player = None

    async def synthesize(self, text: str) -> TTSResult:
        start = time.perf_counter()

        try:
            import edge_tts

            lang = getattr(self.config, "language", "en-us").lower()
            voice = LANGUAGE_VOICE_MAP.get(lang) or LANGUAGE_VOICE_MAP.get(lang[:2]) or self.config.voice

            communicate = edge_tts.Communicate(
                text,
                voice=voice,
                rate=f"{self.config.rate:+d}%",
                volume=f"{self.config.volume:+d}%",
                pitch=f"{self.config.pitch:+d}Hz",
            )

            audio_chunks: list[bytes] = []
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_chunks.append(chunk["data"])

            audio_data = b"".join(audio_chunks) if audio_chunks else None
            elapsed = (time.perf_counter() - start) * 1000

            return TTSResult(
                audio_data=audio_data,
                text=text,
                is_final=True,
                latency_ms=elapsed,
            )
        except ImportError:
            raise ImportError("edge-tts not installed: pip install edge-tts")
        except Exception as e:
            logger.error(f"Edge TTS synthesis failed: {e}")
            return TTSResult(
                text=text,
                is_final=True,
                error=str(e),
            )

    async def synthesize_stream(
        self, text_stream: AsyncGenerator[LLMResponse, None]
    ) -> AsyncGenerator[TTSResult, None]:
        text_buffer = ""
        chunk_count = 0

        async for response in text_stream:
            text_buffer += response.text
            chunk_count += 1

            if response.is_final:
                break

            # Synthesize in sentence-sized chunks (on sentence boundaries or every N tokens)
            should_synthesize = (
                any(c in response.text for c in ".!?\n") or chunk_count >= 5
            )

            if should_synthesize and text_buffer.strip():
                result = await self.synthesize(text_buffer)
                text_buffer = ""
                chunk_count = 0
                yield result

        # Final flush if anything remains
        if text_buffer.strip():
            result = await self.synthesize(text_buffer)
            result.is_final = True
            yield result

    async def play(self, audio_data: bytes) -> None:
        """Play MP3 audio from edge-tts using sounddevice."""
        if not audio_data:
            return

        try:
            import soundfile as sf

            # edge-tts returns MP3 data; decode via soundfile which handles it
            data, sample_rate = sf.read(io.BytesIO(audio_data))
            import sounddevice as sd

            sd.play(data, sample_rate)
            sd.wait()
        except ImportError:
            logger.warning("sounddevice not installed, falling back to ffmpeg")
            # Fallback: write to temp file and play with ffplay/aplay
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(audio_data)
                tmp_path = f.name

            try:
                proc = await asyncio.create_subprocess_exec(
                    "ffplay", "-nodisp", "-autoexit", tmp_path,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await proc.wait()
            finally:
                import os
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        except Exception as e:
            logger.error(f"Audio playback failed: {e}")


class Pyttsx3Engine(TTSEngine):
    """Offline pyttsx3 TTS backend — works without internet, lower quality."""

    def __init__(self, config: TTSConfig):
        super().__init__(config)
        self._engine = None
        self._lock = asyncio.Lock()

    def _get_engine(self):
        if self._engine is None:
            import pyttsx3

            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self.config.pyttsx3_rate)
            self._engine.setProperty("volume", self.config.pyttsx3_volume)
        return self._engine

    async def synthesize(self, text: str) -> TTSResult:
        if not text.strip():
            return TTSResult(text=text, is_final=True)

        start = time.perf_counter()

        loop = asyncio.get_running_loop()

        def _synthesize():
            engine = self._get_engine()
            import os
            import tempfile

            fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            try:
                engine.save_to_file(text, tmp_path)
                engine.runAndWait()
                with open(tmp_path, "rb") as f:
                    return f.read()
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        async with self._lock:
            try:
                audio_data = await loop.run_in_executor(None, _synthesize)
                elapsed = (time.perf_counter() - start) * 1000
                return TTSResult(
                    audio_data=audio_data,
                    text=text,
                    is_final=True,
                    latency_ms=elapsed,
                )
            except Exception as e:
                logger.error(f"pyttsx3 synthesis failed: {e}")
                return TTSResult(text=text, is_final=True, error=str(e))

    async def synthesize_stream(
        self, text_stream: AsyncGenerator[LLMResponse, None]
    ) -> AsyncGenerator[TTSResult, None]:
        """For pyttsx3, collect all text and synthesize at once (no streaming)."""
        full_text = ""

        async for response in text_stream:
            full_text += response.text
            if response.is_final:
                break

        if full_text.strip():
            result = await self.synthesize(full_text)
            result.is_final = True
            yield result

    async def play(self, audio_data: bytes) -> None:
        """Play WAV audio using aplay or sounddevice."""
        if not audio_data:
            return

        try:
            import sounddevice as sd
            import soundfile as sf

            data, sample_rate = sf.read(io.BytesIO(audio_data))
            sd.play(data, sample_rate)
            sd.wait()
        except ImportError:
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_data)
                tmp_path = f.name

            try:
                proc = await asyncio.create_subprocess_exec(
                    "aplay", tmp_path,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await proc.wait()
            finally:
                import os
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass


class MockTTSEngine(TTSEngine):
    """Silent mock TTS for testing without audio hardware."""

    async def synthesize(self, text: str) -> TTSResult:
        await asyncio.sleep(0.1)  # simulate processing
        return TTSResult(
            audio_data=b"",
            text=text,
            is_final=True,
            latency_ms=100.0,
        )

    async def synthesize_stream(
        self, text_stream: AsyncGenerator[LLMResponse, None]
    ) -> AsyncGenerator[TTSResult, None]:
        full_text = ""
        async for response in text_stream:
            full_text += response.text
            if response.is_final:
                break
            # Yield a result per chunk for streaming metrics
            yield TTSResult(
                audio_data=b"",
                text=response.text,
                is_final=False,
                latency_ms=50.0,
            )

        if full_text.strip():
            yield TTSResult(
                audio_data=b"",
                text=full_text,
                is_final=True,
                latency_ms=100.0,
            )

    async def play(self, audio_data: bytes) -> None:
        pass  # Silent


def create_tts_engine(config: TTSConfig) -> TTSEngine:
    """Factory function to create the appropriate TTS engine."""
    if config.backend == TTSBackend.EDGE_TTS:
        return EdgeTTSEngine(config)
    elif config.backend == TTSBackend.PYTTSX3:
        return Pyttsx3Engine(config)
    elif config.backend == TTSBackend.MOCK:
        return MockTTSEngine(config)
    else:
        raise ValueError(f"Unknown TTS backend: {config.backend}")
