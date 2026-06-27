"""Speech Recognition (ASR) engine with multiple backends.

Supported backends:
  - google: Google Web Speech API (free, requires internet)
  - openai: OpenAI Whisper API (paid, requires API key)
  - local_whisper: Hugging Face Transformers Whisper (local, requires model download)
"""

import asyncio
import io
import logging
import time
from abc import ABC, abstractmethod
from typing import AsyncGenerator

import numpy as np

from src.config import ASRBackend, ASRConfig
from src.pipeline.models import AudioChunk, TranscriptionResult

logger = logging.getLogger(__name__)


class DynamicVAD:
    """Dynamic Energy-based Voice Activity Detector with Zero Crossing Rate estimation."""

    def __init__(self, sample_rate: int = 16000, frame_duration: float = 0.5):
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.ambient_energy = 100.0  # initial estimate of noise floor
        self.alpha = 0.95  # decay rate for tracking noise floor
        self.zcr_min = 20.0  # minimum zero crossing rate per second for speech
        self.zcr_max = 600.0  # maximum zero crossing rate per second for speech

    def is_speech(self, audio_data: bytes) -> bool:
        """Determines if the audio data contains speech using dynamic thresholding."""
        if not audio_data:
            return False
            
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        if len(audio_array) == 0:
            return False

        # 1. Compute RMS energy
        energy = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))

        # 2. Track noise floor (ambient energy) using a slow running minimum
        if energy < self.ambient_energy:
            # Quickly descend to new noise floor
            self.ambient_energy = 0.9 * self.ambient_energy + 0.1 * energy
        else:
            # Slowly decay upwards to track slowly changing ambient noise
            self.ambient_energy = self.alpha * self.ambient_energy + (1 - self.alpha) * energy

        # Dynamic threshold: base threshold plus a margin scaled by ambient noise
        threshold = max(250.0, self.ambient_energy * 2.5)

        # 3. Calculate Zero Crossing Rate (ZCR)
        zero_crossings = np.sum(np.diff(np.sign(audio_array)) != 0)
        zcr = zero_crossings / (len(audio_array) / self.sample_rate)

        # 4. Speech classification: energy above threshold and ZCR in human speech range
        is_energy_above = energy > threshold
        is_zcr_valid = (self.zcr_min <= zcr <= self.zcr_max) or (zcr == 0.0 and energy > 1000.0)

        return is_energy_above and is_zcr_valid

    def reset(self) -> None:
        self.ambient_energy = 100.0


class ASREngine(ABC):
    """Abstract base class for ASR engines."""

    def __init__(self, config: ASRConfig):
        self.config = config
        self._buffer: list[bytes] = []
        self._silence_counter: int = 0
        self._is_speaking: bool = False
        self._utterance_start: float = 0.0
        self.vad = DynamicVAD(config.sample_rate, config.chunk_duration)

    @abstractmethod
    async def transcribe(self, audio_data: bytes, sample_rate: int) -> TranscriptionResult:
        """Transcribe a single chunk of audio data."""
        ...

    async def transcribe_stream(
        self, audio_stream: AsyncGenerator[AudioChunk, None]
    ) -> AsyncGenerator[TranscriptionResult, None]:
        """Streaming transcription with voice activity detection.

        Accumulates audio chunks, detects utterance boundaries via silence,
        and transcribes complete utterances.
        """
        async for chunk in audio_stream:
            self._buffer.append(chunk.data)

            # Use Dynamic VAD
            is_silence = not self.vad.is_speech(chunk.data)

            if is_silence:
                self._silence_counter += 1
            else:
                self._silence_counter = 0
                if not self._is_speaking:
                    self._is_speaking = True
                    self._utterance_start = time.perf_counter()

            silence_duration = self._silence_counter * self.config.chunk_duration

            # If we've been speaking and silence exceeds threshold, transcribe
            if self._is_speaking and silence_duration > self.config.silence_threshold:
                utterance_data = b"".join(self._buffer)
                self._buffer.clear()
                self._is_speaking = False
                self._silence_counter = 0

                try:
                    result = await asyncio.wait_for(
                        self.transcribe(utterance_data, chunk.sample_rate),
                        timeout=min(self.config.max_utterance_seconds, 10.0),
                    )
                    result.is_final = True
                    yield result
                except asyncio.TimeoutError:
                    logger.warning("ASR transcription timed out for utterance")
                    yield TranscriptionResult(
                        text="",
                        is_final=True,
                        error="ASR transcription timed out",
                    )
                except Exception as e:
                    logger.error(f"ASR transcription failed: {e}")
                    yield TranscriptionResult(
                        text="",
                        is_final=True,
                        error=str(e),
                    )

    def reset(self) -> None:
        """Reset internal state."""
        self._buffer.clear()
        self._silence_counter = 0
        self._is_speaking = False
        self.vad.reset()


class GoogleASREngine(ASREngine):
    """Google Web Speech API ASR backend — free, requires internet."""

    def __init__(self, config: ASRConfig):
        super().__init__(config)
        self._recognizer = None

    @property
    def recognizer(self):
        if self._recognizer is None:
            try:
                import speech_recognition as sr

                self._recognizer = sr.Recognizer()
                self._recognizer.energy_threshold = 300
                self._recognizer.dynamic_energy_threshold = True
                self._recognizer.pause_threshold = 0.5
            except ImportError:
                raise ImportError(
                    "speech_recognition not installed. Install it with: "
                    "pip install SpeechRecognition"
                )
        return self._recognizer

    async def transcribe(self, audio_data: bytes, sample_rate: int) -> TranscriptionResult:
        start = time.perf_counter()

        # Convert raw bytes to AudioData
        import speech_recognition as sr

        audio = sr.AudioData(audio_data, sample_rate, 2)

        loop = asyncio.get_running_loop()

        def _recognize():
            try:
                text = self.recognizer.recognize_google(audio, language=self.config.language)
                return text, None
            except sr.UnknownValueError:
                return "", "Could not understand audio"
            except sr.RequestError as e:
                return "", f"Google API error: {e}"

        text, error = await loop.run_in_executor(None, _recognize)
        elapsed = (time.perf_counter() - start) * 1000

        return TranscriptionResult(
            text=text,
            is_final=True,
            confidence=1.0,
            latency_ms=elapsed,
            error=error,
        )


class OpenAIASREngine(ASREngine):
    """OpenAI Whisper API ASR backend."""

    def __init__(self, config: ASRConfig):
        super().__init__(config)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI

                self._client = AsyncOpenAI(api_key=self.config.openai_api_key)
            except ImportError:
                raise ImportError("openai package not installed: pip install openai")
        return self._client

    async def transcribe(self, audio_data: bytes, sample_rate: int) -> TranscriptionResult:
        start = time.perf_counter()

        # Convert PCM to WAV in-memory
        import wave

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)
        wav_buffer.seek(0)
        wav_buffer.name = "audio.wav"

        try:
            response = await self.client.audio.transcriptions.create(
                model=self.config.openai_model,
                file=wav_buffer,
                language=self.config.language[:2],
            )
            text = response.text.strip()
            error = None
        except Exception as e:
            text = ""
            error = str(e)

        elapsed = (time.perf_counter() - start) * 1000
        return TranscriptionResult(
            text=text,
            is_final=True,
            confidence=1.0,
            latency_ms=elapsed,
            error=error,
        )


class LocalWhisperASREngine(ASREngine):
    """Hugging Face Transformers Whisper ASR backend — runs locally on CPU/GPU."""

    def __init__(self, config: ASRConfig):
        super().__init__(config)
        self._pipeline = None
        self._processor = None

    async def _load_model(self):
        """Lazy-load the Whisper model. Use tiny for fast startup."""
        if self._pipeline is not None:
            return

        try:
            import torch
            from transformers import (
                AutoModelForSpeechSeq2Seq,
                AutoProcessor,
                pipeline,
            )

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            logger.info(
                f"Loading local Whisper model '{self.config.local_model_name}' "
                f"on {device}..."
            )
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.config.local_model_name,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            model.to(device)

            processor = AutoProcessor.from_pretrained(self.config.local_model_name)

            self._pipeline = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                chunk_length_s=30,
                return_timestamps=False,
                device=device,
            )
            logger.info("Local Whisper model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load local Whisper model: {e}")

    async def transcribe(self, audio_data: bytes, sample_rate: int) -> TranscriptionResult:
        start = time.perf_counter()
        await self._load_model()

        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        loop = asyncio.get_running_loop()

        def _run_inference():
            result = self._pipeline(audio_array)
            return result["text"].strip()

        try:
            text = await loop.run_in_executor(None, _run_inference)
            error = None if text else "No speech detected"
        except Exception as e:
            text = ""
            error = str(e)

        elapsed = (time.perf_counter() - start) * 1000
        return TranscriptionResult(
            text=text,
            is_final=True,
            confidence=1.0,
            latency_ms=elapsed,
            error=error,
        )


class MockASREngine(ASREngine):
    """Mock ASR engine for offline testing and benchmarking."""

    async def transcribe(self, audio_data: bytes, sample_rate: int) -> TranscriptionResult:
        start = time.perf_counter()
        # Simulate processing delay (100ms)
        await asyncio.sleep(0.1)
        elapsed = (time.perf_counter() - start) * 1000
        return TranscriptionResult(
            text="Hello from the user!",
            is_final=True,
            confidence=1.0,
            latency_ms=elapsed,
        )


def create_asr_engine(config: ASRConfig) -> ASREngine:
    """Factory function to create the appropriate ASR engine."""
    if config.backend == ASRBackend.GOOGLE:
        return GoogleASREngine(config)
    elif config.backend == ASRBackend.OPENAI:
        return OpenAIASREngine(config)
    elif config.backend == ASRBackend.LOCAL_WHISPER:
        return LocalWhisperASREngine(config)
    elif config.backend == ASRBackend.MOCK:
        return MockASREngine(config)
    else:
        raise ValueError(f"Unknown ASR backend: {config.backend}")
