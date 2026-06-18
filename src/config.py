"""Global configuration for the real-time multimodal pipeline."""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    if os.path.exists(".env"):
        with open(".env") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip().strip('"').strip("'")



class ASRBackend(str, Enum):
    GOOGLE = "google"
    OPENAI = "openai"
    LOCAL_WHISPER = "local_whisper"
    MOCK = "mock"


class LLMBackend(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MOCK = "mock"


class TTSBackend(str, Enum):
    EDGE_TTS = "edge_tts"
    PYTTSX3 = "pyttsx3"
    MOCK = "mock"


@dataclass
class ASRConfig:
    backend: ASRBackend = ASRBackend.GOOGLE
    model: str = "whisper-1"
    language: str = "en-US"
    sample_rate: int = 16000
    chunk_duration: float = 0.5  # seconds per audio chunk
    silence_threshold: float = 0.5  # seconds of silence to mark utterance end
    vad_enabled: bool = True
    max_utterance_seconds: float = 30.0

    # OpenAI Whisper API
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY")
    )
    openai_model: str = "whisper-1"

    # Local Whisper (via transformers)
    local_model_name: str = "openai/whisper-small" if os.environ.get("CI") else "openai/whisper-tiny"

    def __repr__(self) -> str:
        masked_key = "None"
        if self.openai_api_key:
            masked_key = f"{self.openai_api_key[:4]}...{self.openai_api_key[-4:]}" if len(self.openai_api_key) > 8 else "..."
        return (
            f"ASRConfig(backend={self.backend.value}, model={self.model}, "
            f"language={self.language}, sample_rate={self.sample_rate}, "
            f"chunk_duration={self.chunk_duration}, silence_threshold={self.silence_threshold}, "
            f"vad_enabled={self.vad_enabled}, max_utterance_seconds={self.max_utterance_seconds}, "
            f"openai_api_key={masked_key}, openai_model={self.openai_model}, "
            f"local_model_name={self.local_model_name})"
        )


@dataclass
class LLMConfig:
    backend: LLMBackend = LLMBackend.OPENAI
    model: str = "gpt-4o-mini"
    system_prompt: str = "You are a helpful voice assistant. Keep responses concise and conversational."
    temperature: float = 0.7
    max_tokens: int = 512
    stream: bool = True

    # OpenAI
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY")
    )
    openai_model: str = "gpt-4o-mini"

    # Anthropic
    anthropic_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY")
    )
    anthropic_model: str = "claude-3-5-haiku-latest"

    def __repr__(self) -> str:
        def mask(key):
            if not key:
                return "None"
            return f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "..."
        return (
            f"LLMConfig(backend={self.backend.value}, model={self.model}, "
            f"temperature={self.temperature}, max_tokens={self.max_tokens}, "
            f"stream={self.stream}, openai_api_key={mask(self.openai_api_key)}, "
            f"openai_model={self.openai_model}, anthropic_api_key={mask(self.anthropic_api_key)}, "
            f"anthropic_model={self.anthropic_model})"
        )



@dataclass
class TTSConfig:
    backend: TTSBackend = TTSBackend.EDGE_TTS
    voice: str = "en-US-JennyNeural"
    rate: int = 0  # +/- percentage
    volume: int = 0  # +/- percentage
    pitch: int = 0  # +/- Hz
    streaming: bool = True
    language: str = "en-US"

    # pyttsx3
    pyttsx3_rate: int = 180
    pyttsx3_volume: float = 1.0


@dataclass
class ResilienceConfig:
    # Timeouts (seconds)
    asr_timeout: float = 10.0
    llm_timeout: float = 15.0
    tts_timeout: float = 10.0
    total_pipeline_timeout: float = 40.0

    # Circuit breaker
    circuit_breaker_failure_threshold: int = 3
    circuit_breaker_recovery_timeout: float = 30.0

    # Retry
    max_retries: int = 2
    retry_base_delay: float = 1.0
    retry_max_delay: float = 10.0

    # Degradation
    allow_tts_fallback_to_text: bool = True
    allow_asr_fallback: bool = True
    allow_empty_response_graceful: bool = True
    degradation_message: str = "I'm sorry, I'm having trouble processing your request right now."


@dataclass
class MonitoringConfig:
    latency_window: int = 100  # number of turns to keep in latency history
    visualization_output: str = "latency_report.png"
    log_latency: bool = True
    detailed_tracing: bool = True


@dataclass
class AudioConfig:
    input_device_index: Optional[int] = None
    output_device_index: Optional[int] = None
    input_sample_rate: int = 16000
    output_sample_rate: int = 24000
    channels: int = 1
    input_chunk_size: int = 1024
    output_chunk_size: int = 1024


@dataclass
class PipelineConfig:
    asr: ASRConfig = field(default_factory=ASRConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    resilience: ResilienceConfig = field(default_factory=ResilienceConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)

    def validate(self) -> None:
        """Validate configuration values and presence of required API keys."""
        if self.asr.backend == ASRBackend.OPENAI and not self.asr.openai_api_key:
            raise ValueError(
                "OpenAI API key is required when using OpenAI ASR backend. "
                "Please set the OPENAI_API_KEY environment variable or define it in your .env file."
            )
        if self.llm.backend == LLMBackend.OPENAI and not self.llm.openai_api_key:
            raise ValueError(
                "OpenAI API key is required when using OpenAI LLM backend. "
                "Please set the OPENAI_API_KEY environment variable or define it in your .env file."
            )
        if self.llm.backend == LLMBackend.ANTHROPIC and not self.llm.anthropic_api_key:
            raise ValueError(
                "Anthropic API key is required when using Anthropic LLM backend. "
                "Please set the ANTHROPIC_API_KEY environment variable or define it in your .env file."
            )

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Load config from environment variables with sensible defaults."""
        cfg = cls()

        # ASR backend
        asr_backend = os.environ.get("ASR_BACKEND", "google").lower()
        if asr_backend in ("openai", "whisper"):
            cfg.asr.backend = ASRBackend.OPENAI
        elif asr_backend in ("local", "local_whisper"):
            cfg.asr.backend = ASRBackend.LOCAL_WHISPER
        elif asr_backend == "mock":
            cfg.asr.backend = ASRBackend.MOCK
        else:
            cfg.asr.backend = ASRBackend.GOOGLE

        # LLM backend
        llm_backend = os.environ.get("LLM_BACKEND", "openai").lower()
        if llm_backend == "anthropic":
            cfg.llm.backend = LLMBackend.ANTHROPIC
        elif llm_backend == "mock":
            cfg.llm.backend = LLMBackend.MOCK
        else:
            cfg.llm.backend = LLMBackend.OPENAI

        # TTS backend
        tts_backend = os.environ.get("TTS_BACKEND", "edge_tts").lower()
        if tts_backend == "pyttsx3":
            cfg.tts.backend = TTSBackend.PYTTSX3
        elif tts_backend == "mock":
            cfg.tts.backend = TTSBackend.MOCK
        else:
            cfg.tts.backend = TTSBackend.EDGE_TTS

        cfg.llm.system_prompt = os.environ.get(
            "SYSTEM_PROMPT", cfg.llm.system_prompt
        )

        return cfg


DEFAULT_CONFIG = PipelineConfig()
