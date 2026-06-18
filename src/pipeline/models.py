"""Data models for the streaming pipeline."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class PipelineStage(Enum):
    AUDIO_CAPTURE = "audio_capture"
    ASR = "asr"
    LLM_TTFT = "llm_time_to_first_token"
    LLM_TOTAL = "llm_total"
    TTS = "tts"
    AUDIO_PLAYBACK = "audio_playback"
    END_TO_END = "end_to_end"


@dataclass
class AudioChunk:
    """A chunk of audio data from the microphone."""

    data: bytes
    timestamp: float  # time.perf_counter()
    sample_rate: int
    is_speech: bool = True


@dataclass
class TranscriptionResult:
    """Result from the ASR engine."""

    text: str
    is_final: bool
    confidence: float = 1.0
    latency_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class LLMResponse:
    """A streaming token or complete response from the LLM."""

    text: str
    is_first_token: bool = False
    is_final: bool = False
    latency_ms: float = 0.0
    model: str = ""
    error: Optional[str] = None
    finish_reason: Optional[str] = None


@dataclass
class TTSResult:
    """Result from the TTS engine — either audio data or play signal."""

    audio_data: Optional[bytes] = None
    text: str = ""
    is_final: bool = False
    latency_ms: float = 0.0
    error: Optional[str] = None
    fallback_used: bool = False


@dataclass
class TurnMetrics:
    """Complete latency metrics for one voice interaction turn."""

    turn_number: int = 0
    timestamp: float = 0.0

    # Timing (all in milliseconds)
    audio_capture_ms: float = 0.0
    asr_ms: float = 0.0
    llm_ttft_ms: float = 0.0  # Time to first token
    llm_total_ms: float = 0.0
    tts_ms: float = 0.0
    audio_playback_ms: float = 0.0
    end_to_end_ms: float = 0.0

    # Additional info
    asr_backend: str = ""
    llm_backend: str = ""
    tts_backend: str = ""
    llm_model: str = ""
    error: Optional[str] = None
    degradation_used: bool = False

    # Raw text
    user_text: str = ""
    assistant_text: str = ""

    def as_dict(self) -> dict:
        return {
            "turn": self.turn_number,
            "asr_ms": round(self.asr_ms, 1),
            "llm_ttft_ms": round(self.llm_ttft_ms, 1),
            "llm_total_ms": round(self.llm_total_ms, 1),
            "tts_ms": round(self.tts_ms, 1),
            "e2e_ms": round(self.end_to_end_ms, 1),
            "asr_backend": self.asr_backend,
            "llm_backend": self.llm_backend,
            "tts_backend": self.tts_backend,
            "llm_model": self.llm_model,
            "error": self.error,
            "degradation": self.degradation_used,
        }


@dataclass
class PipelineState:
    """Current state of the pipeline."""

    is_running: bool = False
    is_muted: bool = False
    current_turn: int = 0
    last_error: Optional[str] = None
    degradation_mode: bool = False
    metrics_history: list[TurnMetrics] = field(default_factory=list)
