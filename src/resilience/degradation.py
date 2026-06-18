"""Graceful degradation manager.

Provides fallback chains and degradation strategies so the pipeline
degrades gracefully during partial service outages.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, AsyncGenerator, Awaitable, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class DegradationLevel(Enum):
    """How much the pipeline is degraded."""

    NONE = auto()  # Full service
    ASR_FALLBACK = auto()  # ASR using backup backend
    TTS_FALLBACK = auto()  # TTS using backup or text-only
    LLM_FALLBACK = auto()  # LLM using backup
    TEXT_ONLY = auto()  # Text response only, no audio
    DEGRADED = auto()  # Canned response
    OFFLINE = auto()  # Completely unavailable


@dataclass
class FallbackResult:
    """Result from a fallback chain."""

    success: bool
    value: Any = None
    error: Optional[str] = None
    fallback_used: bool = False
    stage: str = ""


class ServiceDegradationError(Exception):
    """Raised when all fallback options are exhausted."""

    def __init__(self, service: str, message: str):
        self.service = service
        super().__init__(message)


@dataclass
class FallbackChain:
    """A chain of fallback strategies for a service.

    Tries primary first, then falls back through the list of alternatives.
    """

    service_name: str
    primary: Callable[..., Awaitable[Any]]
    fallbacks: list[Callable[..., Awaitable[Any]]] = field(default_factory=list)
    final_fallback: Optional[Callable[..., Any]] = None

    async def execute(self, *args: Any, **kwargs: Any) -> FallbackResult:
        """Try each strategy in order, returning the first success."""
        strategies = [self.primary] + self.fallbacks

        for i, strategy in enumerate(strategies):
            try:
                result = await strategy(*args, **kwargs)
                logger.info(
                    f"FallbackChain '{self.service_name}': strategy {i} succeeded"
                )
                return FallbackResult(
                    success=True,
                    value=result,
                    fallback_used=i > 0,
                    stage=self.service_name,
                )
            except Exception as e:
                logger.warning(
                    f"FallbackChain '{self.service_name}': strategy {i} failed: {e}"
                )
                continue

        # All strategies failed — use final fallback if available
        if self.final_fallback:
            try:
                result = self.final_fallback(*args, **kwargs)
                if isinstance(result, Awaitable):
                    result = await result
                return FallbackResult(
                    success=True,
                    value=result,
                    fallback_used=True,
                    stage=self.service_name,
                )
            except Exception as e:
                logger.error(
                    f"FallbackChain '{self.service_name}': final fallback failed: {e}"
                )

        return FallbackResult(
            success=False,
            error=f"All {len(strategies)} strategies failed for '{self.service_name}'",
            stage=self.service_name,
        )


class DegradationManager:
    """Manages graceful degradation of the pipeline.

    Tracks current degradation level and decides fallback strategies
    when services fail.
    """

    def __init__(self):
        self.current_level = DegradationLevel.NONE
        self._service_status: dict[str, bool] = {}
        self._alert_callbacks: list[Callable[[str, DegradationLevel], None]] = []

    def register_alert(self, callback: Callable[[str, DegradationLevel], None]) -> None:
        """Register a callback for degradation alerts."""
        self._alert_callbacks.append(callback)

    def record_failure(self, service: str) -> None:
        """Record a service failure and potentially escalate degradation."""
        self._service_status[service] = False

        if service == "asr":
            self.current_level = DegradationLevel.ASR_FALLBACK
        elif service == "llm":
            self.current_level = DegradationLevel.LLM_FALLBACK
        elif service == "tts":
            if self.current_level in (DegradationLevel.NONE, DegradationLevel.ASR_FALLBACK):
                self.current_level = DegradationLevel.TTS_FALLBACK
            else:
                self.current_level = DegradationLevel.TEXT_ONLY

        self._notify(f"Service failure: {service}")

    def record_recovery(self, service: str) -> None:
        """Record a service recovery and potentially reduce degradation."""
        self._service_status[service] = True

        # Downgrade degradation level if all services are healthy
        if all(self._service_status.get(s, True) for s in ("asr", "llm", "tts")):
            old_level = self.current_level
            self.current_level = DegradationLevel.NONE
            if old_level != DegradationLevel.NONE:
                self._notify("All services recovered")

    def is_degraded(self) -> bool:
        return self.current_level != DegradationLevel.NONE

    def get_degradation_message(self) -> str:
        """Get a user-facing message about current degradation."""
        messages = {
            DegradationLevel.NONE: "",
            DegradationLevel.ASR_FALLBACK: "Speech recognition is running in fallback mode.",
            DegradationLevel.TTS_FALLBACK: "Speech synthesis is running in fallback mode.",
            DegradationLevel.LLM_FALLBACK: "Language model is running in fallback mode.",
            DegradationLevel.TEXT_ONLY: "Audio output unavailable. Response shown as text.",
            DegradationLevel.DEGRADED: "Some services are unavailable. Responses may be limited.",
            DegradationLevel.OFFLINE: "Pipeline is offline due to critical service failures.",
        }
        return messages.get(self.current_level, "")

    def _notify(self, message: str) -> None:
        """Notify all registered alert callbacks."""
        for callback in self._alert_callbacks:
            try:
                callback(message, self.current_level)
            except Exception as e:
                logger.error(f"Degradation alert callback failed: {e}")

    def reset(self) -> None:
        """Reset degradation state."""
        self.current_level = DegradationLevel.NONE
        self._service_status.clear()

    def __repr__(self) -> str:
        return (
            f"DegradationManager(level={self.current_level.name}, "
            f"services={self._service_status})"
        )
