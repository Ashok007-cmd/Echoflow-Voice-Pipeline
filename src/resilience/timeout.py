"""Timeout handling utilities.

Provides configurable timeouts for each pipeline stage to
prevent indefinite hangs during service outages.
"""

import asyncio
import functools
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class PipelineTimeoutError(asyncio.TimeoutError):
    """Raised when a pipeline stage exceeds its timeout threshold."""

    def __init__(self, stage: str, timeout: float, message: Optional[str] = None):
        self.stage = stage
        self.timeout = timeout
        self.message = message or f"{stage} timed out after {timeout:.1f}s"
        super().__init__(self.message)


@dataclass
class TimeoutSettings:
    """Timeout configuration for a pipeline stage."""

    stage_name: str
    timeout_seconds: float
    graceful_message: str = ""
    raise_on_timeout: bool = True

    @classmethod
    def for_asr(cls, timeout_seconds: float = 10.0) -> "TimeoutSettings":
        return cls(
            stage_name="ASR",
            timeout_seconds=timeout_seconds,
            graceful_message="Speech recognition timed out. Please try again.",
        )

    @classmethod
    def for_llm(cls, timeout_seconds: float = 15.0) -> "TimeoutSettings":
        return cls(
            stage_name="LLM",
            timeout_seconds=timeout_seconds,
            graceful_message="Language model response timed out.",
        )

    @classmethod
    def for_tts(cls, timeout_seconds: float = 10.0) -> "TimeoutSettings":
        return cls(
            stage_name="TTS",
            timeout_seconds=timeout_seconds,
            graceful_message="Speech synthesis timed out.",
        )

    @classmethod
    def for_pipeline(cls, timeout_seconds: float = 40.0) -> "TimeoutSettings":
        return cls(
            stage_name="Pipeline",
            timeout_seconds=timeout_seconds,
            graceful_message="The full pipeline took too long to respond.",
        )


async def with_timeout(
    coro: asyncio.Future | asyncio.Task,
    settings: TimeoutSettings,
    fallback_result: Optional[T] = None,
) -> T:
    """Execute a coroutine with a timeout.

    Args:
        coro: The coroutine to execute.
        settings: Timeout configuration.
        fallback_result: Optional fallback value if timeout occurs.

    Returns:
        The coroutine result, or fallback_result if timeout occurs
        and raise_on_timeout is False.

    Raises:
        PipelineTimeoutError: If timeout occurs and raise_on_timeout is True.
    """
    try:
        result = await asyncio.wait_for(coro, timeout=settings.timeout_seconds)
        return result
    except asyncio.TimeoutError:
        logger.warning(
            f"Timeout in {settings.stage_name} after {settings.timeout_seconds:.1f}s"
        )
        if settings.raise_on_timeout:
            raise PipelineTimeoutError(
                stage=settings.stage_name,
                timeout=settings.timeout_seconds,
            )
        return fallback_result


def timeout(settings: TimeoutSettings):
    """Decorator to add timeout to an async function.

    Usage:
        @timeout(TimeoutSettings.for_llm(10.0))
        async def call_llm(text: str) -> str:
            ...
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            coro = func(*args, **kwargs)
            return await with_timeout(coro, settings)

        return wrapper

    return decorator
