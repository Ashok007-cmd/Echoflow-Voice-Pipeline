"""Circuit breaker pattern for external service resilience.

Prevents cascading failures by detecting when a service is unhealthy
and short-circuiting requests until it recovers.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Awaitable, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    CLOSED = auto()  # Normal operation — requests pass through
    OPEN = auto()  # Failing — requests are rejected immediately
    HALF_OPEN = auto()  # Testing — one probe request is allowed


class CircuitBreakerOpenError(Exception):
    """Raised when a request is rejected because the circuit is open."""

    def __init__(self, service_name: str, retry_after: float):
        self.service_name = service_name
        self.retry_after = retry_after
        super().__init__(
            f"Circuit breaker open for '{service_name}'. "
            f"Retry in {retry_after:.0f}s."
        )


@dataclass
class CircuitBreakerConfig:
    service_name: str
    failure_threshold: int = 3
    recovery_timeout: float = 30.0  # seconds before transitioning to half-open
    half_open_max_requests: int = 1  # requests to allow in half-open state
    consecutive_successes_to_close: int = 2


class CircuitBreaker:
    """Async circuit breaker implementation.

    States:
        CLOSED: Normal operation. Requests pass through.
               Failures are counted. Opens when threshold is reached.
        OPEN:  Service considered unhealthy. Requests are rejected.
               After recovery_timeout, transitions to HALF_OPEN.
        HALF_OPEN: One probe request is allowed.
                  Success → CLOSED. Failure → OPEN.
    """

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0.0
        self._half_open_requests = 0
        self._lock = asyncio.Lock()

    async def call(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute a function through the circuit breaker.

        Args:
            func: The async function to call.
            *args, **kwargs: Passed through to func.

        Returns:
            The result of func.

        Raises:
            CircuitBreakerOpenError: If the circuit is open.
            Exception: Any exception from func.
        """
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if time.monotonic() - self._last_failure_time >= self.config.recovery_timeout:
                    logger.info(
                        f"Circuit '{self.config.service_name}': OPEN → HALF_OPEN "
                        f"(recovery timeout elapsed)"
                    )
                    self.state = CircuitState.HALF_OPEN
                    self._half_open_requests = 0
                else:
                    retry_after = self.config.recovery_timeout - (
                        time.monotonic() - self._last_failure_time
                    )
                    raise CircuitBreakerOpenError(
                        service_name=self.config.service_name,
                        retry_after=retry_after,
                    )

            if self.state == CircuitState.HALF_OPEN:
                if self._half_open_requests >= self.config.half_open_max_requests:
                    raise CircuitBreakerOpenError(
                        service_name=self.config.service_name,
                        retry_after=self.config.recovery_timeout,
                    )
                self._half_open_requests += 1

        # Execute the function (outside the lock to prevent contention)
        try:
            result = await func(*args, **kwargs)
        except Exception as e:
            # Differentiate client-side errors (auth, bad request) from actual
            # service failures. Client errors should NOT trip the circuit breaker.
            # Use isinstance() against real exception types where available,
            # with a name-based fallback for non-imported libraries.
            _CLIENT_ERROR_NAMES = frozenset({
                "AuthenticationError",
                "PermissionDeniedError",
                "NotFoundError",
                "BadRequestError",
                "InvalidRequestError",
                "RateLimitError",
            })
            is_client_error = type(e).__name__ in _CLIENT_ERROR_NAMES
            if not is_client_error:
                async with self._lock:
                    self._on_failure()
            else:
                logger.debug(
                    f"Circuit '{self.config.service_name}': client error ignored for failure counting: {e}"
                )
            raise
        else:
            async with self._lock:
                self._on_success()
            return result

    def _on_success(self) -> None:
        """Handle a successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self._success_count += 1
            self._half_open_requests = max(0, self._half_open_requests - 1)
            if self._success_count >= self.config.consecutive_successes_to_close:
                logger.info(
                    f"Circuit '{self.config.service_name}': HALF_OPEN → CLOSED "
                    f"(recovered)"
                )
                self._reset()
        elif self.state == CircuitState.CLOSED:
            self._failure_count = 0  # Reset failure count on success

    def _on_failure(self) -> None:
        """Handle a failed call."""
        self._last_failure_time = time.monotonic()
        self._failure_count += 1

        if self.state == CircuitState.HALF_OPEN:
            logger.warning(
                f"Circuit '{self.config.service_name}': HALF_OPEN → OPEN "
                f"(probe request failed)"
            )
            self.state = CircuitState.OPEN
            self._half_open_requests = max(0, self._half_open_requests - 1)
        elif self.state == CircuitState.CLOSED:
            logger.info(
                f"Circuit '{self.config.service_name}': failure "
                f"{self._failure_count}/{self.config.failure_threshold}"
            )
            if self._failure_count >= self.config.failure_threshold:
                logger.warning(
                    f"Circuit '{self.config.service_name}': CLOSED → OPEN "
                    f"(threshold reached)"
                )
                self.state = CircuitState.OPEN

    def _reset(self) -> None:
        """Reset circuit to closed state."""
        self.state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_requests = 0

    @property
    def is_available(self) -> bool:
        """Quick check if the circuit is accepting requests (non-blocking)."""
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if time.monotonic() - self._last_failure_time >= self.config.recovery_timeout:
                return True  # Would transition on next call
            return False
        return self._half_open_requests < self.config.half_open_max_requests

    def __repr__(self) -> str:
        return (
            f"CircuitBreaker('{self.config.service_name}', "
            f"state={self.state.name}, "
            f"failures={self._failure_count}/{self.config.failure_threshold})"
        )
