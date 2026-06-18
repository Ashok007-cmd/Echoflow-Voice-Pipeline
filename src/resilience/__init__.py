"""Resilience module: timeouts, circuit breaker, and graceful degradation.

Phase 3 of the project: System resilience with robust timeout handling
and graceful degradation strategies to prevent indefinite system hangs.
"""

from src.resilience.timeout import (
    timeout,
    TimeoutSettings,
    with_timeout,
    PipelineTimeoutError,
)
from src.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitBreakerOpenError,
)
from src.resilience.degradation import (
    DegradationManager,
    FallbackChain,
    FallbackResult,
    ServiceDegradationError,
)

__all__ = [
    "timeout",
    "TimeoutSettings",
    "with_timeout",
    "PipelineTimeoutError",
    "CircuitBreaker",
    "CircuitState",
    "CircuitBreakerOpenError",
    "DegradationManager",
    "FallbackChain",
    "FallbackResult",
    "ServiceDegradationError",
]
