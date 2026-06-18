"""Precision latency tracking for pipeline stages.

Measures time spent in each pipeline stage with nanosecond precision
and provides statistical summaries (percentiles, mean, median, etc.).
"""

import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional

import numpy as np

from src.pipeline.models import TurnMetrics

logger = logging.getLogger(__name__)


@dataclass
class LatencySummary:
    """Statistical summary of latency measurements for a stage."""

    stage_name: str
    count: int = 0
    mean_ms: float = 0.0
    median_ms: float = 0.0
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    std_ms: float = 0.0
    values_ms: list[float] = field(default_factory=list)

    @classmethod
    def from_values(cls, stage_name: str, values: list[float]) -> "LatencySummary":
        if not values:
            return cls(stage_name=stage_name)
        arr = np.array(values)
        return cls(
            stage_name=stage_name,
            count=len(values),
            mean_ms=float(np.mean(arr)),
            median_ms=float(np.median(arr)),
            p50_ms=float(np.percentile(arr, 50)),
            p90_ms=float(np.percentile(arr, 90)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
            std_ms=float(np.std(arr)),
            values_ms=list(arr),
        )


class LatencyTracker:
    """Tracks latency for each pipeline stage across multiple turns.

    Provides nanosecond-precision timing and statistical aggregation.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._stages: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window_size))
        self._turn_metrics: deque[TurnMetrics] = deque(maxlen=window_size)
        self._current_turn: Optional[TurnMetrics] = None
        self._turn_count = 0

    @contextmanager
    def measure_stage(self, stage_name: str) -> "LatencyTracker":
        """Context manager to measure the duration of a code block.

        Usage:
            with latency_tracker.measure_stage("asr"):
                result = await asr.transcribe(audio)
        """
        start = time.perf_counter_ns()
        try:
            yield self
        finally:
            elapsed_ns = time.perf_counter_ns() - start
            elapsed_ms = elapsed_ns / 1_000_000
            self._record(stage_name, elapsed_ms)

    @asynccontextmanager
    async def ameasure_stage(self, stage_name: str):
        """Async context manager to measure a coroutine's duration."""
        start = time.perf_counter_ns()
        try:
            yield self
        finally:
            elapsed_ns = time.perf_counter_ns() - start
            elapsed_ms = elapsed_ns / 1_000_000
            self._record(stage_name, elapsed_ms)

    def _record(self, stage: str, elapsed_ms: float) -> None:
        """Record a latency measurement for a stage."""
        self._stages[stage].append(elapsed_ms)

    def record_stage_value(self, stage_name: str, latency_ms: float) -> None:
        """Record an explicit latency value for a stage (bypasses context manager)."""
        if latency_ms > 0:
            self._record(stage_name, latency_ms)

    def start_turn(self) -> int:
        """Start tracking a new interaction turn."""
        self._turn_count += 1
        self._current_turn = TurnMetrics(
            turn_number=self._turn_count,
            timestamp=time.time(),
        )
        return self._turn_count

    def complete_turn(
        self,
        asr_ms: float = 0.0,
        llm_ttft_ms: float = 0.0,
        llm_total_ms: float = 0.0,
        tts_ms: float = 0.0,
        audio_playback_ms: float = 0.0,
        e2e_ms: float = 0.0,
        asr_backend: str = "",
        llm_backend: str = "",
        tts_backend: str = "",
        llm_model: str = "",
        error: Optional[str] = None,
        degradation_used: bool = False,
        user_text: str = "",
        assistant_text: str = "",
    ) -> TurnMetrics:
        """Finalize the current turn's metrics."""
        if self._current_turn is None:
            self.start_turn()

        self._current_turn.asr_ms = asr_ms
        self._current_turn.llm_ttft_ms = llm_ttft_ms
        self._current_turn.llm_total_ms = llm_total_ms
        self._current_turn.tts_ms = tts_ms
        self._current_turn.audio_playback_ms = audio_playback_ms
        self._current_turn.end_to_end_ms = e2e_ms
        self._current_turn.asr_backend = asr_backend
        self._current_turn.llm_backend = llm_backend
        self._current_turn.tts_backend = tts_backend
        self._current_turn.llm_model = llm_model
        self._current_turn.error = error
        self._current_turn.degradation_used = degradation_used
        self._current_turn.user_text = user_text
        self._current_turn.assistant_text = assistant_text

        self._turn_metrics.append(self._current_turn)

        metrics = self._current_turn
        self._current_turn = None
        return metrics

    def get_summary(self) -> dict[str, LatencySummary]:
        """Get statistical summary for all tracked stages."""
        return {
            stage: LatencySummary.from_values(stage, list(values))
            for stage, values in self._stages.items()
        }

    def get_turn_metrics(self) -> list[TurnMetrics]:
        """Get all recorded turn metrics."""
        return list(self._turn_metrics)

    def reset(self) -> None:
        """Clear all tracked data."""
        self._stages.clear()
        self._turn_metrics.clear()
        self._current_turn = None
        self._turn_count = 0


class PipelineTimer:
    """High-precision timer for measuring specific pipeline stages.

    Provides manual start/stop points for more control over what's timed.
    """

    def __init__(self):
        self._start_time: Optional[float] = None
        self._elapsed_ms: float = 0.0

    def start(self) -> None:
        """Start the timer."""
        self._start_time = time.perf_counter_ns()

    def stop(self) -> float:
        """Stop the timer and return elapsed milliseconds."""
        if self._start_time is None:
            raise RuntimeError("Timer was not started")
        elapsed_ns = time.perf_counter_ns() - self._start_time
        self._elapsed_ms = elapsed_ns / 1_000_000
        self._start_time = None
        return self._elapsed_ms

    @property
    def elapsed_ms(self) -> float:
        """Return elapsed ms so far (if running) or from last stop."""
        if self._start_time is not None:
            return (time.perf_counter_ns() - self._start_time) / 1_000_000
        return self._elapsed_ms

    def reset(self) -> None:
        self._start_time = None
        self._elapsed_ms = 0.0
