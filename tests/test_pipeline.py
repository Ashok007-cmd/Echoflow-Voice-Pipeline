"""Tests for the real-time multimodal pipeline.

Covers:
  - Resilience: circuit breaker, timeout, degradation, fallback chain
  - Monitoring: latency tracker, pipeline timer
  - Pipeline: ASR, LLM, TTS factory creation and basic functionality
"""

import asyncio
import time

import pytest

from src.pipeline.models import (
    AudioChunk,
    LLMResponse,
    TranscriptionResult,
    TTSResult,
    TurnMetrics,
)
from src.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitState,
)
from src.resilience.degradation import DegradationManager, DegradationLevel, FallbackChain, FallbackResult
from src.resilience.timeout import PipelineTimeoutError, TimeoutSettings
from src.monitoring.latency_tracker import LatencyTracker, PipelineTimer


# ═══════════════════════════════════════════════════════════════════
# Phase 3 Tests: Resilience
# ═══════════════════════════════════════════════════════════════════

class TestCircuitBreaker:
    """Circuit breaker state machine tests."""

    @pytest.fixture
    def cb(self):
        return CircuitBreaker(
            CircuitBreakerConfig(
                service_name="test",
                failure_threshold=2,
                recovery_timeout=0.1,  # short for testing
                half_open_max_requests=1,
                consecutive_successes_to_close=2,
            )
        )

    @pytest.mark.asyncio
    async def test_initial_state_closed(self, cb):
        assert cb.state == CircuitState.CLOSED
        assert cb.is_available is True

    @pytest.mark.asyncio
    async def test_opens_after_threshold_failures(self, cb):
        async def fail():
            raise ValueError("service error")

        with pytest.raises(ValueError):
            await cb.call(fail)
        assert cb.state == CircuitState.CLOSED  # not yet at threshold

        with pytest.raises(ValueError):
            await cb.call(fail)
        assert cb.state == CircuitState.OPEN
        assert cb.is_available is False

    @pytest.mark.asyncio
    async def test_rejects_when_open(self, cb):
        async def fail():
            raise ValueError("service error")

        # Trip the breaker
        for _ in range(2):
            try:
                await cb.call(fail)
            except ValueError:
                pass

        assert cb.state == CircuitState.OPEN

        # Should be rejected without calling fail()
        with pytest.raises(CircuitBreakerOpenError):
            await cb.call(fail)

    @pytest.mark.asyncio
    async def test_half_open_after_recovery_timeout(self, cb):
        async def fail():
            raise ValueError("service error")

        # Trip the breaker
        for _ in range(2):
            try:
                await cb.call(fail)
            except ValueError:
                pass

        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Next call should transition to HALF_OPEN and let one request through
        with pytest.raises(ValueError):
            await cb.call(fail)
        assert cb.state == CircuitState.OPEN  # failed in half-open → back to open

    @pytest.mark.asyncio
    async def test_recovery_on_success(self, cb):
        call_count = 0

        async def sometimes_fail():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ValueError("service error")
            return "success"

        # Trip the breaker through cb.call() so failures are tracked
        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.call(sometimes_fail)

        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # First success in half-open
        result = await cb.call(sometimes_fail)
        assert result == "success"
        assert cb.state == CircuitState.HALF_OPEN  # needs 2 consecutive

        # Second success → closed
        result = await cb.call(sometimes_fail)
        assert result == "success"
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_success_resets_failure_count(self, cb):
        async def fail_once_then_succeed():
            if not hasattr(fail_once_then_succeed, "called"):
                fail_once_then_succeed.called = True
                raise ValueError("transient error")
            return "ok"

        # One failure
        with pytest.raises(ValueError):
            await cb.call(fail_once_then_succeed)

        # Success resets
        result = await cb.call(fail_once_then_succeed)
        assert result == "ok"
        assert cb._failure_count == 0


class TestTimeout:
    """Timeout utility tests."""

    @pytest.mark.asyncio
    async def test_with_timeout_success(self):
        from src.resilience.timeout import with_timeout
        async def quick():
            await asyncio.sleep(0.01)
            return "done"

        result = await with_timeout(
            quick(),
            settings=TimeoutSettings.for_llm(5.0),
        )
        assert result == "done"

    @pytest.mark.asyncio
    async def test_with_timeout_triggers(self):
        from src.resilience.timeout import with_timeout
        async def slow():
            await asyncio.sleep(10.0)
            return "never"

        settings = TimeoutSettings(stage_name="test", timeout_seconds=0.05, raise_on_timeout=True)

        with pytest.raises(PipelineTimeoutError):
            await with_timeout(
                slow(),
                settings=settings,
            )

    @pytest.mark.asyncio
    async def test_with_timeout_fallback(self):
        from src.resilience.timeout import with_timeout
        async def slow():
            await asyncio.sleep(10.0)
            return "never"

        settings = TimeoutSettings(stage_name="test", timeout_seconds=0.05, raise_on_timeout=False)

        result = await with_timeout(
            slow(),
            settings=settings,
            fallback_result="fallback_val",
        )
        assert result == "fallback_val"

    @pytest.mark.asyncio
    async def test_timeout_decorator(self):
        from src.resilience.timeout import timeout
        
        @timeout(TimeoutSettings(stage_name="dec_test", timeout_seconds=0.05))
        async def decorated_slow():
            await asyncio.sleep(10.0)
            return "never"

        with pytest.raises(PipelineTimeoutError):
            await decorated_slow()

    @pytest.mark.asyncio
    async def test_timeout_settings_factory(self):
        asr = TimeoutSettings.for_asr(10.0)
        assert asr.stage_name == "ASR"
        assert asr.timeout_seconds == 10.0

        llm = TimeoutSettings.for_llm(15.0)
        assert llm.stage_name == "LLM"
        assert llm.timeout_seconds == 15.0

        tts = TimeoutSettings.for_tts(10.0)
        assert tts.stage_name == "TTS"

        pipeline = TimeoutSettings.for_pipeline(40.0)
        assert pipeline.stage_name == "Pipeline"


class TestDegradationManager:
    """Graceful degradation tests."""

    @pytest.fixture
    def manager(self):
        return DegradationManager()

    def test_initial_state(self, manager):
        assert manager.current_level == DegradationLevel.NONE
        assert manager.is_degraded() is False

    def test_records_failure(self, manager):
        manager.record_failure("asr")
        assert manager.current_level == DegradationLevel.ASR_FALLBACK
        assert manager.is_degraded() is True

    def test_escalates_degradation(self, manager):
        manager.record_failure("asr")
        assert manager.current_level == DegradationLevel.ASR_FALLBACK

        manager.record_failure("llm")
        assert manager.current_level == DegradationLevel.LLM_FALLBACK

        manager.record_failure("tts")
        assert manager.current_level == DegradationLevel.TEXT_ONLY

    def test_recovers(self, manager):
        manager.record_failure("asr")
        assert manager.is_degraded() is True

        manager.record_recovery("asr")
        assert manager.is_degraded() is False

    def test_alert_callbacks(self, manager):
        alerts = []

        def alert_cb(msg, level):
            alerts.append((msg, level))

        manager.register_alert(alert_cb)
        manager.record_failure("asr")

        assert len(alerts) == 1
        assert "asr" in alerts[0][0]

    def test_reset(self, manager):
        manager.record_failure("asr")
        manager.record_failure("llm")
        assert manager.is_degraded() is True

        manager.reset()
        assert manager.is_degraded() is False
        assert manager.current_level == DegradationLevel.NONE


class TestFallbackChain:
    """Fallback chain tests."""

    @pytest.mark.asyncio
    async def test_primary_succeeds(self):
        async def primary():
            return "primary"

        chain = FallbackChain(
            service_name="test",
            primary=primary,
        )
        result = await chain.execute()
        assert result.success is True
        assert result.value == "primary"
        assert result.fallback_used is False

    @pytest.mark.asyncio
    async def test_fallback_on_failure(self):
        call_order = []

        async def primary():
            call_order.append("primary")
            raise ValueError("primary failed")

        async def fallback():
            call_order.append("fallback")
            return "fallback_result"

        chain = FallbackChain(
            service_name="test",
            primary=primary,
            fallbacks=[fallback],
        )
        result = await chain.execute()
        assert result.success is True
        assert result.value == "fallback_result"
        assert result.fallback_used is True
        assert call_order == ["primary", "fallback"]

    @pytest.mark.asyncio
    async def test_all_fail(self):
        async def fail1():
            raise ValueError("fail1")

        async def fail2():
            raise ValueError("fail2")

        chain = FallbackChain(
            service_name="test",
            primary=fail1,
            fallbacks=[fail2],
        )
        result = await chain.execute()
        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_final_fallback(self):
        async def always_fail():
            raise ValueError("fail")

        def final():
            return "final_fallback"

        chain = FallbackChain(
            service_name="test",
            primary=always_fail,
            fallbacks=[always_fail],
            final_fallback=final,
        )
        result = await chain.execute()
        assert result.success is True
        assert result.value == "final_fallback"
        assert result.fallback_used is True


# ═══════════════════════════════════════════════════════════════════
# Phase 2 Tests: Latency Monitoring
# ═══════════════════════════════════════════════════════════════════

class TestPipelineTimer:
    """Pipeline timer tests."""

    def test_start_stop(self):
        timer = PipelineTimer()
        timer.start()
        time.sleep(0.01)
        elapsed = timer.stop()
        assert elapsed >= 9.0  # at least 9ms
        assert isinstance(elapsed, float)

    def test_stop_without_start_raises(self):
        timer = PipelineTimer()
        with pytest.raises(RuntimeError):
            timer.stop()

    def test_reset(self):
        timer = PipelineTimer()
        timer.start()
        time.sleep(0.01)
        timer.stop()
        timer.reset()
        assert timer.elapsed_ms == 0.0

    def test_elapsed_property(self):
        timer = PipelineTimer()
        timer.start()
        time.sleep(0.01)
        timer.stop()
        assert timer.elapsed_ms > 0


class TestLatencyTracker:
    """Latency tracker tests."""

    @pytest.fixture
    def tracker(self):
        return LatencyTracker(window_size=10)

    def test_measure_stage(self, tracker):
        with tracker.measure_stage("test_stage"):
            time.sleep(0.01)

        summary = tracker.get_summary()
        assert "test_stage" in summary
        assert summary["test_stage"].count == 1
        assert summary["test_stage"].mean_ms >= 9.0

    def test_multiple_measurements(self, tracker):
        for _ in range(5):
            with tracker.measure_stage("asr"):
                time.sleep(0.005)

        summary = tracker.get_summary()
        assert summary["asr"].count == 5
        assert summary["asr"].mean_ms >= 4.0
        assert summary["asr"].min_ms <= summary["asr"].max_ms

    def test_turn_tracking(self, tracker):
        turn_num = tracker.start_turn()
        assert turn_num == 1

        metrics = tracker.complete_turn(
            asr_ms=100.0,
            llm_ttft_ms=200.0,
            llm_total_ms=500.0,
            tts_ms=300.0,
            e2e_ms=1000.0,
            asr_backend="google",
            llm_backend="openai",
            tts_backend="edge_tts",
        )

        assert metrics.turn_number == 1
        assert metrics.asr_ms == 100.0
        assert metrics.llm_ttft_ms == 200.0
        assert metrics.llm_total_ms == 500.0
        assert metrics.tts_ms == 300.0
        assert metrics.end_to_end_ms == 1000.0
        assert metrics.asr_backend == "google"

    def test_window_size(self, tracker):
        for i in range(20):
            with tracker.measure_stage("stage"):
                time.sleep(0.001)

        summary = tracker.get_summary()
        assert summary["stage"].count == 10  # capped at window_size

    def test_multiple_stages(self, tracker):
        stages = ["asr", "llm_ttft", "llm_total", "tts", "e2e"]
        for stage in stages:
            with tracker.measure_stage(stage):
                time.sleep(0.005)

        summary = tracker.get_summary()
        assert len(summary) == 5
        for stage in stages:
            assert stage in summary

    def test_reset(self, tracker):
        with tracker.measure_stage("stage"):
            pass
        tracker.reset()
        assert len(tracker.get_summary()) == 0
        assert len(tracker.get_turn_metrics()) == 0

    def test_empty_summary(self, tracker):
        summary = tracker.get_summary()
        assert summary == {}


# ═══════════════════════════════════════════════════════════════════
# Phase 1 Tests: Pipeline Models & Factories
# ═══════════════════════════════════════════════════════════════════

class TestPipelineModels:
    """Pipeline data model tests."""

    def test_audio_chunk(self):
        chunk = AudioChunk(data=b"test", timestamp=100.0, sample_rate=16000)
        assert chunk.data == b"test"
        assert chunk.timestamp == 100.0
        assert chunk.sample_rate == 16000

    def test_transcription_result(self):
        result = TranscriptionResult(text="hello world", is_final=True, latency_ms=150.0)
        assert result.text == "hello world"
        assert result.is_final is True
        assert result.latency_ms == 150.0

    def test_llm_response(self):
        response = LLMResponse(
            text="Hello", is_first_token=True, latency_ms=200.0, model="gpt-4"
        )
        assert response.text == "Hello"
        assert response.is_first_token is True
        assert response.is_final is False
        assert response.model == "gpt-4"

    def test_tts_result(self):
        result = TTSResult(audio_data=b"audio", text="hello", is_final=True, latency_ms=300.0)
        assert result.audio_data == b"audio"
        assert result.text == "hello"
        assert result.latency_ms == 300.0

    def test_turn_metrics_as_dict(self):
        metrics = TurnMetrics(
            turn_number=1,
            asr_ms=100.0,
            llm_ttft_ms=200.0,
            llm_total_ms=500.0,
            tts_ms=300.0,
            end_to_end_ms=1000.0,
            asr_backend="google",
            llm_backend="openai",
            tts_backend="edge_tts",
        )
        d = metrics.as_dict()
        assert d["turn"] == 1
        assert d["asr_ms"] == 100.0
        assert d["e2e_ms"] == 1000.0
        assert d["asr_backend"] == "google"


# ═══════════════════════════════════════════════════════════════════
# Pipeline Orchestrator Integration Test
# ═══════════════════════════════════════════════════════════════════

class TestPipelineOrchestrator:
    """Integration tests for the pipeline orchestrator."""

    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self):
        """Test that orchestrator can be created with default config."""
        from src.pipeline.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()
        assert orchestrator.state.is_running is False
        assert orchestrator.state.current_turn == 0

    @pytest.mark.asyncio
    async def test_orchestrator_start_stop(self):
        """Test lifecycle methods."""
        from src.pipeline.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()
        await orchestrator.start()
        assert orchestrator.state.is_running is True

        await orchestrator.stop()
        assert orchestrator.state.is_running is False

    @pytest.mark.asyncio
    async def test_orchestrator_shutdown(self):
        """Test graceful shutdown."""
        from src.pipeline.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()
        await orchestrator.start()
        await orchestrator.shutdown()
        assert orchestrator.state.is_running is False

    @pytest.mark.asyncio
    async def test_run_turn_with_mock_llm(self):
        """Test a full turn with mock backends and synthetic audio."""
        from src.config import PipelineConfig, LLMBackend, TTSBackend

        config = PipelineConfig()
        config.llm.backend = LLMBackend.MOCK
        config.tts.backend = TTSBackend.MOCK
        config.resilience.asr_timeout = 5.0
        config.resilience.llm_timeout = 10.0
        config.resilience.tts_timeout = 5.0

        from src.pipeline.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator(config)
        await orchestrator.start()

        # Synthetic audio: 1 second of near-silence
        import numpy as np
        audio = (np.random.randn(16000) * 50).astype(np.int16).tobytes()

        # The ASR will likely not detect speech in synthetic noise,
        # but the orchestrator should handle this gracefully
        metrics = await orchestrator.run_turn(audio, 16000)

        assert metrics is not None
        assert metrics.turn_number == 1
        assert isinstance(metrics.end_to_end_ms, float)

        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_multiple_turns_track_metrics(self):
        """Test that multiple turns accumulate metrics correctly."""
        from src.config import PipelineConfig, LLMBackend, TTSBackend
        from src.pipeline.orchestrator import PipelineOrchestrator

        config = PipelineConfig()
        config.llm.backend = LLMBackend.MOCK
        config.tts.backend = TTSBackend.MOCK

        orchestrator = PipelineOrchestrator(config)
        await orchestrator.start()

        import numpy as np
        audio = (np.random.randn(16000) * 50).astype(np.int16).tobytes()

        for i in range(3):
            metrics = await orchestrator.run_turn(audio, 16000)
            assert metrics is not None

        await orchestrator.stop()

        turn_metrics = orchestrator.latency_tracker.get_turn_metrics()
        assert len(turn_metrics) == 3

    @pytest.mark.asyncio
    async def test_orchestrator_with_mock_asr(self):
        """Test a turn where ASR, LLM, and TTS are all mocked."""
        from src.config import PipelineConfig, ASRBackend, LLMBackend, TTSBackend
        from src.pipeline.orchestrator import PipelineOrchestrator

        config = PipelineConfig()
        config.asr.backend = ASRBackend.MOCK
        config.llm.backend = LLMBackend.MOCK
        config.tts.backend = TTSBackend.MOCK

        orchestrator = PipelineOrchestrator(config)
        await orchestrator.start()

        # Synthetic audio
        import numpy as np
        audio = (np.random.randn(16000) * 50).astype(np.int16).tobytes()

        metrics = await orchestrator.run_turn(audio, 16000)
        assert metrics is not None
        assert metrics.user_text == "Hello from the user!"
        assert "mock" in metrics.assistant_text
        assert metrics.asr_ms > 0
        assert metrics.llm_ttft_ms > 0
        assert metrics.tts_ms > 0
        assert metrics.end_to_end_ms > 0
        assert metrics.error is None

        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_orchestrator_llm_circuit_breaker_integration(self):
        """Test that the LLM circuit breaker works when LLM stream fails."""
        from src.config import PipelineConfig, ASRBackend, LLMBackend, TTSBackend
        from src.pipeline.orchestrator import PipelineOrchestrator
        from src.resilience.circuit_breaker import CircuitState

        config = PipelineConfig()
        config.asr.backend = ASRBackend.MOCK
        config.llm.backend = LLMBackend.MOCK
        config.tts.backend = TTSBackend.MOCK
        config.resilience.circuit_breaker_failure_threshold = 2
        config.resilience.circuit_breaker_recovery_timeout = 5.0

        orchestrator = PipelineOrchestrator(config)
        await orchestrator.start()

        # Force LLM client to raise errors
        original_generate_stream = orchestrator.llm_client.generate_stream
        
        async def failing_generate_stream(user_text):
            raise ValueError("llm failure")
            yield  # make it a generator
            
        orchestrator.llm_client.generate_stream = failing_generate_stream

        # Execute turns and watch the breaker trip
        audio = b"\x00" * 32000
        
        # Turn 1: fails, breaker is CLOSED (1st failure)
        metrics1 = await orchestrator.run_turn(audio, 16000)
        assert metrics1 is not None
        assert metrics1.error == "LLM failed: llm failure"
        assert orchestrator.llm_circuit.state == CircuitState.CLOSED

        # Turn 2: fails, breaker trips to OPEN (2nd failure)
        metrics2 = await orchestrator.run_turn(audio, 16000)
        assert metrics2 is not None
        assert "llm failure" in metrics2.error
        assert orchestrator.llm_circuit.state == CircuitState.OPEN

        # Turn 3: short-circuits immediately with CircuitBreakerOpenError
        metrics3 = await orchestrator.run_turn(audio, 16000)
        assert metrics3 is not None
        assert "Circuit breaker open" in metrics3.error

        # Restore
        orchestrator.llm_client.generate_stream = original_generate_stream
        await orchestrator.stop()


# ═══════════════════════════════════════════════════════════════════
# Latency Visualization Tests
# ═══════════════════════════════════════════════════════════════════

class TestLatencyVisualizer:
    """Test latency visualization preparation."""

    @pytest.fixture
    def populated_tracker(self):
        tracker = LatencyTracker(window_size=10)

        # Simulate multiple turns
        for i in range(5):
            turn = tracker.start_turn()
            with tracker.measure_stage("asr"):
                pass
            with tracker.measure_stage("llm_time_to_first_token"):
                pass
            with tracker.measure_stage("llm_total"):
                pass
            with tracker.measure_stage("tts"):
                pass
            tracker.complete_turn(
                asr_ms=100 + i * 10,
                llm_ttft_ms=200 + i * 5,
                llm_total_ms=500 + i * 20,
                tts_ms=300 + i * 15,
                e2e_ms=1000 + i * 50,
            )

        return tracker

    def test_prepare_data(self, populated_tracker):
        from src.monitoring.visualizer import LatencyVisualizer

        viz = LatencyVisualizer(populated_tracker)
        data = viz.prepare_data()

        assert len(data.stage_names) > 0
        assert len(data.mean_latencies) > 0
        assert len(data.turn_numbers) == 5

    def test_generate_summary_table(self, populated_tracker):
        from src.monitoring.visualizer import generate_summary_table

        table = generate_summary_table(populated_tracker)
        assert "LLM" in table
        assert "TTS" in table
        assert "E2E" in table or "End-to-End" in table


# ═══════════════════════════════════════════════════════════════════
# Additional Latency Visualizer & Orchestrator Tests
# ═══════════════════════════════════════════════════════════════════

class TestLatencyVisualizerAdditional:
    def test_render_no_stage_names(self):
        from src.monitoring.visualizer import LatencyVisualizer
        tracker = LatencyTracker(window_size=10)
        viz = LatencyVisualizer(tracker)
        
        path = viz.render_to_file("output.png")
        assert path == ""

    def test_render_degenerate_data(self, tmp_path):
        import os
        from src.monitoring.visualizer import LatencyVisualizer
        tracker = LatencyTracker(window_size=10)
        tracker.start_turn()
        tracker.complete_turn(
            asr_ms=0.0001,
            llm_ttft_ms=0.0001,
            llm_total_ms=0.0001,
            tts_ms=0.0001,
            e2e_ms=0.0001
        )
        tracker.record_stage_value("asr", 0.0001)
        
        viz = LatencyVisualizer(tracker)
        output_png = str(tmp_path / "degenerate.png")
        path = viz.render_to_file(output_png)
        
        assert path == output_png
        assert os.path.exists(output_png)

    def test_plot_breakdown_empty(self):
        from src.monitoring.visualizer import LatencyVisualizer, VisualizationData
        tracker = LatencyTracker(window_size=10)
        viz = LatencyVisualizer(tracker)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        
        data = VisualizationData(
            stage_names=[],
            stage_keys=[],
            mean_latencies=[],
            p50_latencies=[],
            p90_latencies=[],
            p95_latencies=[],
            max_latencies=[],
            turn_numbers=[],
            per_turn_data={}
        )
        viz._plot_breakdown(ax, data)
        plt.close(fig)

    def test_plot_timeline_empty(self):
        from src.monitoring.visualizer import LatencyVisualizer, VisualizationData
        tracker = LatencyTracker(window_size=10)
        viz = LatencyVisualizer(tracker)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        data = VisualizationData(
            stage_names=["ASR"],
            stage_keys=["asr"],
            mean_latencies=[10.0],
            p50_latencies=[10.0],
            p90_latencies=[10.0],
            p95_latencies=[10.0],
            max_latencies=[10.0],
            turn_numbers=[],
            per_turn_data={}
        )
        viz._plot_timeline(ax, data)
        plt.close(fig)

    def test_plot_percentiles_empty(self):
        from src.monitoring.visualizer import LatencyVisualizer, VisualizationData
        tracker = LatencyTracker(window_size=10)
        viz = LatencyVisualizer(tracker)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        data = VisualizationData(
            stage_names=[],
            stage_keys=[],
            mean_latencies=[],
            p50_latencies=[],
            p90_latencies=[],
            p95_latencies=[],
            max_latencies=[],
            turn_numbers=[],
            per_turn_data={}
        )
        viz._plot_percentiles(ax, data)
        plt.close(fig)


class TestPipelineOrchestratorAdditional:
    @pytest.mark.asyncio
    async def test_run_turn_skips_when_degraded_on_turn_0(self):
        from src.pipeline.orchestrator import PipelineOrchestrator
        from src.config import PipelineConfig
        config = PipelineConfig()
        orchestrator = PipelineOrchestrator(config)
        
        orchestrator.state.degradation_mode = True
        orchestrator.state.current_turn = 0
        
        res = await orchestrator.run_turn(b"\x00" * 3200, 16000)
        assert res is None

    @pytest.mark.asyncio
    async def test_run_turn_asr_unexpected_error(self):
        from src.pipeline.orchestrator import PipelineOrchestrator
        from src.config import PipelineConfig
        from unittest.mock import AsyncMock

        config = PipelineConfig()
        orchestrator = PipelineOrchestrator(config)
        await orchestrator.start()
        
        orchestrator.asr_engine.transcribe = AsyncMock(side_effect=RuntimeError("unexpected asr crash"))
        
        res = await orchestrator.run_turn(b"\x00" * 3200, 16000)
        assert res is not None
        assert "unexpected asr crash" in res.error
        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_run_turn_asr_result_error(self):
        from src.pipeline.orchestrator import PipelineOrchestrator
        from src.config import PipelineConfig
        from src.pipeline.models import TranscriptionResult
        from unittest.mock import AsyncMock

        config = PipelineConfig()
        orchestrator = PipelineOrchestrator(config)
        await orchestrator.start()
        
        orchestrator.asr_engine.transcribe = AsyncMock(
            return_value=TranscriptionResult(text="", is_final=True, error="API key invalid")
        )
        
        res = await orchestrator.run_turn(b"\x00" * 3200, 16000)
        assert res is not None
        assert res.user_text == ""
        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_run_turn_empty_speech_skips_llm_tts(self):
        from src.pipeline.orchestrator import PipelineOrchestrator
        from src.config import PipelineConfig
        from src.pipeline.models import TranscriptionResult
        from unittest.mock import AsyncMock

        config = PipelineConfig()
        orchestrator = PipelineOrchestrator(config)
        await orchestrator.start()
        
        orchestrator.asr_engine.transcribe = AsyncMock(
            return_value=TranscriptionResult(text="   ", is_final=True)
        )
        
        res = await orchestrator.run_turn(b"\x00" * 3200, 16000)
        assert res is not None
        assert res.user_text == "   "
        assert res.assistant_text == ""
        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_run_turn_llm_unexpected_error(self):
        from src.pipeline.orchestrator import PipelineOrchestrator
        from src.config import PipelineConfig
        from src.pipeline.models import TranscriptionResult
        from unittest.mock import AsyncMock

        config = PipelineConfig()
        config.resilience.allow_empty_response_graceful = True
        config.resilience.degradation_message = "Fallback message"
        
        orchestrator = PipelineOrchestrator(config)
        await orchestrator.start()
        
        orchestrator.asr_engine.transcribe = AsyncMock(
            return_value=TranscriptionResult(text="test input", is_final=True)
        )
        async def failing_stream(user_text):
            raise RuntimeError("llm internal crash")
            yield None
        orchestrator.llm_client.generate_stream = failing_stream
        
        res = await orchestrator.run_turn(b"\x00" * 3200, 16000)
        assert res is not None
        assert "llm internal crash" in res.error
        assert res.assistant_text == "Fallback message"
        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_run_turn_tts_failure_with_fallback_to_text(self):
        from src.pipeline.orchestrator import PipelineOrchestrator
        from src.config import PipelineConfig
        from src.pipeline.models import TranscriptionResult, LLMResponse, TTSResult
        from unittest.mock import AsyncMock, patch

        config = PipelineConfig()
        config.resilience.allow_tts_fallback_to_text = True
        
        orchestrator = PipelineOrchestrator(config)
        await orchestrator.start()
        
        orchestrator.asr_engine.transcribe = AsyncMock(
            return_value=TranscriptionResult(text="test input", is_final=True)
        )
        async def mock_llm_stream(text):
            yield LLMResponse(text="Hello response.", is_first_token=True, is_final=False)
            yield LLMResponse(text="", is_final=True)
        orchestrator.llm_client.generate_stream = mock_llm_stream
        
        orchestrator.tts_engine.synthesize = AsyncMock(
            return_value=TTSResult(text="Hello response.", is_final=True, error="TTS server down")
        )
        
        with patch("click.style") as mock_style, patch("click.echo") as mock_echo:
            res = await orchestrator.run_turn(b"\x00" * 3200, 16000)
            
        assert res is not None
        assert res.degradation_used is True
        assert res.assistant_text == "Hello response."
        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_capture_audio_stream_muted(self):
        from src.pipeline.orchestrator import PipelineOrchestrator
        from src.config import PipelineConfig
        from unittest.mock import MagicMock, patch
        import sys
        
        config = PipelineConfig()
        orchestrator = PipelineOrchestrator(config)
        orchestrator.state.is_muted = True
        
        async def stop_later():
            await asyncio.sleep(0.05)
            orchestrator.state.is_running = False
            
        orchestrator.state.is_running = True
        asyncio.create_task(stop_later())
        
        chunks = []
        mock_pyaudio = MagicMock()
        mock_py = MagicMock()
        mock_stream = MagicMock()
        mock_stream.read.return_value = b"\x00" * 3200
        mock_py.open.return_value = mock_stream
        mock_pyaudio.PyAudio.return_value = mock_py
        
        with patch.dict(sys.modules, {"pyaudio": mock_pyaudio}):
            async for chunk in orchestrator.capture_audio_stream():
                chunks.append(chunk)
                
        assert len(chunks) == 0

    @pytest.mark.asyncio
    async def test_metrics_database_logging(self, tmp_path):
        import os
        import sqlite3
        from src.pipeline.orchestrator import PipelineOrchestrator
        from src.config import PipelineConfig, ASRBackend, LLMBackend, TTSBackend

        db_file = str(tmp_path / "test_metrics.db")
        config = PipelineConfig()
        config.asr.backend = ASRBackend.MOCK
        config.llm.backend = LLMBackend.MOCK
        config.tts.backend = TTSBackend.MOCK

        orchestrator = PipelineOrchestrator(config)
        from src.monitoring.database import MetricsDatabase
        orchestrator.db = MetricsDatabase(db_path=db_file)
        await orchestrator.start()

        metrics = await orchestrator.run_turn(b"\x00" * 32000, 16000)
        assert metrics is not None

        assert os.path.exists(db_file)
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT turn_number, asr_backend, user_text FROM turn_metrics")
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[0] == 1
        assert row[1] == "mock"
        assert row[2] == "Hello from the user!"

        await orchestrator.stop()


    @pytest.mark.asyncio
    async def test_capture_audio_stream_synthetic_fallback(self):
        from src.pipeline.orchestrator import PipelineOrchestrator
        from src.config import PipelineConfig
        from unittest.mock import patch
        import sys
        
        config = PipelineConfig()
        orchestrator = PipelineOrchestrator(config)
        
        with patch.dict(sys.modules, {"pyaudio": None}):
            orchestrator.state.is_running = True
            chunks = []
            async for chunk in orchestrator.capture_audio_stream():
                chunks.append(chunk)
                if len(chunks) >= 3:
                    orchestrator.state.is_running = False
            
            assert len(chunks) >= 3
            assert all(len(c.data) > 0 for c in chunks)


class TestPipelineConfig:
    def test_from_env_defaults(self, monkeypatch):
        from src.config import PipelineConfig, ASRBackend, LLMBackend, TTSBackend
        monkeypatch.delenv("ASR_BACKEND", raising=False)
        monkeypatch.delenv("LLM_BACKEND", raising=False)
        monkeypatch.delenv("TTS_BACKEND", raising=False)
        monkeypatch.delenv("SYSTEM_PROMPT", raising=False)
        
        config = PipelineConfig.from_env()
        assert config.asr.backend == ASRBackend.GOOGLE
        assert config.llm.backend == LLMBackend.OPENAI
        assert config.tts.backend == TTSBackend.EDGE_TTS

    def test_from_env_custom(self, monkeypatch):
        from src.config import PipelineConfig, ASRBackend, LLMBackend, TTSBackend
        monkeypatch.setenv("ASR_BACKEND", "local_whisper")
        monkeypatch.setenv("LLM_BACKEND", "anthropic")
        monkeypatch.setenv("TTS_BACKEND", "pyttsx3")
        monkeypatch.setenv("SYSTEM_PROMPT", "Test Prompt")
        
        config = PipelineConfig.from_env()
        assert config.asr.backend == ASRBackend.LOCAL_WHISPER
        assert config.llm.backend == LLMBackend.ANTHROPIC
        assert config.tts.backend == TTSBackend.PYTTSX3
        assert config.llm.system_prompt == "Test Prompt"

    def test_from_env_mock(self, monkeypatch):
        from src.config import PipelineConfig, ASRBackend, LLMBackend, TTSBackend
        monkeypatch.setenv("ASR_BACKEND", "mock")
        monkeypatch.setenv("LLM_BACKEND", "mock")
        monkeypatch.setenv("TTS_BACKEND", "mock")
        
        config = PipelineConfig.from_env()
        assert config.asr.backend == ASRBackend.MOCK
        assert config.llm.backend == LLMBackend.MOCK
        assert config.tts.backend == TTSBackend.MOCK


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
