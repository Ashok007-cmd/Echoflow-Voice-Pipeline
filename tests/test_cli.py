"""Integration tests for the CLI interface."""

import os
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd
import pytest
from click.testing import CliRunner

from src.main import cli


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Real-Time Multimodal Voice Assistant" in result.output
    assert "benchmark" in result.output
    assert "interactive" in result.output
    assert "report" in result.output


def test_benchmark_command_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["benchmark", "--help"])
    assert result.exit_code == 0
    assert "turns" in result.output
    assert "mock-llm" in result.output


def test_benchmark_command_runs(tmp_path):
    output_png = str(tmp_path / "test_benchmark.png")
    runner = CliRunner()
    result = runner.invoke(cli, [
        "benchmark",
        "--turns", "2",
        "--mock-asr",
        "--mock-llm",
        "--mock-tts",
        "--output", output_png,
    ])
    assert result.exit_code == 0
    assert "Latendy Benchmark" in result.output or "Latency Benchmark" in result.output
    assert "Visualization saved to" in result.output
    assert os.path.exists(output_png)
    assert os.path.getsize(output_png) > 0


def test_report_command_no_args():
    runner = CliRunner()
    result = runner.invoke(cli, ["report"])
    assert result.exit_code == 0
    assert "To generate a lateny report" in result.output or "To generate a latency report" in result.output


def test_report_command_with_csv(tmp_path):
    csv_file = tmp_path / "metrics.csv"
    # Create synthetic metric dataframe
    df = pd.DataFrame({
        "turn": [1, 2],
        "asr_ms": [100.0, 110.0],
        "llm_ttft_ms": [200.0, 190.0],
        "llm_total_ms": [500.0, 480.0],
        "tts_ms": [300.0, 280.0],
        "e2e_ms": [1000.0, 950.0],
        "asr_backend": ["mock", "mock"],
        "llm_backend": ["mock", "mock"],
        "tts_backend": ["mock", "mock"],
        "llm_model": ["mock", "mock"],
        "error": [None, None],
        "degradation": [False, False],
    })
    df.to_csv(csv_file, index=False)

    runner = CliRunner()
    result = runner.invoke(cli, ["report", str(csv_file)])
    assert result.exit_code == 0
    assert "Loaded 2 metrics from" in result.output
    assert "asr_ms" in result.output


@patch("src.main.PipelineOrchestrator")
def test_interactive_command_mocked(mock_orchestrator_class):
    mock_orchestrator = MagicMock()
    mock_orchestrator.start = AsyncMock()
    mock_orchestrator.stop = AsyncMock()
    mock_orchestrator.capture_audio_stream = MagicMock()
    
    # Simulate zero chunks yielded to exit immediately
    async def empty_stream():
        if False:
            yield None
            
    mock_orchestrator.capture_audio_stream.return_value = empty_stream()
    mock_orchestrator.latency_tracker = MagicMock()
    mock_orchestrator.latency_tracker.get_turn_metrics.return_value = []
    
    mock_orchestrator_class.return_value = mock_orchestrator

    runner = CliRunner()
    result = runner.invoke(cli, ["interactive", "--mock"])
    assert result.exit_code == 0
    assert "Listening... (Ctrl+C to stop)" in result.output


@patch("src.main.PipelineOrchestrator")
def test_interactive_command_with_audio(mock_orchestrator_class):
    from src.pipeline.models import AudioChunk, TurnMetrics

    mock_orchestrator = MagicMock()
    mock_orchestrator.start = AsyncMock()
    mock_orchestrator.stop = AsyncMock()
    
    # We yield: 1 chunk of speech, then 2 chunks of silence to trigger VAD utterance boundary
    speech_chunk = AudioChunk(data=b"\xff\x7f" * 1024, timestamp=0.0, sample_rate=16000)  # loud
    silence_chunk = AudioChunk(data=b"\x00" * 2048, timestamp=0.0, sample_rate=16000)  # silent
    
    async def mock_stream():
        yield speech_chunk
        yield silence_chunk
        yield silence_chunk

    mock_orchestrator.capture_audio_stream.return_value = mock_stream()
    
    # Mock turn processing and metrics
    mock_metrics = TurnMetrics(
        turn_number=1,
        asr_ms=10.0,
        llm_ttft_ms=20.0,
        llm_total_ms=50.0,
        tts_ms=30.0,
        end_to_end_ms=100.0,
        user_text="test input",
        assistant_text="test response",
    )
    mock_orchestrator.run_turn = AsyncMock(return_value=mock_metrics)
    
    mock_orchestrator.latency_tracker = MagicMock()
    mock_orchestrator.latency_tracker.get_turn_metrics.return_value = [mock_metrics]
    
    mock_orchestrator_class.return_value = mock_orchestrator

    runner = CliRunner()
    result = runner.invoke(cli, ["interactive", "--mock"])
    assert result.exit_code == 0
    assert "User speaking..." in result.output
    assert "Processing utterance..." in result.output
    assert "User: test input" in result.output
    assert "Assistant: test response" in result.output
