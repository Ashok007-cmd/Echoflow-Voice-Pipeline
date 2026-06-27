#!/usr/bin/env python3
"""Real-Time Multimodal Application — CLI entry point.

Modes:
  - interactive: Full voice assistant loop (ASR → LLM → TTS)
  - benchmark: Run N turns with synthetic data and generate latency report
  - report:   Generate latency visualization from existing metrics
"""

import asyncio
import logging
import signal
import sys
from typing import Optional

import click

from src.config import (
    ASRBackend,
    LLMBackend,
    PipelineConfig,
    TTSBackend,
)
from src.pipeline.orchestrator import PipelineOrchestrator
from src.monitoring.visualizer import (
    LatencyVisualizer,
    generate_summary_table,
)

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure structured logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# ── Interactive Mode (Phase 1 + 2 + 3) ─────────────────────────────

@click.command()
@click.option("--asr-backend", default=None, help="ASR backend: google, openai, local_whisper, mock")
@click.option("--llm-backend", default=None, help="LLM backend: openai, anthropic, mock")
@click.option("--tts-backend", default=None, help="TTS backend: edge_tts, pyttsx3, mock")
@click.option("--mock", is_flag=True, help="Use mock backends for all services (testing)")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.option("--timeout", default=None, type=float, help="Pipeline stage timeout in seconds")
@click.option("--report", is_flag=True, help="Generate latency report on exit")
def interactive(
    asr_backend: Optional[str],
    llm_backend: Optional[str],
    tts_backend: Optional[str],
    mock: bool,
    verbose: bool,
    timeout: Optional[float],
    report: bool,
) -> None:
    """Run the real-time voice assistant in interactive mode.

    Captures audio from the microphone, processes through ASR → LLM → TTS,
    and plays back the response. Measures and displays latency at every stage.
    """
    setup_logging(verbose)

    config = PipelineConfig.from_env()

    if mock:
        config.asr.backend = ASRBackend.MOCK
        config.llm.backend = LLMBackend.MOCK
        config.tts.backend = TTSBackend.MOCK
    else:
        if asr_backend:
            try:
                config.asr.backend = ASRBackend(asr_backend)
            except ValueError:
                click.echo(f"Invalid ASR backend: {asr_backend}. Options: google, openai, local_whisper, mock")
                sys.exit(1)
        if llm_backend:
            try:
                config.llm.backend = LLMBackend(llm_backend)
            except ValueError:
                click.echo(f"Invalid LLM backend: {llm_backend}. Options: openai, anthropic, mock")
                sys.exit(1)
        if tts_backend:
            try:
                config.tts.backend = TTSBackend(tts_backend)
            except ValueError:
                click.echo(f"Invalid TTS backend: {tts_backend}. Options: edge_tts, pyttsx3, mock")
                sys.exit(1)

    if timeout:
        config.resilience.asr_timeout = timeout
        config.resilience.llm_timeout = timeout
        config.resilience.tts_timeout = timeout

    try:
        config.validate()
    except ValueError as e:
        click.echo(click.style(f"Configuration Error: {e}", fg="red", bold=True))
        sys.exit(1)

    click.echo(click.style("╔══════════════════════════════════════════╗", bold=True, fg="cyan"))
    click.echo(click.style("║  Real-Time Multimodal Voice Assistant   ║", bold=True, fg="cyan"))
    click.echo(click.style("║  Phase 1: Streaming Pipeline            ║", bold=True, fg="cyan"))
    click.echo(click.style("║  Phase 2: Latency Budget                ║", bold=True, fg="cyan"))
    click.echo(click.style("║  Phase 3: System Resilience             ║", bold=True, fg="cyan"))
    click.echo(click.style("╚══════════════════════════════════════════╝", bold=True, fg="cyan"))

    click.echo()
    click.echo(f"ASR: {config.asr.backend.value}  |  LLM: {config.llm.backend.value}  |  "
               f"TTS: {config.tts.backend.value}")
    click.echo(f"Timeouts: ASR={config.resilience.asr_timeout}s, "
               f"LLM={config.resilience.llm_timeout}s, "
               f"TTS={config.resilience.tts_timeout}s")
    click.echo()

    orchestrator = PipelineOrchestrator(config)
    running = True

    def _handle_signal(sig, frame):
        nonlocal running
        running = False
        click.echo("\nShutting down gracefully...")

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    async def _run():
        nonlocal running
        await orchestrator.start()

        click.echo(click.style("🎤 Listening... (Ctrl+C to stop)", fg="green", bold=True))
        click.echo("-" * 60)

        from src.pipeline.asr import DynamicVAD
        vad = DynamicVAD(config.audio.input_sample_rate, config.asr.chunk_duration)
        buffer = []
        is_speaking = False
        silence_counter = 0
        chunk_duration = config.asr.chunk_duration
        silence_threshold = config.asr.silence_threshold

        try:
            async for audio_chunk in orchestrator.capture_audio_stream():
                if not running:
                    break

                buffer.append(audio_chunk.data)

                # Use Dynamic VAD
                is_silence = not vad.is_speech(audio_chunk.data)

                if is_silence:
                    silence_counter += 1
                else:
                    silence_counter = 0
                    if not is_speaking:
                        is_speaking = True
                        click.echo(click.style("\n🎙  User speaking...", fg="blue", bold=True))

                silence_duration = silence_counter * chunk_duration

                if is_speaking and silence_duration > silence_threshold:
                    # Utterance boundary detected!
                    utterance_audio = b"".join(buffer)
                    buffer.clear()
                    is_speaking = False
                    silence_counter = 0

                    click.echo(click.style("⚡ Processing utterance...", fg="yellow"))
                    metrics = await orchestrator.run_turn(utterance_audio, audio_chunk.sample_rate)

                    if metrics:
                        click.echo()
                        if metrics.user_text:
                            click.echo(click.style(f"User: {metrics.user_text}", fg="cyan", bold=True))
                        if metrics.assistant_text:
                            click.echo(click.style(f"Assistant: {metrics.assistant_text}", fg="green", bold=True))
                        click.echo(
                            f"Latency: ASR={metrics.asr_ms:.0f}ms | LLM TTFT={metrics.llm_ttft_ms:.0f}ms | TTS={metrics.tts_ms:.0f}ms | E2E={metrics.end_to_end_ms:.0f}ms"
                        )
                        click.echo("-" * 60)
                    else:
                        click.echo(click.style("No speech detected or empty response.", fg="white", dim=True))

                    click.echo(click.style("🎤 Listening...", fg="green", bold=True))

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
        finally:
            await orchestrator.stop()

            # Generate latency report
            metrics = orchestrator.latency_tracker.get_turn_metrics()
            if metrics:
                click.echo()
                click.echo(click.style("=== Latency Summary ===", bold=True, fg="cyan"))
                click.echo(generate_summary_table(orchestrator.latency_tracker))

                if report:
                    viz = LatencyVisualizer(orchestrator.latency_tracker)
                    path = viz.render_to_file(
                        config.monitoring.visualization_output
                    )
                    if path:
                        click.echo(f"Latency report saved to: {path}")
            else:
                click.echo("No metrics collected.")

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


# ── Benchmark Mode ─────────────────────────────────────────────────

@click.command()
@click.option("--turns", "-n", default=10, type=int, help="Number of benchmark turns")
@click.option("--mock", is_flag=True, help="Use mock backends for all services (offline)")
@click.option("--mock-llm", is_flag=True, help="Use mock LLM for faster benchmarks")
@click.option("--mock-tts", is_flag=True, help="Use mock TTS (no audio output)")
@click.option("--mock-asr", is_flag=True, help="Use mock ASR for offline benchmarks")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.option("--output", default="latency_report.png", help="Output path for visualization")
def benchmark(
    turns: int,
    mock: bool,
    mock_llm: bool,
    mock_tts: bool,
    mock_asr: bool,
    verbose: bool,
    output: str,
) -> None:
    """Run a latency benchmark with synthetic data.

    Measures precise timing for each pipeline stage across N turns
    and generates a visualization.
    """
    setup_logging(verbose)

    config = PipelineConfig.from_env()
    if mock:
        config.asr.backend = ASRBackend.MOCK
        config.llm.backend = LLMBackend.MOCK
        config.tts.backend = TTSBackend.MOCK
    else:
        if mock_asr:
            config.asr.backend = ASRBackend.MOCK
        if mock_llm:
            config.llm.backend = LLMBackend.MOCK
        if mock_tts:
            config.tts.backend = TTSBackend.MOCK
    config.monitoring.visualization_output = output

    try:
        config.validate()
    except ValueError as e:
        click.echo(click.style(f"Configuration Error: {e}", fg="red", bold=True))
        sys.exit(1)

    click.echo(
        click.style("═══ Latency Benchmark ═══", bold=True, fg="cyan")
    )
    click.echo(f"Turns: {turns}  |  ASR: {config.asr.backend.value}  |  "
               f"LLM: {config.llm.backend.value}  |  "
               f"TTS: {config.tts.backend.value}")
    click.echo()

    orchestrator = PipelineOrchestrator(config)

    async def _run_benchmark():
        await orchestrator.start()

        click.echo(f"{'Turn':>5} {'ASR(ms)':>10} {'TTFT(ms)':>10} "
                   f"{'LLM(ms)':>10} {'TTS(ms)':>10} {'E2E(ms)':>10} {'Status':>12}")
        click.echo("-" * 70)

        # Generate synthetic audio data (1 second of silence with slight noise)
        import numpy as np
        sample_rate = config.audio.input_sample_rate
        synthetic_audio = (np.random.randn(sample_rate * 2) * 100).astype(np.int16).tobytes()

        for i in range(turns):
            metrics = await orchestrator.run_turn(synthetic_audio, sample_rate)
            if metrics:
                status = "OK" if not metrics.error else "ERR"
                click.echo(
                    f"{metrics.turn_number:>5} {metrics.asr_ms:>9.1f} "
                    f"{metrics.llm_ttft_ms:>9.1f} "
                    f"{metrics.llm_total_ms:>9.1f} "
                    f"{metrics.tts_ms:>9.1f} "
                    f"{metrics.end_to_end_ms:>9.1f} "
                    f"{status:>12}"
                )

        await orchestrator.stop()

        # Generate report
        click.echo()
        click.echo(click.style("═══ Latency Budget Report ═══", bold=True, fg="cyan"))
        click.echo(generate_summary_table(orchestrator.latency_tracker))

        viz = LatencyVisualizer(orchestrator.latency_tracker)
        path = viz.render_to_file(output)
        if path:
            click.echo(f"\nVisualization saved to: {path}")

    asyncio.run(_run_benchmark())


# ── Latency Report Mode ────────────────────────────────────────────

@click.command()
@click.argument("metrics_file", type=click.Path(exists=True), required=False)
@click.option("--output", default="latency_report.png", help="Output path for visualization")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def report(metrics_file: Optional[str], output: str, verbose: bool) -> None:
    """Generate a latency budget report from collected metrics.

    If METRICS_FILE is provided, loads pre-collected metrics from a CSV file.
    Otherwise, shows instructions for collecting metrics.
    """
    setup_logging(verbose)

    if metrics_file:
        # Load and visualize pre-collected metrics
        import pandas as pd

        df = pd.read_csv(metrics_file)
        click.echo(f"Loaded {len(df)} metrics from {metrics_file}")
        click.echo(df.to_string())
    else:
        click.echo(
            "To generate a latency report:\n"
            "  1. Run a benchmark:  voice-assistant benchmark --turns 20\n"
            "  2. Or run interactive mode with --report flag\n"
            "\n"
            "Example:\n"
            "  voice-assistant benchmark --turns 20 --mock-llm --output latency.png\n"
            "  voice-assistant interactive --report\n"
        )


# ── CLI Group ──────────────────────────────────────────────────────

@click.group()
def cli() -> None:
    """Real-Time Multimodal Voice Assistant.

    A streaming pipeline (ASR → LLM → TTS) with latency budget
    deconstruction and system resilience.
    """
    pass


cli.add_command(interactive)
cli.add_command(benchmark)
cli.add_command(report)


if __name__ == "__main__":
    cli()
