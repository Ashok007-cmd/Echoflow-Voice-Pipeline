"""Latency visualization module.

Generates waterfall charts, stacked bar charts, and statistical plots
to deconstruct the end-to-end latency budget.
"""

import logging
from dataclasses import dataclass

import numpy as np

from src.monitoring.latency_tracker import LatencyTracker

logger = logging.getLogger(__name__)


@dataclass
class VisualizationData:
    """Aggregated data ready for plotting."""

    stage_names: list[str]
    stage_keys: list[str]  # raw stage keys (used for color lookup)
    mean_latencies: list[float]
    p50_latencies: list[float]
    p90_latencies: list[float]
    p95_latencies: list[float]
    max_latencies: list[float]
    turn_numbers: list[int]
    per_turn_data: dict[str, list[float]]


class LatencyVisualizer:
    """Generates latency budget visualizations using matplotlib."""

    STAGE_COLORS = {
        "audio_capture": "#4ECDC4",
        "asr": "#45B7D1",
        "llm_time_to_first_token": "#96CEB4",
        "llm_total": "#FFEAA7",
        "tts": "#DDA0DD",
        "audio_playback": "#FF8C42",
        "end_to_end": "#FF6B6B",
    }

    STAGE_LABELS = {
        "audio_capture": "Audio Capture",
        "asr": "ASR (Speech Recognition)",
        "llm_time_to_first_token": "LLM TTFT",
        "llm_total": "LLM Generation",
        "tts": "TTS (Speech Synthesis)",
        "audio_playback": "Audio Playback",
        "end_to_end": "End-to-End",
    }

    def __init__(self, tracker: LatencyTracker):
        self.tracker = tracker

    def prepare_data(self) -> VisualizationData:
        """Extract and aggregate latency data from the tracker."""
        summary = self.tracker.get_summary()
        metrics = self.tracker.get_turn_metrics()

        stage_names = []
        mean_latencies = []
        p50_latencies = []
        p90_latencies = []
        p95_latencies = []
        max_latencies = []

        ordered_stages = [
            "audio_capture",
            "asr",
            "llm_time_to_first_token",
            "llm_total",
            "tts",
            "audio_playback",
            "end_to_end",
        ]

        stage_keys = []
        for stage in ordered_stages:
            if stage in summary:
                s = summary[stage]
                stage_keys.append(stage)
                stage_names.append(self.STAGE_LABELS.get(stage, stage))
                mean_latencies.append(s.mean_ms)
                p50_latencies.append(s.p50_ms)
                p90_latencies.append(s.p90_ms)
                p95_latencies.append(s.p95_ms)
                max_latencies.append(s.max_ms)

        turn_numbers = [m.turn_number for m in metrics]
        per_turn_data: dict[str, list[float]] = {}
        if metrics:
            per_turn_data = {
                "asr_ms": [m.asr_ms for m in metrics],
                "llm_ttft_ms": [m.llm_ttft_ms for m in metrics],
                "llm_total_ms": [m.llm_total_ms for m in metrics],
                "tts_ms": [m.tts_ms for m in metrics],
                "e2e_ms": [m.end_to_end_ms for m in metrics],
            }

        return VisualizationData(
            stage_names=stage_names,
            stage_keys=stage_keys,
            mean_latencies=mean_latencies,
            p50_latencies=p50_latencies,
            p90_latencies=p90_latencies,
            p95_latencies=p95_latencies,
            max_latencies=max_latencies,
            turn_numbers=turn_numbers,
            per_turn_data=per_turn_data,
        )

    def render_to_file(self, output_path: str = "latency_report.png") -> str:
        """Generate latency visualization and save to file.

        Creates a multi-panel figure:
          1. Average latency breakdown (stacked horizontal bar)
          2. Per-turn latency timeline
          3. Percentile distribution (box plot variant)
        """
        try:
            import matplotlib
            matplotlib.use("Agg")  # Non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib is required for visualization: pip install matplotlib")
            return ""

        data = self.prepare_data()
        if not data.stage_names:
            logger.warning("No data to visualize")
            return ""

        # Determine layout based on data availability
        has_timeline = len(data.turn_numbers) > 0
        n_panels = 3 if has_timeline else 2

        fig, axes = plt.subplots(n_panels, 1, figsize=(14, 5 * n_panels))
        if n_panels == 1:
            axes = [axes]

        # Guard against degenerate data (all zeros or very small values)
        max_val = max(data.mean_latencies) if data.mean_latencies else 0
        if max_val < 0.001:
            logger.warning("No meaningful latency data to visualize")
            fig.text(
                0.5, 0.5,
                "No meaningful latency data collected.\n"
                "Pipeline stages may have failed or no turns were completed.",
                ha="center", va="center", fontsize=14, alpha=0.7,
            )
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return output_path

        fig.suptitle(
            "Pipeline Latency Budget Deconstruction",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        # Panel 1: Average Latency Breakdown (horizontal bar)
        self._plot_breakdown(axes[0], data)

        # Panel 2: Per-Turn Latency Timeline
        if has_timeline:
            self._plot_timeline(axes[1], data)
            self._plot_percentiles(axes[2], data)
        else:
            self._plot_percentiles(axes[1], data)

        try:
            plt.tight_layout(rect=[0, 0, 1, 0.95])
        except ValueError:
            # Gracefully handle layout issues with degenerate data
            pass
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Latency report saved to {output_path}")
        return output_path

    def _plot_breakdown(self, ax, data: VisualizationData) -> None:
        """Panel 1: Average latency breakdown bar chart."""
        stages = data.stage_names[:-1]  # Exclude end-to-end
        keys = data.stage_keys[:-1]     # Raw keys for color lookup
        means = data.mean_latencies[:-1]
        p90s = data.p90_latencies[:-1]

        if not stages:
            ax.text(0.5, 0.5, "No stage data", ha="center", va="center")
            return

        y_pos = np.arange(len(stages))
        bars = ax.barh(y_pos, means, color=[self.STAGE_COLORS.get(k, "#999") for k in keys])

        # Add error bars (P90 as marker)
        for i, (bar, mean, p90) in enumerate(zip(bars, means, p90s)):
            ax.plot(
                [mean, p90], [bar.get_y() + bar.get_height() / 2] * 2,
                color="red", linestyle="--", linewidth=1, alpha=0.6,
            )
            ax.plot(
                p90, bar.get_y() + bar.get_height() / 2,
                "rv", markersize=4, alpha=0.7,
            )
            ax.text(
                bar.get_width() + 5,
                bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.0f}ms",
                va="center", fontsize=9,
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(stages, fontsize=10)
        ax.set_xlabel("Latency (ms)", fontsize=11)
        ax.set_title("Average Latency by Stage (bar = mean,  ⋊  = P90)", fontsize=12)
        ax.grid(axis="x", alpha=0.3)

    def _plot_timeline(self, ax, data: VisualizationData) -> None:
        """Panel 2: Per-turn latency breakdown as stacked bars."""
        turns = data.turn_numbers
        if not turns:
            ax.text(0.5, 0.5, "No per-turn data", ha="center", va="center")
            return

        # Plot each stage's contribution per turn
        stages_to_stack = [
            ("asr_ms", "#45B7D1"),
            ("llm_ttft_ms", "#96CEB4"),
            ("tts_ms", "#DDA0DD"),
        ]

        bottoms = np.zeros(len(turns))
        for stage_name, color in stages_to_stack:
            values = data.per_turn_data.get(stage_name, [])
            if len(values) != len(turns):
                continue

            label = stage_name.replace("_ms", "").replace("_", " ").title()
            ax.bar(turns, values, bottom=bottoms, label=label, color=color, alpha=0.85)
            bottoms += np.array(values)

        # Add e2e line
        if "e2e_ms" in data.per_turn_data:
            e2e = data.per_turn_data["e2e_ms"]
            if len(e2e) == len(turns):
                ax.plot(
                    turns, e2e, "r-o", markersize=3, linewidth=1.5,
                    label="End-to-End", alpha=0.7,
                )
                # Add average line
                avg_e2e = np.mean(e2e)
                ax.axhline(
                    avg_e2e, color="red", linestyle="--", linewidth=1, alpha=0.5,
                )
                ax.text(
                    turns[-1], avg_e2e, f"Avg E2E: {avg_e2e:.0f}ms",
                    fontsize=8, color="red", alpha=0.7,
                )

        ax.set_xlabel("Turn Number", fontsize=11)
        ax.set_ylabel("Latency (ms)", fontsize=11)
        ax.set_title("Per-Turn Latency Breakdown", fontsize=12)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(axis="y", alpha=0.3)

    def _plot_percentiles(self, ax, data: VisualizationData) -> None:
        """Panel 3: P50 / P90 / P95 comparison across stages."""
        stages = data.stage_names[:-1]  # Exclude end-to-end
        if not stages:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return

        x = np.arange(len(stages))
        width = 0.25

        p50 = data.p50_latencies[:-1]
        p90 = data.p90_latencies[:-1]
        p95 = data.p95_latencies[:-1]

        ax.bar(x - width, p50, width, label="P50 (Median)", color="#45B7D1", alpha=0.8)
        ax.bar(x, p90, width, label="P90", color="#FFEAA7", alpha=0.8)
        ax.bar(x + width, p95, width, label="P95", color="#FF6B6B", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(stages, fontsize=8, rotation=15, ha="right")
        ax.set_ylabel("Latency (ms)", fontsize=11)
        ax.set_title("Latency Percentiles by Stage", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)


def generate_summary_table(tracker: LatencyTracker) -> str:
    """Generate a formatted text summary of latency metrics."""
    summary = tracker.get_summary()
    if not summary:
        return "No latency data collected."

    lines = [
        "=" * 80,
        "LATENCY BUDGET REPORT",
        "=" * 80,
        f"{'Stage':<30} {'Count':>6} {'Mean(ms)':>10} {'P50(ms)':>10} "
        f"{'P90(ms)':>10} {'P95(ms)':>10} {'Max(ms)':>10}",
        "-" * 80,
    ]

    ordered = [
        "audio_capture", "asr", "llm_time_to_first_token",
        "llm_total", "tts", "audio_playback", "end_to_end",
    ]

    for stage in ordered:
        if stage in summary:
            s = summary[stage]
            label = LatencyVisualizer.STAGE_LABELS.get(stage, stage)
            lines.append(
                f"{label:<30} {s.count:>6} {s.mean_ms:>9.1f} "
                f"{s.p50_ms:>9.1f} {s.p90_ms:>9.1f} "
                f"{s.p95_ms:>9.1f} {s.max_ms:>9.1f}"
            )

    lines.extend([
        "-" * 80,
        "E2E Latency Budget Allocation:",
    ])

    # Calculate budget percentages
    if "end_to_end" in summary and summary["end_to_end"].mean_ms > 0:
        e2e_mean = summary["end_to_end"].mean_ms
        for stage in ["asr", "llm_total", "tts"]:
            if stage in summary and summary[stage].mean_ms > 0:
                pct = (summary[stage].mean_ms / e2e_mean) * 100
                label = LatencyVisualizer.STAGE_LABELS.get(stage, stage)
                lines.append(f"  {label:<28} {pct:>5.1f}% ({summary[stage].mean_ms:.0f}ms)")

    lines.append("=" * 80)
    return "\n".join(lines)
