"""Comparison plots for real-stream evaluation runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from rt_preqec.utils import ensure_parent


def plot_mode_latency_percentiles(summary_metrics: pd.DataFrame, out_path: str | Path) -> None:
    """Plot p95/p99/p999 latency by mode."""
    if summary_metrics.empty:
        return
    frame = summary_metrics.set_index("mode")
    plt.figure(figsize=(8, 4))
    for column in ["p95_latency_us", "p99_latency_us", "p999_latency_us"]:
        if column in frame.columns:
            plt.plot(frame.index.tolist(), frame[column].tolist(), marker="o", label=column.replace("_latency_us", ""))
    plt.ylabel("Latency (us)")
    plt.xticks(rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ensure_parent(out_path))
    plt.close()


def plot_logical_error_vs_deadline_miss(summary_metrics: pd.DataFrame, out_path: str | Path) -> None:
    """Scatter logical error rate against deadline miss ratio."""
    if summary_metrics.empty:
        return
    plt.figure(figsize=(5, 4))
    plt.scatter(summary_metrics["deadline_miss_ratio"], summary_metrics["logical_error_rate"])
    for _, row in summary_metrics.iterrows():
        plt.annotate(str(row["mode"]), (row["deadline_miss_ratio"], row["logical_error_rate"]))
    plt.xlabel("Deadline miss ratio")
    plt.ylabel("Logical error rate")
    plt.tight_layout()
    plt.savefig(ensure_parent(out_path))
    plt.close()


def plot_backlog_over_time_by_mode(mode_event_paths: dict[str, str | Path], out_path: str | Path) -> None:
    """Overlay backlog-over-time traces for multiple modes."""
    plt.figure(figsize=(8, 4))
    plotted = False
    for mode, event_path in mode_event_paths.items():
        path = Path(event_path)
        if not path.exists():
            continue
        events = pd.read_csv(path)
        if "shot_id" not in events or "backlog" not in events:
            continue
        plt.plot(events["shot_id"], events["backlog"], label=mode)
        plotted = True
    if not plotted:
        plt.close()
        return
    plt.xlabel("Shot ID")
    plt.ylabel("Backlog")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ensure_parent(out_path))
    plt.close()


def plot_decoder_selection_rates(summary_metrics: pd.DataFrame, out_path: str | Path) -> None:
    """Plot fast/accurate selection rates per mode."""
    if summary_metrics.empty:
        return
    frame = summary_metrics.set_index("mode")
    modes = frame.index.tolist()
    fast = frame["fast_selection_rate"].tolist() if "fast_selection_rate" in frame else [0.0] * len(modes)
    accurate = frame["accurate_selection_rate"].tolist() if "accurate_selection_rate" in frame else [0.0] * len(modes)
    x = range(len(modes))
    plt.figure(figsize=(8, 4))
    plt.bar([value - 0.2 for value in x], fast, width=0.4, label="fast")
    plt.bar([value + 0.2 for value in x], accurate, width=0.4, label="accurate")
    plt.xticks(list(x), modes, rotation=30, ha="right")
    plt.ylabel("Selection rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ensure_parent(out_path))
    plt.close()
