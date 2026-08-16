"""Generate plots from a run directory."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import typer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rt_preqec.visualization.plots_ablation import plot_threshold_tradeoff
from rt_preqec.visualization.plots_latency import plot_backlog_over_time, plot_latency_cdf, plot_latency_percentiles
from rt_preqec.visualization.plots_realtime_compare import (
    plot_backlog_over_time_by_mode,
    plot_decoder_selection_rates,
    plot_logical_error_vs_deadline_miss,
    plot_mode_latency_percentiles,
)

app = typer.Typer(add_completion=False)


@app.command()
def main(run_dir: str = "results/runs/smoke_eval", out: str = "results/figures") -> None:
    """Render plots for latency and backlog."""
    run_path = Path(run_dir)
    out_path = Path(out)
    summary_path = run_path / "summary_metrics.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        plot_mode_latency_percentiles(summary, out_path / "mode_latency_percentiles.png")
        plot_logical_error_vs_deadline_miss(summary, out_path / "logical_error_vs_deadline_miss.png")
        plot_decoder_selection_rates(summary, out_path / "decoder_selection_rates.png")
        mode_event_paths = {
            str(mode): run_path / str(mode) / "events.csv"
            for mode in summary["mode"].tolist()
        } if "mode" in summary else {}
        plot_backlog_over_time_by_mode(mode_event_paths, out_path / "backlog_over_time_by_mode.png")
        return
    events = pd.read_csv(run_path / "events.csv")
    latency = pd.read_csv(run_path / "latency.csv")
    latencies = events["latency_us"].to_numpy() if "latency_us" in events else np.asarray([])
    plot_latency_cdf(latencies, out_path / "latency_cdf.png")
    plot_latency_percentiles(latencies, out_path / "latency_percentiles.png")
    plot_backlog_over_time(events, out_path / "backlog_over_time.png")
    if (run_path / "ablation.csv").exists():
        ablation = pd.read_csv(run_path / "ablation.csv")
        if "confidence_threshold" in ablation and "accept_rate" in ablation:
            plot_threshold_tradeoff(
                ablation["confidence_threshold"].to_numpy(),
                ablation["accept_rate"].to_numpy(),
                out_path / "threshold_tradeoff.png",
            )


if __name__ == "__main__":
    app()
