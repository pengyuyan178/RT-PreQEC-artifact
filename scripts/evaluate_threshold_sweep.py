"""Sweep RT-QEC risk/confidence thresholds on a paired real-stream eval set."""

from __future__ import annotations

from pathlib import Path
import copy
import itertools
import json
import sys

import pandas as pd
import typer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rt_preqec.config import load_config
from rt_preqec.evaluation.real_stream import run_real_stream_eval
from rt_preqec.metrics.aggregation import save_metrics_json
from rt_preqec.utils import ensure_parent

app = typer.Typer(add_completion=False)


def _parse_grid(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


@app.command()
def main(
    config: str = "configs/real_stream_eval.yaml",
    risk_dataset: str | None = typer.Option(None, "--risk-dataset"),
    risk_checkpoint: str = typer.Option("none", "--risk-checkpoint"),
    calibration: str | None = typer.Option(None, "--calibration"),
    split: str = typer.Option("test", "--split"),
    out: str = typer.Option("results/runs/threshold_sweep", "--out"),
    risk_thresholds: str = typer.Option("0.25,0.35,0.50,0.65,0.80", "--risk-thresholds"),
    confidence_thresholds: str = typer.Option("0.50,0.70,0.85,0.95", "--confidence-thresholds"),
    mode: str = typer.Option("rt_qec", "--mode"),
) -> None:
    """Run a threshold sweep for the Pareto table used by Figure 1."""
    base_cfg = load_config(config)
    run_root = Path(out)
    rows: list[dict[str, object]] = []
    thresholds = list(itertools.product(_parse_grid(risk_thresholds), _parse_grid(confidence_thresholds)))
    for risk_threshold, confidence_threshold in thresholds:
        cfg = copy.deepcopy(base_cfg)
        cfg.risk_eval.modes = [mode]
        cfg.risk_eval.ai_risk_threshold = float(risk_threshold)
        cfg.risk_eval.ai_confidence_threshold = float(confidence_threshold)
        cfg.predecoder.risk_threshold = float(risk_threshold)
        cfg.predecoder.confidence_threshold = float(confidence_threshold)
        cfg.outputs.save_events = False
        cfg.outputs.save_decisions = False
        cfg.outputs.save_predictions = False
        cfg.outputs.save_plots_ready_csv = False
        run_dir = run_root / f"risk_{risk_threshold:.2f}_conf_{confidence_threshold:.2f}"
        payload = run_real_stream_eval(
            cfg,
            risk_checkpoint=None if str(risk_checkpoint).lower() == "none" else risk_checkpoint,
            out_dir=run_dir,
            risk_dataset_path=risk_dataset,
            split=split,
            calibration_path=calibration,
        )
        summary_rows = payload.get("summary", [])
        metrics = dict(summary_rows[0]) if summary_rows else {}
        rows.append(
            {
                "risk_threshold": float(risk_threshold),
                "confidence_threshold": float(confidence_threshold),
                **metrics,
                "run_dir": str(run_dir),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(ensure_parent(run_root / "threshold_sweep.csv"), index=False)
    save_metrics_json({"rows": rows}, run_root / "threshold_sweep.json")
    with ensure_parent(run_root / "threshold_sweep_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": config,
                "risk_dataset": risk_dataset,
                "risk_checkpoint": None if str(risk_checkpoint).lower() == "none" else risk_checkpoint,
                "calibration": calibration,
                "split": split,
                "mode": mode,
                "num_runs": len(rows),
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    app()
