"""Compare optional AI risk scheduling against the heuristic RT-QEC scheduler."""

from __future__ import annotations

from pathlib import Path
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
from rt_preqec.utils import ensure_parent

app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: str = "configs/real_stream_eval.yaml",
    risk_dataset: str | None = typer.Option(None, "--risk-dataset"),
    split: str = typer.Option("test", "--split"),
    risk_checkpoint: str = typer.Option("checkpoints/risk_lstm_v2.pt", "--risk-checkpoint"),
    calibration: str | None = typer.Option(None, "--calibration"),
    out: str = typer.Option("results/runs/ai_vs_heuristic", "--out"),
) -> None:
    """Run E3-style optional AI-vs-heuristic scheduling comparison."""
    cfg = load_config(config)
    cfg.risk_eval.modes = ["rt_qec", "risk_heuristic", "ai_risk", "oracle_risk"]
    payload = run_real_stream_eval(
        cfg,
        risk_checkpoint=None if str(risk_checkpoint).lower() == "none" else risk_checkpoint,
        out_dir=out,
        risk_dataset_path=risk_dataset,
        split=split,
        calibration_path=calibration,
    )
    summary = pd.DataFrame(payload.get("summary", []))
    rows: list[dict[str, object]] = []
    if not summary.empty and "mode" in summary:
        baseline = summary[summary["mode"] == "rt_qec"]
        ai = summary[summary["mode"] == "ai_risk"]
        if not baseline.empty and not ai.empty:
            base_row = baseline.iloc[0]
            ai_row = ai.iloc[0]
            rows.append(
                {
                    "comparison": "ai_risk_minus_rt_qec",
                    "delta_logical_error_rate": float(ai_row["logical_error_rate"] - base_row["logical_error_rate"]),
                    "delta_p99_latency_us": float(ai_row["p99_latency_us"] - base_row["p99_latency_us"]),
                    "delta_p99_response_time_us": float(
                        ai_row["p99_response_time_us"] - base_row["p99_response_time_us"]
                    ),
                    "delta_deadline_miss_ratio": float(
                        ai_row["deadline_miss_ratio"] - base_row["deadline_miss_ratio"]
                    ),
                    "delta_p99_pauli_frame_lag": float(
                        ai_row.get("p99_pauli_frame_lag", 0.0) - base_row.get("p99_pauli_frame_lag", 0.0)
                    ),
                    "delta_fast_selection_rate": float(ai_row["fast_selection_rate"] - base_row["fast_selection_rate"]),
                    "delta_risk_false_negative_rate": float(
                        ai_row.get("risk_false_negative_rate", 0.0)
                        - base_row.get("risk_false_negative_rate", 0.0)
                    ),
                    "delta_risk_false_positive_rate": float(
                        ai_row.get("risk_false_positive_rate", 0.0)
                        - base_row.get("risk_false_positive_rate", 0.0)
                    ),
                    "delta_predecode_accept_rate": float(
                        ai_row.get("predecode_accept_rate", 0.0) - base_row.get("predecode_accept_rate", 0.0)
                    ),
                    "delta_estimated_residual_reduction": float(
                        ai_row.get("mean_estimated_residual_reduction", 0.0)
                        - base_row.get("mean_estimated_residual_reduction", 0.0)
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(ensure_parent(Path(out) / "ai_vs_heuristic_delta.csv"), index=False)
    with ensure_parent(Path(out) / "ai_vs_heuristic_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": config,
                "risk_dataset": risk_dataset,
                "split": split,
                "risk_checkpoint": None if str(risk_checkpoint).lower() == "none" else risk_checkpoint,
                "calibration": calibration,
                "modes": cfg.risk_eval.modes,
                "delta_rows": len(rows),
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    app()
