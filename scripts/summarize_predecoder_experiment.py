"""Create a compact, paper-ready summary for the predecoder experiment."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import torch
import typer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rt_preqec.utils import dump_json, ensure_parent

app = typer.Typer(add_completion=False)


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _row_by_split(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    for row in rows:
        if row.get("split") == split:
            return row
    raise ValueError(f"Missing split row: {split}")


def _baseline_row(rows: list[dict[str, Any]], split: str, baseline: str) -> dict[str, Any]:
    for row in rows:
        if row.get("split") == split and row.get("baseline") == baseline:
            return row
    raise ValueError(f"Missing baseline row: {split}/{baseline}")


def _write_metric_table(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "model",
        "split",
        "threshold",
        "bit_accuracy",
        "precision",
        "recall",
        "f1",
        "patch_exact_match",
        "confidence_mae",
        "risk_mae",
    ]
    target = ensure_parent(path)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@app.command()
def main(
    checkpoint: str = "checkpoints/predecoder_v1_300k.pt",
    dataset_summary: str = "results/runs/predecoder_dataset_v1_300k_build/summary.json",
    metrics: str = "results/runs/predecoder_v1_300k_eval/metrics.json",
    val_sweep: str = "results/runs/predecoder_v1_300k_threshold_sweep_val/threshold_sweep.json",
    baselines: str = "results/runs/predecoder_v1_300k_baselines/metrics.json",
    out: str = "results/runs/predecoder_v1_300k_paper_summary",
) -> None:
    """Aggregate dataset, training, baseline, and final test metrics."""
    checkpoint_payload = torch.load(checkpoint, map_location="cpu")
    dataset_payload = _load_json(dataset_summary)
    metrics_payload = _load_json(metrics)
    val_sweep_payload = _load_json(val_sweep)
    baseline_payload = _load_json(baselines)

    model_test = _row_by_split(metrics_payload["rows"], "test")
    model_val = _row_by_split(metrics_payload["rows"], "val")
    last_slice_test = _baseline_row(baseline_payload["rows"], "test", "last_slice")
    zero_test = _baseline_row(baseline_payload["rows"], "test", "zero")

    f1_gain_abs = float(model_test["correction_f1"] - last_slice_test["correction_f1"])
    f1_gain_rel = f1_gain_abs / max(float(last_slice_test["correction_f1"]), 1e-8)
    exact_gain_abs = float(model_test["patch_exact_match"] - last_slice_test["patch_exact_match"])
    exact_gain_rel = exact_gain_abs / max(float(last_slice_test["patch_exact_match"]), 1e-8)

    summary = {
        "checkpoint": str(checkpoint),
        "dataset": {
            "path": metrics_payload["data"],
            "num_samples": dataset_payload["num_samples"],
            "patch_shape": dataset_payload["patch_shape"],
            "correction_dim": dataset_payload["correction_dim"],
            "split_sizes": dataset_payload["split_sizes"],
            "settings": dataset_payload["metadata"]["settings"],
            "split_policy": dataset_payload["metadata"]["split_policy"],
            "target_policy": dataset_payload["metadata"]["target_policy"],
            "seed": dataset_payload["metadata"]["seed"],
        },
        "training": {
            "model": checkpoint_payload["config"]["model"],
            "training": checkpoint_payload["config"]["training"],
            "loss_weights": checkpoint_payload["loss_weights"],
            "best_epoch": checkpoint_payload["best_epoch"],
            "best_metric_name": checkpoint_payload["best_metric_name"],
            "best_val_metric": checkpoint_payload["best_val_metric"],
            "epochs_trained": checkpoint_payload["epochs_trained"],
            "early_stopping_patience": checkpoint_payload["early_stopping_patience"],
        },
        "threshold_selection": {
            "selected_on": "validation",
            "selected_threshold": val_sweep_payload["best"]["threshold"],
            "validation_f1": val_sweep_payload["best"]["f1"],
        },
        "final_test": model_test,
        "validation": model_val,
        "baselines_test": {
            "zero": zero_test,
            "last_slice": last_slice_test,
        },
        "improvement_over_last_slice_test": {
            "f1_absolute": f1_gain_abs,
            "f1_relative": f1_gain_rel,
            "patch_exact_absolute": exact_gain_abs,
            "patch_exact_relative": exact_gain_rel,
        },
    }

    out_dir = ensure_parent(Path(out) / "summary.json").parent
    dump_json(summary, out_dir / "summary.json")

    table_rows = []
    for split in ["train", "val", "test"]:
        row = _row_by_split(metrics_payload["rows"], split)
        table_rows.append(
            {
                "model": "TinyNeuralPredecoder",
                "split": split,
                "threshold": row["correction_threshold"],
                "bit_accuracy": row["correction_bit_accuracy"],
                "precision": row["correction_precision"],
                "recall": row["correction_recall"],
                "f1": row["correction_f1"],
                "patch_exact_match": row["patch_exact_match"],
                "confidence_mae": row["confidence_mae"],
                "risk_mae": row["risk_mae"],
            }
        )
    for baseline in ["zero", "last_slice", "temporal_or"]:
        row = _baseline_row(baseline_payload["rows"], "test", baseline)
        table_rows.append(
            {
                "model": f"baseline:{baseline}",
                "split": "test",
                "threshold": "",
                "bit_accuracy": row["correction_bit_accuracy"],
                "precision": row["correction_precision"],
                "recall": row["correction_recall"],
                "f1": row["correction_f1"],
                "patch_exact_match": row["patch_exact_match"],
                "confidence_mae": "",
                "risk_mae": "",
            }
        )
    _write_metric_table(out_dir / "paper_metrics.csv", table_rows)

    training_log = pd.read_csv(Path(checkpoint).with_suffix(".csv"))
    best_rows = training_log[(training_log["split"] == "val") & (training_log["is_best"] == 1.0)]
    best_rows.to_csv(out_dir / "best_val_epochs.csv", index=False)


if __name__ == "__main__":
    app()
