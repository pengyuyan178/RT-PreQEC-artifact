"""Evaluate simple correction baselines for the patch-level predecoder."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pandas as pd
import torch
import typer
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rt_preqec.data.dataset import ArrayPredecoderDataset
from rt_preqec.utils import dump_json, ensure_parent

app = typer.Typer(add_completion=False)


def _metrics_from_prediction(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    tp = float(((pred > 0.5) & (target > 0.5)).sum().item())
    fp = float(((pred > 0.5) & (target <= 0.5)).sum().item())
    fn = float(((pred <= 0.5) & (target > 0.5)).sum().item())
    tn = float(((pred <= 0.5) & (target <= 0.5)).sum().item())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _evaluate_baseline(data: str | Path, split: str, baseline: str, batch_size: int) -> dict[str, Any]:
    dataset = ArrayPredecoderDataset(data, split=split)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    tp = fp = fn = tn = exact = count = 0.0
    for batch in loader:
        target = batch["correction_target"].float()
        if baseline == "last_slice":
            pred = batch["patch"][:, 0, -1, :, :].flatten(start_dim=1).float()
        elif baseline == "temporal_or":
            pred = (batch["patch"][:, 0, :, :, :].sum(dim=1) > 0.5).flatten(start_dim=1).float()
        elif baseline == "zero":
            pred = torch.zeros_like(target)
        else:
            raise ValueError(f"Unknown baseline: {baseline}")
        metrics = _metrics_from_prediction(pred, target)
        tp += metrics["tp"]
        fp += metrics["fp"]
        fn += metrics["fn"]
        tn += metrics["tn"]
        exact += float((pred == target).all(dim=1).float().sum().item())
        count += float(target.shape[0])
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    return {
        "split": split,
        "baseline": baseline,
        "num_samples": int(count),
        "correction_bit_accuracy": float((tp + tn) / max(tp + tn + fp + fn, 1.0)),
        "correction_precision": float(precision),
        "correction_recall": float(recall),
        "correction_f1": float(f1),
        "patch_exact_match": float(exact / max(count, 1.0)),
    }


@app.command()
def main(
    data: str = "data/processed/predecoder_dataset_v1_300k.npz",
    out: str = "results/runs/predecoder_v1_300k_baselines",
    baselines: str = typer.Option("zero,last_slice,temporal_or", "--baselines"),
    batch_size: int = typer.Option(4096, "--batch-size"),
) -> None:
    """Evaluate simple non-neural correction baselines on all dataset splits."""
    baseline_names = [name.strip() for name in baselines.split(",") if name.strip()]
    rows = [
        _evaluate_baseline(data, split, baseline, int(batch_size))
        for split in ["train", "val", "test"]
        for baseline in baseline_names
    ]
    out_dir = ensure_parent(Path(out) / "metrics.csv").parent
    pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)
    dump_json({"data": str(data), "rows": rows}, out_dir / "metrics.json")


if __name__ == "__main__":
    app()
