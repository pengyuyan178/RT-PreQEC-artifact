"""Sweep correction thresholds for a trained predecoder."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import torch
import typer
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.evaluate_predecoder import _load_model
from rt_preqec.data.dataset import ArrayPredecoderDataset
from rt_preqec.utils import dump_json, ensure_parent

app = typer.Typer(add_completion=False)


def _parse_thresholds(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


@app.command()
def main(
    checkpoint: str = "checkpoints/predecoder_v1_300k.pt",
    data: str = "data/processed/predecoder_dataset_v1_300k.npz",
    split: str = "val",
    out: str = "results/runs/predecoder_v1_300k_threshold_sweep",
    thresholds: str = typer.Option("0.30,0.40,0.50,0.60,0.70,0.80", "--thresholds"),
    batch_size: int = typer.Option(2048, "--batch-size"),
    device: str = typer.Option("cpu", "--device"),
) -> None:
    """Sweep binary correction thresholds and save precision/recall/F1."""
    model = _load_model(checkpoint, device)
    dataset = ArrayPredecoderDataset(data, split=split)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    threshold_values = _parse_thresholds(thresholds)
    stats = {threshold: {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0, "exact": 0.0, "n": 0.0} for threshold in threshold_values}
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            probs = torch.sigmoid(model(batch["patch"])["correction_logits"])
            target = batch["correction_target"].float()
            for threshold in threshold_values:
                pred = (probs >= float(threshold)).float()
                item = stats[threshold]
                item["tp"] += float(((pred > 0.5) & (target > 0.5)).sum().item())
                item["fp"] += float(((pred > 0.5) & (target <= 0.5)).sum().item())
                item["fn"] += float(((pred <= 0.5) & (target > 0.5)).sum().item())
                item["tn"] += float(((pred <= 0.5) & (target <= 0.5)).sum().item())
                item["exact"] += float((pred == target).all(dim=1).float().sum().item())
                item["n"] += float(target.shape[0])
    rows = []
    for threshold, item in stats.items():
        precision = item["tp"] / max(item["tp"] + item["fp"], 1.0)
        recall = item["tp"] / max(item["tp"] + item["fn"], 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
        rows.append(
            {
                "split": split,
                "threshold": float(threshold),
                "bit_accuracy": float((item["tp"] + item["tn"]) / max(item["tp"] + item["tn"] + item["fp"] + item["fn"], 1.0)),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "patch_exact_match": float(item["exact"] / max(item["n"], 1.0)),
            }
        )
    rows.sort(key=lambda row: row["threshold"])
    out_dir = ensure_parent(Path(out) / "threshold_sweep.csv").parent
    pd.DataFrame(rows).to_csv(out_dir / "threshold_sweep.csv", index=False)
    best = max(rows, key=lambda row: row["f1"]) if rows else {}
    dump_json({"checkpoint": str(checkpoint), "data": str(data), "split": split, "best": best, "rows": rows}, out_dir / "threshold_sweep.json")


if __name__ == "__main__":
    app()
