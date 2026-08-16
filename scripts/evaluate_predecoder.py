"""Evaluate a trained patch-level predecoder checkpoint."""

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
from rt_preqec.models.predecoder import TinyNeuralPredecoder
from rt_preqec.utils import dump_json, ensure_parent

app = typer.Typer(add_completion=False)


def _load_model(checkpoint: str | Path, device: str) -> TinyNeuralPredecoder:
    payload = torch.load(checkpoint, map_location=device)
    cfg = payload.get("config", {})
    predecoder_cfg = cfg.get("predecoder", {}) if isinstance(cfg, dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    model = TinyNeuralPredecoder(
        temporal_window=int(predecoder_cfg.get("temporal_window", 3)),
        patch_size=int(predecoder_cfg.get("patch_size", 5)),
        hidden_channels=int(model_cfg.get("hidden_channels", 16)),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model


def _evaluate_split(
    model: TinyNeuralPredecoder,
    data: str | Path,
    split: str,
    batch_size: int,
    device: str,
    threshold: float,
) -> dict[str, Any]:
    dataset = ArrayPredecoderDataset(data, split=split)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    tp = fp = fn = tn = exact = count = 0.0
    confidence_abs = risk_abs = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(batch["patch"])
            target = batch["correction_target"].float()
            pred = (torch.sigmoid(outputs["correction_logits"]) >= float(threshold)).float()
            tp += float(((pred > 0.5) & (target > 0.5)).sum().item())
            fp += float(((pred > 0.5) & (target <= 0.5)).sum().item())
            fn += float(((pred <= 0.5) & (target > 0.5)).sum().item())
            tn += float(((pred <= 0.5) & (target <= 0.5)).sum().item())
            exact += float((pred == target).all(dim=1).float().sum().item())
            count += float(target.shape[0])
            confidence_abs += float(
                torch.abs(torch.sigmoid(outputs["confidence_logit"]) - batch["confidence_target"].float()).sum().item()
            )
            risk_abs += float(torch.abs(torch.sigmoid(outputs["risk_logit"]) - batch["risk_target"].float()).sum().item())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    return {
        "split": split,
        "correction_threshold": float(threshold),
        "num_samples": int(count),
        "correction_bit_accuracy": float((tp + tn) / max(tp + tn + fp + fn, 1.0)),
        "correction_precision": float(precision),
        "correction_recall": float(recall),
        "correction_f1": float(f1),
        "patch_exact_match": float(exact / max(count, 1.0)),
        "confidence_mae": float(confidence_abs / max(count, 1.0)),
        "risk_mae": float(risk_abs / max(count, 1.0)),
    }


@app.command()
def main(
    checkpoint: str = "checkpoints/predecoder_v1_300k.pt",
    data: str = "data/processed/predecoder_dataset_v1_300k.npz",
    out: str = "results/runs/predecoder_v1_300k_eval",
    batch_size: int = typer.Option(1024, "--batch-size"),
    device: str = typer.Option("cpu", "--device"),
    threshold: float = typer.Option(0.5, "--threshold"),
) -> None:
    """Evaluate a predecoder checkpoint on all dataset splits."""
    model = _load_model(checkpoint, device)
    rows = [_evaluate_split(model, data, split, int(batch_size), device, float(threshold)) for split in ["train", "val", "test"]]
    out_dir = ensure_parent(Path(out) / "metrics.json").parent
    dump_json({"checkpoint": str(checkpoint), "data": str(data), "rows": rows}, out_dir / "metrics.json")
    pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)


if __name__ == "__main__":
    app()
