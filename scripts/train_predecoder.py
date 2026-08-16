"""Train the tiny selective predecoder."""

from __future__ import annotations

import copy
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import torch
import typer
from torch.optim import Adam
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rt_preqec.config import load_config
from rt_preqec.data.dataset import ArrayPredecoderDataset
from rt_preqec.logging_utils import configure_logging, get_logger
from rt_preqec.models.losses import compute_loss_breakdown, predecoder_loss
from rt_preqec.models.predecoder import TinyNeuralPredecoder
from rt_preqec.utils import ensure_parent

app = typer.Typer(add_completion=False)
logger = get_logger(__name__)


def _empty_accumulator() -> dict[str, float]:
    return {
        "loss_sum": 0.0,
        "correction_loss_sum": 0.0,
        "confidence_loss_sum": 0.0,
        "risk_loss_sum": 0.0,
        "tp": 0.0,
        "fp": 0.0,
        "fn": 0.0,
        "tn": 0.0,
        "exact": 0.0,
        "samples": 0.0,
        "bits": 0.0,
        "confidence_abs": 0.0,
        "risk_abs": 0.0,
    }


def _update_accumulator(
    accumulator: dict[str, float],
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    loss: torch.Tensor,
    breakdown: dict[str, torch.Tensor],
    *,
    threshold: float,
) -> None:
    correction_target = batch["correction_target"].float()
    correction_pred = (torch.sigmoid(outputs["correction_logits"].detach()) >= float(threshold)).float()
    confidence_pred = torch.sigmoid(outputs["confidence_logit"])
    risk_pred = torch.sigmoid(outputs["risk_logit"])
    samples = float(correction_target.shape[0])
    accumulator["loss_sum"] += float(loss.detach().item()) * samples
    accumulator["correction_loss_sum"] += float(breakdown["correction"].detach().item()) * samples
    accumulator["confidence_loss_sum"] += float(breakdown["confidence"].detach().item()) * samples
    accumulator["risk_loss_sum"] += float(breakdown["risk"].detach().item()) * samples
    accumulator["tp"] += float(((correction_pred > 0.5) & (correction_target > 0.5)).sum().item())
    accumulator["fp"] += float(((correction_pred > 0.5) & (correction_target <= 0.5)).sum().item())
    accumulator["fn"] += float(((correction_pred <= 0.5) & (correction_target > 0.5)).sum().item())
    accumulator["tn"] += float(((correction_pred <= 0.5) & (correction_target <= 0.5)).sum().item())
    accumulator["exact"] += float((correction_pred == correction_target).all(dim=1).float().sum().item())
    accumulator["samples"] += samples
    accumulator["bits"] += float(correction_target.numel())
    accumulator["confidence_abs"] += float(torch.abs(confidence_pred - batch["confidence_target"].float()).sum().item())
    accumulator["risk_abs"] += float(torch.abs(risk_pred - batch["risk_target"].float()).sum().item())


def _finalize_metrics(accumulator: dict[str, float]) -> dict[str, float]:
    samples = max(accumulator["samples"], 1.0)
    bits = max(accumulator["bits"], 1.0)
    tp = accumulator["tp"]
    fp = accumulator["fp"]
    fn = accumulator["fn"]
    tn = accumulator["tn"]
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    return {
        "loss": float(accumulator["loss_sum"] / samples),
        "correction_loss": float(accumulator["correction_loss_sum"] / samples),
        "confidence_loss": float(accumulator["confidence_loss_sum"] / samples),
        "risk_loss": float(accumulator["risk_loss_sum"] / samples),
        "correction_bit_accuracy": float((tp + tn) / bits),
        "correction_precision": float(precision),
        "correction_recall": float(recall),
        "correction_f1": float(f1),
        "patch_exact_match": float(accumulator["exact"] / samples),
        "confidence_mae": float(accumulator["confidence_abs"] / samples),
        "risk_mae": float(accumulator["risk_abs"] / samples),
    }


def _loss_weights(dataset: ArrayPredecoderDataset) -> dict[str, float]:
    targets = dataset.correction_targets[dataset.indices]
    positives = max(float(targets.sum()), 1.0)
    negatives = max(float(targets.size - targets.sum()), 1.0)
    pos_weight = min(negatives / positives, 20.0)
    return {
        "correction": 2.0,
        "confidence": 0.5,
        "risk": 0.5,
        "correction_pos_weight": float(pos_weight),
        "correction_dice": 0.75,
    }


@app.command()
def main(
    config: str = "configs/train_predecoder.yaml",
    data: str = "data/processed/predecoder_dataset_v1_300k.npz",
    out: str = "checkpoints/predecoder_v1_300k.pt",
    epochs: int | None = typer.Option(None, "--epochs"),
    batch_size: int | None = typer.Option(None, "--batch-size"),
    hidden_channels: int | None = typer.Option(None, "--hidden-channels"),
    train_split: str = typer.Option("train", "--train-split"),
    val_split: str = typer.Option("val", "--val-split"),
    patience: int | None = typer.Option(None, "--patience"),
    min_delta: float = typer.Option(1e-5, "--min-delta"),
) -> None:
    """Train a tiny CNN on local patch samples."""
    cfg = load_config(config)
    configure_logging(cfg.log_level)
    if epochs is not None:
        cfg.training.epochs = int(epochs)
    if batch_size is not None:
        cfg.training.batch_size = int(batch_size)
    if hidden_channels is not None:
        cfg.model.hidden_channels = int(hidden_channels)
    train_dataset = ArrayPredecoderDataset(data, split=train_split)
    val_dataset = ArrayPredecoderDataset(data, split=val_split)
    dataset_metadata = dict(train_dataset.metadata)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )
    loss_weights = _loss_weights(train_dataset)
    model = TinyNeuralPredecoder(
        temporal_window=cfg.predecoder.temporal_window,
        patch_size=cfg.predecoder.patch_size,
        hidden_channels=cfg.model.hidden_channels,
    )
    model.to(cfg.device)
    optimizer = Adam(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    logs: list[dict[str, float]] = []
    threshold = float(cfg.predecoder.correction_threshold)
    best_state: dict[str, Any] | None = None
    best_epoch = -1
    best_val_metric = float("-inf")
    epochs_without_improvement = 0
    for epoch in range(cfg.training.epochs):
        model.train()
        train_accumulator = _empty_accumulator()
        for batch_idx, batch in enumerate(train_loader):
            batch = {key: value.to(cfg.device) for key, value in batch.items()}
            outputs = model(batch["patch"])
            loss = predecoder_loss(outputs, batch, weights=loss_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            breakdown = compute_loss_breakdown(outputs, batch, weights=loss_weights)
            _update_accumulator(
                train_accumulator,
                outputs,
                batch,
                loss,
                breakdown,
                threshold=threshold,
            )
        train_metrics = _finalize_metrics(train_accumulator)
        logs.append({"epoch": float(epoch), "split": "train", "is_best": 0.0, **train_metrics})
        model.eval()
        val_accumulator = _empty_accumulator()
        with torch.no_grad():
            for batch in val_loader:
                batch = {key: value.to(cfg.device) for key, value in batch.items()}
                outputs = model(batch["patch"])
                loss = predecoder_loss(outputs, batch, weights=loss_weights)
                breakdown = compute_loss_breakdown(outputs, batch, weights=loss_weights)
                _update_accumulator(
                    val_accumulator,
                    outputs,
                    batch,
                    loss,
                    breakdown,
                    threshold=threshold,
                )
        val_metrics = _finalize_metrics(val_accumulator)
        improved = val_metrics["correction_f1"] > best_val_metric + float(min_delta)
        if improved:
            best_val_metric = float(val_metrics["correction_f1"])
            best_epoch = int(epoch)
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        logs.append({"epoch": float(epoch), "split": "val", "is_best": float(improved), **val_metrics})
        logger.info(
            "epoch=%s train_loss=%.6f val_loss=%.6f val_f1=%.6f best_epoch=%s best_val_f1=%.6f",
            epoch,
            train_metrics["loss"],
            val_metrics["loss"],
            val_metrics["correction_f1"],
            best_epoch,
            best_val_metric,
        )
        if patience is not None and epochs_without_improvement >= int(patience):
            logger.info("early stopping at epoch %s after %s stale validation epochs", epoch, epochs_without_improvement)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    target = ensure_parent(out)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": cfg.to_dict(),
            "dataset_metadata": dataset_metadata,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "data_path": str(data),
            "loss_weights": loss_weights,
            "correction_threshold": float(cfg.predecoder.correction_threshold),
            "best_epoch": int(best_epoch),
            "best_metric_name": "val_correction_f1",
            "best_val_metric": float(best_val_metric),
            "epochs_trained": int(max((row["epoch"] for row in logs), default=-1.0) + 1),
            "early_stopping_patience": None if patience is None else int(patience),
        },
        target,
    )
    pd.DataFrame(logs).to_csv(target.with_suffix(".csv"), index=False)
    logger.info("saved checkpoint to %s", out)


if __name__ == "__main__":
    app()
