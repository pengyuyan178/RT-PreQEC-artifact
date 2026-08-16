"""Training loops for RT-PreQEC model components."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import time

import numpy as np
import pandas as pd
import torch

from rt_preqec.models.checkpointing import save_model_checkpoint
from rt_preqec.models.losses import (
    candidate_predecoder_loss,
    compute_candidate_metrics,
    compute_risk_runtime_metrics,
    risk_runtime_loss,
)


def _to_device(batch: dict[str, torch.Tensor], device: str | torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _mean_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {}
    keys = sorted({key for item in metrics for key in item})
    return {key: float(np.mean([item.get(key, 0.0) for item in metrics])) for key in keys}


def _format_metric(value: Any, precision: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


def _format_risk_epoch_summary(
    epoch: int,
    epochs: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float] | None,
) -> str:
    display_metrics = val_metrics if val_metrics is not None else train_metrics
    if "combined_risk_accuracy" in display_metrics:
        parts = [
            f"Epoch {epoch + 1:03d}/{epochs:03d}",
            f"train_loss={_format_metric(train_metrics.get('loss'))}",
        ]
        if val_metrics is not None:
            parts.append(f"val_loss={_format_metric(val_metrics.get('loss'))}")
        parts.extend(
            [
                f"combined_acc={_format_metric(display_metrics.get('combined_risk_accuracy'))}",
                f"combined_precision={_format_metric(display_metrics.get('combined_risk_precision'))}",
                f"combined_recall={_format_metric(display_metrics.get('combined_risk_recall'))}",
                f"combined_fnr={_format_metric(display_metrics.get('combined_risk_fnr'))}",
                f"combined_fpr={_format_metric(display_metrics.get('combined_risk_fpr'))}",
                f"fast_wrong_precision={_format_metric(display_metrics.get('fast_wrong_precision'))}",
                f"fast_wrong_recall={_format_metric(display_metrics.get('fast_wrong_recall'))}",
                f"fast_wrong_fnr={_format_metric(display_metrics.get('fast_wrong_fnr'))}",
                f"fast_wrong_fpr={_format_metric(display_metrics.get('fast_wrong_fpr'))}",
                f"fast_logical_fail_precision={_format_metric(display_metrics.get('fast_logical_fail_precision'))}",
                f"fast_logical_fail_recall={_format_metric(display_metrics.get('fast_logical_fail_recall'))}",
                f"fast_logical_fail_fnr={_format_metric(display_metrics.get('fast_logical_fail_fnr'))}",
                f"fast_logical_fail_fpr={_format_metric(display_metrics.get('fast_logical_fail_fpr'))}",
                f"safe_fast_precision={_format_metric(display_metrics.get('safe_fast_precision'))}",
                f"safe_fast_recall={_format_metric(display_metrics.get('safe_fast_recall'))}",
                f"hard_runtime_acc={_format_metric(display_metrics.get('hard_runtime_accuracy'))}",
                f"runtime_mae={_format_metric(display_metrics.get('runtime_mae'))}",
                f"samples/sec={_format_metric(train_metrics.get('samples_per_sec'), precision=2)}",
            ]
        )
        return " | ".join(parts)
    parts = [
        f"Epoch {epoch + 1:03d}/{epochs:03d}",
        f"train_loss={_format_metric(train_metrics.get('loss'))}",
    ]
    if val_metrics is not None:
        parts.append(f"val_loss={_format_metric(val_metrics.get('loss'))}")
    parts.extend(
        [
            f"acc={_format_metric(display_metrics.get('risk_accuracy'))}",
            f"precision={_format_metric(display_metrics.get('precision'))}",
            f"recall={_format_metric(display_metrics.get('recall'))}",
            f"fpr={_format_metric(display_metrics.get('fpr'))}",
            f"fnr={_format_metric(display_metrics.get('fnr'))}",
            f"runtime_mae={_format_metric(display_metrics.get('runtime_mae'))}",
            f"samples/sec={_format_metric(train_metrics.get('samples_per_sec'), precision=2)}",
        ]
    )
    return " | ".join(parts)


def _format_candidate_epoch_summary(
    epoch: int,
    epochs: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float] | None = None,
) -> str:
    display_metrics = val_metrics if val_metrics is not None else train_metrics
    parts = [
        f"Epoch {epoch + 1:03d}/{epochs:03d}",
        f"train_loss={_format_metric(train_metrics.get('loss'))}",
    ]
    if val_metrics is not None:
        parts.append(f"val_loss={_format_metric(val_metrics.get('loss'))}")
    parts.extend(
        [
            f"candidate_acc={_format_metric(display_metrics.get('candidate_accuracy'))}",
            f"abstain_rate={_format_metric(display_metrics.get('abstain_rate'))}",
            f"false_accept_rate={_format_metric(display_metrics.get('false_accept_rate'))}",
        ]
    )
    return " | ".join(parts)


class RiskRuntimeTrainer:
    """Trainer for the RT-PreQEC Risk/Runtime Profiler.

    Input: dataloaders yielding current features and optional
    `history_features`.
    Output: per-epoch train/validation loss and scheduler-facing metrics.
    RT-PreQEC role: optimizes risk, runtime hardness, runtime regression, and
    confidence heads for the risk-aware scheduler.
    Realtime fit: training mirrors online inference by passing causal history
    only when present and never requiring future shots.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str | torch.device,
        loss_weights: dict[str, float] | None = None,
        grad_clip_norm: float | None = None,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.loss_weights = loss_weights or {}
        self.grad_clip_norm = grad_clip_norm
        self.logs: list[dict[str, Any]] = []

    def _forward(self, batch: dict[str, torch.Tensor]) -> Any:
        if "history_features" in batch:
            return self.model(features=batch["features"], history_features=batch["history_features"])
        return self.model(features=batch["features"])

    def train_one_epoch(self, loader: Any) -> dict[str, float]:
        self.model.train()
        start_time = time.perf_counter()
        losses: list[float] = []
        metrics: list[dict[str, float]] = []
        for batch in loader:
            batch = _to_device(batch, self.device)
            output = self._forward(batch)
            loss = risk_runtime_loss(output, batch, self.loss_weights)
            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()
            losses.append(float(loss.item()))
            metrics.append(compute_risk_runtime_metrics(output, batch))
        summary = _mean_metrics(metrics)
        summary["loss"] = float(np.mean(losses)) if losses else 0.0
        elapsed = max(time.perf_counter() - start_time, 1e-9)
        sample_count = int(len(getattr(loader, "dataset", [])))
        summary["epoch_time_sec"] = float(elapsed)
        summary["samples_per_sec"] = float(sample_count / elapsed) if sample_count else 0.0
        return summary

    @torch.no_grad()
    def validate(self, loader: Any) -> dict[str, float]:
        self.model.eval()
        losses: list[float] = []
        metrics: list[dict[str, float]] = []
        for batch in loader:
            batch = _to_device(batch, self.device)
            output = self._forward(batch)
            loss = risk_runtime_loss(output, batch, self.loss_weights)
            losses.append(float(loss.item()))
            metrics.append(compute_risk_runtime_metrics(output, batch))
        summary = _mean_metrics(metrics)
        summary["loss"] = float(np.mean(losses)) if losses else 0.0
        return summary

    def fit(
        self,
        train_loader: Any,
        val_loader: Any | None = None,
        epochs: int = 1,
        verbose: bool = True,
        log_every_epoch: bool = True,
    ) -> list[dict[str, Any]]:
        total_epochs = int(epochs)
        for epoch in range(total_epochs):
            train_metrics = self.train_one_epoch(train_loader)
            self.logs.append({"epoch": epoch, "split": "train", **train_metrics})
            val_metrics = None
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                self.logs.append({"epoch": epoch, "split": "val", **val_metrics})
            if verbose and log_every_epoch:
                print(_format_risk_epoch_summary(epoch, total_epochs, train_metrics, val_metrics))
        return self.logs

    def save_checkpoint(
        self,
        path: str | Path,
        model_type: str,
        model_config: dict[str, Any],
        normalization: dict[str, Any] | None,
        feature_names: list[str],
        metrics: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        save_model_checkpoint(
            path=path,
            model=self.model,
            model_type=model_type,
            model_config=model_config,
            normalization=normalization,
            feature_names=feature_names,
            metrics=metrics or {},
            extra=extra or {},
        )
        pd.DataFrame(self.logs).to_csv(Path(path).with_suffix(".training_log.csv"), index=False)


class CandidatePredecoderTrainer:
    """Scaffold trainer for the selective DEM-candidate predecoder.

    This is intentionally minimal until Milestone 3B connects real
    DetectorPatch + LocalErrorCandidate supervision.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str | torch.device,
        loss_weights: dict[str, float] | None = None,
        grad_clip_norm: float | None = None,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.loss_weights = loss_weights or {}
        self.grad_clip_norm = grad_clip_norm
        self.logs: list[dict[str, Any]] = []

    def train_one_epoch(self, loader: Any) -> dict[str, float]:
        self.model.train()
        losses: list[float] = []
        metrics: list[dict[str, float]] = []
        for batch in loader:
            batch = _to_device(batch, self.device)
            output = self.model(
                batch["detector_features"],
                batch["detector_mask"],
                batch["candidate_features"],
                batch["candidate_mask"],
            )
            loss = candidate_predecoder_loss(output, batch, self.loss_weights)
            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()
            losses.append(float(loss.item()))
            metrics.append(compute_candidate_metrics(output, batch))
        summary = _mean_metrics(metrics)
        summary["loss"] = float(np.mean(losses)) if losses else 0.0
        return summary

    @torch.no_grad()
    def validate(self, loader: Any) -> dict[str, float]:
        self.model.eval()
        losses: list[float] = []
        metrics: list[dict[str, float]] = []
        for batch in loader:
            batch = _to_device(batch, self.device)
            output = self.model(
                batch["detector_features"],
                batch["detector_mask"],
                batch["candidate_features"],
                batch["candidate_mask"],
            )
            loss = candidate_predecoder_loss(output, batch, self.loss_weights)
            losses.append(float(loss.item()))
            metrics.append(compute_candidate_metrics(output, batch))
        summary = _mean_metrics(metrics)
        summary["loss"] = float(np.mean(losses)) if losses else 0.0
        return summary

    def fit(
        self,
        train_loader: Any,
        val_loader: Any | None = None,
        epochs: int = 1,
        verbose: bool = True,
        log_every_epoch: bool = True,
    ) -> list[dict[str, Any]]:
        total_epochs = int(epochs)
        for epoch in range(total_epochs):
            train_metrics = self.train_one_epoch(train_loader)
            self.logs.append({"epoch": epoch, "split": "train", **train_metrics})
            val_metrics = None
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                self.logs.append({"epoch": epoch, "split": "val", **val_metrics})
            if verbose and log_every_epoch:
                print(_format_candidate_epoch_summary(epoch, total_epochs, train_metrics, val_metrics))
        return self.logs
