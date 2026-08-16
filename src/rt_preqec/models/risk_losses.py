"""Losses and metrics for the risk-only profiler."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

try:
    from sklearn.metrics import roc_auc_score
except ImportError:  # pragma: no cover
    roc_auc_score = None


def risk_profiler_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Compute the aggregate loss for the risk-only profiler."""
    weights = weights or {
        "risk": 1.0,
        "hard_runtime": 0.5,
        "runtime_regression": 0.25,
        "confidence": 0.1,
    }
    runtime_target = torch.log1p(batch["accurate_runtime_us"].float())
    runtime_pred = outputs["runtime_pred"].float()
    confidence_target = 1.0 - torch.abs(torch.sigmoid(outputs["risk_logit"]).detach() - batch["risk_label"].float())
    risk_loss = F.binary_cross_entropy_with_logits(outputs["risk_logit"], batch["risk_label"].float())
    hard_runtime_valid = batch.get("hard_runtime_label_valid")
    hard_runtime_valid_flag = True
    if hard_runtime_valid is not None:
        hard_runtime_valid_flag = bool((hard_runtime_valid.float() > 0.5).all().item())
    if hard_runtime_valid_flag and float(weights.get("hard_runtime", 0.5)) > 0.0:
        hard_runtime_loss = F.binary_cross_entropy_with_logits(outputs["runtime_logit"], batch["hard_runtime"].float())
    else:
        hard_runtime_loss = outputs["runtime_logit"].float().sum() * 0.0
    runtime_loss = F.mse_loss(runtime_pred, runtime_target)
    confidence_loss = F.binary_cross_entropy_with_logits(outputs["confidence_logit"], confidence_target.clamp(0.0, 1.0))
    return (
        weights["risk"] * risk_loss
        + (weights["hard_runtime"] if hard_runtime_valid_flag else 0.0) * hard_runtime_loss
        + weights["runtime_regression"] * runtime_loss
        + weights["confidence"] * confidence_loss
    )


def compute_risk_metrics(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute binary classification metrics for risk prediction."""
    scores = torch.sigmoid(outputs["risk_logit"]).detach().cpu().numpy().astype(float)
    labels = batch["risk_label"].detach().cpu().numpy().astype(int)
    predictions = (scores >= threshold).astype(int)
    positives = max(int((labels == 1).sum()), 1)
    negatives = max(int((labels == 0).sum()), 1)
    tp = int(((predictions == 1) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    metrics: dict[str, Any] = {
        "accuracy": float((tp + tn) / max(len(labels), 1)),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / positives),
        "false_negative_rate": float(fn / positives),
        "false_positive_rate": float(fp / negatives),
    }
    if roc_auc_score is not None and len(np.unique(labels)) > 1:
        metrics["risk_auc"] = float(roc_auc_score(labels, scores))
    else:
        metrics["risk_auc"] = None
    return metrics
