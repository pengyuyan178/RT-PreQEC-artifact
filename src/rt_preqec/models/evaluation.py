"""Inference and evaluation utilities for RT-PreQEC models."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from rt_preqec.models.losses import compute_risk_runtime_metrics
from rt_preqec.models.normalization import apply_normalization


def _device_of(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@torch.no_grad()
def evaluate_risk_runtime_model(
    model: torch.nn.Module,
    dataloader: Any,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Evaluate a risk/runtime model over a dataloader."""
    model.eval()
    metrics = []
    for batch in dataloader:
        device = _device_of(model)
        tensor_batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        if "history_features" in tensor_batch:
            output = model(tensor_batch["features"], tensor_batch["history_features"])
        else:
            output = model(tensor_batch["features"])
        metrics.append(compute_risk_runtime_metrics(output, tensor_batch, threshold=threshold))
    if not metrics:
        return {}
    keys = sorted({key for item in metrics for key in item})
    return {key: float(np.mean([item.get(key, 0.0) for item in metrics])) for key in keys}


def sweep_risk_threshold(
    model: torch.nn.Module,
    dataloader: Any,
    thresholds: list[float] | np.ndarray,
) -> list[dict[str, float]]:
    """Evaluate risk/runtime metrics over multiple scheduler thresholds."""
    return [
        {"threshold": float(threshold), **evaluate_risk_runtime_model(model, dataloader, threshold=float(threshold))}
        for threshold in thresholds
    ]


@torch.no_grad()
def predict_risk_runtime(
    model: torch.nn.Module,
    features: np.ndarray | torch.Tensor,
    history_features: np.ndarray | torch.Tensor | None = None,
    normalization: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Run normalized risk/runtime inference and return numpy arrays.

    Output keys are adapted for `evaluate_real_stream.py`:
    `risk_score`, `hard_runtime_score`, `runtime_pred`, and `confidence`.
    Legacy `runtime_score` is also returned as an alias for hard-runtime score.
    """
    device = _device_of(model)
    if isinstance(features, torch.Tensor):
        feature_tensor = features.detach().to(device=device, dtype=torch.float32)
    else:
        feature_tensor = torch.as_tensor(np.asarray(features, dtype=np.float32), device=device)
    if feature_tensor.ndim == 1:
        feature_tensor = feature_tensor.unsqueeze(0)
    feature_tensor = apply_normalization(feature_tensor, normalization)
    history_tensor = None
    if history_features is not None:
        if isinstance(history_features, torch.Tensor):
            history_tensor = history_features.detach().to(device=device, dtype=torch.float32)
        else:
            history_tensor = torch.as_tensor(np.asarray(history_features, dtype=np.float32), device=device)
        if history_tensor.ndim == 2:
            history_tensor = history_tensor.unsqueeze(0)
        history_tensor = apply_normalization(history_tensor, normalization)
    model.eval()
    if history_tensor is None:
        output = model(feature_tensor)
    else:
        output = model(feature_tensor, history_tensor)
    values = output.to_dict() if hasattr(output, "to_dict") else output
    hard_runtime = values.get("hard_runtime_score", values.get("runtime_score"))
    return {
        "risk_score": values["risk_score"].detach().cpu().numpy(),
        "hard_runtime_score": hard_runtime.detach().cpu().numpy(),
        "runtime_score": hard_runtime.detach().cpu().numpy(),
        "runtime_pred": values["runtime_pred"].detach().cpu().numpy(),
        "confidence": values["confidence"].detach().cpu().numpy(),
    }
