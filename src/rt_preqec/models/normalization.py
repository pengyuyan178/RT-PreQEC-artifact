"""Normalization helpers for RT-PreQEC model training and inference."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn


def compute_normalization_stats(
    features: np.ndarray | torch.Tensor,
    feature_names: list[str] | None = None,
    eps: float = 1e-6,
) -> dict[str, Any]:
    """Compute mean/std statistics for feature normalization."""
    if isinstance(features, torch.Tensor):
        array = features.detach().cpu().numpy().astype(np.float32)
    else:
        array = np.asarray(features, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("features must be a 2D array")
    mean = array.mean(axis=0).astype(np.float32) if len(array) else np.zeros(array.shape[1], dtype=np.float32)
    std = array.std(axis=0).astype(np.float32) if len(array) else np.ones(array.shape[1], dtype=np.float32)
    std = np.where(np.abs(std) > eps, std, 1.0).astype(np.float32)
    return {
        "mean": mean,
        "std": std,
        "eps": float(eps),
        "feature_names": list(feature_names or []),
    }


def apply_normalization(
    features: np.ndarray | torch.Tensor,
    stats: dict[str, Any] | None,
) -> np.ndarray | torch.Tensor:
    """Apply saved normalization stats to numpy or torch features."""
    if stats is None:
        return features
    mean_value = stats.get("mean", [])
    std_value = stats.get("std", [])
    if isinstance(features, torch.Tensor):
        mean = torch.as_tensor(mean_value, dtype=features.dtype, device=features.device)
        std = torch.as_tensor(std_value, dtype=features.dtype, device=features.device)
        if mean.numel() != features.shape[-1] or std.numel() != features.shape[-1]:
            return features
        std = torch.where(std.abs() > 1e-6, std, torch.ones_like(std))
        return (features - mean) / std
    array = np.asarray(features, dtype=np.float32)
    mean = np.asarray(mean_value, dtype=np.float32)
    std = np.asarray(std_value, dtype=np.float32)
    if mean.size != array.shape[-1] or std.size != array.shape[-1]:
        return array
    std = np.where(np.abs(std) > 1e-6, std, 1.0).astype(np.float32)
    return (array - mean) / std


class NormalizationLayer(nn.Module):
    """Fixed feature normalization layer.

    Input: current or history features ending in feature dimension `F`.
    Output: normalized tensor with the same shape.
    RT-PreQEC role: keeps scheduler feature scales stable across syndrome,
    patch-aggregate, and runtime-state inputs.
    Realtime fit: buffer subtraction/division is deterministic, tiny, and
    checkpointable with feature names.
    """

    def __init__(self, mean: np.ndarray | torch.Tensor, std: np.ndarray | torch.Tensor, eps: float = 1e-6) -> None:
        super().__init__()
        mean_tensor = torch.as_tensor(mean, dtype=torch.float32)
        std_tensor = torch.as_tensor(std, dtype=torch.float32)
        std_tensor = torch.where(std_tensor.abs() > eps, std_tensor, torch.ones_like(std_tensor))
        self.register_buffer("mean", mean_tensor)
        self.register_buffer("std", std_tensor)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.mean) / self.std
