"""Tiny neural predecoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelStats:
    """Simple model statistics."""

    num_parameters: int


class TinyNeuralPredecoder(nn.Module):
    """Small 3D CNN over `[B, 1, W, K, K]` syndrome patches."""

    def __init__(self, temporal_window: int, patch_size: int, hidden_channels: int = 16) -> None:
        super().__init__()
        self.temporal_window = temporal_window
        self.patch_size = patch_size
        self.encoder = nn.Sequential(
            nn.Conv3d(1, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(),
        )
        self.correction_head = nn.Sequential(
            nn.Conv2d(hidden_channels + 2, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.confidence_head = nn.Linear(hidden_channels, 1)
        self.risk_head = nn.Linear(hidden_channels, 1)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(inputs)
        current_features = features[:, :, -1, :, :]
        raw_current = inputs[:, :, -1, :, :]
        raw_temporal_mean = inputs.mean(dim=2)
        correction_input = torch.cat([current_features, raw_current, raw_temporal_mean], dim=1)
        correction_logits = self.correction_head(correction_input).flatten(start_dim=1)
        pooled = self.pool(features).flatten(start_dim=1)
        return {
            "correction_logits": correction_logits,
            "confidence_logit": self.confidence_head(pooled).squeeze(-1),
            "risk_logit": self.risk_head(pooled).squeeze(-1),
        }


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
