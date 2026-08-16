"""Model heads for RT-PreQEC selective and scheduler outputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from rt_preqec.models.encoders import PatchCandidateCompatibility


@dataclass
class HeadDimensions:
    """Simple container for output dimensions."""

    hidden: int
    correction_dim: int


class MultiHeadOutput(nn.Module):
    """Shared hidden layer with correction, confidence, and risk heads."""

    def __init__(self, dims: HeadDimensions) -> None:
        super().__init__()
        self.correction = nn.Linear(dims.hidden, dims.correction_dim)
        self.confidence = nn.Linear(dims.hidden, 1)
        self.risk = nn.Linear(dims.hidden, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "correction_logits": self.correction(features),
            "confidence_logit": self.confidence(features).squeeze(-1),
            "risk_logit": self.risk(features).squeeze(-1),
        }


class RiskHead(nn.Module):
    """Predict fast-decoder risk for scheduler routing.

    Input: hidden tensor `[B, H]` from current and optional causal-history
    features.
    Output: `risk_logit` tensor `[B, 1]`.
    RT-PreQEC role: estimates whether the fast backend decoder may disagree,
    logically fail, or otherwise be unsafe for the current job.
    Realtime fit: a single bounded linear head adds negligible latency and does
    not emit corrections, only a scheduler hint that can fall back.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden)


class HardRuntimeHead(nn.Module):
    """Predict accurate-decoder runtime tail membership.

    Input: hidden tensor `[B, H]`.
    Output: `hard_runtime_logit` tensor `[B, 1]`.
    RT-PreQEC role: identifies jobs likely to exceed a p90/p95 accurate-decoder
    runtime threshold so the risk-aware scheduler can estimate service risk.
    Realtime fit: fixed linear compute provides a bounded online runtime signal.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden)


class RuntimeRegressionHead(nn.Module):
    """Regress accurate backend service time for scheduler cost estimation.

    Input: hidden tensor `[B, H]`.
    Output: `runtime_pred` tensor `[B, 1]`, trained on
    `log1p(accurate_runtime_us)`.
    RT-PreQEC role: gives the lag-bounded scheduler a small service-time
    estimate for accurate backend decoding.
    Realtime fit: bounded linear regression head with fallback-safe output.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden)


class ConfidenceHead(nn.Module):
    """Predict whether the model output should be trusted.

    Input: hidden tensor `[B, H]`.
    Output: `confidence_logit` tensor `[B, 1]`.
    RT-PreQEC role: calibrates trust in risk or candidate outputs for scheduler
    and validation gating.
    Realtime fit: confidence is not risk; it is an explicit small head that
    allows conservative fallback when model evidence is weak.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden)


class AbstainHead(nn.Module):
    """Predict whether the selective predecoder should refuse a candidate.

    Input: patch hidden tensor `[B, H]`.
    Output: `abstain_logit` tensor `[B, 1]`.
    RT-PreQEC role: rejects ambiguous local patches, observable-touching or
    low-overlap candidates, and clusters too risky for local predecoding.
    Realtime fit: abstention is a first-class bounded output that routes to
    validation or backend fallback instead of forcing a neural correction.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden)


class CandidateHead(nn.Module):
    """Rank DEM local candidates against a detector patch.

    Input: patch embedding `[B,H]`, candidate embeddings `[B,C,H]`, candidate
    mask `[B,C]`.
    Output: masked candidate logits `[B,C]`.
    RT-PreQEC role: wraps `PatchCandidateCompatibility` so the predecoder only
    chooses among DEM local candidates or abstains.
    Realtime fit: candidate masking and fixed local candidate count keep compute
    bounded and prevent invalid candidate selection.
    """

    def __init__(self, hidden_dim: int = 64, scorer: str = "bilinear") -> None:
        super().__init__()
        self.compatibility = PatchCandidateCompatibility(hidden_dim=hidden_dim, scorer=scorer)

    def forward(
        self,
        patch_embedding: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.compatibility(patch_embedding, candidate_embeddings, candidate_mask)
