"""Decomposed risk/runtime profiler for RT-PreQEC scheduling."""

from __future__ import annotations

import torch
from torch import nn

from rt_preqec.models.encoders import CausalHistoryEncoder, FeatureProjectionEncoder
from rt_preqec.models.heads import ConfidenceHead, HardRuntimeHead, RuntimeRegressionHead
from rt_preqec.models.outputs import DecomposedRiskOutput


def _prob_to_logit(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    value = value.clamp(float(eps), 1.0 - float(eps))
    return torch.log(value / (1.0 - value))


class RiskDecompositionModel(nn.Module):
    """Risk/runtime profiler with explicit scheduler-risk component heads."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        feature_layers: int = 2,
        history_encoder_type: str = "none",
        history_length: int = 1,
        history_hidden_dim: int = 64,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
        num_layers: int = 1,
        bidirectional: bool = False,
        combination_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.history_encoder_type = str(history_encoder_type).lower()
        self.history_length = int(history_length)
        self.history_hidden_dim = int(history_hidden_dim)
        self.combination_weights = dict(
            combination_weights
            or {
                "fast_wrong": 1.0,
                "fast_logical_fail": 1.0,
                "hard_runtime": 0.5,
                "syndrome_tail": 0.2,
            }
        )
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if bidirectional:
            raise ValueError("RiskDecompositionModel history encoder must be causal and unidirectional")

        self.current_encoder = FeatureProjectionEncoder(
            input_dim=self.feature_dim,
            hidden_dim=hidden_dim,
            num_layers=feature_layers,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )
        self.use_history = self.history_length > 1 or self.history_encoder_type != "none"
        self.history_encoder = CausalHistoryEncoder(
            input_dim=self.feature_dim,
            hidden_dim=history_hidden_dim,
            encoder_type=self.history_encoder_type,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        fusion_dim = int(hidden_dim) + int(history_hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.fast_wrong_head = nn.Linear(hidden_dim, 1)
        self.fast_logical_fail_head = nn.Linear(hidden_dim, 1)
        self.hard_runtime_head = HardRuntimeHead(hidden_dim)
        self.syndrome_tail_head = nn.Linear(hidden_dim, 1)
        self.safe_fast_head = nn.Linear(hidden_dim, 1)
        self.runtime_regression_head = RuntimeRegressionHead(hidden_dim)
        self.confidence_head = ConfidenceHead(hidden_dim)

    def _coerce_history(self, features: torch.Tensor, history_features: torch.Tensor | None) -> torch.Tensor:
        if history_features is None:
            return features.unsqueeze(1)
        if history_features.ndim != 3:
            raise ValueError(f"history_features must be [B,T,F], got {tuple(history_features.shape)}")
        if history_features.shape[0] != features.shape[0] or history_features.shape[2] != features.shape[1]:
            raise ValueError("history_features must match features batch and feature dimensions")
        return history_features.float()

    def forward(
        self,
        features: torch.Tensor,
        history_features: torch.Tensor | None = None,
    ) -> DecomposedRiskOutput:
        """Run decomposed scheduler-risk heads over current and causal history features."""
        if features.ndim != 2:
            raise ValueError(f"features must be [B,F], got {tuple(features.shape)}")
        if features.shape[1] != self.feature_dim:
            raise ValueError(f"expected feature_dim={self.feature_dim}, got {features.shape[1]}")
        features = features.float()
        current_embedding = self.current_encoder(features)
        causal_history = self._coerce_history(features, history_features)
        history_embedding = self.history_encoder(causal_history)
        fused_embedding = self.fusion(torch.cat([current_embedding, history_embedding], dim=-1))
        fast_wrong_logit = self.fast_wrong_head(fused_embedding)
        fast_logical_fail_logit = self.fast_logical_fail_head(fused_embedding)
        hard_runtime_logit = self.hard_runtime_head(fused_embedding)
        syndrome_tail_logit = self.syndrome_tail_head(fused_embedding)
        safe_fast_logit = self.safe_fast_head(fused_embedding)
        weights = self.combination_weights
        fast_wrong_weight = float(weights.get("fast_wrong", 1.0))
        fast_fail_weight = float(weights.get("fast_logical_fail", 1.0))
        hard_weight = float(weights.get("hard_runtime", 0.5))
        tail_weight = float(weights.get("syndrome_tail", 0.2))
        total_weight = max(fast_wrong_weight + fast_fail_weight + hard_weight + tail_weight, 1e-6)
        scheduler_risk_prob = (
            fast_wrong_weight * torch.sigmoid(fast_wrong_logit)
            + fast_fail_weight * torch.sigmoid(fast_logical_fail_logit)
            + hard_weight * torch.sigmoid(hard_runtime_logit)
            + tail_weight * torch.sigmoid(syndrome_tail_logit)
        ) / total_weight
        return DecomposedRiskOutput(
            risk_logit=_prob_to_logit(scheduler_risk_prob),
            confidence_logit=self.confidence_head(fused_embedding),
            runtime_pred=self.runtime_regression_head(fused_embedding),
            fast_wrong_logit=fast_wrong_logit,
            fast_logical_fail_logit=fast_logical_fail_logit,
            hard_runtime_logit=hard_runtime_logit,
            syndrome_tail_logit=syndrome_tail_logit,
            safe_fast_logit=safe_fast_logit,
            embeddings={
                "current_embedding": current_embedding,
                "history_embedding": history_embedding,
                "fused_embedding": fused_embedding,
            },
            metadata={
                "model_type": "risk_decomposed",
                "history_encoder_type": self.history_encoder_type,
                "history_length": self.history_length,
            },
            combination_weights=self.combination_weights,
        )

    @torch.no_grad()
    def predict_proba(
        self,
        features: torch.Tensor,
        history_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return legacy-compatible and decomposed probabilities for inference."""
        return self.forward(features, history_features).to_dict()

    def count_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


RiskRuntimeModelV2 = RiskDecompositionModel
