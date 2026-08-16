"""Risk/runtime profiler model for RT-PreQEC scheduling."""

from __future__ import annotations

import torch
from torch import nn

from rt_preqec.models.encoders import CausalHistoryEncoder, FeatureProjectionEncoder
from rt_preqec.models.heads import ConfidenceHead, HardRuntimeHead, RiskHead, RuntimeRegressionHead
from rt_preqec.models.outputs import RiskRuntimeOutput


class RiskRuntimeModel(nn.Module):
    """RT-PreQEC Risk/Runtime Profiler for lag-bounded scheduling.

    Input: current shot/job features `[B,F]` and optional causal history
    features `[B,T,F]`.
    Output: `RiskRuntimeOutput` with risk, hard-runtime, runtime regression, and
    confidence logits.
    RT-PreQEC role: helps the scheduler decide fast versus accurate backend
    decoding. It never emits global logical corrections and is not a QEC
    decoder.
    Realtime fit: small feature projection plus optional causal GRU/LSTM/TCN
    history encoder uses only past/current jobs, fixed hidden widths, and
    bounded heads.
    """

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
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.history_encoder_type = history_encoder_type.lower()
        self.history_length = int(history_length)
        self.history_hidden_dim = int(history_hidden_dim)
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if bidirectional:
            raise ValueError("RiskRuntimeModel history encoder must be causal and unidirectional")

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
        self.risk_head = RiskHead(hidden_dim)
        self.hard_runtime_head = HardRuntimeHead(hidden_dim)
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
    ) -> RiskRuntimeOutput:
        """Run the scheduler profiler over `[B,F]` and optional `[B,T,F]` history."""
        if features.ndim != 2:
            raise ValueError(f"features must be [B,F], got {tuple(features.shape)}")
        if features.shape[1] != self.feature_dim:
            raise ValueError(f"expected feature_dim={self.feature_dim}, got {features.shape[1]}")
        features = features.float()
        current_embedding = self.current_encoder(features)
        causal_history = self._coerce_history(features, history_features)
        history_embedding = self.history_encoder(causal_history)
        fused_embedding = self.fusion(torch.cat([current_embedding, history_embedding], dim=-1))
        return RiskRuntimeOutput(
            risk_logit=self.risk_head(fused_embedding),
            hard_runtime_logit=self.hard_runtime_head(fused_embedding),
            runtime_pred=self.runtime_regression_head(fused_embedding),
            confidence_logit=self.confidence_head(fused_embedding),
            embeddings={
                "current_embedding": current_embedding,
                "history_embedding": history_embedding,
                "fused_embedding": fused_embedding,
            },
            metadata={
                "model_type": "risk_runtime",
                "history_encoder_type": self.history_encoder_type,
                "history_length": self.history_length,
            },
        )

    @torch.no_grad()
    def predict_proba(
        self,
        features: torch.Tensor,
        history_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return legacy-compatible risk/runtime probabilities for inference."""
        return self.forward(features, history_features).to_dict()

    def count_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
