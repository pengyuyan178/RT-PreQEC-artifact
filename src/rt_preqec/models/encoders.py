"""Paper-driven encoders for RT-PreQEC model components."""

from __future__ import annotations

import torch
from torch import nn


class FeatureProjectionEncoder(nn.Module):
    """Encode heterogeneous shot-level features for risk-aware scheduling.

    Input: tensor `[B, F]` containing current syndrome features, patch aggregate
    features, and runtime-state features.
    Output: tensor `[B, H]` in a bounded hidden space.
    RT-PreQEC role: base encoder for the Risk/Runtime Profiler, corresponding
    to lightweight features for risk-aware scheduling.
    Realtime fit: fixed input width and a small MLP give bounded compute while
    LayerNorm stabilizes different feature scales such as syndrome weight,
    candidate counts, and backlog/runtime state.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        use_layer_norm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        layers: list[nn.Module] = []
        current_dim = int(input_dim)
        for _ in range(max(int(num_layers), 1)):
            layers.append(nn.Linear(current_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        self.network = nn.Sequential(*layers)
        self.output_dim = int(hidden_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Encode current shot features from `[B, F]` to `[B, H]`."""
        if features.ndim != 2:
            raise ValueError(f"FeatureProjectionEncoder expects [B,F], got {tuple(features.shape)}")
        return self.network(features.float())


class _CausalTCN(nn.Module):
    """Small causal temporal convolution for bounded recent-history encoding.

    Input: tensor `[B, T, F]` containing only past and current job features.
    Output: tensor `[B, H]` from the last causal time step.
    RT-PreQEC role: captures recent burst, drift, repeated high-density
    syndrome, and backlog trend without recurrent state.
    Realtime fit: causal left padding and fixed convolution depth make inference
    timing predictable and prevent future-shot leakage.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = int(input_dim)
        for layer_idx in range(max(int(num_layers), 1)):
            dilation = 2**layer_idx
            conv = nn.Conv1d(
                current_dim,
                hidden_dim,
                kernel_size=3,
                padding=0,
                dilation=dilation,
            )
            layers.append(nn.ConstantPad1d((2 * dilation, 0), 0.0))
            layers.append(conv)
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        self.network = nn.Sequential(*layers)
        self.output_dim = int(hidden_dim)

    def forward(self, history_features: torch.Tensor) -> torch.Tensor:
        values = history_features.float().transpose(1, 2)
        encoded = self.network(values)
        return encoded[:, :, -1]


class CausalHistoryEncoder(nn.Module):
    """Encode recent syndrome/job history without looking into the future.

    Input: tensor `[B, T, F]` where time is ordered from oldest to current.
    Output: tensor `[B, H]` summarizing causal history.
    RT-PreQEC role: models recent syndrome/job history such as bursts, drift,
    consecutive high-density syndromes, and backlog growth for the scheduler.
    Realtime fit: all modes use only past/current features. GRU/LSTM are
    unidirectional recurrent modules; TCN uses left-padded causal convolutions;
    `none` is a fixed MLP over the current time step for non-temporal baseline.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        encoder_type: str = "gru",
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        if bidirectional:
            raise ValueError("RT-PreQEC online history encoders must be causal and unidirectional")
        self.encoder_type = encoder_type.lower()
        self.output_dim = int(hidden_dim)
        rnn_dropout = float(dropout) if int(num_layers) > 1 else 0.0
        if self.encoder_type == "none":
            self.encoder: nn.Module = FeatureProjectionEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=1,
                use_layer_norm=True,
                dropout=dropout,
            )
        elif self.encoder_type == "gru":
            self.encoder = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=max(int(num_layers), 1),
                dropout=rnn_dropout,
                batch_first=True,
                bidirectional=False,
            )
        elif self.encoder_type == "lstm":
            self.encoder = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=max(int(num_layers), 1),
                dropout=rnn_dropout,
                batch_first=True,
                bidirectional=False,
            )
        elif self.encoder_type == "tcn":
            self.encoder = _CausalTCN(input_dim, hidden_dim, num_layers, dropout)
        else:
            raise ValueError(f"Unsupported history encoder_type: {encoder_type}")

    def forward(self, history_features: torch.Tensor) -> torch.Tensor:
        """Encode `[B,T,F]` history and return the current-time embedding `[B,H]`."""
        if history_features.ndim != 3:
            raise ValueError(f"CausalHistoryEncoder expects [B,T,F], got {tuple(history_features.shape)}")
        if history_features.shape[1] < 1:
            raise ValueError("history_features must contain at least one time step")
        history_features = history_features.float()
        if self.encoder_type == "none":
            return self.encoder(history_features[:, -1, :])
        if self.encoder_type == "gru":
            _, hidden = self.encoder(history_features)
            return hidden[-1]
        if self.encoder_type == "lstm":
            _, (hidden, _) = self.encoder(history_features)
            return hidden[-1]
        return self.encoder(history_features)


class DetectorPatchEncoder(nn.Module):
    """Encode a variable-size layout-aware local DetectorPatch.

    Input: detector features `[B, P, D]` and detector mask `[B, P]`, where each
    detector feature can include syndrome bit, coordinates, detector type, and
    time coordinate.
    Output: patch embedding `[B, H]`.
    RT-PreQEC role: represents the local detector patch used by the selective
    neural predecoder.
    Realtime fit: per-detector MLP plus masked mean/max pooling handles bounded
    local patches without assuming a toy grid or running an unbounded graph
    network. Pooling supports variable detector count and fixed compute caps.
    """

    def __init__(self, detector_feature_dim: int, hidden_dim: int = 64, pooling: str = "mean_max") -> None:
        super().__init__()
        if pooling not in {"mean", "max", "mean_max"}:
            raise ValueError("pooling must be one of: mean, max, mean_max")
        self.pooling = pooling
        self.per_detector = nn.Sequential(
            nn.Linear(detector_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        pooled_dim = hidden_dim * (2 if pooling == "mean_max" else 1)
        self.project = nn.Sequential(nn.Linear(pooled_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())
        self.output_dim = int(hidden_dim)

    def forward(self, detector_features: torch.Tensor, detector_mask: torch.Tensor) -> torch.Tensor:
        """Encode detector features `[B,P,D]` with mask `[B,P]` to `[B,H]`."""
        if detector_features.ndim != 3:
            raise ValueError(f"detector_features must be [B,P,D], got {tuple(detector_features.shape)}")
        if detector_mask.shape != detector_features.shape[:2]:
            raise ValueError("detector_mask must match detector_features [B,P]")
        encoded = self.per_detector(detector_features.float())
        mask = detector_mask.to(dtype=torch.bool, device=encoded.device)
        mask_f = mask.unsqueeze(-1).float()
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        mean_pool = (encoded * mask_f).sum(dim=1) / denom
        very_negative = torch.finfo(encoded.dtype).min
        max_values = encoded.masked_fill(~mask.unsqueeze(-1), very_negative).max(dim=1).values
        max_pool = torch.where(mask.any(dim=1, keepdim=True), max_values, torch.zeros_like(max_values))
        if self.pooling == "mean":
            pooled = mean_pool
        elif self.pooling == "max":
            pooled = max_pool
        else:
            pooled = torch.cat([mean_pool, max_pool], dim=-1)
        return self.project(pooled)


class DEMCandidateEncoder(nn.Module):
    """Encode DEM local candidate features for selective predecoding.

    Input: candidate features `[B, C, K]` and candidate mask `[B, C]`.
    Output: candidate embeddings `[B, C, H]`.
    RT-PreQEC role: represents local DEM candidates, including probability or
    weight, detector span, observable-touching flag, active-detector overlap,
    and local/nonlocal indicators.
    Realtime fit: candidates come from a bounded local DEM list and invalid
    positions are masked, so the model ranks only validated candidate slots
    rather than generating arbitrary global corrections.
    """

    def __init__(self, candidate_feature_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(candidate_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.output_dim = int(hidden_dim)

    def forward(self, candidate_features: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
        """Encode candidate features `[B,C,K]` to masked embeddings `[B,C,H]`."""
        if candidate_features.ndim != 3:
            raise ValueError(f"candidate_features must be [B,C,K], got {tuple(candidate_features.shape)}")
        if candidate_mask.shape != candidate_features.shape[:2]:
            raise ValueError("candidate_mask must match candidate_features [B,C]")
        encoded = self.network(candidate_features.float())
        return encoded * candidate_mask.to(dtype=encoded.dtype, device=encoded.device).unsqueeze(-1)


class PatchCandidateCompatibility(nn.Module):
    """Score compatibility between a DetectorPatch and each DEM candidate.

    Input: patch embedding `[B,H]`, candidate embeddings `[B,C,H]`, and mask
    `[B,C]`.
    Output: candidate logits `[B,C]`.
    RT-PreQEC role: answers whether a local DEM candidate matches the observed
    patch syndrome, forming the candidate logits for selective predecoding.
    Realtime fit: bilinear or small concat-MLP scoring is bounded in candidate
    count, applies explicit candidate masking, and never emits arbitrary global
    corrections.
    """

    def __init__(self, hidden_dim: int = 64, scorer: str = "bilinear") -> None:
        super().__init__()
        if scorer not in {"bilinear", "concat_mlp"}:
            raise ValueError("scorer must be 'bilinear' or 'concat_mlp'")
        self.scorer = scorer
        if scorer == "bilinear":
            self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, 1)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(
        self,
        patch_embedding: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return masked compatibility logits `[B,C]`."""
        if patch_embedding.ndim != 2:
            raise ValueError("patch_embedding must be [B,H]")
        if candidate_embeddings.ndim != 3:
            raise ValueError("candidate_embeddings must be [B,C,H]")
        if candidate_embeddings.shape[0] != patch_embedding.shape[0]:
            raise ValueError("patch and candidate batch dimensions differ")
        if candidate_mask.shape != candidate_embeddings.shape[:2]:
            raise ValueError("candidate_mask must match candidate_embeddings [B,C]")
        batch, candidates, _ = candidate_embeddings.shape
        patch_expanded = patch_embedding.unsqueeze(1).expand(batch, candidates, -1)
        if self.scorer == "bilinear":
            logits = self.bilinear(
                patch_expanded.reshape(batch * candidates, -1),
                candidate_embeddings.reshape(batch * candidates, -1),
            ).reshape(batch, candidates)
        else:
            pair = torch.cat(
                [patch_expanded, candidate_embeddings, patch_expanded * candidate_embeddings],
                dim=-1,
            )
            logits = self.mlp(pair).squeeze(-1)
        mask = candidate_mask.to(dtype=torch.bool, device=logits.device)
        return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
