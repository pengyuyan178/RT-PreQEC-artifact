"""Selective DEM-candidate predecoder for RT-PreQEC."""

from __future__ import annotations

import torch
from torch import nn

from rt_preqec.models.encoders import DEMCandidateEncoder, DetectorPatchEncoder
from rt_preqec.models.heads import AbstainHead, CandidateHead, ConfidenceHead, RiskHead
from rt_preqec.models.outputs import CandidatePredecoderOutput


class CandidatePredecoderModel(nn.Module):
    """Selective neural predecoder over DetectorPatch and DEM candidates.

    Input: detector patch features `[B,P,D]`, detector mask `[B,P]`, DEM local
    candidate features `[B,C,K]`, and candidate mask `[B,C]`.
    Output: `CandidatePredecoderOutput` with candidate logits, abstain,
    confidence, and local risk logits.
    RT-PreQEC role: chooses a local DEM candidate or abstains. It is not a full
    decoder and never generates arbitrary global corrections.
    Realtime fit: bounded detector/candidate sets, masked candidate scoring,
    explicit abstention, and downstream validation make every AI output
    fallback-safe.
    """

    def __init__(
        self,
        detector_feature_dim: int,
        candidate_feature_dim: int,
        hidden_dim: int = 64,
        pooling: str = "mean_max",
        scorer: str = "bilinear",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.detector_feature_dim = int(detector_feature_dim)
        self.candidate_feature_dim = int(candidate_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.detector_encoder = DetectorPatchEncoder(
            detector_feature_dim=detector_feature_dim,
            hidden_dim=hidden_dim,
            pooling=pooling,
        )
        self.candidate_encoder = DEMCandidateEncoder(
            candidate_feature_dim=candidate_feature_dim,
            hidden_dim=hidden_dim,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.candidate_head = CandidateHead(hidden_dim=hidden_dim, scorer=scorer)
        self.confidence_head = ConfidenceHead(hidden_dim)
        self.risk_head = RiskHead(hidden_dim)
        self.abstain_head = AbstainHead(hidden_dim)

    def forward(
        self,
        detector_features: torch.Tensor,
        detector_mask: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> CandidatePredecoderOutput:
        """Rank local DEM candidates for each patch and expose abstention."""
        if detector_features.ndim != 3:
            raise ValueError("detector_features must be [B,P,D]")
        if candidate_features.ndim != 3:
            raise ValueError("candidate_features must be [B,C,K]")
        if detector_features.shape[0] != candidate_features.shape[0]:
            raise ValueError("detector and candidate batches must match")
        if detector_features.shape[-1] != self.detector_feature_dim:
            raise ValueError("detector feature dimension mismatch")
        if candidate_features.shape[-1] != self.candidate_feature_dim:
            raise ValueError("candidate feature dimension mismatch")
        patch_embedding = self.dropout(self.detector_encoder(detector_features, detector_mask))
        candidate_embeddings = self.dropout(self.candidate_encoder(candidate_features, candidate_mask))
        candidate_logits = self.candidate_head(patch_embedding, candidate_embeddings, candidate_mask)
        return CandidatePredecoderOutput(
            candidate_logits=candidate_logits,
            abstain_logit=self.abstain_head(patch_embedding),
            confidence_logit=self.confidence_head(patch_embedding),
            risk_logit=self.risk_head(patch_embedding),
            candidate_mask=candidate_mask,
            embeddings={
                "patch_embedding": patch_embedding,
                "candidate_embeddings": candidate_embeddings,
            },
            metadata={
                "model_type": "candidate_predecoder",
                "selective": True,
                "requires_validation": True,
            },
        )

    def count_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
