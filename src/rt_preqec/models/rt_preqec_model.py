"""Combined RT-PreQEC model wrapper."""

from __future__ import annotations

from typing import Any

from torch import nn

from rt_preqec.models.candidate_predecoder_model import CandidatePredecoderModel
from rt_preqec.models.outputs import RTPreQECOutput
from rt_preqec.models.risk_runtime_model import RiskRuntimeModel


class RTPreQECModel(nn.Module):
    """Unified wrapper for RT-PreQEC's two fallback-safe AI paths.

    Input: a batch dict containing risk features and/or detector patch candidate
    tensors.
    Output: `RTPreQECOutput` with risk-runtime and/or candidate-predecode
    outputs.
    RT-PreQEC role: exposes the Risk/Runtime Profiler and Selective Candidate
    Predecoder through a single interface for future joint experiments.
    Realtime fit: each path remains optional, bounded, selective, and safe to
    bypass when validation or scheduler policy rejects AI output.
    """

    def __init__(
        self,
        risk_runtime_model: RiskRuntimeModel | None,
        candidate_predecoder_model: CandidatePredecoderModel | None,
    ) -> None:
        super().__init__()
        self.risk_runtime_model = risk_runtime_model
        self.candidate_predecoder_model = candidate_predecoder_model

    def forward(self, batch: dict[str, Any]) -> RTPreQECOutput:
        """Run available RT-PreQEC paths over a structured batch dict."""
        risk_output = None
        candidate_output = None
        if self.risk_runtime_model is not None and "features" in batch:
            risk_output = self.risk_runtime_model(
                features=batch["features"],
                history_features=batch.get("history_features"),
            )
        if self.candidate_predecoder_model is not None and "detector_features" in batch:
            candidate_output = self.candidate_predecoder_model(
                detector_features=batch["detector_features"],
                detector_mask=batch["detector_mask"],
                candidate_features=batch["candidate_features"],
                candidate_mask=batch["candidate_mask"],
            )
        return RTPreQECOutput(
            risk_runtime=risk_output,
            candidate_predecode=candidate_output,
            metadata={"model_type": "rt_preqec_combined"},
        )
