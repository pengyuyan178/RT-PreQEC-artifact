"""Typed outputs for RT-PreQEC model components."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


def _squeeze_last(value: torch.Tensor) -> torch.Tensor:
    if value.ndim > 1 and value.shape[-1] == 1:
        return value.squeeze(-1)
    return value


def _prob_to_logit(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    value = value.clamp(float(eps), 1.0 - float(eps))
    return torch.log(value / (1.0 - value))


@dataclass
class RiskRuntimeOutput:
    """Output of the RT-PreQEC Risk/Runtime Profiler.

    The profiler does not decode. It predicts risk, runtime hardness,
    runtime cost, and model confidence so the lag-bounded scheduler can
    choose between fast and accurate backend decoders.
    """

    risk_logit: torch.Tensor
    hard_runtime_logit: torch.Tensor
    runtime_pred: torch.Tensor
    confidence_logit: torch.Tensor
    embeddings: dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def risk_score(self) -> torch.Tensor:
        """Return `sigmoid(risk_logit)` as the fast-decoder risk score."""
        return torch.sigmoid(_squeeze_last(self.risk_logit))

    def hard_runtime_score(self) -> torch.Tensor:
        """Return `sigmoid(hard_runtime_logit)` as accurate-runtime tail risk."""
        return torch.sigmoid(_squeeze_last(self.hard_runtime_logit))

    def confidence_score(self) -> torch.Tensor:
        """Return `sigmoid(confidence_logit)` as trust in model outputs."""
        return torch.sigmoid(_squeeze_last(self.confidence_logit))

    def to_dict(self) -> dict:
        """Return a dict compatible with legacy risk-profiler code."""
        risk_logit = _squeeze_last(self.risk_logit)
        hard_runtime_logit = _squeeze_last(self.hard_runtime_logit)
        confidence_logit = _squeeze_last(self.confidence_logit)
        runtime_pred = _squeeze_last(self.runtime_pred)
        return {
            "risk_logit": risk_logit,
            "hard_runtime_logit": hard_runtime_logit,
            "runtime_logit": hard_runtime_logit,
            "runtime_pred": runtime_pred,
            "confidence_logit": confidence_logit,
            "risk_score": torch.sigmoid(risk_logit),
            "hard_runtime_score": torch.sigmoid(hard_runtime_logit),
            "runtime_score": torch.sigmoid(hard_runtime_logit),
            "confidence": torch.sigmoid(confidence_logit),
            "embeddings": self.embeddings,
            "metadata": self.metadata,
        }


@dataclass
class DecomposedRiskOutput:
    """Output of the decomposed risk/runtime profiler.

    The decomposed profiler keeps the scheduler-facing legacy `risk_logit`
    contract, but derives that score from explicit task heads instead of a
    single coarse risk head.
    """

    risk_logit: torch.Tensor
    confidence_logit: torch.Tensor
    runtime_pred: torch.Tensor
    fast_wrong_logit: torch.Tensor
    fast_logical_fail_logit: torch.Tensor
    hard_runtime_logit: torch.Tensor
    syndrome_tail_logit: torch.Tensor
    safe_fast_logit: torch.Tensor
    embeddings: dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    combination_weights: dict[str, float] = field(
        default_factory=lambda: {
            "fast_wrong": 1.0,
            "fast_logical_fail": 1.0,
            "hard_runtime": 0.5,
            "syndrome_tail": 0.2,
        }
    )

    def fast_wrong_score(self) -> torch.Tensor:
        return torch.sigmoid(_squeeze_last(self.fast_wrong_logit))

    def fast_logical_fail_score(self) -> torch.Tensor:
        return torch.sigmoid(_squeeze_last(self.fast_logical_fail_logit))

    def hard_runtime_score(self) -> torch.Tensor:
        return torch.sigmoid(_squeeze_last(self.hard_runtime_logit))

    def syndrome_tail_score(self) -> torch.Tensor:
        return torch.sigmoid(_squeeze_last(self.syndrome_tail_logit))

    def safe_fast_score(self) -> torch.Tensor:
        return torch.sigmoid(_squeeze_last(self.safe_fast_logit))

    def confidence_score(self) -> torch.Tensor:
        return torch.sigmoid(_squeeze_last(self.confidence_logit))

    def combined_fast_risk(self) -> torch.Tensor:
        return torch.maximum(self.fast_wrong_score(), self.fast_logical_fail_score())

    def combined_scheduler_risk(self) -> torch.Tensor:
        weights = self.combination_weights or {}
        fast_wrong_weight = float(weights.get("fast_wrong", 1.0))
        fast_fail_weight = float(weights.get("fast_logical_fail", 1.0))
        hard_weight = float(weights.get("hard_runtime", 0.5))
        tail_weight = float(weights.get("syndrome_tail", 0.2))
        total_weight = max(fast_wrong_weight + fast_fail_weight + hard_weight + tail_weight, 1e-6)
        return (
            fast_wrong_weight * self.fast_wrong_score()
            + fast_fail_weight * self.fast_logical_fail_score()
            + hard_weight * self.hard_runtime_score()
            + tail_weight * self.syndrome_tail_score()
        ) / total_weight

    def risk_score(self) -> torch.Tensor:
        return self.combined_scheduler_risk()

    def to_dict(self) -> dict:
        """Return decomposed heads plus legacy risk/runtime keys."""
        fast_wrong_logit = _squeeze_last(self.fast_wrong_logit)
        fast_logical_fail_logit = _squeeze_last(self.fast_logical_fail_logit)
        hard_runtime_logit = _squeeze_last(self.hard_runtime_logit)
        syndrome_tail_logit = _squeeze_last(self.syndrome_tail_logit)
        safe_fast_logit = _squeeze_last(self.safe_fast_logit)
        confidence_logit = _squeeze_last(self.confidence_logit)
        runtime_pred = _squeeze_last(self.runtime_pred)
        combined_fast_risk = self.combined_fast_risk()
        combined_scheduler_risk = self.combined_scheduler_risk()
        risk_logit = _prob_to_logit(combined_scheduler_risk)
        return {
            "risk_logit": risk_logit,
            "confidence_logit": confidence_logit,
            "runtime_pred": runtime_pred,
            "fast_wrong_logit": fast_wrong_logit,
            "fast_logical_fail_logit": fast_logical_fail_logit,
            "hard_runtime_logit": hard_runtime_logit,
            "runtime_logit": hard_runtime_logit,
            "syndrome_tail_logit": syndrome_tail_logit,
            "safe_fast_logit": safe_fast_logit,
            "fast_wrong_prob": torch.sigmoid(fast_wrong_logit),
            "fast_logical_fail_prob": torch.sigmoid(fast_logical_fail_logit),
            "hard_runtime_prob": torch.sigmoid(hard_runtime_logit),
            "hard_runtime_score": torch.sigmoid(hard_runtime_logit),
            "runtime_score": torch.sigmoid(hard_runtime_logit),
            "syndrome_tail_prob": torch.sigmoid(syndrome_tail_logit),
            "safe_fast_prob": torch.sigmoid(safe_fast_logit),
            "combined_fast_risk": combined_fast_risk,
            "combined_scheduler_risk": combined_scheduler_risk,
            "risk_score": combined_scheduler_risk,
            "confidence": torch.sigmoid(confidence_logit),
            "embeddings": self.embeddings,
            "metadata": {
                **self.metadata,
                "combination_weights": dict(self.combination_weights or {}),
            },
        }


@dataclass
class CandidatePredecoderOutput:
    """Output of the selective DEM-candidate predecoder.

    Candidate logits rank local DEM candidates only. Abstention, risk, and
    confidence are explicit outputs because any accepted candidate must still
    pass validation and can always fall back to the backend decoder.
    """

    candidate_logits: torch.Tensor
    abstain_logit: torch.Tensor
    confidence_logit: torch.Tensor
    risk_logit: torch.Tensor
    candidate_mask: torch.Tensor | None = None
    embeddings: dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def _masked_candidate_logits(self) -> torch.Tensor:
        logits = self.candidate_logits
        if self.candidate_mask is None:
            return logits
        mask = self.candidate_mask.to(dtype=torch.bool, device=logits.device)
        return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)

    def candidate_probs(self) -> torch.Tensor:
        """Return softmax probabilities over valid DEM local candidates."""
        return F.softmax(self._masked_candidate_logits(), dim=-1)

    def abstain_prob(self) -> torch.Tensor:
        """Return `sigmoid(abstain_logit)` for selective fallback gating."""
        return torch.sigmoid(_squeeze_last(self.abstain_logit))

    def confidence_score(self) -> torch.Tensor:
        """Return trust in candidate ranking, separate from low-risk scoring."""
        return torch.sigmoid(_squeeze_last(self.confidence_logit))

    def risk_score(self) -> torch.Tensor:
        """Return local patch risk score used for candidate acceptance gating."""
        return torch.sigmoid(_squeeze_last(self.risk_logit))

    def selected_candidate(self) -> torch.Tensor:
        """Return the best valid candidate index; invalid-only rows return -1."""
        masked_logits = self._masked_candidate_logits()
        selected = torch.argmax(masked_logits, dim=-1)
        if self.candidate_mask is None:
            return selected
        has_candidate = self.candidate_mask.to(dtype=torch.bool, device=masked_logits.device).any(dim=-1)
        return torch.where(has_candidate, selected, torch.full_like(selected, -1))

    def to_dict(self) -> dict:
        """Return a dict with explicit candidate, abstain, risk, and confidence fields."""
        return {
            "candidate_logits": self._masked_candidate_logits(),
            "abstain_logit": _squeeze_last(self.abstain_logit),
            "confidence_logit": _squeeze_last(self.confidence_logit),
            "risk_logit": _squeeze_last(self.risk_logit),
            "candidate_mask": self.candidate_mask,
            "candidate_probs": self.candidate_probs(),
            "abstain_prob": self.abstain_prob(),
            "confidence": self.confidence_score(),
            "risk_score": self.risk_score(),
            "selected_candidate": self.selected_candidate(),
            "embeddings": self.embeddings,
            "metadata": self.metadata,
        }


@dataclass
class RTPreQECOutput:
    """Combined output container for the two RT-PreQEC AI paths."""

    risk_runtime: RiskRuntimeOutput | None
    candidate_predecode: CandidatePredecoderOutput | None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return nested outputs while preserving risk-only legacy keys."""
        payload: dict = {"metadata": self.metadata}
        if self.risk_runtime is not None:
            risk_dict = self.risk_runtime.to_dict()
            payload["risk_runtime"] = risk_dict
            payload.update({key: value for key, value in risk_dict.items() if key != "embeddings"})
        if self.candidate_predecode is not None:
            payload["candidate_predecode"] = self.candidate_predecode.to_dict()
        return payload
