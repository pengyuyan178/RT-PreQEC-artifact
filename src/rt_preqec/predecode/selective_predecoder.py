"""Selective neural predecoder orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from rt_preqec.data.layout import DetectorLayout
from rt_preqec.data.schemas import DetectorPatch, LocalErrorCandidate
from rt_preqec.predecode.confidence import sigmoid
from rt_preqec.predecode.validator import (
    batch_validate_local_corrections,
    select_best_candidate_for_patch,
)


@dataclass
class PredecodeResult:
    """Outputs of selective predecoding."""

    accepted_mask: np.ndarray
    local_corrections: np.ndarray
    confidence: np.ndarray
    risk: np.ndarray
    validation_pass: np.ndarray
    accepted_candidates: list[LocalErrorCandidate] | None = None
    patch_ids: np.ndarray | None = None
    center_detector_ids: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SelectivePredecoder:
    """Neural or heuristic selective predecoder."""

    def __init__(
        self,
        model: Any | None,
        confidence_threshold: float,
        risk_threshold: float,
        correction_threshold: float = 0.5,
        enable_validation: bool = True,
        enable_abstention: bool = True,
        device: str = "cpu",
        layout: DetectorLayout | None = None,
        candidates_by_detector: dict[int, list[LocalErrorCandidate]] | None = None,
        mode: str = "toy",
    ) -> None:
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.risk_threshold = risk_threshold
        self.correction_threshold = float(correction_threshold)
        self.enable_validation = enable_validation
        self.enable_abstention = enable_abstention
        self.device = device
        self.layout = layout
        self.candidates_by_detector = candidates_by_detector or {}
        self.mode = mode

    def accept_mask(
        self,
        confidence: np.ndarray,
        risk: np.ndarray,
        validation_pass: np.ndarray,
    ) -> np.ndarray:
        """Decide which local outputs are accepted."""
        accepted = confidence >= self.confidence_threshold
        accepted &= risk <= self.risk_threshold
        if self.enable_validation:
            accepted &= validation_pass
        if not self.enable_abstention:
            accepted = validation_pass if self.enable_validation else np.ones_like(accepted, dtype=bool)
        return accepted

    def _heuristic_outputs(self, patches: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        corrections = []
        confidence = []
        risk = []
        for patch in patches:
            last_slice = patch[-1]
            correction = last_slice.reshape(-1).astype(np.float32)
            density = float(last_slice.mean())
            corrections.append(correction)
            confidence.append(max(0.0, 1.0 - density))
            risk.append(min(1.0, density))
        return np.stack(corrections), np.asarray(confidence), np.asarray(risk)

    def _run_candidate_mode(self, patches: list[DetectorPatch]) -> PredecodeResult:
        accepted_candidates: list[LocalErrorCandidate] = []
        confidence: list[float] = []
        risk: list[float] = []
        validation_pass: list[bool] = []
        patch_ids: list[int] = []
        center_detector_ids: list[int] = []
        for patch in patches:
            result = select_best_candidate_for_patch(patch, self.candidates_by_detector)
            patch_ids.append(patch.patch_id)
            center_detector_ids.append(patch.center_detector_id)
            validation_pass.append(result.passed)
            overlap_ratio = float(result.metadata.get("overlap_ratio", 0.0))
            candidate = None
            if result.candidate_id is not None:
                for local_candidates in self.candidates_by_detector.values():
                    for local_candidate in local_candidates:
                        if local_candidate.candidate_id == result.candidate_id:
                            candidate = local_candidate
                            break
                    if candidate is not None:
                        break
            if candidate is not None:
                confidence_value = max(overlap_ratio, float(candidate.probability or 0.0))
                risk_value = 1.0 - confidence_value
                accepted_candidates.append(candidate)
            else:
                confidence_value = 0.0
                risk_value = 1.0
            confidence.append(float(np.clip(confidence_value, 0.0, 1.0)))
            risk.append(float(np.clip(risk_value, 0.0, 1.0)))

        confidence_array = np.asarray(confidence, dtype=float)
        risk_array = np.asarray(risk, dtype=float)
        validation_array = np.asarray(validation_pass, dtype=bool)
        accepted = self.accept_mask(confidence_array, risk_array, validation_array)
        accepted_candidates_filtered = [candidate for candidate, keep in zip(accepted_candidates, accepted) if keep]
        return PredecodeResult(
            accepted_mask=accepted,
            local_corrections=np.empty((len(patches), 0), dtype=np.float32),
            confidence=confidence_array,
            risk=risk_array,
            validation_pass=validation_array,
            accepted_candidates=accepted_candidates_filtered,
            patch_ids=np.asarray(patch_ids, dtype=np.int32),
            center_detector_ids=np.asarray(center_detector_ids, dtype=np.int32),
            metadata={"mode": "candidate", "model": "heuristic_candidate_selector" if self.model is None else type(self.model).__name__},
        )

    def _run_risk_only_mode(self, patches: list[DetectorPatch]) -> PredecodeResult:
        patch_ids = np.asarray([patch.patch_id for patch in patches], dtype=np.int32)
        center_detector_ids = np.asarray([patch.center_detector_id for patch in patches], dtype=np.int32)
        confidence = np.asarray([0.5 if len(patch.active_detector_ids) > 0 else 0.1 for patch in patches], dtype=float)
        risk = 1.0 - confidence
        validation_pass = np.ones(len(patches), dtype=bool)
        accepted = np.zeros(len(patches), dtype=bool)
        return PredecodeResult(
            accepted_mask=accepted,
            local_corrections=np.empty((len(patches), 0), dtype=np.float32),
            confidence=confidence,
            risk=risk,
            validation_pass=validation_pass,
            accepted_candidates=[],
            patch_ids=patch_ids,
            center_detector_ids=center_detector_ids,
            metadata={"mode": "risk_only"},
        )

    def run(self, patches: list[Any]) -> PredecodeResult:
        """Run predecoding over extracted patches."""
        if self.mode == "candidate":
            typed_patches = [patch for patch in patches if isinstance(patch, DetectorPatch)]
            if len(typed_patches) == 0:
                empty = np.asarray([], dtype=bool)
                return PredecodeResult(empty, np.empty((0, 0)), np.asarray([]), np.asarray([]), empty, accepted_candidates=[], metadata={"empty": True, "mode": "candidate"})
            return self._run_candidate_mode(typed_patches)

        if self.mode == "risk_only":
            typed_patches = [patch for patch in patches if isinstance(patch, DetectorPatch)]
            if len(typed_patches) == 0:
                empty = np.asarray([], dtype=bool)
                return PredecodeResult(empty, np.empty((0, 0)), np.asarray([]), np.asarray([]), empty, accepted_candidates=[], metadata={"empty": True, "mode": "risk_only"})
            return self._run_risk_only_mode(typed_patches)

        patch_arrays = [np.asarray(item["patch"], dtype=np.float32) for item in patches]
        if len(patch_arrays) == 0:
            empty = np.asarray([], dtype=bool)
            return PredecodeResult(empty, np.empty((0, 0)), np.asarray([]), np.asarray([]), empty, {"empty": True})

        if self.model is None or torch is None:
            local_corrections, confidence, risk = self._heuristic_outputs(patch_arrays)
            metadata = {"model": "heuristic", "mode": "toy"}
        else:
            self.model.eval()
            with torch.no_grad():
                batch = torch.tensor(np.stack(patch_arrays)[:, None, ...], dtype=torch.float32, device=self.device)
                outputs = self.model(batch)
                local_corrections = (
                    torch.sigmoid(outputs["correction_logits"]) >= self.correction_threshold
                ).float().cpu().numpy()
                confidence = sigmoid(outputs["confidence_logit"].cpu().numpy())
                risk = sigmoid(outputs["risk_logit"].cpu().numpy())
            metadata = {"model": type(self.model).__name__, "mode": "toy"}

        validation_pass = (
            batch_validate_local_corrections(np.stack(patch_arrays), local_corrections)
            if self.enable_validation
            else np.ones(len(patch_arrays), dtype=bool)
        )
        accepted = self.accept_mask(confidence, risk, validation_pass)
        return PredecodeResult(
            accepted_mask=accepted,
            local_corrections=local_corrections,
            confidence=confidence,
            risk=risk,
            validation_pass=validation_pass,
            accepted_candidates=None,
            patch_ids=None,
            center_detector_ids=None,
            metadata=metadata,
        )
