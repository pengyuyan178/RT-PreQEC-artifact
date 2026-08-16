"""Toy and candidate-based syndrome-consistency validation."""

from __future__ import annotations

from typing import Any

import numpy as np

from rt_preqec.data.schemas import CandidateValidationResult, DetectorPatch, LocalErrorCandidate


def build_placeholder_local_check_matrix(patch_size: int) -> np.ndarray:
    """Build a placeholder local check matrix.

    TODO: replace this with a local surface-code stabilizer incidence matrix
    derived from detector graph structure and code layout.
    """
    size = patch_size * patch_size
    return np.eye(size, dtype=np.int8)


def validate_local_correction(
    patch_syndrome: np.ndarray,
    local_correction: np.ndarray,
    local_check_matrix: np.ndarray | None = None,
) -> bool:
    """Validate a local correction against a placeholder parity rule."""
    patch = np.asarray(patch_syndrome)
    correction = np.asarray(local_correction).reshape(-1)
    matrix = local_check_matrix if local_check_matrix is not None else build_placeholder_local_check_matrix(patch.shape[-1])
    syndrome_vec = patch[-1].reshape(-1) % 2
    correction_vec = correction[: matrix.shape[1]] % 2
    predicted = (matrix @ correction_vec) % 2
    mismatch = np.abs(predicted - syndrome_vec[: matrix.shape[0]]).sum()
    return bool(mismatch <= max(1, syndrome_vec.sum()))


def batch_validate_local_corrections(
    patch_syndromes: np.ndarray,
    local_corrections: np.ndarray,
    local_check_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Validate a batch of local corrections."""
    return np.asarray(
        [
            validate_local_correction(patch_syndromes[idx], local_corrections[idx], local_check_matrix)
            for idx in range(len(patch_syndromes))
        ],
        dtype=bool,
    )


def validate_candidate_against_patch(
    patch: DetectorPatch,
    candidate: LocalErrorCandidate,
    allow_observable_flip: bool = False,
    require_exact_match: bool = False,
) -> CandidateValidationResult:
    """Validate one local DEM candidate against a detector patch."""
    active = set(int(value) for value in patch.active_detector_ids.tolist())
    cand = set(int(value) for value in candidate.detector_ids.tolist())
    matched = np.asarray(sorted(active & cand), dtype=np.int32)
    unmatched = np.asarray(sorted(cand - active), dtype=np.int32)
    touches_observable = len(candidate.observable_ids) > 0
    if touches_observable and not allow_observable_flip:
        return CandidateValidationResult(
            passed=False,
            candidate_id=candidate.candidate_id,
            reason="touches_observable",
            matched_detector_ids=matched,
            unmatched_detector_ids=unmatched,
            touches_observable=True,
            metadata={"active_count": len(active), "candidate_count": len(cand)},
        )
    if len(cand) == 0:
        return CandidateValidationResult(
            passed=False,
            candidate_id=candidate.candidate_id,
            reason="empty_candidate",
            matched_detector_ids=matched,
            unmatched_detector_ids=unmatched,
            touches_observable=touches_observable,
            metadata={},
        )
    if require_exact_match:
        passed = cand == active
        reason = "exact_match" if passed else "exact_mismatch"
    else:
        overlap_ratio = len(active & cand) / max(len(cand), 1)
        passed = cand.issubset(active) or overlap_ratio >= 0.5
        reason = "subset_or_overlap" if passed else "insufficient_overlap"
    return CandidateValidationResult(
        passed=passed,
        candidate_id=candidate.candidate_id,
        reason=reason,
        matched_detector_ids=matched,
        unmatched_detector_ids=unmatched,
        touches_observable=touches_observable,
        metadata={
            "active_count": len(active),
            "candidate_count": len(cand),
            "overlap_ratio": len(active & cand) / max(len(cand), 1),
        },
    )


def select_best_candidate_for_patch(
    patch: DetectorPatch,
    candidates_by_detector: dict[int, list[LocalErrorCandidate]],
    allow_observable_flip: bool = False,
    require_exact_match: bool = False,
) -> CandidateValidationResult:
    """Select the best local candidate for a patch."""
    unique_candidates: dict[int, LocalErrorCandidate] = {}
    for detector_id in patch.active_detector_ids.tolist():
        for candidate in candidates_by_detector.get(int(detector_id), []):
            unique_candidates[candidate.candidate_id] = candidate
    if not unique_candidates:
        return CandidateValidationResult(
            passed=False,
            candidate_id=None,
            reason="no_candidate",
            matched_detector_ids=np.asarray([], dtype=np.int32),
            unmatched_detector_ids=np.asarray([], dtype=np.int32),
            touches_observable=False,
            metadata={},
        )

    scored: list[tuple[float, CandidateValidationResult]] = []
    for candidate in unique_candidates.values():
        result = validate_candidate_against_patch(
            patch,
            candidate,
            allow_observable_flip=allow_observable_flip,
            require_exact_match=require_exact_match,
        )
        overlap = float(result.metadata.get("overlap_ratio", 0.0))
        probability_bonus = float(candidate.probability or 0.0)
        weight_penalty = float(candidate.weight or 0.0)
        observable_penalty = 1.0 if result.touches_observable else 0.0
        score = overlap * 10.0 + probability_bonus - 0.1 * len(candidate.detector_ids) - weight_penalty - observable_penalty * 100.0
        if result.passed:
            score += 100.0
        scored.append((score, result))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def batch_validate_candidates(
    patches: list[DetectorPatch],
    candidates_by_detector: dict[int, list[LocalErrorCandidate]],
    allow_observable_flip: bool = False,
    require_exact_match: bool = False,
) -> list[CandidateValidationResult]:
    """Validate best candidates over a batch of patches."""
    return [
        select_best_candidate_for_patch(
            patch,
            candidates_by_detector,
            allow_observable_flip=allow_observable_flip,
            require_exact_match=require_exact_match,
        )
        for patch in patches
    ]
