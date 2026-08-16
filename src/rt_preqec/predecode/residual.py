"""Residual syndrome construction for toy and candidate-based paths."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from rt_preqec.data.schemas import LocalErrorCandidate


def apply_local_corrections_to_syndrome(
    original_syndrome: np.ndarray,
    corrections: Sequence[np.ndarray],
    locations: Sequence[tuple[int, int, int]],
) -> np.ndarray:
    """Apply local binary corrections to the final time slice around each location."""
    residual = np.asarray(original_syndrome).copy()
    if residual.ndim != 3:
        raise ValueError("Expected syndrome shape [T, H, W].")
    padded = residual.copy()
    for correction, (t, i, j) in zip(corrections, locations):
        corr = np.asarray(correction)
        side = int(np.sqrt(corr.size))
        half = side // 2
        patch = (corr.reshape(side, side) >= 0.5).astype(residual.dtype)
        padded_t = padded[t]
        padded_slice = np.pad(padded_t, ((half, half), (half, half)), mode="constant")
        padded_slice[i : i + side, j : j + side] ^= patch
        padded[t] = padded_slice[half:-half, half:-half]
    return padded


def apply_candidate_to_flat_syndrome(syndrome: np.ndarray, candidate: LocalErrorCandidate) -> np.ndarray:
    """Apply one local candidate to a flat detector syndrome."""
    residual = np.asarray(syndrome, dtype=np.int8).copy()
    if residual.ndim != 1:
        raise ValueError("Expected flat syndrome shape [num_detectors].")
    detector_ids = np.asarray(candidate.detector_ids, dtype=np.int32)
    if np.any(detector_ids < 0) or np.any(detector_ids >= residual.shape[0]):
        raise ValueError("Candidate detector id out of bounds for flat syndrome.")
    residual[detector_ids] ^= 1
    return residual


def apply_candidates_to_flat_syndrome(
    syndrome: np.ndarray,
    candidates: list[LocalErrorCandidate],
) -> np.ndarray:
    """Apply a list of local candidates to a flat detector syndrome."""
    residual = np.asarray(syndrome, dtype=np.int8).copy()
    for candidate in candidates:
        residual = apply_candidate_to_flat_syndrome(residual, candidate)
    return residual


def compute_flat_residual_stats(original: np.ndarray, residual: np.ndarray) -> dict[str, float]:
    """Compute weight and reduction statistics for flat residuals."""
    original_bits = np.asarray(original, dtype=np.int8)
    residual_bits = np.asarray(residual, dtype=np.int8)
    original_weight = float(original_bits.sum())
    residual_weight = float(residual_bits.sum())
    removed_weight = float(original_weight - residual_weight)
    return {
        "original_weight": original_weight,
        "residual_weight": residual_weight,
        "removed_weight": removed_weight,
        "reduction_ratio": removed_weight / max(original_weight, 1.0),
    }


def compute_residual_density(original_syndrome: np.ndarray, residual_syndrome: np.ndarray) -> float:
    """Compute density ratio of residual active syndrome."""
    original = np.asarray(original_syndrome)
    residual = np.asarray(residual_syndrome)
    return float(residual.sum() / max(original.sum(), 1))


def remove_accepted_clusters(
    original_syndrome: np.ndarray,
    corrections: Sequence[np.ndarray],
    locations: Sequence[tuple[int, int, int]],
    accepted_mask: np.ndarray,
) -> np.ndarray:
    """Apply only accepted corrections to build residual syndrome."""
    accepted_corrections = [corr for corr, keep in zip(corrections, accepted_mask) if keep]
    accepted_locations = [loc for loc, keep in zip(locations, accepted_mask) if keep]
    if not accepted_corrections:
        return np.asarray(original_syndrome).copy()
    return apply_local_corrections_to_syndrome(original_syndrome, accepted_corrections, accepted_locations)
