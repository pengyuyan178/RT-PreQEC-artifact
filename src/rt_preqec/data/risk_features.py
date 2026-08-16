"""Feature extraction utilities for the risk-only AI profiler."""

from __future__ import annotations

from typing import Any

import numpy as np

from rt_preqec.data.layout import DetectorLayout
from rt_preqec.data.schemas import DetectorPatch, LocalErrorCandidate


def _safe_mean(values: list[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    return float(array.mean()) if array.size else 0.0


def _safe_max(values: list[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    return float(array.max()) if array.size else 0.0


def _span(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.max() - finite.min()) if finite.size else 0.0


def _active_coord_stats(active_ids: np.ndarray, layout: DetectorLayout | None) -> tuple[list[float], list[str]]:
    names = [
        "layout_present",
        "num_active_time_bins",
        "active_time_span",
        "active_x_span",
        "active_y_span",
        "layout_missing_flag",
    ]
    if layout is None or active_ids.size == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 1.0 if layout is None else 0.0], names
    coords = layout.get_coord_array(active_ids.tolist())
    t_vals = coords[:, 0]
    x_vals = coords[:, 1]
    y_vals = coords[:, 2]
    finite_t = t_vals[np.isfinite(t_vals)]
    num_active_time_bins = float(len(np.unique(np.round(finite_t, 6)))) if finite_t.size else 0.0
    return [
        1.0,
        num_active_time_bins,
        _span(t_vals),
        _span(x_vals),
        _span(y_vals),
        0.0,
    ], names


def _candidate_stats(
    active_ids: np.ndarray,
    candidates_by_detector: dict[int, list[LocalErrorCandidate]] | None,
) -> tuple[list[float], list[str]]:
    names = [
        "mean_candidate_count_active",
        "max_candidate_count_active",
        "fraction_active_with_candidate",
        "candidates_missing_flag",
    ]
    if candidates_by_detector is None:
        return [0.0, 0.0, 0.0, 1.0], names
    if active_ids.size == 0:
        return [0.0, 0.0, 0.0, 0.0], names
    counts = [float(len(candidates_by_detector.get(int(detector_id), []))) for detector_id in active_ids.tolist()]
    with_candidate = [count > 0 for count in counts]
    fraction = float(np.mean(with_candidate)) if counts else 0.0
    return [_safe_mean(counts), _safe_max(counts), fraction, 0.0], names


def extract_syndrome_features(
    syndrome: np.ndarray,
    layout: DetectorLayout | None = None,
    candidates_by_detector: dict[int, list[LocalErrorCandidate]] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Extract fixed-length shot-level features from one syndrome."""
    bits = np.asarray(syndrome, dtype=np.float32).reshape(-1)
    num_detectors = float(bits.size)
    active_ids = np.flatnonzero(bits > 0)
    weight = float(active_ids.size)
    density = weight / num_detectors if num_detectors else 0.0
    base_values = [
        weight,
        density,
        num_detectors,
        1.0 if weight > 0 else 0.0,
        density,
    ]
    base_names = [
        "syndrome_weight",
        "syndrome_density",
        "num_detectors",
        "has_any_detection",
        "active_detector_fraction",
    ]
    coord_values, coord_names = _active_coord_stats(active_ids.astype(np.int32), layout)
    candidate_values, candidate_names = _candidate_stats(active_ids.astype(np.int32), candidates_by_detector)
    interaction_values = [weight * weight, float(np.log1p(weight))]
    interaction_names = ["weight_squared", "log1p_weight"]
    features = np.asarray(base_values + coord_values + candidate_values + interaction_values, dtype=np.float32)
    feature_names = base_names + coord_names + candidate_names + interaction_names
    return features, feature_names


def extract_patch_aggregate_features(patches: list[DetectorPatch]) -> tuple[np.ndarray, list[str]]:
    """Aggregate layout-aware detector patch features into a fixed-size vector."""
    names = [
        "num_patches",
        "mean_patch_active",
        "max_patch_active",
        "mean_patch_size",
        "max_patch_size",
        "mean_patch_density",
        "max_patch_density",
    ]
    if not patches:
        return np.zeros(len(names), dtype=np.float32), names
    active_counts = [float(len(patch.active_detector_ids)) for patch in patches]
    patch_sizes = [float(len(patch.detector_ids)) for patch in patches]
    densities = [
        (active / size) if size > 0 else 0.0
        for active, size in zip(active_counts, patch_sizes)
    ]
    values = [
        float(len(patches)),
        _safe_mean(active_counts),
        _safe_max(active_counts),
        _safe_mean(patch_sizes),
        _safe_max(patch_sizes),
        _safe_mean(densities),
        _safe_max(densities),
    ]
    return np.asarray(values, dtype=np.float32), names


def combine_feature_blocks(*blocks: tuple[np.ndarray, list[str]]) -> tuple[np.ndarray, list[str]]:
    """Concatenate feature blocks while preserving names."""
    arrays: list[np.ndarray] = []
    names: list[str] = []
    for values, block_names in blocks:
        array = np.asarray(values, dtype=np.float32).reshape(-1)
        arrays.append(array)
        names.extend(block_names)
    if not arrays:
        return np.asarray([], dtype=np.float32), []
    return np.concatenate(arrays).astype(np.float32), names


def extract_features_from_patch_metadata(patches: list[DetectorPatch], metadata: dict[str, Any] | None = None) -> tuple[np.ndarray, list[str]]:
    """Convenience wrapper for patch-only feature extraction."""
    del metadata
    return extract_patch_aggregate_features(patches)
