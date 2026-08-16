"""Local syndrome patch extraction for toy and layout-aware paths."""

from __future__ import annotations

from typing import Any

import numpy as np

from rt_preqec.data.layout import DetectorLayout, nearest_detectors
from rt_preqec.data.schemas import DetectorPatch


def _check_syndrome_shape(syndrome: np.ndarray) -> np.ndarray:
    array = np.asarray(syndrome)
    if array.ndim != 3:
        raise ValueError("Expected syndrome shape [T, H, W]. Flat detector vectors are TODO.")
    return array


def extract_local_patches(
    syndrome: np.ndarray,
    patch_size: int,
    temporal_window: int,
) -> list[dict[str, Any]]:
    """Extract centered local patches from a syndrome tensor shaped [T, H, W]."""
    array = _check_syndrome_shape(syndrome)
    t_size, height, width = array.shape
    half = patch_size // 2
    patches: list[dict[str, Any]] = []
    if temporal_window > t_size:
        return patches
    padded = np.pad(array, ((0, 0), (half, half), (half, half)), mode="constant")
    for t in range(temporal_window - 1, t_size):
        for i in range(height):
            for j in range(width):
                patch = padded[
                    t - temporal_window + 1 : t + 1,
                    i : i + patch_size,
                    j : j + patch_size,
                ]
                patches.append(
                    {
                        "patch": patch.astype(np.float32),
                        "location": (t, i, j),
                        "features": estimate_cluster_features(patch),
                    }
                )
    return patches


def extract_detector_patches_from_flat_syndrome(
    syndrome: np.ndarray,
    layout: DetectorLayout,
    patch_radius: float,
    time_radius: float | None = None,
    active_only: bool = True,
    max_patches: int | None = None,
    shot_id: int | None = None,
) -> list[DetectorPatch]:
    """Extract layout-aware patches from a flat detector syndrome vector."""
    bits = np.asarray(syndrome, dtype=np.int8)
    if bits.ndim != 1:
        raise ValueError("Expected flat syndrome shape [num_detectors].")
    active_ids = np.flatnonzero(bits > 0)
    centers = active_ids.tolist() if active_only and len(active_ids) > 0 else [coord.detector_id for coord in layout.coords]
    patches: list[DetectorPatch] = []
    for patch_id, center_detector_id in enumerate(centers):
        detector_ids = nearest_detectors(layout, int(center_detector_id), patch_radius, time_radius=time_radius)
        detector_ids_array = np.asarray(sorted(detector_ids), dtype=np.int32)
        syndrome_bits = bits[detector_ids_array]
        active_detector_ids = detector_ids_array[syndrome_bits == 1]
        center_coord = layout.get_coord(int(center_detector_id))
        patches.append(
            DetectorPatch(
                patch_id=patch_id,
                shot_id=shot_id,
                center_detector_id=int(center_detector_id),
                detector_ids=detector_ids_array,
                detector_coords=layout.get_coord_array(detector_ids_array.tolist()),
                syndrome_bits=syndrome_bits.astype(np.int8),
                active_detector_ids=active_detector_ids.astype(np.int32),
                metadata={
                    "center_coord": center_coord.raw_coord,
                    "time_radius": time_radius,
                    "spatial_radius": patch_radius,
                },
            )
        )
        if max_patches is not None and len(patches) >= max_patches:
            break
    return patches


def extract_detector_patches_batch(
    syndromes: np.ndarray,
    layout: DetectorLayout,
    patch_radius: float,
    time_radius: float | None = None,
    active_only: bool = True,
    max_patches: int | None = None,
) -> list[DetectorPatch]:
    """Extract layout-aware patches from a batch of flat detector syndromes."""
    array = np.asarray(syndromes, dtype=np.int8)
    if array.ndim != 2:
        raise ValueError("Expected batched flat syndromes shape [num_shots, num_detectors].")
    patches: list[DetectorPatch] = []
    for shot_id, syndrome in enumerate(array):
        shot_patches = extract_detector_patches_from_flat_syndrome(
            syndrome,
            layout,
            patch_radius=patch_radius,
            time_radius=time_radius,
            active_only=active_only,
            max_patches=max_patches,
            shot_id=shot_id,
        )
        patches.extend(shot_patches)
    return patches


def estimate_cluster_features(patch: np.ndarray | DetectorPatch) -> dict[str, float]:
    """Compute simple local cluster features for toy or layout-aware patches."""
    if isinstance(patch, DetectorPatch):
        coords = np.asarray(patch.detector_coords, dtype=float)
        active = float(len(patch.active_detector_ids))
        patch_size_num_detectors = float(len(patch.detector_ids))
        density = active / patch_size_num_detectors if patch_size_num_detectors else 0.0
        center_is_active = float(int(patch.center_detector_id in set(patch.active_detector_ids.tolist())))

        def _span(column: int) -> float:
            if coords.size == 0 or column >= coords.shape[1]:
                return 0.0
            values = coords[:, column]
            finite = values[np.isfinite(values)]
            return float(finite.max() - finite.min()) if len(finite) else 0.0

        return {
            "num_active": active,
            "active_density": density,
            "coord_span_time": _span(0),
            "coord_span_x": _span(1),
            "coord_span_y": _span(2),
            "patch_size_num_detectors": patch_size_num_detectors,
            "center_is_active": center_is_active,
            "touches_boundary": 0.0,
        }

    array = np.asarray(patch)
    active = float(array.sum())
    volume = float(array.size)
    density = active / volume if volume else 0.0
    center_slice = array[-1]
    edge_sum = float(
        center_slice[0, :].sum()
        + center_slice[-1, :].sum()
        + center_slice[:, 0].sum()
        + center_slice[:, -1].sum()
    )
    return {"active": active, "density": density, "edge_activity": edge_sum}


def is_candidate_easy_cluster(
    patch: np.ndarray | DetectorPatch,
    max_cluster_size: int,
    min_boundary_distance: int,
) -> bool:
    """Heuristic filter for small and interior clusters."""
    if isinstance(patch, DetectorPatch):
        features = estimate_cluster_features(patch)
        return (
            features["num_active"] > 0
            and features["num_active"] <= max_cluster_size
            and features["coord_span_x"] <= max(max_cluster_size, 1)
            and features["coord_span_y"] <= max(max_cluster_size, 1)
        )

    array = np.asarray(patch)
    if array.ndim != 3:
        return False
    current = array[-1]
    active_coords = np.argwhere(current > 0)
    if len(active_coords) == 0 or len(active_coords) > max_cluster_size:
        return False
    height, width = current.shape
    boundary_distances = [
        min(i, j, height - 1 - i, width - 1 - j)
        for i, j in active_coords
    ]
    return min(boundary_distances) >= min_boundary_distance
