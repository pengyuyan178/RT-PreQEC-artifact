"""Detector error model parser scaffold for local candidates."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from rt_preqec.data.layout import DetectorLayout
from rt_preqec.data.schemas import LocalErrorCandidate

logger = logging.getLogger(__name__)


def _safe_probability(instruction: Any) -> float | None:
    try:
        args = instruction.args_copy()
        if len(args) > 0:
            return float(args[0])
    except Exception:  # pragma: no cover
        pass
    return None


def _target_lists(targets: list[Any]) -> tuple[list[int], list[int]]:
    detector_ids: list[int] = []
    observable_ids: list[int] = []
    for target in targets:
        try:
            if target.is_relative_detector_id():
                detector_ids.append(int(target.relative_detector_id()))
            elif target.is_logical_observable_id():
                observable_ids.append(int(target.logical_observable_id()))
        except Exception:  # pragma: no cover
            text = str(target)
            if text.startswith("D"):
                detector_ids.append(int(text[1:]))
            elif text.startswith("L"):
                observable_ids.append(int(text[1:]))
    return detector_ids, observable_ids


def _coord_span(detector_ids: list[int], layout: DetectorLayout | None) -> dict[str, Any]:
    if layout is None or len(detector_ids) == 0:
        return {}
    coords = layout.get_coord_array(detector_ids)
    t_vals = coords[:, 0]
    x_vals = coords[:, 1]
    y_vals = coords[:, 2]

    def _minmax(values: np.ndarray) -> tuple[float | None, float | None]:
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            return None, None
        return float(np.min(finite)), float(np.max(finite))

    t_min, t_max = _minmax(t_vals)
    x_min, x_max = _minmax(x_vals)
    y_min, y_max = _minmax(y_vals)
    spatial_points = coords[:, 1:3]
    finite_points = spatial_points[np.all(np.isfinite(spatial_points), axis=1)]
    spatial_diameter = 0.0
    if len(finite_points) >= 2:
        diffs = finite_points[:, None, :] - finite_points[None, :, :]
        spatial_diameter = float(np.max(np.sqrt(np.sum(diffs * diffs, axis=-1))))
    time_diameter = 0.0 if t_min is None or t_max is None else float(t_max - t_min)
    return {
        "min_time": t_min,
        "max_time": t_max,
        "min_x": x_min,
        "max_x": x_max,
        "min_y": y_min,
        "max_y": y_max,
        "spatial_diameter": spatial_diameter,
        "time_diameter": time_diameter,
    }


def parse_dem_error_candidates(
    dem: Any,
    layout: DetectorLayout | None = None,
    max_detectors_per_candidate: int = 4,
) -> list[LocalErrorCandidate]:
    """Parse local candidates from a Stim detector error model."""
    if dem is None:
        return []
    candidates: list[LocalErrorCandidate] = []
    try:
        instructions = list(dem)
    except Exception as exc:  # pragma: no cover
        logger.warning("failed to iterate DEM: %s", exc)
        return []

    for instruction in instructions:
        try:
            inst_type = getattr(instruction, "type", None)
        except Exception:  # pragma: no cover
            inst_type = None
        if inst_type != "error":
            continue
        try:
            targets = list(instruction.targets_copy())
        except Exception as exc:  # pragma: no cover
            logger.warning("failed to parse DEM targets: %s", exc)
            continue
        detector_ids, observable_ids = _target_lists(targets)
        probability = _safe_probability(instruction)
        nonlocal_candidate = len(detector_ids) > max_detectors_per_candidate
        candidate = LocalErrorCandidate(
            candidate_id=len(candidates),
            detector_ids=np.asarray(detector_ids, dtype=np.int32),
            observable_ids=np.asarray(observable_ids, dtype=np.int32),
            probability=probability,
            weight=None if probability is None or probability <= 0 else float(-np.log(probability)),
            coord_span=_coord_span(detector_ids, layout),
            metadata={"nonlocal": nonlocal_candidate, "instruction_tag": getattr(instruction, "tag", None)},
        )
        candidates.append(candidate)
    return candidates


def filter_local_candidates(
    candidates: list[LocalErrorCandidate],
    max_spatial_diameter: float,
    max_time_diameter: float,
    allow_observable_flip: bool = False,
) -> list[LocalErrorCandidate]:
    """Filter candidates by locality and observable interaction."""
    filtered: list[LocalErrorCandidate] = []
    for candidate in candidates:
        if not allow_observable_flip and len(candidate.observable_ids) > 0:
            continue
        if candidate.metadata.get("nonlocal", False):
            continue
        spatial = float(candidate.coord_span.get("spatial_diameter", 0.0))
        time_diameter = float(candidate.coord_span.get("time_diameter", 0.0))
        if spatial <= max_spatial_diameter and time_diameter <= max_time_diameter:
            filtered.append(candidate)
    return filtered


def index_candidates_by_detector(candidates: list[LocalErrorCandidate]) -> dict[int, list[LocalErrorCandidate]]:
    """Index candidates by participating detector id."""
    indexed: dict[int, list[LocalErrorCandidate]] = {}
    for candidate in candidates:
        for detector_id in candidate.detector_ids.tolist():
            indexed.setdefault(int(detector_id), []).append(candidate)
    return indexed


def candidate_to_syndrome_mask(candidate: LocalErrorCandidate, num_detectors: int) -> np.ndarray:
    """Convert a local candidate into a binary syndrome XOR mask."""
    mask = np.zeros(num_detectors, dtype=np.int8)
    valid_ids = candidate.detector_ids[(candidate.detector_ids >= 0) & (candidate.detector_ids < num_detectors)]
    mask[valid_ids] = 1
    return mask
