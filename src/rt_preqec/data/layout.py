"""Detector layout scaffold built from Stim detector coordinates."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from rt_preqec.utils import ensure_parent


def _normalize_raw_coord(values: Sequence[float] | None) -> list[float]:
    if values is None:
        return []
    return [float(value) for value in values]


def _infer_coord_views(raw_coord: list[float], semantics: str = "auto") -> dict[str, float | None]:
    if semantics not in {"auto", "stim_default", "txy", "xyt"}:
        semantics = "auto"
    if len(raw_coord) == 0:
        return {"inferred_time": None, "inferred_x": None, "inferred_y": None, "time_first": None, "time_last": None}
    if len(raw_coord) == 1:
        value = raw_coord[0]
        return {"inferred_time": value, "inferred_x": value, "inferred_y": None, "time_first": value, "time_last": value}
    if len(raw_coord) == 2:
        x, y = raw_coord[0], raw_coord[1]
        return {"inferred_time": None, "inferred_x": x, "inferred_y": y, "time_first": x, "time_last": y}
    if semantics == "txy":
        inferred_time, inferred_x, inferred_y = raw_coord[0], raw_coord[1], raw_coord[2]
    elif semantics == "xyt":
        inferred_x, inferred_y, inferred_time = raw_coord[0], raw_coord[1], raw_coord[2]
    else:
        inferred_x, inferred_y, inferred_time = raw_coord[0], raw_coord[1], raw_coord[-1]
    return {
        "inferred_time": inferred_time,
        "inferred_x": inferred_x,
        "inferred_y": inferred_y,
        "time_first": raw_coord[0],
        "time_last": raw_coord[-1],
    }


@dataclass
class DetectorCoord:
    """Coordinate record for one detector."""

    detector_id: int
    raw_coord: list[float] = field(default_factory=list)
    inferred_time: float | None = None
    inferred_x: float | None = None
    inferred_y: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def coord_0(self) -> float | None:
        return self.raw_coord[0] if len(self.raw_coord) >= 1 else None

    @property
    def coord_1(self) -> float | None:
        return self.raw_coord[1] if len(self.raw_coord) >= 2 else None

    @property
    def coord_2(self) -> float | None:
        return self.raw_coord[2] if len(self.raw_coord) >= 3 else None


@dataclass
class DetectorLayout:
    """Detector layout scaffold with ID-to-coordinate mapping."""

    coords: list[DetectorCoord]
    metadata: dict[str, Any] = field(default_factory=dict)
    _spatial_index: np.ndarray | None = None

    @property
    def detector_id_to_coord(self) -> dict[int, DetectorCoord]:
        return {coord.detector_id: coord for coord in self.coords}

    def get_coord(self, detector_id: int) -> DetectorCoord:
        return self.detector_id_to_coord[detector_id]

    def get_coord_array(self, detector_ids: Sequence[int]) -> np.ndarray:
        rows: list[list[float]] = []
        for detector_id in detector_ids:
            coord = self.get_coord(int(detector_id))
            rows.append(
                [
                    np.nan if coord.inferred_time is None else coord.inferred_time,
                    np.nan if coord.inferred_x is None else coord.inferred_x,
                    np.nan if coord.inferred_y is None else coord.inferred_y,
                ]
            )
        return np.asarray(rows, dtype=float)


def group_detectors_by_time(layout: DetectorLayout, round_ndigits: int = 6) -> dict[float | None, list[int]]:
    """Group detectors by inferred time."""
    grouped: dict[float | None, list[int]] = {}
    for coord in layout.coords:
        key = None if coord.inferred_time is None else round(coord.inferred_time, round_ndigits)
        grouped.setdefault(key, []).append(coord.detector_id)
    return grouped


def build_spatial_index(layout: DetectorLayout) -> DetectorLayout:
    """Build a simple dense spatial index using inferred coordinates."""
    layout._spatial_index = layout.get_coord_array([coord.detector_id for coord in layout.coords])
    return layout


def nearest_detectors(
    layout: DetectorLayout,
    center_detector_id: int,
    spatial_radius: float,
    time_radius: float | None = None,
) -> list[int]:
    """Return nearby detectors using brute-force coordinate distance."""
    center = layout.get_coord(center_detector_id)
    center_t = center.inferred_time
    center_x = center.inferred_x
    center_y = center.inferred_y
    nearby: list[int] = []
    for coord in layout.coords:
        if time_radius is not None and center_t is not None and coord.inferred_time is not None:
            if abs(coord.inferred_time - center_t) > time_radius:
                continue
        dx = 0.0 if center_x is None or coord.inferred_x is None else coord.inferred_x - center_x
        dy = 0.0 if center_y is None or coord.inferred_y is None else coord.inferred_y - center_y
        spatial_distance = float(np.sqrt(dx * dx + dy * dy))
        if spatial_distance <= spatial_radius:
            nearby.append(coord.detector_id)
    return nearby


def build_detector_layout_from_dem(dem: Any, coordinate_semantics: str = "auto") -> DetectorLayout | None:
    """Build detector layout from a detector error model when coordinates exist."""
    if dem is None or not hasattr(dem, "get_detector_coordinates"):
        return None
    try:
        coord_map = dem.get_detector_coordinates()
    except Exception:  # pragma: no cover
        return None
    coords: list[DetectorCoord] = []
    for detector_id, values in coord_map.items():
        raw_coord = _normalize_raw_coord(values)
        inferred = _infer_coord_views(raw_coord, semantics=coordinate_semantics)
        coords.append(
            DetectorCoord(
                detector_id=int(detector_id),
                raw_coord=raw_coord,
                inferred_time=inferred["inferred_time"],
                inferred_x=inferred["inferred_x"],
                inferred_y=inferred["inferred_y"],
                metadata={"time_first": inferred["time_first"], "time_last": inferred["time_last"]},
            )
        )
    layout = DetectorLayout(
        coords=coords,
        metadata={
            "num_detectors": len(coords),
            "source": "dem",
            "coordinate_semantics": coordinate_semantics,
            "note": "Auto-inferred time/x/y should be checked against the specific Stim circuit layout.",
        },
    )
    return build_spatial_index(layout)


def detector_layout_to_dataframe(layout: DetectorLayout) -> pd.DataFrame:
    """Convert layout into a tabular DataFrame."""
    return pd.DataFrame(
        [
            {
                "detector_id": coord.detector_id,
                "coord_0": coord.coord_0,
                "coord_1": coord.coord_1,
                "coord_2": coord.coord_2,
                "raw_coord": json.dumps(coord.raw_coord),
                "inferred_time": coord.inferred_time,
                "inferred_x": coord.inferred_x,
                "inferred_y": coord.inferred_y,
            }
            for coord in layout.coords
        ]
    )


def save_detector_layout(layout: DetectorLayout, path: str | Path) -> None:
    """Save layout as CSV."""
    target = ensure_parent(path)
    detector_layout_to_dataframe(layout).to_csv(target, index=False)


def load_detector_layout(path: str | Path) -> DetectorLayout:
    """Load layout from CSV."""
    frame = pd.read_csv(path)
    coords = [
        DetectorCoord(
            detector_id=int(row["detector_id"]),
            raw_coord=json.loads(row["raw_coord"]) if isinstance(row["raw_coord"], str) else [],
            inferred_time=None if pd.isna(row["inferred_time"]) else float(row["inferred_time"]),
            inferred_x=None if pd.isna(row["inferred_x"]) else float(row["inferred_x"]),
            inferred_y=None if pd.isna(row["inferred_y"]) else float(row["inferred_y"]),
            metadata={},
        )
        for _, row in frame.iterrows()
    ]
    return build_spatial_index(DetectorLayout(coords=coords, metadata={"num_detectors": len(coords), "source": "csv"}))
