"""Core data schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SyndromeShot:
    """Single syndrome sample with optional metadata."""

    syndrome: np.ndarray
    observable: np.ndarray
    distance: int
    rounds: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PatchSample:
    """Local patch training sample."""

    patch: np.ndarray
    location: tuple[int, int, int]
    correction_target: np.ndarray
    is_correct: bool
    confidence_target: float
    risk_target: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectorPatch:
    """Layout-aware detector patch over a flat detector syndrome."""

    patch_id: int
    shot_id: int | None
    center_detector_id: int
    detector_ids: np.ndarray
    detector_coords: np.ndarray
    syndrome_bits: np.ndarray
    active_detector_ids: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalErrorCandidate:
    """Local error mechanism parsed from a detector error model."""

    candidate_id: int
    detector_ids: np.ndarray
    observable_ids: np.ndarray
    probability: float | None
    weight: float | None
    coord_span: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateValidationResult:
    """Result of validating a local candidate against a detector patch."""

    passed: bool
    candidate_id: int | None
    reason: str
    matched_detector_ids: np.ndarray
    unmatched_detector_ids: np.ndarray
    touches_observable: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamEvent:
    """Streamed syndrome event entering the runtime."""

    event_id: int
    syndrome: np.ndarray
    timestamp_us: float
    deadline_us: float
    logical_boundary: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
