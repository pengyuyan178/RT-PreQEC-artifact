"""Decoding job schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DecodingJob:
    """Single residual decoding job."""

    job_id: int
    syndrome: np.ndarray
    created_us: float
    deadline_us: float
    risk_score: float
    predicted_runtime_us: float
    logical_boundary: bool = False
    ai_risk_score: float | None = None
    ai_runtime_score: float | None = None
    ai_confidence: float | None = None
    feature_vector: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
