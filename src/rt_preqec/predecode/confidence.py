"""Confidence utilities."""

from __future__ import annotations

import numpy as np


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-values))
