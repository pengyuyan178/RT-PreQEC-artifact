"""Data transforms for patch tensors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NormalizeBinaryPatch:
    """Normalize binary patches to float32 tensors."""

    scale: float = 1.0

    def __call__(self, patch: np.ndarray) -> np.ndarray:
        return np.asarray(patch, dtype=np.float32) / self.scale
