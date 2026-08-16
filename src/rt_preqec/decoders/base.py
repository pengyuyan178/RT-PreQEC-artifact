"""Decoder base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DecodeResult:
    """Result from a backend decoder."""

    correction: np.ndarray
    success: bool
    latency_us: float
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseDecoder(ABC):
    """Abstract decoder interface."""

    name: str

    @abstractmethod
    def decode(self, syndrome: np.ndarray) -> DecodeResult:
        """Decode a syndrome tensor."""
