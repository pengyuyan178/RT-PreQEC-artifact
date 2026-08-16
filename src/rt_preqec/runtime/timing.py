"""Timing helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


def now_us() -> float:
    """Current monotonic time in microseconds."""
    return time.perf_counter() * 1e6


def measure_latency_us(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Run a function and measure latency in microseconds."""
    start = now_us()
    result = fn(*args, **kwargs)
    return result, now_us() - start


@dataclass
class RunningLatencyStats:
    """Collect latency samples and compute summary statistics."""

    values: list[float] = field(default_factory=list)

    def update(self, latency_us: float) -> None:
        self.values.append(float(latency_us))

    def summary(self) -> dict[str, float]:
        if not self.values:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "p999": 0.0, "max": 0.0}
        array = np.asarray(self.values, dtype=float)
        return {
            "p50": float(np.percentile(array, 50)),
            "p95": float(np.percentile(array, 95)),
            "p99": float(np.percentile(array, 99)),
            "p999": float(np.percentile(array, 99.9)),
            "max": float(array.max()),
        }
