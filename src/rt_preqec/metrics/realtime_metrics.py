"""Real-time runtime metrics."""

from __future__ import annotations

import numpy as np


def deadline_miss_ratio(latencies_us: np.ndarray, deadlines_us: np.ndarray) -> float:
    """Compute deadline miss ratio."""
    latencies = np.asarray(latencies_us, dtype=float)
    deadlines = np.asarray(deadlines_us, dtype=float)
    if len(latencies) == 0:
        return 0.0
    return float((latencies > deadlines).mean())


def max_pauli_frame_lag(lags: np.ndarray) -> float:
    """Maximum lag."""
    values = np.asarray(lags, dtype=float)
    return float(values.max()) if values.size else 0.0


def average_pauli_frame_lag(lags: np.ndarray) -> float:
    """Average lag."""
    values = np.asarray(lags, dtype=float)
    return float(values.mean()) if values.size else 0.0


def backlog_stats(backlogs: np.ndarray) -> dict[str, float]:
    """Backlog mean and max."""
    values = np.asarray(backlogs, dtype=float)
    if values.size == 0:
        return {"mean": 0.0, "max": 0.0}
    return {"mean": float(values.mean()), "max": float(values.max())}


def latency_percentiles(latencies_us: np.ndarray) -> dict[str, float]:
    """Latency percentile summary."""
    values = np.asarray(latencies_us, dtype=float)
    if values.size == 0:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "p999": 0.0}
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "p999": float(np.percentile(values, 99.9)),
    }
