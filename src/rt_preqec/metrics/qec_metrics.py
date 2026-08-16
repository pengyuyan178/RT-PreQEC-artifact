"""QEC-side metrics."""

from __future__ import annotations

from typing import Any

import numpy as np


def observable_mismatch_mask(predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return per-shot observable mismatch mask."""
    predicted_array = np.asarray(predicted, dtype=np.int8)
    actual_array = np.asarray(actual, dtype=np.int8)
    if predicted_array.shape != actual_array.shape:
        raise ValueError(f"Shape mismatch: predicted {predicted_array.shape}, actual {actual_array.shape}")
    if predicted_array.ndim == 1:
        return predicted_array != actual_array
    return np.any(predicted_array != actual_array, axis=1)


def logical_error_rate(predicted_observables: np.ndarray, actual_observables: np.ndarray) -> float:
    """Logical error rate from predicted versus actual observables."""
    mismatch = observable_mismatch_mask(predicted_observables, actual_observables)
    values = np.asarray(mismatch, dtype=float)
    return float(values.mean()) if values.size else 0.0


def mean_rounds_to_failure(rounds_to_failure: np.ndarray) -> float:
    """Average rounds until failure."""
    values = np.asarray(rounds_to_failure, dtype=float)
    return float(values.mean()) if values.size else 0.0


def failure_rate(failures: np.ndarray) -> float:
    """Alias for binary failure rate."""
    values = np.asarray(failures, dtype=float)
    return float(values.mean()) if values.size else 0.0


def placeholder_threshold_summary(values: np.ndarray) -> dict[str, float]:
    """Simple summary for toy experiments."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {"mean": 0.0, "max": 0.0}
    return {"mean": float(array.mean()), "max": float(array.max())}


def summarize_qec_baseline(
    predicted_observables: np.ndarray,
    actual_observables: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize a real or placeholder QEC baseline run."""
    mismatch = observable_mismatch_mask(predicted_observables, actual_observables)
    summary = {
        "logical_error_rate": logical_error_rate(predicted_observables, actual_observables),
        "num_shots": int(np.asarray(actual_observables).shape[0]) if np.asarray(actual_observables).ndim >= 1 else 0,
        "num_observables": int(np.asarray(actual_observables).shape[1]) if np.asarray(actual_observables).ndim == 2 else 1,
        "num_mismatches": int(mismatch.sum()),
    }
    if metadata:
        summary.update(metadata)
    return summary
