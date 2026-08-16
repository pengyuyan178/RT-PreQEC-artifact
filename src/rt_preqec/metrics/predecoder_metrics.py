"""Predecoder-specific metrics."""

from __future__ import annotations

import numpy as np


def accept_rate(accepted_mask: np.ndarray) -> float:
    """Fraction of accepted patches."""
    values = np.asarray(accepted_mask, dtype=bool)
    return float(values.mean()) if values.size else 0.0


def abstention_rate(accepted_mask: np.ndarray) -> float:
    """Fraction of abstained patches."""
    return 1.0 - accept_rate(accepted_mask)


def false_accept_rate(accepted_mask: np.ndarray, is_correct: np.ndarray) -> float:
    """Fraction of incorrect accepted patches."""
    accepted = np.asarray(accepted_mask, dtype=bool)
    correct = np.asarray(is_correct, dtype=bool)
    accepted_count = accepted.sum()
    if accepted_count == 0:
        return 0.0
    return float((accepted & ~correct).sum() / accepted_count)


def accepted_error_rate(accepted_mask: np.ndarray, is_correct: np.ndarray) -> float:
    """Alias for false accept rate."""
    return false_accept_rate(accepted_mask, is_correct)


def validation_pass_rate(validation_pass: np.ndarray) -> float:
    """Fraction of patches passing validation."""
    values = np.asarray(validation_pass, dtype=bool)
    return float(values.mean()) if values.size else 0.0


def residual_density_reduction(original_density: float, residual_density: float) -> float:
    """Relative residual density reduction."""
    return float((original_density - residual_density) / max(original_density, 1e-9))


def residual_graph_size_reduction(original_size: int, residual_size: int) -> float:
    """Relative graph size reduction."""
    return float((original_size - residual_size) / max(original_size, 1))
