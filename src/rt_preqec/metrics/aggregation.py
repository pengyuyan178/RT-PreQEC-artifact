"""Metric aggregation and persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from rt_preqec.utils import dump_json, ensure_parent


def aggregate_run_metrics(metric_groups: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge metric groups into one dictionary."""
    merged: dict[str, Any] = {}
    for group in metric_groups:
        merged.update(group)
    return merged


def save_metrics_json(metrics: dict[str, Any], path: str | Path) -> None:
    """Save metrics as JSON."""
    dump_json(metrics, path)


def save_metrics_csv(metrics: dict[str, Any], path: str | Path) -> None:
    """Save metrics as a single-row CSV."""
    target = ensure_parent(path)
    pd.DataFrame([metrics]).to_csv(target, index=False)


def compare_modes_summary(mode_metrics: list[dict[str, Any]]) -> pd.DataFrame:
    """Return a mode-comparison summary table."""
    if not mode_metrics:
        return pd.DataFrame()
    return pd.DataFrame(mode_metrics).sort_values(by="mode").reset_index(drop=True)


def save_summary_metrics_csv(summary_metrics: pd.DataFrame, path: str | Path) -> None:
    """Save a mode-comparison summary CSV."""
    target = ensure_parent(path)
    summary_metrics.to_csv(target, index=False)
