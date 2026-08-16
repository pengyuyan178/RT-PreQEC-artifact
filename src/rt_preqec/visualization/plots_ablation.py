"""Ablation plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rt_preqec.utils import ensure_parent


def plot_threshold_tradeoff(thresholds: np.ndarray, metric_values: np.ndarray, path: str | Path) -> None:
    """Plot threshold trade-off curve."""
    plt.figure()
    plt.plot(thresholds, metric_values, marker="o")
    plt.xlabel("Confidence threshold")
    plt.ylabel("Metric")
    plt.tight_layout()
    plt.savefig(ensure_parent(path))
    plt.close()


def plot_patch_size_ablation(patch_sizes: np.ndarray, metric_values: np.ndarray, path: str | Path) -> None:
    """Plot patch-size ablation."""
    plt.figure()
    plt.plot(patch_sizes, metric_values, marker="o")
    plt.xlabel("Patch size")
    plt.ylabel("Metric")
    plt.tight_layout()
    plt.savefig(ensure_parent(path))
    plt.close()
