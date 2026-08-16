"""QEC-oriented plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rt_preqec.utils import ensure_parent


def plot_logical_error_vs_physical_error(
    physical_error_rates: np.ndarray,
    logical_error_rates: np.ndarray,
    path: str | Path,
) -> None:
    """Plot logical error rate versus physical error rate."""
    plt.figure()
    plt.plot(physical_error_rates, logical_error_rates, marker="o")
    plt.xlabel("Physical error rate")
    plt.ylabel("Logical error rate")
    plt.tight_layout()
    plt.savefig(ensure_parent(path))
    plt.close()


def plot_logical_error_vs_deadline_miss(
    deadline_miss: np.ndarray,
    logical_error_rates: np.ndarray,
    path: str | Path,
) -> None:
    """Plot logical error rate versus deadline misses."""
    plt.figure()
    plt.scatter(deadline_miss, logical_error_rates)
    plt.xlabel("Deadline miss ratio")
    plt.ylabel("Logical error rate")
    plt.tight_layout()
    plt.savefig(ensure_parent(path))
    plt.close()
