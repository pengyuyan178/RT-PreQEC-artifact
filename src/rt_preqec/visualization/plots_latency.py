"""Latency and backlog plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rt_preqec.utils import ensure_parent


def plot_latency_cdf(latencies_us: np.ndarray, path: str | Path) -> None:
    """Plot latency CDF."""
    values = np.sort(np.asarray(latencies_us, dtype=float))
    probs = np.linspace(0, 1, len(values), endpoint=True) if len(values) else np.asarray([])
    plt.figure()
    plt.plot(values, probs)
    plt.xlabel("Latency (us)")
    plt.ylabel("CDF")
    plt.tight_layout()
    plt.savefig(ensure_parent(path))
    plt.close()


def plot_latency_percentiles(latencies_us: np.ndarray, path: str | Path) -> None:
    """Plot key latency percentiles."""
    percentiles = [50, 95, 99, 99.9]
    values = [np.percentile(latencies_us, p) if len(latencies_us) else 0.0 for p in percentiles]
    plt.figure()
    plt.bar([str(p) for p in percentiles], values)
    plt.ylabel("Latency (us)")
    plt.tight_layout()
    plt.savefig(ensure_parent(path))
    plt.close()


def plot_backlog_over_time(events: pd.DataFrame, path: str | Path) -> None:
    """Plot backlog over time."""
    plt.figure()
    if not events.empty and "event_id" in events and "backlog" in events:
        plt.plot(events["event_id"], events["backlog"])
    plt.xlabel("Event")
    plt.ylabel("Backlog")
    plt.tight_layout()
    plt.savefig(ensure_parent(path))
    plt.close()
