"""Causal sequence helpers for RT-PreQEC online history features."""

from __future__ import annotations

from typing import Any

import numpy as np

from rt_preqec.models.normalization import apply_normalization


def build_causal_history_matrix(
    features: np.ndarray,
    history_length: int,
    normalization: dict[str, Any] | None = None,
    pad_mode: str = "edge",
) -> np.ndarray:
    """Build rolling causal histories `[N,T,F]` from shot features `[N,F]`.

    The i-th row uses only `max(0, i-T+1) ... i`; no future shot features are
    read. `pad_mode="edge"` repeats the first available shot at stream start,
    while `pad_mode="zero"` pads missing history with zeros.
    """
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("features must be [N,F]")
    history_length = max(int(history_length), 1)
    histories: list[np.ndarray] = []
    for idx in range(len(array)):
        rows: list[np.ndarray] = []
        for source in range(idx - history_length + 1, idx + 1):
            if source < 0:
                if pad_mode == "zero":
                    rows.append(np.zeros(array.shape[1], dtype=np.float32))
                else:
                    rows.append(array[0].astype(np.float32))
            else:
                rows.append(array[source].astype(np.float32))
        histories.append(np.stack(rows, axis=0))
    history = np.stack(histories, axis=0).astype(np.float32) if histories else np.zeros((0, history_length, array.shape[1]), dtype=np.float32)
    return np.asarray(apply_normalization(history, normalization), dtype=np.float32)
