"""Validation-only risk/confidence threshold calibration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rt_preqec.config import ProjectConfig
from rt_preqec.models.calibration import (
    select_calibration_thresholds,
    sweep_decomposed_thresholds,
    sweep_risk_confidence_thresholds,
)
from rt_preqec.models.datasets import HistoryRiskDataset, ModelRiskDataset, make_risk_dataloader
from rt_preqec.models.risk_profiler import load_risk_profiler_checkpoint, predict_risk_scores
from rt_preqec.models.sequence_builder import build_causal_history_matrix
from rt_preqec.utils import dump_json, ensure_parent


def _is_decomposed_model_type(model_type: str) -> bool:
    return str(model_type).lower().startswith("risk_decomposed")


def _prediction_arrays(
    data_path: str | Path,
    split: str,
    checkpoint: str | Path,
    config: ProjectConfig,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    model, normalization, metadata = load_risk_profiler_checkpoint(checkpoint, device=config.device)
    dataset = ModelRiskDataset(data_path, split=split, normalization_stats=None, seed=int(config.seed))
    features = dataset.features[dataset.indices].astype(np.float32)
    labels = dataset.labels[dataset.indices, dataset._risk_idx].astype(int)
    model_type = str(metadata.get("model_type", ""))
    model_cfg = dict(metadata.get("model_config", {}))
    history_length = int(model_cfg.get("history_length", metadata.get("history_length", 1)))
    history_encoder_type = str(model_cfg.get("history_encoder_type", "none"))
    requires_history = model_type in {
        "risk_gru",
        "risk_lstm",
        "risk_tcn",
        "risk_decomposed_gru",
        "risk_decomposed_lstm",
        "risk_decomposed_tcn",
    } or history_length > 1 or history_encoder_type != "none"
    history = (
        build_causal_history_matrix(
            features,
            history_length=max(history_length, 1),
            normalization=None,
            pad_mode=str(model_cfg.get("pad_mode", "edge")),
        )
        if requires_history
        else None
    )
    predictions = predict_risk_scores(model, features, normalization, history_features=history)
    return (
        {key: np.asarray(value, dtype=float) for key, value in predictions.items()},
        labels,
        {
            "model_type": model_type,
            "num_samples": int(len(dataset)),
            "split_policy": dataset.split_policy,
            "split_indices": dataset.indices.tolist(),
            "checkpoint_metadata": metadata,
            "fast_wrong_labels": np.asarray(
                dataset.labels[dataset.indices, dataset._fast_wrong_idx],
                dtype=int,
            ),
            "fast_logical_fail_labels": np.asarray(
                dataset.labels[dataset.indices, dataset._fast_fail_idx],
                dtype=int,
            ),
        },
    )


def run_calibration(
    config: ProjectConfig,
    data_path: str | Path,
    checkpoint: str | Path,
    split: str,
    out_path: str | Path,
    objective: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Calibrate thresholds on a validation split and persist JSON/CSV."""
    split = str(split).lower()
    if split == "test":
        raise ValueError("Threshold calibration cannot use split=test. Use validation split only.")
    predictions, labels, metadata = _prediction_arrays(data_path, split, checkpoint, config)
    objective_cfg = objective or getattr(config, "calibration_objective", None) or {"type": "maximize_f1"}
    if _is_decomposed_model_type(str(metadata.get("model_type", ""))) or "safe_fast_prob" in predictions:
        sweep_rows = sweep_decomposed_thresholds(
            combined_fast_risk=np.asarray(
                predictions.get("combined_fast_risk", predictions.get("risk_score")),
                dtype=float,
            ),
            safe_fast_prob=np.asarray(predictions.get("safe_fast_prob"), dtype=float),
            confidence_scores=np.asarray(predictions["confidence"], dtype=float),
            labels=labels,
            fast_wrong_labels=np.asarray(metadata.get("fast_wrong_labels"), dtype=int),
            fast_logical_fail_labels=np.asarray(metadata.get("fast_logical_fail_labels"), dtype=int),
        )
    else:
        sweep_rows = sweep_risk_confidence_thresholds(
            np.asarray(predictions["risk_score"], dtype=float),
            np.asarray(predictions["confidence"], dtype=float),
            labels,
        )
    selected = select_calibration_thresholds(sweep_rows, objective_cfg)
    target = ensure_parent(out_path)
    sweep_csv = target.with_suffix(".sweep.csv")
    pd.DataFrame(sweep_rows).to_csv(sweep_csv, index=False)
    payload = {
        "selected_ai_risk_threshold": float(selected["risk_threshold"]),
        "selected_ai_confidence_threshold": float(selected["confidence_threshold"]),
        "selected_safe_fast_threshold": float(selected.get("safe_fast_threshold", 0.5)),
        "objective": objective_cfg,
        "val_metrics": selected,
        "split": split,
        "data": str(data_path),
        "checkpoint": str(checkpoint),
        "split_policy": metadata.get("split_policy"),
        "num_val_samples": int(len(labels)),
        "sweep_csv": str(sweep_csv),
    }
    dump_json(payload, target)
    return payload
