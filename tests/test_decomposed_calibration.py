import json
from pathlib import Path

import numpy as np
import torch

from experiments.calibrate_risk_thresholds import run_calibration
from rt_preqec.config import ProjectConfig
from rt_preqec.data.risk_dataset import create_split_indices, save_risk_dataset_splits, splits_from_dict
from rt_preqec.models.calibration import select_calibration_thresholds, sweep_decomposed_thresholds
from rt_preqec.models.risk_decomposition_model import RiskDecompositionModel


def _write_dataset(path: Path, n: int = 12, f: int = 3) -> None:
    features = np.random.default_rng(0).normal(size=(n, f)).astype(np.float32)
    labels = np.zeros((n, 6), dtype=np.int8)
    labels[:, 0] = np.arange(n) % 3 == 0
    labels[:, 1] = np.arange(n) % 4 == 0
    labels[:, 3] = np.arange(n) % 5 == 0
    labels[:, 4] = np.logical_or.reduce([labels[:, 0], labels[:, 1], labels[:, 3]]).astype(np.int8)
    labels[:, 5] = np.arange(n) >= n // 2
    np.savez_compressed(
        path,
        features=features,
        syndromes=(np.arange(n * 2).reshape(n, 2) % 2).astype(np.int8),
        actual_observables=np.zeros((n, 1), dtype=np.int8),
        accurate_predictions=np.zeros((n, 1), dtype=np.int8),
        fast_predictions=np.zeros((n, 1), dtype=np.int8),
        labels=labels,
        label_names=np.asarray(
            [
                "fast_wrong_vs_accurate",
                "fast_logical_fail",
                "accurate_logical_fail",
                "hard_runtime",
                "scheduler_risk_label",
                "syndrome_weight_tail",
            ],
            dtype="<U64",
        ),
        runtimes=np.stack([np.linspace(1.0, 4.0, n), np.ones(n)], axis=1).astype(np.float32),
        runtime_names=np.asarray(["accurate_runtime_us", "fast_runtime_us"], dtype="<U64"),
        feature_names=np.asarray([f"f{i}" for i in range(f)], dtype="<U64"),
        metadata_json=json.dumps({"num_samples": n, "hard_runtime_label_valid": True}),
    )
    splits = splits_from_dict(create_split_indices(n, "stream_block", 0.5, 0.25, 0.25, seed=1))
    save_risk_dataset_splits(splits, path.with_name("risk_dataset_splits.json"))


def _write_checkpoint(path: Path, f: int = 3) -> None:
    model = RiskDecompositionModel(feature_dim=f, hidden_dim=4, feature_layers=1)
    torch.save(
        {
            "model_type": "risk_decomposed_mlp",
            "model_config": {
                "feature_dim": f,
                "hidden_dim": 4,
                "feature_layers": 1,
                "history_encoder_type": "none",
                "history_length": 1,
                "history_hidden_dim": 64,
            },
            "state_dict": model.state_dict(),
            "feature_names": [f"f{i}" for i in range(f)],
            "normalization": {"mean": [0.0] * f, "std": [1.0] * f},
        },
        path,
    )


def test_decomposed_threshold_sweep_selects_three_thresholds() -> None:
    rows = sweep_decomposed_thresholds(
        combined_fast_risk=np.asarray([0.1, 0.8, 0.2, 0.7]),
        safe_fast_prob=np.asarray([0.9, 0.1, 0.8, 0.2]),
        confidence_scores=np.asarray([0.9, 0.9, 0.6, 0.8]),
        labels=np.asarray([0, 1, 0, 1]),
        fast_wrong_labels=np.asarray([0, 1, 0, 0]),
        fast_logical_fail_labels=np.asarray([0, 0, 0, 1]),
    )
    selected = select_calibration_thresholds(
        rows,
        {"type": "minimize_fnr_under_fast_rate", "max_false_negative_rate": 0.05, "min_fast_selection_rate": 0.2},
    )
    assert "risk_threshold" in selected
    assert "safe_fast_threshold" in selected
    assert "confidence_threshold" in selected


def test_decomposed_calibration_outputs_safe_fast_threshold(tmp_path: Path) -> None:
    data = tmp_path / "risk_dataset.npz"
    ckpt = tmp_path / "risk_decomposed.pt"
    out = tmp_path / "cal.json"
    _write_dataset(data)
    _write_checkpoint(ckpt)
    payload = run_calibration(
        ProjectConfig(),
        data,
        ckpt,
        "val",
        out,
        objective={"type": "minimize_fnr_under_fast_rate", "max_false_negative_rate": 0.5, "min_fast_selection_rate": 0.0},
    )
    assert out.exists()
    assert "selected_ai_risk_threshold" in payload
    assert "selected_safe_fast_threshold" in payload
