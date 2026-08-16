import json
from pathlib import Path

import numpy as np
import pytest

from experiments.calibrate_risk_thresholds import run_calibration
from rt_preqec.config import ProjectConfig
from rt_preqec.data.risk_dataset import create_split_indices, save_risk_dataset_splits, splits_from_dict
from rt_preqec.models.calibration import select_calibration_thresholds, sweep_risk_confidence_thresholds
from rt_preqec.models.risk_profiler import TinyRiskProfiler
import torch


def _write_dataset(path: Path, n: int = 12, f: int = 3) -> None:
    features = np.random.default_rng(0).normal(size=(n, f)).astype(np.float32)
    labels = np.zeros((n, 5), dtype=np.int8)
    labels[:, 4] = np.arange(n) % 2
    labels[:, 3] = labels[:, 4]
    np.savez_compressed(
        path,
        features=features,
        syndromes=np.zeros((n, 2), dtype=np.int8),
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
            ],
            dtype="<U64",
        ),
        runtimes=np.ones((n, 2), dtype=np.float32),
        runtime_names=np.asarray(["accurate_runtime_us", "fast_runtime_us"], dtype="<U64"),
        feature_names=np.asarray([f"f{i}" for i in range(f)], dtype="<U64"),
        metadata_json=json.dumps({"num_samples": n, "hard_runtime_label_valid": True}),
    )
    splits = splits_from_dict(create_split_indices(n, "stream_block", 0.5, 0.25, 0.25, seed=1))
    save_risk_dataset_splits(splits, path.with_name("risk_dataset_splits.json"))


def _write_checkpoint(path: Path, f: int = 3) -> None:
    model = TinyRiskProfiler(input_dim=f, hidden_dim=4, num_layers=1)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": f,
            "feature_names": [f"f{i}" for i in range(f)],
            "normalization": {"mean": [0.0] * f, "std": [1.0] * f},
            "model_hparams": {"hidden_dim": 4, "num_layers": 1, "dropout": 0.0},
        },
        path,
    )


def test_threshold_sweep_selects_thresholds() -> None:
    rows = sweep_risk_confidence_thresholds(
        risk_scores=np.asarray([0.1, 0.8, 0.7, 0.2]),
        confidence_scores=np.asarray([0.9, 0.9, 0.4, 0.8]),
        labels=np.asarray([0, 1, 1, 0]),
    )
    selected = select_calibration_thresholds(rows, {"type": "maximize_f1"})
    assert "risk_threshold" in selected
    assert "confidence_threshold" in selected


def test_calibration_rejects_test_split(tmp_path: Path) -> None:
    data = tmp_path / "risk_dataset.npz"
    ckpt = tmp_path / "risk.pt"
    _write_dataset(data)
    _write_checkpoint(ckpt)
    with pytest.raises(ValueError):
        run_calibration(ProjectConfig(), data, ckpt, "test", tmp_path / "cal.json")


def test_calibration_outputs_selected_thresholds(tmp_path: Path) -> None:
    data = tmp_path / "risk_dataset.npz"
    ckpt = tmp_path / "risk.pt"
    out = tmp_path / "cal.json"
    _write_dataset(data)
    _write_checkpoint(ckpt)
    payload = run_calibration(ProjectConfig(), data, ckpt, "val", out, objective={"type": "maximize_f1"})
    assert out.exists()
    assert "selected_ai_risk_threshold" in payload
    assert payload["split"] == "val"
