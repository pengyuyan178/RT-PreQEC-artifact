import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rt_preqec.config import ProjectConfig
from rt_preqec.data.risk_dataset import create_split_indices, save_risk_dataset_splits, splits_from_dict
from rt_preqec.evaluation.real_stream import run_real_stream_eval
from rt_preqec.models.risk_decomposition_model import RiskDecompositionModel


def _config() -> ProjectConfig:
    cfg = ProjectConfig()
    cfg.runtime.round_period_us = 1.0
    cfg.runtime.decode_deadline_us = 2.0
    cfg.runtime.max_pauli_frame_lag = 8
    cfg.runtime.logical_boundary_interval = 4
    cfg.risk_eval.modes = ["ai_risk"]
    cfg.risk_eval.ai_risk_threshold = 0.75
    cfg.risk_eval.ai_confidence_threshold = 0.25
    return cfg


def _write_dataset(path: Path, n: int = 10, f: int = 2) -> None:
    features = np.zeros((n, f), dtype=np.float32)
    labels = np.zeros((n, 6), dtype=np.int8)
    labels[:, 0] = np.arange(n) % 2
    labels[:, 1] = 0
    labels[:, 3] = 0
    labels[:, 4] = labels[:, 0]
    labels[:, 5] = np.arange(n) >= n // 2
    metadata = {
        "num_samples": n,
        "hard_runtime_label_valid": True,
        "samples": [
            {
                "sample_id": idx,
                "shot_id": idx,
                "metadata": {"timing_mode": "loop_per_shot", "hard_runtime_label_valid": True, "toy": True},
            }
            for idx in range(n)
        ],
    }
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
                "syndrome_weight_tail",
            ],
            dtype="<U64",
        ),
        runtimes=np.stack([np.full(n, 3.0), np.full(n, 1.0)], axis=1).astype(np.float32),
        runtime_names=np.asarray(["accurate_runtime_us", "fast_runtime_us"], dtype="<U64"),
        feature_names=np.asarray(["f0", "f1"], dtype="<U64"),
        metadata_json=json.dumps(metadata),
    )
    splits = splits_from_dict(create_split_indices(n, "stream_block", 0.5, 0.2, 0.3, seed=42))
    save_risk_dataset_splits(splits, path.with_name("risk_dataset_splits.json"))


def _write_fast_checkpoint(path: Path) -> None:
    model = RiskDecompositionModel(feature_dim=2, hidden_dim=4, feature_layers=1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.safe_fast_head.bias.fill_(4.0)
        model.confidence_head.linear.bias.fill_(4.0)
        model.fast_wrong_head.bias.fill_(-4.0)
        model.fast_logical_fail_head.bias.fill_(-4.0)
        model.hard_runtime_head.linear.bias.fill_(-4.0)
        model.syndrome_tail_head.bias.fill_(-4.0)
    torch.save(
        {
            "model_type": "risk_decomposed_mlp",
            "model_config": {
                "feature_dim": 2,
                "hidden_dim": 4,
                "feature_layers": 1,
                "history_encoder_type": "none",
                "history_length": 1,
                "history_hidden_dim": 64,
            },
            "state_dict": model.state_dict(),
            "feature_names": ["f0", "f1"],
            "normalization": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
            "train_indices_hash": "different",
        },
        path,
    )


def test_real_stream_decomposed_checkpoint_writes_prob_columns(tmp_path: Path) -> None:
    data = tmp_path / "risk_dataset.npz"
    ckpt = tmp_path / "risk_decomposed.pt"
    cal = tmp_path / "cal.json"
    out = tmp_path / "out"
    _write_dataset(data)
    _write_fast_checkpoint(ckpt)
    cal.write_text(
        json.dumps(
            {
                "selected_ai_risk_threshold": 0.5,
                "selected_safe_fast_threshold": 0.5,
                "selected_ai_confidence_threshold": 0.5,
            }
        ),
        encoding="utf-8",
    )
    payload = run_real_stream_eval(
        _config(),
        risk_checkpoint=str(ckpt),
        out_dir=out,
        risk_dataset_path=data,
        split="test",
        calibration_path=cal,
    )
    events_path = out / "ai_risk" / "events.csv"
    assert payload["metadata"]["ai_risk_available"] is True
    assert events_path.exists()
    events = pd.read_csv(events_path)
    for column in [
        "fast_wrong_prob",
        "fast_logical_fail_prob",
        "hard_runtime_prob",
        "safe_fast_prob",
        "combined_fast_risk",
        "selected_decoder",
    ]:
        assert column in events.columns
    assert "fast" in set(events["selected_decoder"].tolist())


def test_rt_qec_ai_uses_learned_risk_with_predecode_contract(tmp_path: Path) -> None:
    data = tmp_path / "risk_dataset.npz"
    ckpt = tmp_path / "risk_decomposed.pt"
    cal = tmp_path / "cal.json"
    out = tmp_path / "out_rtqec_ai"
    _write_dataset(data)
    _write_fast_checkpoint(ckpt)
    cal.write_text(
        json.dumps(
            {
                "selected_ai_risk_threshold": 0.5,
                "selected_safe_fast_threshold": 0.5,
                "selected_ai_confidence_threshold": 0.5,
            }
        ),
        encoding="utf-8",
    )
    cfg = _config()
    cfg.risk_eval.modes = ["rt_qec_ai"]
    cfg.predecoder.confidence_threshold = 0.1
    cfg.predecoder.risk_threshold = 1.0
    cfg.predecoder.max_cluster_size = 6
    payload = run_real_stream_eval(
        cfg,
        risk_checkpoint=str(ckpt),
        out_dir=out,
        risk_dataset_path=data,
        split="test",
        calibration_path=cal,
    )
    decisions = pd.read_csv(out / "rt_qec_ai" / "decisions.csv")
    events = pd.read_csv(out / "rt_qec_ai" / "events.csv")
    assert payload["metadata"]["ai_risk_available"] is True
    assert "rt_qec_ai" in set(decisions["mode"].tolist())
    assert decisions["selection_reason"].str.contains("learned_risk").any()
    assert decisions["fast_path_certified"].astype(bool).all()
    assert "combined_fast_risk" in events.columns
    assert "fast" in set(events["selected_decoder"].tolist())
