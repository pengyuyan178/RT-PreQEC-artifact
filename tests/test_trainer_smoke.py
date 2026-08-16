import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rt_preqec.models.datasets import ModelRiskDataset, make_risk_dataloader
from rt_preqec.models.model_factory import build_model
from rt_preqec.models.normalization import compute_normalization_stats
from rt_preqec.models.trainer import RiskRuntimeTrainer


def _write_smoke_risk_dataset(path: Path, n: int = 16, f: int = 4) -> None:
    features = np.random.default_rng(7).normal(size=(n, f)).astype(np.float32)
    labels = np.zeros((n, 5), dtype=np.int8)
    labels[:, 0] = np.arange(n) % 2
    labels[:, 1] = (np.arange(n) + 1) % 2
    labels[:, 3] = np.arange(n) % 4 == 0
    labels[:, 4] = np.logical_or(labels[:, 0], labels[:, 3]).astype(np.int8)
    runtimes = np.stack([np.linspace(1.0, 3.0, n), np.ones(n)], axis=1).astype(np.float32)
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
        runtimes=runtimes,
        runtime_names=np.asarray(["accurate_runtime_us", "fast_runtime_us"], dtype="<U64"),
        feature_names=np.asarray([f"feature_{idx}" for idx in range(f)], dtype="<U64"),
        metadata_json=json.dumps({"num_samples": n, "hard_runtime_label_valid": True}),
    )


def test_risk_runtime_trainer_verbose_fit_saves_training_log(tmp_path: Path) -> None:
    data_path = tmp_path / "risk_dataset.npz"
    _write_smoke_risk_dataset(data_path)
    base = ModelRiskDataset(data_path, split="all")
    stats = compute_normalization_stats(base.features, base.feature_names)
    dataset = ModelRiskDataset(data_path, split="all", normalization_stats=stats)
    loader = make_risk_dataloader(dataset, batch_size=8, shuffle=False)
    model = build_model("risk_mlp", input_dim=4, config={"hidden_dim": 8, "feature_layers": 1})
    trainer = RiskRuntimeTrainer(model, torch.optim.Adam(model.parameters(), lr=1e-3), "cpu")

    logs = trainer.fit(loader, loader, epochs=1, verbose=True)

    assert len(logs) == 2
    assert {row["split"] for row in logs} == {"train", "val"}
    assert all("loss" in row for row in logs)

    checkpoint = tmp_path / "risk_mlp.pt"
    trainer.save_checkpoint(
        checkpoint,
        model_type="risk_mlp",
        model_config={
            "feature_dim": 4,
            "hidden_dim": 8,
            "feature_layers": 1,
            "history_encoder_type": "none",
            "history_length": 1,
        },
        normalization=stats,
        feature_names=base.feature_names,
        metrics=logs[-1],
    )
    training_log_path = checkpoint.with_suffix(".training_log.csv")
    assert training_log_path.exists()
    saved = pd.read_csv(training_log_path)
    assert list(saved["split"]) == ["train", "val"]
    assert "loss" in saved.columns
