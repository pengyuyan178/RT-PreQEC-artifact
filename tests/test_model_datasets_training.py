import json
from pathlib import Path

import numpy as np
import torch

from rt_preqec.data.risk_dataset import create_split_indices, save_risk_dataset_splits, splits_from_dict
from scripts.train_risk_profiler import _ensure_training_splits
from rt_preqec.models.checkpointing import load_model_for_inference
from rt_preqec.models.datasets import HistoryRiskDataset, ModelRiskDataset, assert_no_history_cross_split, make_risk_dataloader
from rt_preqec.models.model_factory import build_model
from rt_preqec.models.normalization import compute_normalization_stats
from rt_preqec.models.trainer import RiskRuntimeTrainer


def _write_fake_risk_dataset(path: Path, n: int = 24, f: int = 6) -> None:
    features = np.random.default_rng(0).normal(size=(n, f)).astype(np.float32)
    labels = np.zeros((n, 5), dtype=np.int8)
    labels[:, 0] = np.arange(n) % 2
    labels[:, 1] = (np.arange(n) + 1) % 2
    labels[:, 3] = np.arange(n) % 3 == 0
    labels[:, 4] = np.logical_or(labels[:, 0], labels[:, 3]).astype(np.int8)
    runtimes = np.stack([np.linspace(1.0, 5.0, n), np.ones(n)], axis=1).astype(np.float32)
    np.savez_compressed(
        path,
        features=features,
        syndromes=np.zeros((n, 4), dtype=np.int8),
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
        metadata_json=json.dumps(
            {
                "num_samples": n,
                "hard_runtime_label_valid": True,
                "samples": [
                    {
                        "sample_id": idx,
                        "shot_id": idx,
                        "metadata": {
                            "setting_id": idx // max(n // 3, 1),
                            "episode_id": idx // max(n // 3, 1),
                            "stream_id": idx // max(n // 3, 1),
                        },
                    }
                    for idx in range(n)
                ],
            }
        ),
    )


def test_model_risk_dataset_and_history(tmp_path: Path) -> None:
    data_path = tmp_path / "risk_dataset.npz"
    _write_fake_risk_dataset(data_path)
    dataset = ModelRiskDataset(data_path, split="all")
    item = dataset[0]
    assert item["features"].shape == (6,)
    assert "risk_label" in item
    history = HistoryRiskDataset(dataset, history_length=4)
    hist_item = history[0]
    assert hist_item["history_features"].shape == (4, 6)


def test_train_one_epoch_mlp_and_lstm_save_load(tmp_path: Path) -> None:
    data_path = tmp_path / "risk_dataset.npz"
    _write_fake_risk_dataset(data_path)
    base = ModelRiskDataset(data_path, split="all")
    stats = compute_normalization_stats(base.features, base.feature_names)
    dataset = ModelRiskDataset(data_path, split="all", normalization_stats=stats)
    loader = make_risk_dataloader(dataset, batch_size=8, shuffle=False)
    model = build_model("risk_mlp", input_dim=6, config={"hidden_dim": 12})
    trainer = RiskRuntimeTrainer(model, torch.optim.Adam(model.parameters(), lr=1e-3), "cpu")
    metrics = trainer.train_one_epoch(loader)
    assert metrics["loss"] >= 0.0

    history_dataset = HistoryRiskDataset(dataset, history_length=3)
    history_loader = make_risk_dataloader(history_dataset, batch_size=8, shuffle=False)
    lstm = build_model(
        "risk_lstm",
        input_dim=6,
        config={"hidden_dim": 12, "history_hidden_dim": 10, "history_length": 3, "num_layers": 1},
    )
    lstm_trainer = RiskRuntimeTrainer(lstm, torch.optim.Adam(lstm.parameters(), lr=1e-3), "cpu")
    lstm_metrics = lstm_trainer.train_one_epoch(history_loader)
    assert lstm_metrics["loss"] >= 0.0

    checkpoint = tmp_path / "risk_lstm.pt"
    lstm_trainer.save_checkpoint(
        checkpoint,
        model_type="risk_lstm",
        model_config={
            "feature_dim": 6,
            "hidden_dim": 12,
            "history_hidden_dim": 10,
            "history_length": 3,
            "history_encoder_type": "lstm",
            "feature_layers": 2,
            "num_layers": 1,
        },
        normalization=stats,
        feature_names=base.feature_names,
        metrics=lstm_metrics,
    )
    loaded, normalization, metadata = load_model_for_inference(checkpoint)
    assert metadata["model_type"] == "risk_lstm"
    assert normalization["mean"].shape == (6,)
    assert loaded is not None


def test_history_dataset_does_not_cross_stream_block_split(tmp_path: Path) -> None:
    data_path = tmp_path / "risk_dataset.npz"
    _write_fake_risk_dataset(data_path, n=12, f=4)
    splits = splits_from_dict(
        create_split_indices(12, "stream_block", train_fraction=0.5, val_fraction=0.25, test_fraction=0.25, seed=3)
    )
    save_risk_dataset_splits(splits, tmp_path / "risk_dataset_splits.json")
    val_base = ModelRiskDataset(data_path, split="val")
    history = HistoryRiskDataset(val_base, history_length=4)
    assert_no_history_cross_split(history)
    first = history[0]
    assert int(first["sample_id"].item()) == 6
    assert set(int(idx) for idx in first["history_indices"].tolist()) == {6}


def test_history_dataset_does_not_cross_setting_boundary(tmp_path: Path) -> None:
    data_path = tmp_path / "risk_dataset.npz"
    _write_fake_risk_dataset(data_path, n=18, f=4)
    splits = splits_from_dict(
        create_split_indices(
            18,
            "setting_stratified",
            train_fraction=0.6,
            val_fraction=0.2,
            test_fraction=0.2,
            seed=3,
            setting_ids=np.asarray([0] * 6 + [1] * 6 + [2] * 6),
        )
    )
    save_risk_dataset_splits(splits, tmp_path / "risk_dataset_splits.json")
    train_base = ModelRiskDataset(data_path, split="train")
    history = HistoryRiskDataset(train_base, history_length=4)

    first_setting_one_pos = int(np.where(train_base.indices == 6)[0][0])
    first_setting_one = history[first_setting_one_pos]
    assert int(first_setting_one["sample_id"].item()) == 6
    assert set(int(idx) for idx in first_setting_one["history_indices"].tolist()) == {6}

    later_setting_one_pos = int(np.where(train_base.indices == 8)[0][0])
    later_setting_one = history[later_setting_one_pos]
    assert set(int(idx) for idx in later_setting_one["history_indices"].tolist()) == {6, 7, 8}


def test_training_split_sidecar_policy_mismatch_regenerates(tmp_path: Path, caplog) -> None:
    data_path = tmp_path / "risk_dataset.npz"
    _write_fake_risk_dataset(data_path, n=30, f=4)
    old_splits = splits_from_dict(create_split_indices(30, "stream_block", 0.6, 0.2, 0.2, seed=1))
    save_risk_dataset_splits(old_splits, tmp_path / "risk_dataset_splits.json")

    cfg = type("Cfg", (), {})()
    cfg.risk_training = type("RiskTraining", (), {"val_fraction": 0.2})()
    cfg.qec = type("Qec", (), {"test_fraction": 0.2})()
    caplog.set_level("INFO")

    payload = _ensure_training_splits(data_path, "setting_stratified", seed=1, cfg=cfg)

    assert payload["split_policy"] == "setting_stratified"
    assert payload["train_indices"] == [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25]
    assert "split policy mismatch; regenerating splits" in caplog.text
    with (tmp_path / "risk_dataset_splits.json").open("r", encoding="utf-8") as handle:
        persisted = json.load(handle)
    assert persisted["split_policy"] == "setting_stratified"


def test_model_dataset_uses_dataset_specific_split_sidecar_and_columnar_setting_ids(tmp_path: Path) -> None:
    data_path = tmp_path / "risk_dataset_custom.npz"
    _write_fake_risk_dataset(data_path, n=18, f=4)
    archive = np.load(data_path, allow_pickle=False)
    payload = {key: archive[key] for key in archive.files}
    payload["setting_ids"] = np.asarray([0] * 6 + [1] * 6 + [2] * 6, dtype=np.int16)
    metadata = json.loads(str(payload["metadata_json"]))
    metadata.pop("samples", None)
    payload["metadata_json"] = json.dumps(metadata)
    np.savez_compressed(data_path, **payload)
    splits = splits_from_dict(
        create_split_indices(
            18,
            "setting_stratified",
            train_fraction=0.6,
            val_fraction=0.2,
            test_fraction=0.2,
            seed=3,
            setting_ids=payload["setting_ids"],
        )
    )
    save_risk_dataset_splits(splits, tmp_path / "risk_dataset_custom_splits.json")

    train_base = ModelRiskDataset(data_path, split="train")
    assert train_base.split_policy == "setting_stratified"
    history = HistoryRiskDataset(train_base, history_length=4)
    first_setting_one_pos = int(np.where(train_base.indices == 6)[0][0])
    first_setting_one = history[first_setting_one_pos]
    assert int(first_setting_one["sample_id"].item()) == 6
    assert set(int(idx) for idx in first_setting_one["history_indices"].tolist()) == {6}


def test_reproducible_dataloader_batches_with_same_seed(tmp_path: Path) -> None:
    data_path = tmp_path / "risk_dataset.npz"
    _write_fake_risk_dataset(data_path, n=20, f=3)
    dataset = ModelRiskDataset(data_path, split="all")
    loader_a = make_risk_dataloader(dataset, batch_size=5, shuffle=True, seed=123)
    loader_b = make_risk_dataloader(dataset, batch_size=5, shuffle=True, seed=123)
    batch_a = next(iter(loader_a))["sample_id"]
    batch_b = next(iter(loader_b))["sample_id"]
    assert torch.equal(batch_a, batch_b)


def test_weighted_sampler_is_seeded(tmp_path: Path) -> None:
    data_path = tmp_path / "risk_dataset.npz"
    _write_fake_risk_dataset(data_path, n=20, f=3)
    dataset = ModelRiskDataset(data_path, split="all")
    loader_a = make_risk_dataloader(dataset, batch_size=6, shuffle=True, weighted_sampler=True, seed=77)
    loader_b = make_risk_dataloader(dataset, batch_size=6, shuffle=True, weighted_sampler=True, seed=77)
    assert torch.equal(next(iter(loader_a))["sample_id"], next(iter(loader_b))["sample_id"])
