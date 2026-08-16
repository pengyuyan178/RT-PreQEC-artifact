import numpy as np
import pytest

from rt_preqec.data.risk_dataset import (
    RiskDatasetSplits,
    RiskProfilerDataset,
    RiskSample,
    create_split_indices,
    load_risk_dataset,
    load_risk_dataset_splits,
    save_risk_dataset,
    save_risk_dataset_splits,
)
from rt_preqec.models.datasets import validate_split_policy_for_model


def _sample(idx: int) -> RiskSample:
    features = np.asarray([float(idx), float(idx + 1)], dtype=np.float32)
    return RiskSample(
        sample_id=idx,
        shot_id=idx,
        syndrome=np.asarray([idx % 2, 1], dtype=np.int8),
        features=features,
        feature_names=["f0", "f1"],
        actual_observable=np.asarray([idx % 2], dtype=np.int8),
        accurate_prediction=np.asarray([0], dtype=np.int8),
        fast_prediction=np.asarray([1], dtype=np.int8),
        accurate_runtime_us=10.0 + idx,
        fast_runtime_us=5.0 + idx,
        fast_wrong_vs_accurate=1,
        fast_logical_fail=1,
        accurate_logical_fail=0,
        hard_runtime=idx % 2,
        scheduler_risk_label=1,
        metadata={"row": idx},
    )


def test_save_load_and_dataset_getitem(tmp_path) -> None:
    path = tmp_path / "risk_dataset.npz"
    samples = [_sample(0), _sample(1), _sample(2)]
    save_risk_dataset(samples, path)
    loaded = load_risk_dataset(path)
    assert len(loaded) == 3
    dataset = RiskProfilerDataset(loaded)
    item = dataset[0]
    assert set(item.keys()) == {
        "features",
        "risk_label",
        "hard_runtime",
        "fast_wrong",
        "fast_logical_fail",
        "accurate_runtime_us",
    }
    assert item["features"].shape[0] == 2


def test_save_load_split_sidecar(tmp_path) -> None:
    split_path = tmp_path / "risk_dataset_splits.json"
    splits = RiskDatasetSplits(train_indices=[2, 3], val_indices=[1], test_indices=[0], split_seed=42)
    save_risk_dataset_splits(splits, split_path)
    loaded = load_risk_dataset_splits(split_path)
    assert loaded.train_indices == [2, 3]
    assert loaded.val_indices == [1]
    assert loaded.test_indices == [0]
    assert loaded.split_seed == 42


def test_stream_block_split_is_contiguous_and_disjoint() -> None:
    splits = create_split_indices(
        num_samples=10,
        split_policy="stream_block",
        train_fraction=0.6,
        val_fraction=0.2,
        test_fraction=0.2,
        seed=7,
    )
    assert splits["train_indices"] == [0, 1, 2, 3, 4, 5]
    assert splits["val_indices"] == [6, 7]
    assert splits["test_indices"] == [8, 9]
    combined = splits["train_indices"] + splits["val_indices"] + splits["test_indices"]
    assert sorted(combined) == list(range(10))
    assert splits["leakage_safe_for_temporal"] is True


def test_random_split_is_reproducible_and_disjoint() -> None:
    first = create_split_indices(20, "random", 0.6, 0.2, 0.2, seed=9)
    second = create_split_indices(20, "random", 0.6, 0.2, 0.2, seed=9)
    assert first["train_indices"] == second["train_indices"]
    combined = first["train_indices"] + first["val_indices"] + first["test_indices"]
    assert sorted(combined) == list(range(20))
    assert first["leakage_safe_for_temporal"] is False


def test_setting_stratified_split_slices_every_setting() -> None:
    setting_ids = np.asarray([0] * 10 + [1] * 10 + [2] * 10)
    splits = create_split_indices(
        num_samples=30,
        split_policy="setting_stratified",
        train_fraction=0.6,
        val_fraction=0.2,
        test_fraction=0.2,
        seed=7,
        setting_ids=setting_ids,
    )
    assert splits["split_policy"] == "setting_stratified"
    assert splits["train_indices"] == [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25]
    assert splits["val_indices"] == [6, 7, 16, 17, 26, 27]
    assert splits["test_indices"] == [8, 9, 18, 19, 28, 29]
    for key in ["train_indices", "val_indices", "test_indices"]:
        assert set(setting_ids[splits[key]].tolist()) == {0, 1, 2}
    combined = splits["train_indices"] + splits["val_indices"] + splits["test_indices"]
    assert sorted(combined) == list(range(30))


def test_setting_stratified_requires_setting_metadata() -> None:
    with pytest.raises(ValueError, match="setting_stratified split requires setting_id metadata"):
        create_split_indices(10, "setting_stratified", 0.6, 0.2, 0.2, seed=1)


def test_setting_stratified_label_distribution_is_closer_than_stream_block() -> None:
    setting_ids = np.asarray([0] * 10 + [1] * 10 + [2] * 10)
    labels = np.asarray(
        [0] * 10
        + [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        + [1] * 10,
        dtype=np.float32,
    )
    overall_rate = float(labels.mean())
    stream = create_split_indices(30, "stream_block", 0.6, 0.2, 0.2, seed=1)
    stratified = create_split_indices(30, "setting_stratified", 0.6, 0.2, 0.2, seed=1, setting_ids=setting_ids)

    def total_drift(splits: dict[str, object]) -> float:
        return float(
            sum(
                abs(float(labels[np.asarray(splits[key], dtype=np.int64)].mean()) - overall_rate)
                for key in ["train_indices", "val_indices", "test_indices"]
            )
        )

    assert total_drift(stratified) < total_drift(stream)


def test_temporal_model_rejects_random_split_by_default() -> None:
    with pytest.raises(ValueError):
        validate_split_policy_for_model("risk_lstm", "random")
    validate_split_policy_for_model("risk_lstm", "random", allow_temporal_random_split=True)
    validate_split_policy_for_model("risk_lstm", "setting_stratified")


def test_episode_split_keeps_episode_together() -> None:
    episode_ids = np.asarray([0, 0, 1, 1, 2, 2, 3, 3])
    splits = create_split_indices(
        num_samples=8,
        split_policy="episode",
        train_fraction=0.5,
        val_fraction=0.25,
        test_fraction=0.25,
        seed=1,
        episode_ids=episode_ids,
    )
    groups = []
    for key in ["train_indices", "val_indices", "test_indices"]:
        groups.append(set(episode_ids[splits[key]].tolist()))
    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])
