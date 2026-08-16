import json
from pathlib import Path

import numpy as np

from experiments.build_risk_dataset import run_build_risk_dataset
from rt_preqec.config import ProjectConfig


def test_rtss_16settings_preset_small_build(tmp_path: Path) -> None:
    out = tmp_path / "risk_dataset_v3_16settings_480k.npz"
    cfg = ProjectConfig()
    cfg.seed = 42
    cfg.risk_dataset.split_policy = "setting_stratified"

    summary = run_build_risk_dataset(
        cfg,
        out,
        preset="rtss_16settings_480k",
        shots_per_setting=30,
        verbose=False,
    )

    assert summary["total_samples"] == 480
    assert (tmp_path / "risk_dataset_v3_16settings_480k_splits.json").exists()
    archive = np.load(out, allow_pickle=False)
    metadata = json.loads(str(archive["metadata_json"]))
    assert metadata["dataset_role"] == "main-dev"
    assert metadata["preset"] == "rtss_16settings_480k"
    assert archive["labels"].shape[0] == 480
    feature_names = set(archive["feature_names"].tolist())
    forbidden_features = {
        "estimated_fast_runtime_us",
        "measured_fast_runtime_us",
        "measured_accurate_runtime_us",
        "hard_runtime_label",
        "residual_or_candidate_complexity",
        "backlog_proxy",
        "fast_selection_oracle",
    }
    assert archive["features"].shape[1] == 33
    assert feature_names.isdisjoint(forbidden_features)
    assert archive["setting_ids"].shape == (480,)
    assert set(archive["setting_ids"].tolist()) == set(range(16))
    splits = metadata["splits"]
    assert len(splits["train_indices"]) == 288
    assert len(splits["val_indices"]) == 96
    assert len(splits["test_indices"]) == 96
    for split_name, expected in [("train_indices", 18), ("val_indices", 6), ("test_indices", 6)]:
        ids, counts = np.unique(archive["setting_ids"][np.asarray(splits[split_name], dtype=np.int64)], return_counts=True)
        assert set(ids.tolist()) == set(range(16))
        assert set(counts.tolist()) == {expected}
