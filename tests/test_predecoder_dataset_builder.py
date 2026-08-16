from pathlib import Path

import numpy as np

from scripts.build_predecoder_dataset import main as build_predecoder_dataset_main
from rt_preqec.data.dataset import ArrayPredecoderDataset, predecoder_dataset_split_sidecar_path


def test_build_predecoder_dataset_array_format(tmp_path: Path) -> None:
    out = tmp_path / "predecoder_dataset_v1_300k.npz"
    summary = tmp_path / "summary.json"

    build_predecoder_dataset_main(
        out=str(out),
        summary_out=str(summary),
        num_samples=1000,
        patch_size=5,
        temporal_window=3,
        seed=123,
    )

    assert out.exists()
    assert summary.exists()
    assert predecoder_dataset_split_sidecar_path(out).exists()
    archive = np.load(out, allow_pickle=False)
    assert archive["patches"].shape == (1000, 3, 5, 5)
    assert archive["correction_targets"].shape == (1000, 25)
    assert archive["confidence_targets"].shape == (1000,)
    assert archive["risk_targets"].shape == (1000,)
    assert archive["setting_ids"].shape == (1000,)

    train = ArrayPredecoderDataset(out, split="train")
    val = ArrayPredecoderDataset(out, split="val")
    test = ArrayPredecoderDataset(out, split="test")
    assert len(train) == 600
    assert len(val) == 200
    assert len(test) == 200
    item = train[0]
    assert item["patch"].shape == (1, 3, 5, 5)
    assert item["correction_target"].shape == (25,)
