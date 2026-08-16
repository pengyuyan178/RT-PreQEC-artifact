from pathlib import Path

import torch

from rt_preqec.models.checkpointing import load_model_for_inference, save_model_checkpoint
from rt_preqec.models.risk_decomposition_model import RiskDecompositionModel


def test_decomposed_checkpoint_preserves_combination_weights(tmp_path: Path) -> None:
    model = RiskDecompositionModel(
        feature_dim=3,
        hidden_dim=4,
        feature_layers=1,
        combination_weights={
            "fast_wrong": 2.0,
            "fast_logical_fail": 1.0,
            "hard_runtime": 0.5,
            "syndrome_tail": 0.25,
        },
    )
    checkpoint = tmp_path / "risk_decomposed.pt"
    model_config = {
        "feature_dim": 3,
        "hidden_dim": 4,
        "feature_layers": 1,
        "history_encoder_type": "none",
        "history_length": 1,
        "history_hidden_dim": 64,
        "combination_weights": dict(model.combination_weights),
    }
    save_model_checkpoint(
        checkpoint,
        model,
        model_type="risk_decomposed_mlp",
        model_config=model_config,
        normalization={"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
        feature_names=["f0", "f1", "f2"],
        metrics={"loss": 0.0},
        extra={"train_split_metadata": {"train_indices_hash": "abc"}},
    )

    loaded, _, metadata = load_model_for_inference(checkpoint)

    assert isinstance(loaded, RiskDecompositionModel)
    assert metadata["model_config"]["combination_weights"]["fast_wrong"] == 2.0
    assert metadata["model_config"]["combination_weights"]["syndrome_tail"] == 0.25
    output = loaded(torch.zeros(2, 3)).to_dict()
    assert output["metadata"]["combination_weights"]["fast_wrong"] == 2.0
