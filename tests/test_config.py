from pathlib import Path

from rt_preqec.config import load_config, merge_dicts


def test_load_default_config() -> None:
    config = load_config(Path("configs/default.yaml"))
    assert config.seed == 42
    assert config.qec.distances == [3, 5, 7]


def test_merge_override() -> None:
    merged = merge_dicts({"a": {"b": 1}, "c": 1}, {"a": {"d": 2}, "c": 3})
    assert merged["a"]["b"] == 1
    assert merged["a"]["d"] == 2
    assert merged["c"] == 3


def test_load_decomposed_lstm_config() -> None:
    cfg = load_config("configs/risk_decomposed_lstm.yaml")
    assert cfg.model.type == "risk_decomposed_lstm"
    assert cfg.model.combination_weights["fast_wrong"] == 1.0
    assert cfg.model.combination_weights["hard_runtime"] == 0.5


def test_load_decomposed_mlp_config() -> None:
    cfg = load_config("configs/risk_decomposed_mlp.yaml")
    assert cfg.model.type == "risk_decomposed_mlp"
    assert cfg.model.combination_weights["fast_logical_fail"] == 1.0
    assert cfg.model.history_encoder_type == "none"
