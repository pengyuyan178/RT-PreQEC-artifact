import pytest
import torch

from rt_preqec.models.model_factory import build_model, infer_input_requirements
from rt_preqec.models.risk_decomposition_model import RiskDecompositionModel, RiskRuntimeModelV2


def test_decomposed_mlp_forward_has_legacy_and_component_keys() -> None:
    model = RiskDecompositionModel(feature_dim=8, hidden_dim=16, history_encoder_type="none")
    output = model(torch.randn(4, 8))
    payload = output.to_dict()
    for key in [
        "risk_logit",
        "confidence_logit",
        "runtime_pred",
        "fast_wrong_logit",
        "fast_logical_fail_logit",
        "hard_runtime_logit",
        "syndrome_tail_logit",
        "safe_fast_logit",
        "combined_fast_risk",
        "combined_scheduler_risk",
    ]:
        assert key in payload
        assert payload[key].shape == (4,)


def test_combined_scheduler_risk_uses_combination_weights() -> None:
    model = RiskDecompositionModel(
        feature_dim=8,
        hidden_dim=16,
        history_encoder_type="none",
        combination_weights={
            "fast_wrong": 1.0,
            "fast_logical_fail": 0.0,
            "hard_runtime": 0.0,
            "syndrome_tail": 0.0,
        },
    )
    payload = model(torch.randn(4, 8)).to_dict()
    assert payload["combined_fast_risk"].shape == (4,)
    assert payload["combined_scheduler_risk"].shape == (4,)
    assert torch.allclose(payload["combined_scheduler_risk"], payload["fast_wrong_prob"])
    assert payload["metadata"]["combination_weights"]["fast_wrong"] == 1.0


def test_decomposed_lstm_is_causal_and_factory_registered() -> None:
    model = build_model(
        "risk_decomposed_lstm",
        input_dim=8,
        config={"hidden_dim": 16, "history_hidden_dim": 12, "history_length": 5},
    )
    assert isinstance(model, RiskDecompositionModel)
    output = model(torch.randn(4, 8), torch.randn(4, 5, 8))
    assert output.fast_wrong_logit.shape == (4, 1)
    requirements = infer_input_requirements("risk_decomposed_lstm", {"bidirectional": False})
    assert requirements["requires_history"]
    assert requirements["is_online_safe"]
    with pytest.raises(ValueError):
        build_model("risk_decomposed_lstm", input_dim=8, config={"bidirectional": True})


def test_risk_runtime_model_v2_alias() -> None:
    model = RiskRuntimeModelV2(feature_dim=3, hidden_dim=6)
    assert isinstance(model, RiskDecompositionModel)
