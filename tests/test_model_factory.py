from rt_preqec.models.candidate_predecoder_model import CandidatePredecoderModel
from rt_preqec.models.model_factory import build_model, infer_input_requirements
from rt_preqec.models.risk_decomposition_model import RiskDecompositionModel
from rt_preqec.models.risk_runtime_model import RiskRuntimeModel


def test_build_risk_models() -> None:
    for model_type in ["risk_mlp", "risk_gru", "risk_lstm"]:
        model = build_model(model_type, input_dim=5, config={"hidden_dim": 8, "history_length": 4})
        assert isinstance(model, RiskRuntimeModel)
    for model_type in ["risk_decomposed_mlp", "risk_decomposed_gru", "risk_decomposed_lstm", "risk_decomposed_tcn"]:
        model = build_model(model_type, input_dim=5, config={"hidden_dim": 8, "history_length": 4})
        assert isinstance(model, RiskDecompositionModel)
    weighted = build_model(
        "risk_decomposed_mlp",
        input_dim=5,
        config={
            "hidden_dim": 8,
            "combination_weights": {
                "fast_wrong": 2.0,
                "fast_logical_fail": 1.0,
                "hard_runtime": 0.25,
                "syndrome_tail": 0.1,
            },
        },
    )
    assert isinstance(weighted, RiskDecompositionModel)
    assert weighted.combination_weights["fast_wrong"] == 2.0
    assert weighted.combination_weights["hard_runtime"] == 0.25


def test_build_candidate_predecoder() -> None:
    model = build_model(
        "candidate_predecoder",
        detector_feature_dim=4,
        candidate_feature_dim=6,
        config={"hidden_dim": 8},
    )
    assert isinstance(model, CandidatePredecoderModel)


def test_infer_input_requirements() -> None:
    mlp = infer_input_requirements("risk_mlp")
    lstm = infer_input_requirements("risk_lstm", {"bidirectional": False})
    candidate = infer_input_requirements("candidate_predecoder")
    decomposed = infer_input_requirements("risk_decomposed_lstm", {"bidirectional": False})
    unsafe = infer_input_requirements("risk_lstm", {"bidirectional": True})
    assert mlp["requires_features"]
    assert not mlp["requires_history"]
    assert lstm["requires_history"]
    assert lstm["is_online_safe"]
    assert candidate["requires_detector_patch"]
    assert candidate["requires_candidates"]
    assert decomposed["requires_history"]
    assert decomposed["is_online_safe"]
    assert not unsafe["is_online_safe"]
