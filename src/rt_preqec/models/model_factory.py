"""Factory functions for RT-PreQEC model variants."""

from __future__ import annotations

from typing import Any

from rt_preqec.models.candidate_predecoder_model import CandidatePredecoderModel
from rt_preqec.models.risk_decomposition_model import RiskDecompositionModel
from rt_preqec.models.risk_runtime_model import RiskRuntimeModel
from rt_preqec.models.rt_preqec_model import RTPreQECModel


def _merged_config(config: dict[str, Any] | None, **kwargs: Any) -> dict[str, Any]:
    merged = dict(config or {})
    for key, value in kwargs.items():
        if value is not None:
            merged[key] = value
    return merged


def build_model(
    model_type: str,
    input_dim: int | None = None,
    detector_feature_dim: int | None = None,
    candidate_feature_dim: int | None = None,
    config: dict[str, Any] | None = None,
) -> object:
    """Build a paper-driven RT-PreQEC model by type."""
    model_type = model_type.lower()
    cfg = _merged_config(config)
    if input_dim is not None:
        cfg.setdefault("feature_dim", input_dim)
    if detector_feature_dim is not None:
        cfg.setdefault("detector_feature_dim", detector_feature_dim)
    if candidate_feature_dim is not None:
        cfg.setdefault("candidate_feature_dim", candidate_feature_dim)

    risk_aliases = {
        "risk_runtime": cfg.get("history_encoder_type", "none"),
        "risk_mlp": "none",
        "risk_tcn": "tcn",
        "risk_gru": "gru",
        "risk_lstm": "lstm",
    }
    if model_type in risk_aliases:
        feature_dim = int(cfg.get("feature_dim", cfg.get("input_dim", 0)))
        if feature_dim <= 0:
            raise ValueError("input_dim/feature_dim is required for risk models")
        history_type = risk_aliases[model_type]
        model_config = {
            "feature_dim": feature_dim,
            "hidden_dim": int(cfg.get("hidden_dim", 64)),
            "feature_layers": int(cfg.get("feature_layers", 2)),
            "history_encoder_type": history_type,
            "history_length": int(cfg.get("history_length", 1)),
            "history_hidden_dim": int(cfg.get("history_hidden_dim", cfg.get("hidden_dim", 64))),
            "dropout": float(cfg.get("dropout", 0.0)),
            "use_layer_norm": bool(cfg.get("use_layer_norm", True)),
            "num_layers": int(cfg.get("num_layers", 1)),
            "bidirectional": bool(cfg.get("bidirectional", False)),
        }
        return RiskRuntimeModel(**model_config)

    decomposed_aliases = {
        "risk_decomposed_mlp": "none",
        "risk_decomposed_tcn": "tcn",
        "risk_decomposed_gru": "gru",
        "risk_decomposed_lstm": "lstm",
        "risk_decomposition": cfg.get("history_encoder_type", "none"),
    }
    if model_type in decomposed_aliases:
        feature_dim = int(cfg.get("feature_dim", cfg.get("input_dim", 0)))
        if feature_dim <= 0:
            raise ValueError("input_dim/feature_dim is required for decomposed risk models")
        history_type = decomposed_aliases[model_type]
        model_config = {
            "feature_dim": feature_dim,
            "hidden_dim": int(cfg.get("hidden_dim", 64)),
            "feature_layers": int(cfg.get("feature_layers", 2)),
            "history_encoder_type": history_type,
            "history_length": int(cfg.get("history_length", 1)),
            "history_hidden_dim": int(cfg.get("history_hidden_dim", cfg.get("hidden_dim", 64))),
            "dropout": float(cfg.get("dropout", 0.0)),
            "use_layer_norm": bool(cfg.get("use_layer_norm", True)),
            "num_layers": int(cfg.get("num_layers", 1)),
            "bidirectional": bool(cfg.get("bidirectional", False)),
            "combination_weights": dict(cfg.get("combination_weights", {})),
        }
        return RiskDecompositionModel(**model_config)

    if model_type == "candidate_predecoder":
        if int(cfg.get("detector_feature_dim", 0)) <= 0 or int(cfg.get("candidate_feature_dim", 0)) <= 0:
            raise ValueError("detector_feature_dim and candidate_feature_dim are required")
        return CandidatePredecoderModel(
            detector_feature_dim=int(cfg["detector_feature_dim"]),
            candidate_feature_dim=int(cfg["candidate_feature_dim"]),
            hidden_dim=int(cfg.get("hidden_dim", 64)),
            pooling=str(cfg.get("pooling", "mean_max")),
            scorer=str(cfg.get("scorer", "bilinear")),
            dropout=float(cfg.get("dropout", 0.0)),
        )

    if model_type == "rt_preqec_combined":
        risk_model = None
        candidate_model = None
        if cfg.get("risk_runtime_model") is not None:
            risk_cfg = dict(cfg["risk_runtime_model"])
            risk_type = str(risk_cfg.pop("type", "risk_runtime"))
            risk_model = build_model(risk_type, input_dim=input_dim, config=risk_cfg)
        if cfg.get("candidate_predecoder_model") is not None:
            candidate_cfg = dict(cfg["candidate_predecoder_model"])
            candidate_type = str(candidate_cfg.pop("type", "candidate_predecoder"))
            candidate_model = build_model(candidate_type, config=candidate_cfg)
        return RTPreQECModel(risk_model, candidate_model)

    raise ValueError(f"Unsupported model_type: {model_type}")


def infer_input_requirements(model_type: str, config: dict[str, Any] | None = None) -> dict[str, bool]:
    """Return input and online-safety requirements for a model type."""
    model_type = model_type.lower()
    cfg = config or {}
    requires_history = model_type in {
        "risk_tcn",
        "risk_gru",
        "risk_lstm",
        "risk_decomposed_tcn",
        "risk_decomposed_gru",
        "risk_decomposed_lstm",
    } or (
        model_type == "risk_runtime" and cfg.get("history_encoder_type", "none") != "none"
    ) or (
        model_type == "risk_decomposition" and cfg.get("history_encoder_type", "none") != "none"
    )
    bidirectional = bool(cfg.get("bidirectional", False))
    if model_type in {
        "risk_runtime",
        "risk_mlp",
        "risk_tcn",
        "risk_gru",
        "risk_lstm",
        "risk_decomposition",
        "risk_decomposed_mlp",
        "risk_decomposed_tcn",
        "risk_decomposed_gru",
        "risk_decomposed_lstm",
    }:
        return {
            "requires_features": True,
            "requires_history": bool(requires_history),
            "requires_detector_patch": False,
            "requires_candidates": False,
            "is_online_safe": not bidirectional,
        }
    if model_type == "candidate_predecoder":
        return {
            "requires_features": False,
            "requires_history": False,
            "requires_detector_patch": True,
            "requires_candidates": True,
            "is_online_safe": True,
        }
    if model_type == "rt_preqec_combined":
        return {
            "requires_features": True,
            "requires_history": bool(requires_history),
            "requires_detector_patch": True,
            "requires_candidates": True,
            "is_online_safe": not bidirectional,
        }
    raise ValueError(f"Unsupported model_type: {model_type}")
