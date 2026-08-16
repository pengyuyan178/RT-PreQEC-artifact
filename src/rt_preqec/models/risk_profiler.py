"""Tiny MLP risk-only profiler for scheduler hints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from rt_preqec.models.model_factory import build_model
from rt_preqec.models.normalization import apply_normalization


class TinyRiskProfiler(nn.Module):
    """Risk-only AI profiler.

    This model is not a QEC decoder and does not output corrections.
    It only predicts risk, runtime hardness, confidence, and a runtime
    regression target to help the scheduler choose between fast and
    accurate decoders.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(max(num_layers, 1)):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.risk_head = nn.Linear(current_dim, 1)
        self.runtime_head = nn.Linear(current_dim, 1)
        self.confidence_head = nn.Linear(current_dim, 1)
        self.runtime_regressor = nn.Linear(current_dim, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run a forward pass over `[B, F]` features."""
        hidden = self.backbone(features)
        return {
            "risk_logit": self.risk_head(hidden).squeeze(-1),
            "runtime_logit": self.runtime_head(hidden).squeeze(-1),
            "confidence_logit": self.confidence_head(hidden).squeeze(-1),
            "runtime_pred": self.runtime_regressor(hidden).squeeze(-1),
        }

    @torch.no_grad()
    def predict_proba(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return sigmoid probabilities and runtime prediction."""
        outputs = self.forward(features)
        return {
            "risk_score": torch.sigmoid(outputs["risk_logit"]),
            "runtime_score": torch.sigmoid(outputs["runtime_logit"]),
            "confidence": torch.sigmoid(outputs["confidence_logit"]),
            "runtime_pred": outputs["runtime_pred"],
        }

    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def load_risk_profiler_checkpoint(
    path: str | Path,
    device: str = "cpu",
) -> tuple[nn.Module, dict[str, np.ndarray], dict[str, Any]]:
    """Load legacy or new RT-PreQEC risk/runtime checkpoints.

    Old checkpoints without `model_type` are loaded as `TinyRiskProfiler`.
    New checkpoints are built through `model_factory` and may include causal
    history encoders.
    """
    payload = torch.load(Path(path), map_location=device)
    if "model_type" in payload:
        model_type = str(payload.get("model_type"))
        model_config = dict(payload.get("model_config", {}))
        if "feature_dim" not in model_config and payload.get("feature_names"):
            model_config["feature_dim"] = len(payload["feature_names"])
        model = build_model(model_type, config=model_config)
        model.load_state_dict(payload["state_dict"])
        model.to(device)
        model.eval()
        norm_payload = payload.get("normalization", {})
        normalization = {
            "mean": np.asarray(norm_payload.get("mean", []), dtype=np.float32),
            "std": np.asarray(norm_payload.get("std", []), dtype=np.float32),
        }
        metadata = {key: value for key, value in payload.items() if key != "state_dict"}
        metadata["legacy_checkpoint"] = False
        return model, normalization, metadata

    input_dim = int(payload.get("input_dim", len(payload.get("feature_names", []))))
    model = TinyRiskProfiler(
        input_dim=input_dim,
        hidden_dim=int(payload.get("model_hparams", {}).get("hidden_dim", payload.get("hidden_dim", 64))),
        num_layers=int(payload.get("model_hparams", {}).get("num_layers", payload.get("num_layers", 2))),
        dropout=float(payload.get("model_hparams", {}).get("dropout", payload.get("dropout", 0.0))),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    norm_payload = payload.get("normalization", {})
    normalization = {
        "mean": np.asarray(norm_payload.get("mean", []), dtype=np.float32),
        "std": np.asarray(norm_payload.get("std", []), dtype=np.float32),
    }
    metadata = {key: value for key, value in payload.items() if key != "state_dict"}
    metadata["legacy_checkpoint"] = True
    metadata.setdefault("model_type", "legacy_tiny_risk_profiler")
    return model, normalization, metadata


@torch.no_grad()
def predict_risk_scores(
    model: nn.Module,
    features: np.ndarray | torch.Tensor,
    normalization: dict[str, np.ndarray] | None,
    history_features: np.ndarray | torch.Tensor | None = None,
) -> dict[str, np.ndarray]:
    """Run normalized risk prediction and return numpy arrays."""
    if isinstance(features, torch.Tensor):
        feature_tensor = features.detach().to(next(model.parameters()).device).float()
    else:
        feature_tensor = torch.tensor(np.asarray(features, dtype=np.float32), dtype=torch.float32, device=next(model.parameters()).device)
    if feature_tensor.ndim == 1:
        feature_tensor = feature_tensor.unsqueeze(0)
    feature_tensor = apply_normalization(feature_tensor, normalization)
    history_tensor = None
    if history_features is not None:
        if isinstance(history_features, torch.Tensor):
            history_tensor = history_features.detach().to(next(model.parameters()).device).float()
        else:
            history_tensor = torch.tensor(
                np.asarray(history_features, dtype=np.float32),
                dtype=torch.float32,
                device=next(model.parameters()).device,
            )
        if history_tensor.ndim == 2:
            history_tensor = history_tensor.unsqueeze(0)
        history_tensor = apply_normalization(history_tensor, normalization)
    if history_tensor is not None:
        raw_outputs = model(feature_tensor, history_tensor)
        outputs = raw_outputs.to_dict() if hasattr(raw_outputs, "to_dict") else raw_outputs
    elif hasattr(model, "predict_proba"):
        outputs = model.predict_proba(feature_tensor)
    else:
        raw_outputs = model(feature_tensor)
        outputs = raw_outputs.to_dict() if hasattr(raw_outputs, "to_dict") else raw_outputs
    hard_runtime = outputs.get("hard_runtime_score", outputs.get("runtime_score"))
    result = {
        "risk_score": outputs["risk_score"],
        "runtime_score": hard_runtime,
        "hard_runtime_score": hard_runtime,
        "confidence": outputs["confidence"],
        "runtime_pred": outputs["runtime_pred"],
    }
    for key in [
        "fast_wrong_prob",
        "fast_logical_fail_prob",
        "hard_runtime_prob",
        "syndrome_tail_prob",
        "safe_fast_prob",
        "combined_fast_risk",
        "combined_scheduler_risk",
    ]:
        if key in outputs:
            result[key] = outputs[key]
    return {key: value.detach().cpu().numpy() for key, value in result.items()}
