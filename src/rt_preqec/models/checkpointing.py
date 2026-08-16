"""Checkpoint save/load helpers for RT-PreQEC models."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rt_preqec.models.model_factory import build_model
from rt_preqec.models.risk_profiler import TinyRiskProfiler
from rt_preqec.utils import ensure_parent


def _jsonable_normalization(normalization: dict[str, Any] | None) -> dict[str, Any]:
    if normalization is None:
        return {}
    payload = dict(normalization)
    if "mean" in payload:
        payload["mean"] = np.asarray(payload["mean"], dtype=np.float32).tolist()
    if "std" in payload:
        payload["std"] = np.asarray(payload["std"], dtype=np.float32).tolist()
    if "feature_names" in payload:
        payload["feature_names"] = [str(name) for name in payload["feature_names"]]
    return payload


def save_model_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    model_type: str,
    model_config: dict[str, Any],
    normalization: dict[str, Any] | None,
    feature_names: list[str],
    metrics: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a new-format RT-PreQEC checkpoint."""
    target = ensure_parent(path)
    payload = {
        "model_type": model_type,
        "model_config": dict(model_config),
        "state_dict": model.state_dict(),
        "normalization": _jsonable_normalization(normalization),
        "feature_names": [str(name) for name in feature_names],
        "metrics": metrics or {},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_version": "rt_preqec_model_v1",
        "train_split_metadata": dict((extra or {}).get("train_split_metadata", {})),
        "metadata": {key: value for key, value in dict(extra or {}).items() if key != "train_split_metadata"},
    }
    torch.save(payload, target)


def load_model_checkpoint(path: str | Path, device: str = "cpu") -> dict[str, Any]:
    """Load checkpoint payload and mark legacy checkpoints when needed."""
    payload = torch.load(Path(path), map_location=device)
    if "model_type" not in payload:
        payload = dict(payload)
        payload.setdefault("metadata", {})
        payload["metadata"] = dict(payload["metadata"])
        payload["metadata"]["legacy_checkpoint"] = True
        payload["model_type"] = "legacy_tiny_risk_profiler"
    return payload


def load_model_for_inference(path: str | Path, device: str = "cpu") -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    """Load a new or legacy model checkpoint for inference."""
    payload = load_model_checkpoint(path, device=device)
    model_type = str(payload.get("model_type"))
    if model_type == "legacy_tiny_risk_profiler":
        input_dim = int(payload.get("input_dim", len(payload.get("feature_names", []))))
        hparams = payload.get("model_hparams", {})
        model = TinyRiskProfiler(
            input_dim=input_dim,
            hidden_dim=int(hparams.get("hidden_dim", payload.get("hidden_dim", 64))),
            num_layers=int(hparams.get("num_layers", payload.get("num_layers", 2))),
            dropout=float(hparams.get("dropout", payload.get("dropout", 0.0))),
        )
        model.load_state_dict(payload["state_dict"])
    else:
        model_config = dict(payload.get("model_config", {}))
        if "feature_dim" not in model_config and payload.get("feature_names"):
            model_config["feature_dim"] = len(payload["feature_names"])
        model = build_model(model_type, config=model_config)
        model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    normalization = dict(payload.get("normalization", {}))
    if "mean" in normalization:
        normalization["mean"] = np.asarray(normalization["mean"], dtype=np.float32)
    if "std" in normalization:
        normalization["std"] = np.asarray(normalization["std"], dtype=np.float32)
    metadata = {key: value for key, value in payload.items() if key != "state_dict"}
    return model, normalization, metadata
