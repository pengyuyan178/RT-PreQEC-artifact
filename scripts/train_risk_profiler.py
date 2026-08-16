"""CLI for training RT-PreQEC risk/runtime profiler models."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import typer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rt_preqec.config import load_config
from rt_preqec.config import seed_everything
from rt_preqec.data.risk_dataset import (
    create_split_indices,
    load_risk_dataset_metadata,
    metadata_field_values,
    risk_dataset_split_sidecar_path,
    save_risk_dataset_splits,
    splits_from_dict,
)
from rt_preqec.logging_utils import configure_logging, get_logger
from rt_preqec.models.datasets import (
    HistoryRiskDataset,
    ModelRiskDataset,
    make_risk_dataloader,
    split_indices_hashes,
    validate_split_policy_for_model,
)
from rt_preqec.models.model_factory import build_model
from rt_preqec.models.normalization import compute_normalization_stats
from rt_preqec.models.trainer import RiskRuntimeTrainer
from rt_preqec.utils import dump_json

app = typer.Typer(add_completion=False)
logger = get_logger(__name__)


def _model_config_from_project(cfg: Any, model_type: str, history_length: int | None) -> dict[str, Any]:
    model_cfg = dict(cfg.model.__dict__)
    model_cfg["type"] = model_type or model_cfg.get("type", "risk_mlp")
    if history_length is not None:
        model_cfg["history_length"] = int(history_length)
    if model_type == "risk_mlp":
        model_cfg["history_encoder_type"] = "none"
        model_cfg.setdefault("history_length", 1)
    elif model_type == "risk_gru":
        model_cfg["history_encoder_type"] = "gru"
    elif model_type == "risk_lstm":
        model_cfg["history_encoder_type"] = "lstm"
    elif model_type == "risk_tcn":
        model_cfg["history_encoder_type"] = "tcn"
    elif model_type == "risk_decomposed_mlp":
        model_cfg["history_encoder_type"] = "none"
        model_cfg.setdefault("history_length", 1)
    elif model_type == "risk_decomposed_gru":
        model_cfg["history_encoder_type"] = "gru"
    elif model_type == "risk_decomposed_lstm":
        model_cfg["history_encoder_type"] = "lstm"
    elif model_type == "risk_decomposed_tcn":
        model_cfg["history_encoder_type"] = "tcn"
    return model_cfg


def _loss_weights(cfg: Any) -> dict[str, float]:
    return dict(cfg.loss) if cfg.loss else {
        "risk_weight": 1.0,
        "hard_runtime_weight": 0.5,
        "runtime_weight": 0.2,
        "confidence_weight": 0.1,
    }


def _pos_weight(dataset: ModelRiskDataset, label_key: str) -> float:
    labels = []
    for idx in range(len(dataset)):
        labels.append(float(dataset[idx][label_key].item()))
    if not labels:
        return 1.0
    positives = max(float(np.sum(labels)), 1.0)
    negatives = max(float(len(labels) - np.sum(labels)), 1.0)
    return float(negatives / positives)


def _is_decomposed_model_type(model_type: str) -> bool:
    return str(model_type).lower().startswith("risk_decomposed")


def _parse_bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected boolean true/false, got {value!r}")


def _metric_value(row: dict[str, Any] | None, key: str) -> float | None:
    if not row or key not in row:
        return None
    try:
        return float(row[key])
    except (TypeError, ValueError):
        return None


def _format_summary_value(value: float | None, precision: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{precision}f}"


def _risk_f1(row: dict[str, Any]) -> float | None:
    precision = _metric_value(row, "precision")
    recall = _metric_value(row, "recall")
    if precision is None or recall is None or precision + recall <= 0:
        return None
    return float(2.0 * precision * recall / (precision + recall))


def _best_row(rows: list[dict[str, Any]], key: str, maximize: bool) -> dict[str, Any] | None:
    scored = [(row, _metric_value(row, key)) for row in rows]
    scored = [(row, value) for row, value in scored if value is not None]
    if not scored:
        return None
    return (max if maximize else min)(scored, key=lambda item: item[1])[0]


def _print_training_completion_summary(
    *,
    checkpoint_path: str | Path,
    logs: list[dict[str, Any]],
    split_policy: str,
    hard_runtime_label_valid: bool,
    model_type: str,
    history_length: int,
) -> None:
    val_rows = [row for row in logs if row.get("split") == "val"]
    best_loss_row = _best_row(val_rows, "loss", maximize=False)
    best_accuracy_row = _best_row(val_rows, "risk_accuracy", maximize=True)
    best_f1 = None
    for row in val_rows:
        value = _risk_f1(row)
        if value is not None:
            best_f1 = value if best_f1 is None else max(best_f1, value)
    final_val = val_rows[-1] if val_rows else None
    training_log_path = Path(checkpoint_path).with_suffix(".training_log.csv")
    print("Training complete")
    print(f"checkpoint path: {checkpoint_path}")
    print(f"training_log path: {training_log_path}")
    print(f"best val loss: {_format_summary_value(_metric_value(best_loss_row, 'loss'))}")
    if best_f1 is not None:
        print(f"best val f1: {_format_summary_value(best_f1)}")
    else:
        print(f"best val risk_accuracy: {_format_summary_value(_metric_value(best_accuracy_row, 'risk_accuracy'))}")
    print(
        "final val fnr/fpr: "
        f"{_format_summary_value(_metric_value(final_val, 'fnr'))}/"
        f"{_format_summary_value(_metric_value(final_val, 'fpr'))}"
    )
    print(f"split_policy: {split_policy}")
    print(f"hard_runtime_label_valid: {bool(hard_runtime_label_valid)}")
    print(f"model_type: {model_type}")
    print(f"history_length: {int(history_length)}")


def _ensure_training_splits(data_path: str | Path, split_policy: str, seed: int, cfg: Any) -> dict[str, Any]:
    path = Path(data_path)
    split_path = risk_dataset_split_sidecar_path(path)
    legacy_split_path = path.with_name("risk_dataset_splits.json")
    requested_policy = str(split_policy or "random").lower()
    existing_payload: dict[str, Any] | None = None
    existing_policy: str | None = None
    sidecar_policy_mismatch = False
    existing_split_path = split_path if split_path.exists() else legacy_split_path if legacy_split_path.exists() else None
    if existing_split_path is not None:
        with existing_split_path.open("r", encoding="utf-8") as handle:
            import json

            existing_payload = json.load(handle)
        existing_policy = str(existing_payload.get("split_policy", "random")).lower()
        logger.info("requested split_policy: %s", requested_policy)
        logger.info("existing split_policy: %s", existing_policy)
        if existing_policy == requested_policy:
            logger.info("split policy matches; reusing existing splits")
            if existing_split_path != split_path:
                save_risk_dataset_splits(splits_from_dict(existing_payload), split_path)
            return existing_payload
        sidecar_policy_mismatch = True
        logger.info("split policy mismatch; regenerating splits")
    else:
        logger.info("requested split_policy: %s", requested_policy)
        logger.info("existing split_policy: none")
        logger.info("no split sidecar found; generating splits")

    metadata = load_risk_dataset_metadata(path)
    if isinstance(metadata.get("splits"), dict) and metadata["splits"].get("train_indices") is not None:
        metadata_payload = dict(metadata["splits"])
        metadata_policy = str(metadata_payload.get("split_policy", "random")).lower()
        logger.info("metadata split_policy: %s", metadata_policy)
        if metadata_policy == requested_policy and not sidecar_policy_mismatch:
            logger.info("metadata split policy matches; reusing embedded splits")
            return metadata_payload
        logger.info("metadata split policy mismatch; regenerating splits")
    archive = np.load(path, allow_pickle=False)
    num_samples = int(len(archive["features"]))
    risk_dataset_cfg = getattr(cfg, "risk_dataset", None)
    val_fraction = (
        float(getattr(risk_dataset_cfg, "val_fraction"))
        if risk_dataset_cfg is not None and getattr(risk_dataset_cfg, "val_fraction", None) is not None
        else float(cfg.risk_training.val_fraction)
    )
    test_fraction = (
        float(getattr(risk_dataset_cfg, "test_fraction"))
        if risk_dataset_cfg is not None and getattr(risk_dataset_cfg, "test_fraction", None) is not None
        else float(cfg.qec.test_fraction)
    )
    train_fraction = (
        float(getattr(risk_dataset_cfg, "train_fraction"))
        if risk_dataset_cfg is not None and getattr(risk_dataset_cfg, "train_fraction", None) is not None
        else max(0.0, 1.0 - val_fraction - test_fraction)
    )
    split_payload = create_split_indices(
        num_samples=num_samples,
        split_policy=requested_policy,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=int(seed),
        metadata=metadata,
    )
    splits = splits_from_dict(split_payload)
    save_risk_dataset_splits(splits, split_path)
    return split_payload


def _split_label_distribution(dataset: ModelRiskDataset, indices: np.ndarray) -> dict[str, float]:
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) == 0:
        return {
            "scheduler_risk": 0.0,
            "fast_wrong": 0.0,
            "fast_logical_fail": 0.0,
            "hard_runtime": 0.0,
            "syndrome_tail": 0.0,
            "safe_fast": 0.0,
        }
    fast_wrong = np.asarray(dataset.labels[indices, dataset._fast_wrong_idx], dtype=np.float32)
    fast_fail = np.asarray(dataset.labels[indices, dataset._fast_fail_idx], dtype=np.float32)
    safe_fast = np.logical_and(fast_wrong < 0.5, fast_fail < 0.5).astype(np.float32)
    return {
        "scheduler_risk": float(np.mean(dataset.labels[indices, dataset._risk_idx])),
        "fast_wrong": float(np.mean(fast_wrong)),
        "fast_logical_fail": float(np.mean(fast_fail)),
        "hard_runtime": float(np.mean(dataset.labels[indices, dataset._hard_runtime_idx])),
        "syndrome_tail": float(np.mean(dataset.syndrome_tail[indices])),
        "safe_fast": float(np.mean(safe_fast)),
    }


def _split_setting_counts(dataset: ModelRiskDataset, indices: np.ndarray) -> dict[str, int]:
    setting_ids = metadata_field_values(dataset.metadata, "setting_id", len(dataset.features))
    if setting_ids is None:
        return {}
    counts: dict[str, int] = {}
    for raw_idx in np.asarray(indices, dtype=np.int64).tolist():
        key = str(setting_ids[int(raw_idx)])
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (item[0])))


def _print_split_diagnostics(dataset: ModelRiskDataset) -> None:
    for split_name in ("train", "val", "test"):
        indices = np.asarray(dataset.split_indices.get(split_name, []), dtype=np.int64)
        distribution = _split_label_distribution(dataset, indices)
        setting_counts = _split_setting_counts(dataset, indices)
        print(f"{split_name} label distribution: {distribution}")
        if setting_counts:
            print(f"{split_name} setting counts: {setting_counts}")


def _set_runtime_threads(cfg: Any) -> dict[str, int | None]:
    torch_threads = cfg.timing.set_torch_num_threads
    if torch_threads is not None:
        torch.set_num_threads(int(torch_threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return {
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
    }


@app.command()
def main(
    config: str = "configs/risk_mlp.yaml",
    data: str = "data/processed/risk_dataset.npz",
    out: str = "checkpoints/risk_mlp.pt",
    model_type: str = typer.Option("risk_mlp", "--model-type"),
    history_length: int | None = typer.Option(None, "--history-length"),
    epochs: int | None = typer.Option(None, "--epochs"),
    split_policy: str = typer.Option("stream_block", "--split-policy"),
    allow_temporal_random_split: str = typer.Option("false", "--allow-temporal-random-split"),
    deterministic: str = typer.Option("false", "--deterministic"),
    verbose: str = typer.Option("true", "--verbose"),
    seed: int | None = typer.Option(None, "--seed"),
) -> None:
    """Train a RiskRuntimeModel variant for scheduler risk/runtime hints."""
    cfg = load_config(config)
    configure_logging(cfg.log_level)
    if seed is not None:
        cfg.seed = int(seed)
    deterministic_flag = _parse_bool(deterministic)
    allow_temporal_random_split_flag = _parse_bool(allow_temporal_random_split)
    verbose_flag = _parse_bool(verbose)
    seed_everything(int(cfg.seed), deterministic=deterministic_flag)
    runtime_thread_info = _set_runtime_threads(cfg)
    model_cfg = _model_config_from_project(cfg, model_type, history_length)
    validate_split_policy_for_model(model_type, split_policy, allow_temporal_random_split_flag)
    split_payload = _ensure_training_splits(data, split_policy, int(cfg.seed), cfg)
    effective_split_policy = str(split_payload.get("split_policy", split_policy))
    validate_split_policy_for_model(model_type, effective_split_policy, allow_temporal_random_split_flag)
    if epochs is not None:
        cfg.training.epochs = int(epochs)
    base_train = ModelRiskDataset(data, split="train", seed=int(cfg.seed))
    train_indices = base_train.indices
    train_features = base_train.features[train_indices] if len(train_indices) else base_train.features
    normalization = compute_normalization_stats(train_features, feature_names=base_train.feature_names)
    train_base = ModelRiskDataset(data, split="train", normalization_stats=normalization, seed=int(cfg.seed))
    val_base = ModelRiskDataset(data, split="val", normalization_stats=normalization, seed=int(cfg.seed))
    _print_split_diagnostics(train_base)
    use_history = int(model_cfg.get("history_length", 1)) > 1 or str(model_cfg.get("history_encoder_type", "none")) != "none"
    train_dataset = (
        HistoryRiskDataset(train_base, history_length=int(model_cfg.get("history_length", 1)))
        if use_history
        else train_base
    )
    val_dataset = (
        HistoryRiskDataset(val_base, history_length=int(model_cfg.get("history_length", 1)))
        if use_history
        else val_base
    )
    train_loader = make_risk_dataloader(
        train_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        weighted_sampler=bool(cfg.training.weighted_sampler),
        num_workers=int(cfg.training.num_workers),
        seed=int(cfg.seed),
    )
    val_loader = make_risk_dataloader(
        val_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        weighted_sampler=False,
        num_workers=int(cfg.training.num_workers),
        seed=int(cfg.seed) + 1,
    )
    model_cfg["feature_dim"] = len(base_train.feature_names)
    model = build_model(model_type, input_dim=len(base_train.feature_names), config=model_cfg)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )
    loss_weights = _loss_weights(cfg)
    if not bool(train_base.hard_runtime_label_valid):
        loss_weights["hard_runtime_weight"] = 0.0
        loss_weights["hard_runtime"] = 0.0
    loss_weights["risk_pos_weight"] = _pos_weight(train_base, "risk_label")
    loss_weights["hard_runtime_pos_weight"] = _pos_weight(train_base, "hard_runtime")
    if _is_decomposed_model_type(model_type):
        loss_weights["fast_wrong_pos_weight"] = _pos_weight(train_base, "fast_wrong")
        loss_weights["fast_logical_fail_pos_weight"] = _pos_weight(train_base, "fast_logical_fail")
        loss_weights["syndrome_tail_pos_weight"] = _pos_weight(train_base, "syndrome_tail")
        loss_weights["safe_fast_pos_weight"] = _pos_weight(train_base, "safe_fast")
    trainer = RiskRuntimeTrainer(
        model=model,
        optimizer=optimizer,
        device=cfg.device,
        loss_weights=loss_weights,
        grad_clip_norm=cfg.training.grad_clip_norm,
    )
    logs = trainer.fit(train_loader, val_loader, epochs=int(cfg.training.epochs), verbose=verbose_flag)
    final_metrics = logs[-1] if logs else {}
    split_hashes = split_indices_hashes(train_base)
    risk_labels = [float(train_base[idx]["risk_label"].item()) for idx in range(len(train_base))]
    hard_labels = [float(train_base[idx]["hard_runtime"].item()) for idx in range(len(train_base))]
    checkpoint_cfg = {
        "feature_dim": len(base_train.feature_names),
        "hidden_dim": int(model_cfg.get("hidden_dim", 64)),
        "feature_layers": int(model_cfg.get("feature_layers", 2)),
        "history_encoder_type": str(model_cfg.get("history_encoder_type", "none")),
        "history_length": int(model_cfg.get("history_length", 1)),
        "history_hidden_dim": int(model_cfg.get("history_hidden_dim", model_cfg.get("hidden_dim", 64))),
        "dropout": float(model_cfg.get("dropout", 0.0)),
        "use_layer_norm": bool(model_cfg.get("use_layer_norm", True)),
        "num_layers": int(model_cfg.get("num_layers", 1)),
        "bidirectional": bool(model_cfg.get("bidirectional", False)),
        "pad_mode": "edge",
    }
    if _is_decomposed_model_type(model_type):
        checkpoint_cfg["combination_weights"] = dict(model_cfg.get("combination_weights", {}))
    trainer.save_checkpoint(
        out,
        model_type=model_type,
        model_config=checkpoint_cfg,
        normalization=normalization,
        feature_names=base_train.feature_names,
        metrics=final_metrics,
        extra={
            "train_split_metadata": {
                "train_indices": train_base.indices.tolist(),
                "val_indices": val_base.indices.tolist(),
                "test_indices": train_base.split_indices.get("test", np.asarray([], dtype=np.int64)).tolist(),
                "split_policy": effective_split_policy,
                "split_seed": int(cfg.seed),
                "split_boundaries": train_base.split_boundaries,
                "leakage_safe_for_temporal": bool(split_payload.get("leakage_safe_for_temporal", False)),
                **split_hashes,
            },
            "seed": int(cfg.seed),
            "deterministic": deterministic_flag,
            "num_workers": int(cfg.training.num_workers),
            "train_samples": int(len(train_base)),
            "val_samples": int(len(val_base)),
            "class_balance": {
                "risk_positive_rate": float(np.mean(risk_labels)) if risk_labels else 0.0,
                "hard_runtime_positive_rate": float(np.mean(hard_labels)) if hard_labels else 0.0,
            },
            "hard_runtime_label_valid": bool(train_base.hard_runtime_label_valid),
            **runtime_thread_info,
            "parameter_count": model.count_parameters() if hasattr(model, "count_parameters") else None,
        },
    )
    dump_json({"feature_names": base_train.feature_names}, Path(out).with_suffix(".features.json"))
    dump_json(
        {
            "mean": np.asarray(normalization["mean"], dtype=np.float32).tolist(),
            "std": np.asarray(normalization["std"], dtype=np.float32).tolist(),
            "feature_names": base_train.feature_names,
        },
        Path(out).with_suffix(".norm.json"),
    )
    if verbose_flag:
        _print_training_completion_summary(
            checkpoint_path=out,
            logs=logs,
            split_policy=effective_split_policy,
            hard_runtime_label_valid=bool(train_base.hard_runtime_label_valid),
            model_type=model_type,
            history_length=int(model_cfg.get("history_length", 1)),
        )
    logger.info("saved risk/runtime checkpoint to %s", out)
    logger.info("training summary: %s", final_metrics)


if __name__ == "__main__":
    app()
