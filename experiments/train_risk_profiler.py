"""Training loop for the risk-only AI profiler."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

from rt_preqec.config import ProjectConfig, make_torch_generator, seed_worker
from rt_preqec.data.risk_dataset import (
    RiskDatasetSplits,
    create_split_indices,
    RiskProfilerDataset,
    load_risk_dataset,
    load_risk_dataset_splits,
    splits_from_dict,
)
from rt_preqec.models.risk_losses import compute_risk_metrics, risk_profiler_loss
from rt_preqec.models.risk_profiler import TinyRiskProfiler
from rt_preqec.utils import dump_json, ensure_parent


def _collate_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {key: torch.stack([item[key] for item in batch], dim=0) for key in keys}


def _split_indices(num_samples: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples, dtype=np.int64)
    rng.shuffle(indices)
    val_size = int(round(num_samples * val_fraction))
    val_size = min(max(val_size, 1 if num_samples > 1 else 0), max(num_samples - 1, 0))
    return indices[val_size:], indices[:val_size]


def _resolve_training_splits(
    config: ProjectConfig,
    data_path: str | Path,
    num_samples: int,
    val_fraction: float,
    seed: int,
) -> RiskDatasetSplits:
    split_path = Path(data_path).with_name("risk_dataset_splits.json")
    if split_path.exists():
        splits = load_risk_dataset_splits(split_path)
        if splits.train_indices or splits.val_indices:
            return splits
    return splits_from_dict(
        create_split_indices(
            num_samples=num_samples,
            split_policy=str(config.risk_dataset.split_policy),
            train_fraction=max(0.0, 1.0 - float(val_fraction) - float(config.qec.test_fraction)),
            val_fraction=float(val_fraction),
            test_fraction=float(config.qec.test_fraction),
            seed=int(seed),
        )
    )


def train_risk_profiler(config: ProjectConfig, data_path: str | Path, out_checkpoint: str | Path) -> dict[str, Any]:
    """Train the risk-only profiler and persist logs/checkpoint."""
    samples = load_risk_dataset(data_path)
    dataset = RiskProfilerDataset(samples)
    splits = _resolve_training_splits(config, data_path, len(dataset), config.risk_training.val_fraction, config.seed)
    train_indices = np.asarray(splits.train_indices, dtype=np.int64)
    val_indices = np.asarray(splits.val_indices, dtype=np.int64)
    train_dataset = Subset(dataset, train_indices.tolist())
    val_dataset = Subset(dataset, val_indices.tolist()) if len(val_indices) > 0 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.risk_training.batch_size,
        shuffle=True,
        num_workers=config.risk_training.num_workers,
        collate_fn=_collate_batch,
        generator=make_torch_generator(config.seed),
        worker_init_fn=seed_worker,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=config.risk_training.batch_size,
            shuffle=False,
            num_workers=config.risk_training.num_workers,
            collate_fn=_collate_batch,
            generator=make_torch_generator(config.seed + 1),
            worker_init_fn=seed_worker,
        )
        if val_dataset is not None
        else None
    )
    train_features = np.stack([samples[int(idx)].features for idx in train_indices]) if len(train_indices) else np.zeros((0, len(samples[0].features)))
    mean = train_features.mean(axis=0).astype(np.float32) if len(train_features) else np.zeros(len(samples[0].features), dtype=np.float32)
    std = train_features.std(axis=0).astype(np.float32) if len(train_features) else np.ones(len(samples[0].features), dtype=np.float32)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    model = TinyRiskProfiler(
        input_dim=len(samples[0].features),
        hidden_dim=config.risk_model.hidden_dim,
        num_layers=config.risk_model.num_layers,
        dropout=config.risk_model.dropout,
    ).to(config.device)
    optimizer = Adam(model.parameters(), lr=config.risk_training.lr, weight_decay=config.risk_training.weight_decay)
    logs: list[dict[str, float | int | None]] = []

    def _normalize(features: torch.Tensor) -> torch.Tensor:
        return (features - torch.tensor(mean, dtype=torch.float32, device=features.device)) / torch.tensor(
            std, dtype=torch.float32, device=features.device
        )

    for epoch in range(config.risk_training.epochs):
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            batch = {key: value.to(config.device) for key, value in batch.items()}
            outputs = model(_normalize(batch["features"]))
            loss = risk_profiler_loss(outputs, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            metrics = compute_risk_metrics(outputs, batch)
            logs.append(
                {
                    "epoch": epoch,
                    "batch": batch_idx,
                    "split": "train",
                    "loss": float(loss.item()),
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "risk_auc": metrics["risk_auc"],
                }
            )
        if val_loader is not None:
            model.eval()
            val_losses: list[float] = []
            val_metrics_accum: list[dict[str, Any]] = []
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    batch = {key: value.to(config.device) for key, value in batch.items()}
                    outputs = model(_normalize(batch["features"]))
                    loss = risk_profiler_loss(outputs, batch)
                    val_losses.append(float(loss.item()))
                    val_metrics_accum.append(compute_risk_metrics(outputs, batch))
            mean_metrics = {
                key: float(np.mean([0.0 if metrics[key] is None else metrics[key] for metrics in val_metrics_accum]))
                for key in ["accuracy", "precision", "recall", "false_negative_rate", "false_positive_rate"]
            }
            auc_values = [metrics["risk_auc"] for metrics in val_metrics_accum if metrics["risk_auc"] is not None]
            logs.append(
                {
                    "epoch": epoch,
                    "batch": -1,
                    "split": "val",
                    "loss": float(np.mean(val_losses)) if val_losses else 0.0,
                    "accuracy": mean_metrics["accuracy"],
                    "precision": mean_metrics["precision"],
                    "recall": mean_metrics["recall"],
                    "risk_auc": float(np.mean(auc_values)) if auc_values else None,
                }
            )

    target = ensure_parent(out_checkpoint)
    payload = {
        "state_dict": model.state_dict(),
        "config": config.to_dict(),
        "feature_names": samples[0].feature_names,
        "normalization": {"mean": mean.tolist(), "std": std.tolist()},
        "input_dim": len(samples[0].features),
        "parameter_count": model.count_parameters(),
        "model_hparams": {
            "hidden_dim": int(config.risk_model.hidden_dim),
            "num_layers": int(config.risk_model.num_layers),
            "dropout": float(config.risk_model.dropout),
        },
        "train_indices": train_indices.tolist(),
        "val_indices": val_indices.tolist(),
        "test_indices": [int(idx) for idx in splits.test_indices],
        "split_seed": int(splits.split_seed),
        "split_policy": str(splits.split_policy),
        "split_boundaries": splits.split_boundaries,
        "leakage_safe_for_temporal": bool(splits.leakage_safe_for_temporal),
        "train_indices_hash": splits.train_indices_hash,
        "val_indices_hash": splits.val_indices_hash,
        "test_indices_hash": splits.test_indices_hash,
        "seed": int(config.seed),
        "num_workers": int(config.risk_training.num_workers),
        "class_balance": {
            "risk_positive_rate": float(np.mean([samples[int(idx)].scheduler_risk_label for idx in train_indices])) if len(train_indices) else 0.0,
            "hard_runtime_positive_rate": float(np.mean([samples[int(idx)].hard_runtime for idx in train_indices])) if len(train_indices) else 0.0,
        },
        "dataset_hash": f"{len(samples)}:{len(samples[0].features)}:{samples[0].feature_names}",
    }
    torch.save(payload, target)
    pd.DataFrame(logs).to_csv(target.with_suffix(".csv"), index=False)
    dump_json(payload["normalization"], target.with_suffix(".norm.json"))
    dump_json({"feature_names": samples[0].feature_names}, target.with_suffix(".features.json"))
    final_val = [row for row in logs if row["split"] == "val"]
    return {
        "checkpoint": str(target),
        "num_train": int(len(train_indices)),
        "num_val": int(len(val_indices)),
        "num_test": int(len(splits.test_indices)),
        "parameter_count": model.count_parameters(),
        "feature_dim": len(samples[0].features),
        "last_val": final_val[-1] if final_val else None,
    }
