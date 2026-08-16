"""Training datasets for RT-PreQEC risk/runtime models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from rt_preqec.config import make_torch_generator, seed_worker
from rt_preqec.data.risk_dataset import (
    create_split_indices,
    hash_indices,
    metadata_field_values,
    risk_dataset_split_sidecar_path,
)
from rt_preqec.models.normalization import apply_normalization


TEMPORAL_MODEL_TYPES = {
    "risk_gru",
    "risk_lstm",
    "risk_tcn",
    "risk_decomposed_gru",
    "risk_decomposed_lstm",
    "risk_decomposed_tcn",
}


def validate_split_policy_for_model(
    model_type: str,
    split_policy: str,
    allow_temporal_random_split: bool = False,
) -> None:
    """Reject random splits for temporal models unless explicitly allowed."""
    if (
        str(model_type).lower() in TEMPORAL_MODEL_TYPES
        and str(split_policy).lower() == "random"
        and not bool(allow_temporal_random_split)
    ):
        raise ValueError(
            "Temporal risk models require stream_block, episode, or setting_stratified split_policy to avoid history leakage. "
            "Pass --allow-temporal-random-split true only for explicit ablations."
        )


def _metadata_from_npz(path: Path) -> dict[str, Any]:
    try:
        archive = np.load(path, allow_pickle=False)
        metadata = json.loads(str(archive["metadata_json"])) if "metadata_json" in archive.files else {}
        return _merge_metadata_sidecar(path, metadata)
    except Exception:
        return {}


def _merge_metadata_sidecar(path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    sidecar = path.with_suffix(".metadata.json")
    if not sidecar.exists():
        return metadata
    try:
        with sidecar.open("r", encoding="utf-8") as handle:
            sidecar_payload = json.load(handle)
        merged = dict(metadata)
        merged.update(dict(sidecar_payload))
        return merged
    except Exception:
        return metadata


def _merge_archive_columnar_metadata(
    archive: np.lib.npyio.NpzFile,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Expose columnar per-sample metadata arrays through metadata helpers."""
    merged = dict(metadata)
    columnar_keys = [
        "setting_ids",
        "episode_ids",
        "stream_ids",
        "shot_index_within_setting",
        "global_index",
        "arrival_order",
        "difficulty_tier_ids",
    ]
    for key in columnar_keys:
        if key in archive.files and key not in merged:
            merged[key] = np.asarray(archive[key])
    return merged


def _load_split_payload(path: Path, num_samples: int, seed: int = 42) -> dict[str, Any]:
    split_paths = [risk_dataset_split_sidecar_path(path), path.with_name("risk_dataset_splits.json")]
    for split_path in split_paths:
        if not split_path.exists():
            continue
        with split_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload
    metadata = _metadata_from_npz(path)
    if isinstance(metadata.get("splits"), dict) and metadata["splits"].get("train_indices") is not None:
        return dict(metadata["splits"])
    return create_split_indices(
        num_samples=num_samples,
        split_policy="random",
        train_fraction=0.6,
        val_fraction=0.2,
        test_fraction=0.2,
        seed=seed,
    )


def _split_indices_from_payload(payload: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        "train": np.asarray(payload.get("train_indices", []), dtype=np.int64),
        "val": np.asarray(payload.get("val_indices", []), dtype=np.int64),
        "test": np.asarray(payload.get("test_indices", []), dtype=np.int64),
    }


def _syndrome_weights_from_archive(archive: np.lib.npyio.NpzFile, features: np.ndarray, feature_names: list[str]) -> np.ndarray:
    if "syndromes_padded" in archive.files:
        syndromes = np.asarray(archive["syndromes_padded"], dtype=np.float32)
        if "syndrome_mask" in archive.files:
            return (syndromes * np.asarray(archive["syndrome_mask"], dtype=np.float32)).sum(axis=1)
        if "syndrome_lengths" in archive.files:
            lengths = np.asarray(archive["syndrome_lengths"], dtype=np.int64)
            return np.asarray([float(syndromes[idx, : int(lengths[idx])].sum()) for idx in range(len(syndromes))], dtype=np.float32)
        return syndromes.sum(axis=1)
    if "syndromes" in archive.files:
        return np.asarray(archive["syndromes"], dtype=np.float32).reshape(len(features), -1).sum(axis=1)
    if "syndrome_weight" in feature_names:
        return np.asarray(features[:, feature_names.index("syndrome_weight")], dtype=np.float32)
    return np.asarray(features[:, 0], dtype=np.float32)


def _json_safe_metadata_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size <= 1000:
            return value.tolist()
        return {
            "storage": "columnar_npz_array",
            "shape": [int(dim) for dim in value.shape],
            "dtype": str(value.dtype),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe_metadata_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_metadata_value(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe_metadata_value(item) for item in value]
    return value


def _write_metadata_sidecar(path: Path, metadata: dict[str, Any]) -> None:
    sidecar = path.with_suffix(".metadata.json")
    with sidecar.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe_metadata_value(metadata), handle, indent=2)


def _contiguous_boundaries(indices: np.ndarray) -> tuple[int, int] | None:
    if len(indices) == 0:
        return None
    sorted_indices = np.sort(np.asarray(indices, dtype=np.int64))
    if np.array_equal(sorted_indices, np.arange(int(sorted_indices[0]), int(sorted_indices[-1]) + 1)):
        return int(sorted_indices[0]), int(sorted_indices[-1]) + 1
    return None


def _sequence_context_from_metadata(metadata: dict[str, Any], num_samples: int) -> dict[str, np.ndarray]:
    context: dict[str, np.ndarray] = {}
    episode_ids = metadata_field_values(metadata, "episode_id", num_samples)
    setting_ids = metadata_field_values(metadata, "setting_id", num_samples)
    stream_ids = metadata_field_values(metadata, "stream_id", num_samples)
    if episode_ids is not None:
        context["episode_id"] = episode_ids
    if setting_ids is not None:
        context["setting_id"] = setting_ids
    if stream_ids is not None:
        context["stream_id"] = stream_ids
    return context


class ModelRiskDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset wrapper for `data/processed/risk_dataset.npz`.

    Input source: processed NPZ containing shot-level syndrome features,
    labels, runtimes, predictions, feature names, and optional split metadata.
    Output item: tensors for current features, risk label, hard-runtime label,
    accurate runtime, fast-decoder failure labels, and sample id.
    RT-PreQEC role: trains the Risk/Runtime Profiler used by the scheduler.
    Realtime fit: items are shot-level fixed feature vectors; no future context
    is exposed unless wrapped by `HistoryRiskDataset`, which remains causal.
    """

    def __init__(
        self,
        path: str | Path,
        split: str = "train",
        normalization_stats: dict[str, Any] | None = None,
        return_history: bool = False,
        history_length: int = 1,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.path = Path(path)
        archive = np.load(self.path, allow_pickle=False)
        self.features = np.asarray(archive["features"], dtype=np.float32)
        self.labels = np.asarray(archive["labels"])
        self.runtimes = np.asarray(archive["runtimes"], dtype=np.float32)
        self.feature_names = [str(name) for name in archive["feature_names"].tolist()]
        self.label_names = [str(name) for name in archive["label_names"].tolist()]
        self.runtime_names = [str(name) for name in archive["runtime_names"].tolist()]
        self.metadata = json.loads(str(archive["metadata_json"])) if "metadata_json" in archive.files else {}
        self.metadata = _merge_archive_columnar_metadata(archive, self.metadata)
        self.metadata = _merge_metadata_sidecar(self.path, self.metadata)
        self.hard_runtime_label_valid = bool(self.metadata.get("hard_runtime_label_valid", True))
        label_index = {name: idx for idx, name in enumerate(self.label_names)}
        runtime_index = {name: idx for idx, name in enumerate(self.runtime_names)}
        self._fast_wrong_idx = label_index.get("fast_wrong_vs_accurate", 0)
        self._fast_fail_idx = label_index.get("fast_logical_fail", 1)
        self._hard_runtime_idx = label_index.get("hard_runtime", 3)
        self._risk_idx = label_index.get("scheduler_risk_label", self._hard_runtime_idx)
        self._syndrome_tail_idx = label_index.get("syndrome_weight_tail")
        self._accurate_runtime_idx = runtime_index.get("accurate_runtime_us", 0)
        self.normalization_stats = normalization_stats
        self.split_metadata = _load_split_payload(self.path, len(self.features), seed=seed)
        split_indices = _split_indices_from_payload(self.split_metadata)
        split = split.lower()
        self.split = split
        if split == "all":
            self.indices = np.arange(len(self.features), dtype=np.int64)
        elif split in split_indices and len(split_indices[split]) > 0:
            self.indices = split_indices[split]
        else:
            self.indices = np.arange(len(self.features), dtype=np.int64)
        self.split_indices = {
            "train": split_indices["train"],
            "val": split_indices["val"],
            "test": split_indices["test"],
            "all": np.arange(len(self.features), dtype=np.int64),
        }
        self.split_policy = str(self.split_metadata.get("split_policy", "random"))
        self.split_boundaries = dict(self.split_metadata.get("split_boundaries", {}))
        if not self.split_boundaries:
            for name, values in self.split_indices.items():
                if name == "all":
                    continue
                boundary = _contiguous_boundaries(values)
                if boundary is not None:
                    self.split_boundaries[name] = [boundary[0], boundary[1]]
        self.return_history = bool(return_history)
        self.history_length = int(history_length)
        self.sequence_context = _sequence_context_from_metadata(self.metadata, len(self.features))
        self.syndrome_tail = self._load_or_generate_syndrome_tail(archive)

    def _load_or_generate_syndrome_tail(self, archive: np.lib.npyio.NpzFile) -> np.ndarray:
        if self._syndrome_tail_idx is not None:
            values = np.asarray(self.labels[:, int(self._syndrome_tail_idx)], dtype=np.float32)
            self.metadata.setdefault("syndrome_weight_tail_source", "labels")
            return values
        sample_metadata = list(self.metadata.get("samples", []))
        metadata_values: list[float] = []
        for item in sample_metadata:
            raw = dict(item.get("metadata", {})).get("syndrome_weight_tail")
            if raw is None:
                metadata_values = []
                break
            metadata_values.append(float(raw))
        if len(metadata_values) == len(self.features):
            self.metadata.setdefault("syndrome_weight_tail_source", "sample_metadata")
            return np.asarray(metadata_values, dtype=np.float32)

        percentile = float(self.metadata.get("syndrome_weight_tail_percentile", 90.0))
        train_indices = np.asarray(self.split_indices.get("train", []), dtype=np.int64)
        if len(train_indices) == 0:
            train_indices = np.arange(len(self.features), dtype=np.int64)
        syndrome_weights = _syndrome_weights_from_archive(archive, self.features, self.feature_names)
        threshold = float(np.percentile(syndrome_weights[train_indices], percentile)) if len(syndrome_weights) else 0.0
        values = (syndrome_weights >= threshold).astype(np.float32)
        self.metadata["syndrome_weight_tail_source"] = "generated_train_split_percentile"
        self.metadata["syndrome_weight_tail_percentile"] = percentile
        self.metadata["syndrome_weight_tail_threshold"] = threshold
        self.metadata["syndrome_weight_tail_generated_sidecar"] = str(self.path.with_suffix(".metadata.json"))
        try:
            _write_metadata_sidecar(self.path, self.metadata)
        except OSError:
            pass
        return values

    def __len__(self) -> int:
        return int(len(self.indices))

    def _features_at(self, raw_index: int) -> np.ndarray:
        features = self.features[int(raw_index)].astype(np.float32)
        return np.asarray(apply_normalization(features, self.normalization_stats), dtype=np.float32)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        raw_index = int(self.indices[index])
        item: dict[str, torch.Tensor] = {
            "features": torch.tensor(self._features_at(raw_index), dtype=torch.float32),
            "risk_label": torch.tensor(float(self.labels[raw_index, self._risk_idx]), dtype=torch.float32),
            "hard_runtime": torch.tensor(float(self.labels[raw_index, self._hard_runtime_idx]), dtype=torch.float32),
            "syndrome_tail": torch.tensor(float(self.syndrome_tail[raw_index]), dtype=torch.float32),
            "syndrome_weight_tail": torch.tensor(float(self.syndrome_tail[raw_index]), dtype=torch.float32),
            "accurate_runtime_us": torch.tensor(float(self.runtimes[raw_index, self._accurate_runtime_idx]), dtype=torch.float32),
            "fast_wrong": torch.tensor(float(self.labels[raw_index, self._fast_wrong_idx]), dtype=torch.float32),
            "fast_logical_fail": torch.tensor(float(self.labels[raw_index, self._fast_fail_idx]), dtype=torch.float32),
            "safe_fast": torch.tensor(
                float(
                    not bool(self.labels[raw_index, self._fast_wrong_idx])
                    and not bool(self.labels[raw_index, self._fast_fail_idx])
                ),
                dtype=torch.float32,
            ),
            "sample_id": torch.tensor(raw_index, dtype=torch.long),
            "hard_runtime_label_valid": torch.tensor(float(self.hard_runtime_label_valid), dtype=torch.float32),
        }
        if self.return_history:
            history, history_indices = _build_causal_history(
                self.features,
                raw_index,
                self.history_length,
                self.normalization_stats,
                allowed_indices=self.indices,
                sequence_context=self.sequence_context,
                return_indices=True,
            )
            item["history_features"] = torch.tensor(
                history,
                dtype=torch.float32,
            )
            item["history_indices"] = torch.tensor(history_indices, dtype=torch.long)
        return item


def _build_causal_history(
    features: np.ndarray,
    raw_index: int,
    history_length: int,
    normalization_stats: dict[str, Any] | None,
    pad_mode: str = "edge",
    allowed_indices: np.ndarray | list[int] | None = None,
    sequence_context: dict[str, np.ndarray] | None = None,
    return_indices: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    history_length = max(int(history_length), 1)
    raw_index = int(raw_index)
    if allowed_indices is None:
        allowed = np.arange(len(features), dtype=np.int64)
    else:
        allowed = np.asarray(allowed_indices, dtype=np.int64).reshape(-1)
        if len(allowed) == 0:
            allowed = np.asarray([raw_index], dtype=np.int64)
    allowed_set = set(int(idx) for idx in allowed.tolist())
    causal_allowed = np.sort(allowed[allowed <= raw_index])
    if raw_index not in allowed_set:
        causal_allowed = np.asarray([raw_index], dtype=np.int64)
    if sequence_context:
        context_mask = np.ones(len(causal_allowed), dtype=bool)
        for values in sequence_context.values():
            array = np.asarray(values, dtype=object).reshape(-1)
            if raw_index >= len(array):
                continue
            context_mask &= array[causal_allowed] == array[raw_index]
        causal_allowed = causal_allowed[context_mask]
        if len(causal_allowed) == 0 or int(causal_allowed[-1]) != raw_index:
            causal_allowed = np.asarray([raw_index], dtype=np.int64)
    selected = causal_allowed[-history_length:].astype(np.int64) if len(causal_allowed) else np.asarray([raw_index], dtype=np.int64)
    if len(selected) < history_length:
        pad_count = history_length - len(selected)
        if pad_mode == "zero":
            pad_indices = np.full(pad_count, -1, dtype=np.int64)
        else:
            pad_value = int(selected[0]) if len(selected) else raw_index
            pad_indices = np.full(pad_count, pad_value, dtype=np.int64)
        selected = np.concatenate([pad_indices, selected.astype(np.int64)])
    rows: list[np.ndarray] = []
    for source_idx in selected.tolist():
        if int(source_idx) == -1:
            row = np.zeros(features.shape[1], dtype=np.float32)
        else:
            row = features[int(source_idx)].astype(np.float32)
        rows.append(row)
    history = np.stack(rows, axis=0).astype(np.float32)
    history = np.asarray(apply_normalization(history, normalization_stats), dtype=np.float32)
    if return_indices:
        return history, selected.astype(np.int64)
    return history


class HistoryRiskDataset(Dataset[dict[str, torch.Tensor]]):
    """Causal history wrapper for `ModelRiskDataset`.

    Input: a `ModelRiskDataset` and history length `T`.
    Output item: same current labels plus `history_features` `[T,F]` built from
    indices `[i-T+1, ..., i]` with padding at stream start.
    RT-PreQEC role: trains temporal risk/runtime encoders on burst, drift,
    repeated high-density syndrome, and backlog trends.
    Realtime fit: only past and current samples are used; future shots are never
    visible to GRU/LSTM/TCN history encoders.
    """

    def __init__(
        self,
        base_dataset: ModelRiskDataset,
        history_length: int = 1,
        pad_mode: str = "edge",
        split_indices: np.ndarray | list[int] | None = None,
        split_boundaries: dict[str, Any] | None = None,
        allow_cross_split_history: bool = False,
        indices: np.ndarray | list[int] | None = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.history_length = int(history_length)
        self.pad_mode = pad_mode
        self.allow_cross_split_history = bool(allow_cross_split_history)
        self.split_indices = (
            np.asarray(split_indices, dtype=np.int64)
            if split_indices is not None
            else np.asarray(indices, dtype=np.int64)
            if indices is not None
            else np.asarray(base_dataset.indices, dtype=np.int64)
        )
        self.split_boundaries = dict(split_boundaries or getattr(base_dataset, "split_boundaries", {}))
        self.sequence_context = dict(getattr(base_dataset, "sequence_context", {}))

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.base_dataset[index]
        raw_index = int(item["sample_id"].item())
        allowed_indices = None if self.allow_cross_split_history else self.split_indices
        history, history_indices = _build_causal_history(
            self.base_dataset.features,
            raw_index,
            self.history_length,
            self.base_dataset.normalization_stats,
            pad_mode=self.pad_mode,
            allowed_indices=allowed_indices,
            sequence_context=self.sequence_context,
            return_indices=True,
        )
        item["history_features"] = torch.tensor(
            history,
            dtype=torch.float32,
        )
        item["history_indices"] = torch.tensor(history_indices, dtype=torch.long)
        return item

    def assert_no_history_cross_split(self) -> None:
        """Raise if any item history includes a raw index outside this split."""
        if self.allow_cross_split_history:
            return
        allowed = set(int(idx) for idx in np.asarray(self.split_indices, dtype=np.int64).tolist())
        for idx in range(len(self)):
            item = self[idx]
            history_indices = [int(value) for value in item["history_indices"].detach().cpu().tolist()]
            raw_index = int(item["sample_id"].item())
            for history_idx in history_indices:
                if history_idx == -1:
                    continue
                if history_idx not in allowed:
                    raise AssertionError(f"history for raw index {raw_index} crosses split at {history_idx}")


def assert_no_history_cross_split(dataset: HistoryRiskDataset) -> None:
    """Test helper for split-safe temporal history."""
    dataset.assert_no_history_cross_split()


def risk_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate fixed-size risk/runtime tensors into a training batch."""
    keys = batch[0].keys()
    return {key: torch.stack([item[key] for item in batch], dim=0) for key in keys}


def make_risk_dataloader(
    dataset: Dataset[dict[str, torch.Tensor]],
    batch_size: int = 256,
    shuffle: bool = True,
    weighted_sampler: bool = False,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader:
    """Build a DataLoader with optional risk-label weighted sampling."""
    generator = make_torch_generator(seed)
    sampler = None
    if weighted_sampler and len(dataset) > 0:
        labels = torch.stack([dataset[idx]["risk_label"] for idx in range(len(dataset))]).float()
        positives = labels.sum().clamp_min(1.0)
        negatives = (1.0 - labels).sum().clamp_min(1.0)
        weights = torch.where(labels > 0.5, 0.5 / positives, 0.5 / negatives)
        sampler = WeightedRandomSampler(
            weights=weights.double(),
            num_samples=len(weights),
            replacement=True,
            generator=generator,
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=risk_collate_fn,
        generator=generator,
        worker_init_fn=seed_worker,
    )


def split_indices_hashes(dataset: ModelRiskDataset) -> dict[str, str]:
    """Return train/val/test split hashes for checkpoint metadata."""
    return {
        "train_indices_hash": hash_indices(dataset.split_indices.get("train", [])),
        "val_indices_hash": hash_indices(dataset.split_indices.get("val", [])),
        "test_indices_hash": hash_indices(dataset.split_indices.get("test", [])),
    }
