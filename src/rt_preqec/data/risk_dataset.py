"""Risk-profiler dataset utilities."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from rt_preqec.data.patch_extractor import extract_detector_patches_from_flat_syndrome
from rt_preqec.data.risk_features import (
    combine_feature_blocks,
    extract_patch_aggregate_features,
    extract_syndrome_features,
)
from rt_preqec.utils import ensure_parent


@dataclass
class RiskSample:
    """Serializable sample for risk-only scheduling experiments."""

    sample_id: int
    shot_id: int
    syndrome: np.ndarray
    features: np.ndarray
    feature_names: list[str]
    actual_observable: np.ndarray | int | None
    accurate_prediction: np.ndarray | int | None
    fast_prediction: np.ndarray | int | None
    accurate_runtime_us: float
    fast_runtime_us: float
    fast_wrong_vs_accurate: int
    fast_logical_fail: int
    accurate_logical_fail: int
    hard_runtime: int
    scheduler_risk_label: int
    syndrome_weight_tail: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskDatasetSplits:
    """Persistent dataset split metadata."""

    train_indices: list[int]
    val_indices: list[int]
    test_indices: list[int]
    split_seed: int = 42
    split_policy: str = "random"
    seed: int | None = None
    split_boundaries: dict[str, Any] = field(default_factory=dict)
    leakage_safe_for_temporal: bool = False
    fallback_reason: str | None = None
    train_indices_hash: str | None = None
    val_indices_hash: str | None = None
    test_indices_hash: str | None = None

    def __post_init__(self) -> None:
        if self.seed is None:
            self.seed = int(self.split_seed)
        self.train_indices = [int(idx) for idx in self.train_indices]
        self.val_indices = [int(idx) for idx in self.val_indices]
        self.test_indices = [int(idx) for idx in self.test_indices]
        self.split_seed = int(self.seed)
        self.train_indices_hash = self.train_indices_hash or hash_indices(self.train_indices)
        self.val_indices_hash = self.val_indices_hash or hash_indices(self.val_indices)
        self.test_indices_hash = self.test_indices_hash or hash_indices(self.test_indices)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON/NPZ-serializable split metadata."""
        return {
            "train_indices": self.train_indices,
            "val_indices": self.val_indices,
            "test_indices": self.test_indices,
            "split_seed": int(self.split_seed),
            "seed": int(self.seed if self.seed is not None else self.split_seed),
            "split_policy": str(self.split_policy),
            "split_boundaries": dict(self.split_boundaries),
            "leakage_safe_for_temporal": bool(self.leakage_safe_for_temporal),
            "fallback_reason": self.fallback_reason,
            "train_indices_hash": self.train_indices_hash,
            "val_indices_hash": self.val_indices_hash,
            "test_indices_hash": self.test_indices_hash,
        }


def hash_indices(indices: list[int] | np.ndarray) -> str:
    """Hash split indices for checkpoint/dataset compatibility checks."""
    array = np.asarray(indices, dtype=np.int64).reshape(-1)
    return hashlib.sha256(array.tobytes()).hexdigest()[:16]


def risk_dataset_split_sidecar_path(path: str | Path) -> Path:
    """Return the dataset-specific split sidecar path.

    `risk_dataset.npz` keeps the historical `risk_dataset_splits.json` name,
    while custom datasets get their own sidecar, e.g.
    `risk_dataset_v3_16settings_480k_splits.json`.
    """
    dataset_path = Path(path)
    return dataset_path.with_name(f"{dataset_path.stem}_splits.json")


def _split_sizes(
    num_samples: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> tuple[int, int, int]:
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    fractions = np.asarray([train_fraction, val_fraction, test_fraction], dtype=float)
    if np.any(fractions < 0) or float(fractions.sum()) <= 0:
        raise ValueError("split fractions must be non-negative and sum to a positive value")
    fractions = fractions / float(fractions.sum())
    train_size = int(round(num_samples * float(fractions[0])))
    val_size = int(round(num_samples * float(fractions[1])))
    train_size = min(max(train_size, 0), num_samples)
    val_size = min(max(val_size, 0), max(num_samples - train_size, 0))
    test_size = max(num_samples - train_size - val_size, 0)
    if test_size == 0 and num_samples >= 3 and float(fractions[2]) > 0:
        if train_size >= val_size and train_size > 1:
            train_size -= 1
        elif val_size > 1:
            val_size -= 1
        test_size = 1
    if val_size == 0 and num_samples >= 3 and float(fractions[1]) > 0:
        if train_size > 1:
            train_size -= 1
            val_size = 1
        elif test_size > 1:
            test_size -= 1
            val_size = 1
    return train_size, val_size, test_size


def _is_missing_metadata_value(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False


def metadata_field_values(
    metadata: dict[str, Any],
    field_name: str,
    num_samples: int,
) -> np.ndarray | None:
    """Return per-sample metadata values for a field when present.

    Supports the current saved NPZ metadata shape:

        {"samples": [{"metadata": {"setting_id": ...}}, ...]}

    and simple top-level arrays such as ``setting_ids`` for future callers.
    """
    direct_names = [field_name]
    if field_name.endswith("_id"):
        direct_names.append(f"{field_name[:-3]}_ids")
    for name in direct_names:
        direct = metadata.get(name)
        if isinstance(direct, (list, tuple, np.ndarray)) and len(direct) == int(num_samples):
            values = np.asarray(direct, dtype=object)
            return None if all(_is_missing_metadata_value(value) for value in values.tolist()) else values

    sample_metadata = metadata.get("samples", metadata.get("sample_metadata", []))
    if not isinstance(sample_metadata, list) or len(sample_metadata) < int(num_samples):
        return None

    values: list[Any] = []
    found = False
    for idx in range(int(num_samples)):
        item = sample_metadata[idx]
        if not isinstance(item, dict):
            values.append(None)
            continue
        nested = item.get("metadata", {})
        nested = nested if isinstance(nested, dict) else {}
        if field_name in nested:
            values.append(nested.get(field_name))
            found = True
        elif field_name in item:
            values.append(item.get(field_name))
            found = True
        else:
            values.append(None)
    if not found or all(_is_missing_metadata_value(value) for value in values):
        return None
    return np.asarray(values, dtype=object)


def _stream_block_split(
    num_samples: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    *,
    split_policy: str = "stream_block",
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    train_size, val_size, test_size = _split_sizes(num_samples, train_fraction, val_fraction, test_fraction)
    train_end = train_size
    val_end = train_size + val_size
    train_indices = list(range(0, train_end))
    val_indices = list(range(train_end, val_end))
    test_indices = list(range(val_end, val_end + test_size))
    return RiskDatasetSplits(
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        split_seed=int(seed),
        split_policy=split_policy,
        seed=int(seed),
        split_boundaries={
            "train": [0, train_end],
            "val": [train_end, val_end],
            "test": [val_end, val_end + test_size],
            "effective_split_policy": "stream_block",
        },
        leakage_safe_for_temporal=True,
        fallback_reason=fallback_reason,
    ).to_dict()


def _group_sort_key(value: Any) -> tuple[str, str]:
    if isinstance(value, np.generic):
        value = value.item()
    return (type(value).__name__, str(value))


def _setting_stratified_split(
    num_samples: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    setting_ids: np.ndarray | list[Any] | None,
) -> dict[str, Any]:
    if setting_ids is None or len(setting_ids) != int(num_samples):
        raise ValueError("setting_stratified split requires setting_id metadata")

    setting_array = np.asarray(setting_ids, dtype=object).reshape(-1)
    if any(_is_missing_metadata_value(value) for value in setting_array.tolist()):
        raise ValueError("setting_stratified split requires setting_id metadata for every sample")

    groups: dict[Any, list[int]] = {}
    for idx, setting_id in enumerate(setting_array.tolist()):
        key = setting_id.item() if isinstance(setting_id, np.generic) else setting_id
        try:
            groups.setdefault(key, []).append(int(idx))
        except TypeError:
            groups.setdefault(str(key), []).append(int(idx))

    positive_splits = sum(float(value) > 0.0 for value in (train_fraction, val_fraction, test_fraction))
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    per_setting: dict[str, dict[str, int]] = {}

    for setting_id in sorted(groups, key=_group_sort_key):
        ids = sorted(groups[setting_id])
        n = len(ids)
        if n < positive_splits:
            raise ValueError(
                "setting_stratified split requires enough samples per setting "
                f"to populate non-zero splits; setting_id {setting_id!r} has {n} samples"
            )
        n_train, n_val, _ = _split_sizes(n, train_fraction, val_fraction, test_fraction)
        train_part = ids[:n_train]
        val_part = ids[n_train : n_train + n_val]
        test_part = ids[n_train + n_val :]
        if float(train_fraction) > 0.0 and not train_part:
            raise ValueError(f"setting_id {setting_id!r} produced an empty train split")
        if float(val_fraction) > 0.0 and not val_part:
            raise ValueError(f"setting_id {setting_id!r} produced an empty val split")
        if float(test_fraction) > 0.0 and not test_part:
            raise ValueError(f"setting_id {setting_id!r} produced an empty test split")
        train_indices.extend(train_part)
        val_indices.extend(val_part)
        test_indices.extend(test_part)
        per_setting[str(setting_id)] = {
            "total": int(n),
            "train": int(len(train_part)),
            "val": int(len(val_part)),
            "test": int(len(test_part)),
        }

    return RiskDatasetSplits(
        train_indices=sorted(train_indices),
        val_indices=sorted(val_indices),
        test_indices=sorted(test_indices),
        split_seed=int(seed),
        split_policy="setting_stratified",
        seed=int(seed),
        split_boundaries={
            "settings": per_setting,
            "effective_split_policy": "setting_stratified",
        },
        leakage_safe_for_temporal=True,
    ).to_dict()


def create_split_indices(
    num_samples: int,
    split_policy: str,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    episode_ids: np.ndarray | None = None,
    setting_ids: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create train/val/test indices under a paper-safe split protocol."""
    policy = str(split_policy or "random").lower()
    if policy not in {"random", "stream_block", "episode", "setting_stratified"}:
        raise ValueError(f"Unsupported split_policy: {split_policy}")
    if policy == "stream_block":
        return _stream_block_split(num_samples, train_fraction, val_fraction, test_fraction, seed)
    if policy == "setting_stratified":
        if setting_ids is None and metadata is not None:
            setting_ids = metadata_field_values(metadata, "setting_id", num_samples)
        return _setting_stratified_split(
            num_samples,
            train_fraction,
            val_fraction,
            test_fraction,
            seed,
            setting_ids,
        )
    if policy == "random":
        train_size, val_size, test_size = _split_sizes(num_samples, train_fraction, val_fraction, test_fraction)
        rng = np.random.default_rng(seed)
        indices = np.arange(num_samples, dtype=np.int64)
        rng.shuffle(indices)
        train_indices = indices[:train_size]
        val_indices = indices[train_size : train_size + val_size]
        test_indices = indices[train_size + val_size : train_size + val_size + test_size]
        return RiskDatasetSplits(
            train_indices=train_indices.astype(np.int64).tolist(),
            val_indices=val_indices.astype(np.int64).tolist(),
            test_indices=test_indices.astype(np.int64).tolist(),
            split_seed=int(seed),
            split_policy="random",
            seed=int(seed),
            split_boundaries={
                "train_size": int(len(train_indices)),
                "val_size": int(len(val_indices)),
                "test_size": int(len(test_indices)),
                "effective_split_policy": "random",
            },
            leakage_safe_for_temporal=False,
        ).to_dict()

    if episode_ids is None and metadata is not None:
        episode_ids = metadata_field_values(metadata, "episode_id", num_samples)
    if episode_ids is None or len(episode_ids) != int(num_samples):
        return _stream_block_split(
            num_samples,
            train_fraction,
            val_fraction,
            test_fraction,
            seed,
            split_policy="episode",
            fallback_reason="episode_ids_missing_or_wrong_length",
        )

    episode_array = np.asarray(episode_ids)
    first_positions: dict[str, int] = {}
    grouped: dict[str, list[int]] = {}
    for idx, episode_id in enumerate(episode_array.tolist()):
        key = str(episode_id)
        grouped.setdefault(key, []).append(int(idx))
        first_positions.setdefault(key, int(idx))
    ordered_episodes = sorted(grouped, key=lambda item: first_positions[item])
    target_train, target_val, _ = _split_sizes(num_samples, train_fraction, val_fraction, test_fraction)
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    train_episodes: list[str] = []
    val_episodes: list[str] = []
    test_episodes: list[str] = []
    for episode in ordered_episodes:
        target = grouped[episode]
        if len(train_indices) < target_train:
            train_indices.extend(target)
            train_episodes.append(episode)
        elif len(val_indices) < target_val:
            val_indices.extend(target)
            val_episodes.append(episode)
        else:
            test_indices.extend(target)
            test_episodes.append(episode)
    return RiskDatasetSplits(
        train_indices=sorted(train_indices),
        val_indices=sorted(val_indices),
        test_indices=sorted(test_indices),
        split_seed=int(seed),
        split_policy="episode",
        seed=int(seed),
        split_boundaries={
            "train_episodes": train_episodes,
            "val_episodes": val_episodes,
            "test_episodes": test_episodes,
            "effective_split_policy": "episode",
        },
        leakage_safe_for_temporal=True,
    ).to_dict()


def splits_from_dict(payload: dict[str, Any]) -> RiskDatasetSplits:
    """Build a RiskDatasetSplits object from old or new metadata."""
    return RiskDatasetSplits(
        train_indices=[int(idx) for idx in payload.get("train_indices", [])],
        val_indices=[int(idx) for idx in payload.get("val_indices", [])],
        test_indices=[int(idx) for idx in payload.get("test_indices", [])],
        split_seed=int(payload.get("split_seed", payload.get("seed", 42))),
        split_policy=str(payload.get("split_policy", "random")),
        seed=int(payload.get("seed", payload.get("split_seed", 42))),
        split_boundaries=dict(payload.get("split_boundaries", {})),
        leakage_safe_for_temporal=bool(payload.get("leakage_safe_for_temporal", False)),
        fallback_reason=payload.get("fallback_reason"),
        train_indices_hash=payload.get("train_indices_hash"),
        val_indices_hash=payload.get("val_indices_hash"),
        test_indices_hash=payload.get("test_indices_hash"),
    )


class RiskProfilerDataset(Dataset[dict[str, torch.Tensor]]):
    """PyTorch dataset for risk-only profiling."""

    def __init__(self, samples: list[RiskSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        return {
            "features": torch.tensor(sample.features, dtype=torch.float32),
            "risk_label": torch.tensor(float(sample.scheduler_risk_label), dtype=torch.float32),
            "hard_runtime": torch.tensor(float(sample.hard_runtime), dtype=torch.float32),
            "fast_wrong": torch.tensor(float(sample.fast_wrong_vs_accurate), dtype=torch.float32),
            "fast_logical_fail": torch.tensor(float(sample.fast_logical_fail), dtype=torch.float32),
            "accurate_runtime_us": torch.tensor(float(sample.accurate_runtime_us), dtype=torch.float32),
        }


def _to_array(value: np.ndarray | int | None) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=np.int8)
    return np.asarray(value, dtype=np.int8).reshape(-1)


def _prediction_equal(left: np.ndarray | int | None, right: np.ndarray | int | None) -> bool:
    return bool(np.array_equal(_to_array(left), _to_array(right)))


def pad_1d_arrays(
    arrays: list[np.ndarray],
    pad_value: int = 0,
    dtype: np.dtype | type = np.int8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad variable-length 1D arrays into a dense matrix.

    Risk datasets may mix Stim circuits with different code distances or
    rounds. Those settings can produce different numbers of detectors, so
    flattened syndrome vectors do not always have the same length.

    Returns:
        padded: Array with shape [N, max_len].
        lengths: Original 1D lengths with shape [N].
        mask: Boolean mask with shape [N, max_len], True for valid entries.
    """
    flat_arrays = [np.asarray(arr, dtype=dtype).reshape(-1) for arr in arrays]
    lengths = np.asarray([len(arr) for arr in flat_arrays], dtype=np.int32)
    max_len = int(lengths.max()) if len(lengths) else 0

    padded = np.full((len(flat_arrays), max_len), pad_value, dtype=dtype)
    mask = np.zeros((len(flat_arrays), max_len), dtype=bool)

    for idx, arr in enumerate(flat_arrays):
        length = int(lengths[idx])
        if length:
            padded[idx, :length] = arr
            mask[idx, :length] = True

    return padded, lengths, mask


def _restore_padded_1d(
    padded: np.ndarray,
    lengths: np.ndarray | None,
    index: int,
    dtype: np.dtype | type = np.int8,
) -> np.ndarray:
    """Restore one unpadded 1D array from padded storage."""
    row = np.asarray(padded[index], dtype=dtype).reshape(-1)
    if lengths is None:
        return row
    return row[: int(lengths[index])]


def save_risk_dataset(
    samples: list[RiskSample],
    path: str | Path,
    splits: RiskDatasetSplits | dict[str, Any] | None = None,
    metadata_extra: dict[str, Any] | None = None,
) -> None:
    """Save risk samples to a compressed NPZ file.

    Raw flat syndromes may have variable length when the dataset mixes multiple
    Stim circuit settings, e.g., different code distances or different rounds.
    Therefore syndromes are stored with padded representation:

        syndromes_padded: [N, max_syndrome_len]
        syndrome_lengths: [N]
        syndrome_mask: [N, max_syndrome_len]

    For backward compatibility, a `syndromes` key is also saved and points to
    the padded matrix. New loaders should prefer syndromes_padded + lengths.
    """
    if not samples:
        raise ValueError("Cannot save an empty risk dataset.")

    target = ensure_parent(path)

    feature_names = np.asarray(samples[0].feature_names, dtype="<U128")
    features = np.stack([np.asarray(sample.features, dtype=np.float32) for sample in samples])

    syndrome_arrays = [
        np.asarray(sample.syndrome, dtype=np.int8).reshape(-1)
        for sample in samples
    ]
    syndromes_padded, syndrome_lengths, syndrome_mask = pad_1d_arrays(
        syndrome_arrays,
        pad_value=0,
        dtype=np.int8,
    )
    unique_syndrome_lengths = sorted({int(length) for length in syndrome_lengths.tolist()})
    variable_length_syndromes = len(unique_syndrome_lengths) > 1

    actual = np.stack([_to_array(sample.actual_observable) for sample in samples])
    accurate = np.stack([_to_array(sample.accurate_prediction) for sample in samples])
    fast = np.stack([_to_array(sample.fast_prediction) for sample in samples])

    labels = np.asarray(
        [
            [
                sample.fast_wrong_vs_accurate,
                sample.fast_logical_fail,
                sample.accurate_logical_fail,
                sample.hard_runtime,
                sample.scheduler_risk_label,
                sample.syndrome_weight_tail,
            ]
            for sample in samples
        ],
        dtype=np.int8,
    )
    label_names = np.asarray(
        [
            "fast_wrong_vs_accurate",
            "fast_logical_fail",
            "accurate_logical_fail",
            "hard_runtime",
            "scheduler_risk_label",
            "syndrome_weight_tail",
        ],
        dtype="<U64",
    )

    runtimes = np.asarray(
        [[sample.accurate_runtime_us, sample.fast_runtime_us] for sample in samples],
        dtype=np.float32,
    )
    runtime_names = np.asarray(["accurate_runtime_us", "fast_runtime_us"], dtype="<U64")

    split_payload = splits.to_dict() if isinstance(splits, RiskDatasetSplits) else dict(splits or {})
    hard_runtime_valid_values = [
        bool(sample.metadata.get("hard_runtime_label_valid", True))
        for sample in samples
    ]

    metadata = {
        "num_samples": len(samples),
        "feature_dim": int(features.shape[1]),
        "feature_names": samples[0].feature_names,
        "hard_runtime_label_valid": bool(all(hard_runtime_valid_values)),
        "splits": split_payload,
        "syndrome_storage": "padded",
        "variable_length_syndromes": bool(variable_length_syndromes),
        "max_syndrome_len": int(syndromes_padded.shape[1]) if syndromes_padded.ndim == 2 else 0,
        "min_syndrome_len": int(min(unique_syndrome_lengths)) if unique_syndrome_lengths else 0,
        "unique_syndrome_lengths": unique_syndrome_lengths,
        "samples": [
            {
                "sample_id": sample.sample_id,
                "shot_id": sample.shot_id,
                "metadata": sample.metadata,
            }
            for sample in samples
        ],
    }
    metadata.update(dict(metadata_extra or {}))

    np.savez_compressed(
        target,
        features=features,
        # Backward-compatible key. This matrix is padded when syndrome lengths differ.
        syndromes=syndromes_padded,
        # Preferred variable-length storage.
        syndromes_padded=syndromes_padded,
        syndrome_lengths=syndrome_lengths,
        syndrome_mask=syndrome_mask,
        syndrome_storage=np.asarray("padded"),
        actual_observables=actual,
        accurate_predictions=accurate,
        fast_predictions=fast,
        labels=labels,
        label_names=label_names,
        runtimes=runtimes,
        runtime_names=runtime_names,
        feature_names=feature_names,
        metadata_json=json.dumps(metadata),
    )

def save_risk_dataset_splits(splits: RiskDatasetSplits, path: str | Path) -> None:
    """Persist split metadata as JSON."""
    target = ensure_parent(path)
    payload = splits.to_dict()
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_risk_dataset_splits(path: str | Path) -> RiskDatasetSplits:
    """Load split metadata JSON."""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return splits_from_dict(payload)


def load_risk_dataset_metadata(path: str | Path) -> dict[str, Any]:
    """Load metadata JSON from a risk dataset NPZ."""
    archive = np.load(Path(path), allow_pickle=False)
    metadata = json.loads(str(archive["metadata_json"])) if "metadata_json" in archive.files else {}
    for key in [
        "setting_ids",
        "episode_ids",
        "stream_ids",
        "shot_index_within_setting",
        "global_index",
        "arrival_order",
        "difficulty_tier_ids",
    ]:
        if key in archive.files and key not in metadata:
            metadata[key] = np.asarray(archive[key])
    return metadata


def load_risk_dataset(path: str | Path) -> list[RiskSample]:
    """Load a saved risk dataset from NPZ.

    Supports both old fixed-shape format:

        syndromes

    and new variable-length padded format:

        syndromes_padded
        syndrome_lengths
        syndrome_mask
    """
    archive = np.load(Path(path), allow_pickle=False)
    features = archive["features"]

    if "syndromes_padded" in archive.files:
        syndromes_padded = archive["syndromes_padded"]
        syndrome_lengths = archive["syndrome_lengths"] if "syndrome_lengths" in archive.files else None
    elif "syndromes" in archive.files:
        syndromes_padded = archive["syndromes"]
        syndrome_lengths = None
    else:
        raise ValueError(
            "risk dataset is missing syndrome storage. Expected 'syndromes' "
            "or 'syndromes_padded' + 'syndrome_lengths'."
        )

    actual = archive["actual_observables"]
    accurate = archive["accurate_predictions"]
    fast = archive["fast_predictions"]
    labels = archive["labels"]
    runtimes = archive["runtimes"]
    feature_names = [str(name) for name in archive["feature_names"].tolist()]
    metadata = json.loads(str(archive["metadata_json"])) if "metadata_json" in archive.files else {}
    label_index = {str(name): idx for idx, name in enumerate(archive["label_names"].tolist())}
    columnar_metadata = {
        "setting_id": np.asarray(archive["setting_ids"]).tolist() if "setting_ids" in archive.files else None,
        "episode_id": np.asarray(archive["episode_ids"]).tolist() if "episode_ids" in archive.files else None,
        "stream_id": np.asarray(archive["stream_ids"]).tolist() if "stream_ids" in archive.files else None,
        "shot_index_within_setting": np.asarray(archive["shot_index_within_setting"]).tolist()
        if "shot_index_within_setting" in archive.files
        else None,
        "global_index": np.asarray(archive["global_index"]).tolist() if "global_index" in archive.files else None,
        "arrival_order": np.asarray(archive["arrival_order"]).tolist() if "arrival_order" in archive.files else None,
    }

    samples: list[RiskSample] = []
    sample_metadata = metadata.get("samples", [])
    for idx in range(len(features)):
        item_meta = sample_metadata[idx] if idx < len(sample_metadata) else {}
        merged_metadata = dict(item_meta.get("metadata", {}))
        for field_name, values in columnar_metadata.items():
            if values is not None and field_name not in merged_metadata:
                merged_metadata[field_name] = values[idx]
        syndrome = _restore_padded_1d(
            syndromes_padded,
            syndrome_lengths,
            idx,
            dtype=np.int8,
        )
        samples.append(
            RiskSample(
                sample_id=int(item_meta.get("sample_id", idx)),
                shot_id=int(item_meta.get("shot_id", idx)),
                syndrome=syndrome,
                features=np.asarray(features[idx], dtype=np.float32),
                feature_names=feature_names,
                actual_observable=np.asarray(actual[idx], dtype=np.int8),
                accurate_prediction=np.asarray(accurate[idx], dtype=np.int8),
                fast_prediction=np.asarray(fast[idx], dtype=np.int8),
                accurate_runtime_us=float(runtimes[idx, 0]),
                fast_runtime_us=float(runtimes[idx, 1]),
                fast_wrong_vs_accurate=int(labels[idx, label_index["fast_wrong_vs_accurate"]]),
                fast_logical_fail=int(labels[idx, label_index["fast_logical_fail"]]),
                accurate_logical_fail=int(labels[idx, label_index["accurate_logical_fail"]]),
                hard_runtime=int(labels[idx, label_index["hard_runtime"]]),
                scheduler_risk_label=int(labels[idx, label_index["scheduler_risk_label"]]),
                syndrome_weight_tail=int(labels[idx, label_index.get("syndrome_weight_tail", -1)])
                if "syndrome_weight_tail" in label_index
                else int(merged_metadata.get("syndrome_weight_tail", 0)),
                metadata=merged_metadata,
            )
        )
    return samples

def build_risk_samples_from_decoding_records(
    records: list[dict[str, Any]],
    feature_extractor_config: dict[str, Any] | None = None,
) -> list[RiskSample]:
    """Build risk samples from decoding records emitted by dataset builders."""
    feature_extractor_config = feature_extractor_config or {}
    hard_percentile = float(feature_extractor_config.get("hard_runtime_percentile", 90.0))
    hard_runtime_label_valid = bool(feature_extractor_config.get("hard_runtime_label_valid", True))
    syndrome_tail_percentile = float(feature_extractor_config.get("syndrome_weight_tail_percentile", 90.0))
    combined_definition = list(
        feature_extractor_config.get(
            "combined_definition",
            ["fast_wrong_vs_accurate", "fast_logical_fail", "hard_runtime"],
        )
    )
    runtime_values = np.asarray(
        [float(record.get("accurate_runtime_us", 0.0)) for record in records],
        dtype=float,
    )
    threshold = (
        float(np.percentile(runtime_values, hard_percentile))
        if runtime_values.size and hard_runtime_label_valid
        else float("inf")
    )
    syndrome_weights = np.asarray(
        [float(np.asarray(record["syndrome"], dtype=np.int8).reshape(-1).sum()) for record in records],
        dtype=float,
    )
    syndrome_tail_threshold = (
        float(np.percentile(syndrome_weights, syndrome_tail_percentile))
        if syndrome_weights.size
        else 0.0
    )
    samples: list[RiskSample] = []
    for sample_id, record in enumerate(records):
        syndrome = np.asarray(record["syndrome"], dtype=np.int8).reshape(-1)
        layout = record.get("layout")
        candidates_by_detector = record.get("candidates_by_detector")
        if record.get("patches") is not None:
            patches = list(record["patches"])
        elif layout is not None:
            patches = extract_detector_patches_from_flat_syndrome(
                syndrome,
                layout=layout,
                patch_radius=float(feature_extractor_config.get("patch_radius", 2.5)),
                time_radius=feature_extractor_config.get("time_radius"),
                active_only=True,
                max_patches=feature_extractor_config.get("max_patches"),
                shot_id=int(record.get("shot_id", sample_id)),
            )
        else:
            patches = []
        syndrome_features, syndrome_names = extract_syndrome_features(
            syndrome,
            layout=layout,
            candidates_by_detector=candidates_by_detector,
        )
        patch_features, patch_names = extract_patch_aggregate_features(patches)
        features, feature_names = combine_feature_blocks(
            (syndrome_features, syndrome_names),
            (patch_features, patch_names),
        )
        actual = record.get("actual_observable")
        accurate_prediction = record.get("accurate_prediction")
        fast_prediction = record.get("fast_prediction")
        accurate_runtime_us = float(record.get("accurate_runtime_us", 0.0))
        fast_runtime_us = float(record.get("fast_runtime_us", 0.0))
        fast_wrong = int(not _prediction_equal(fast_prediction, accurate_prediction))
        fast_fail = int(not _prediction_equal(fast_prediction, actual))
        accurate_fail = int(not _prediction_equal(accurate_prediction, actual))
        hard_runtime = int(accurate_runtime_us > threshold) if hard_runtime_label_valid else 0
        syndrome_weight_tail = int(float(syndrome.sum()) >= syndrome_tail_threshold)
        component_values = {
            "fast_wrong_vs_accurate": fast_wrong,
            "fast_logical_fail": fast_fail,
            "hard_runtime": hard_runtime,
            "syndrome_weight_tail": syndrome_weight_tail,
        }
        scheduler_risk_label = int(any(bool(component_values.get(name, 0)) for name in combined_definition))
        samples.append(
            RiskSample(
                sample_id=sample_id,
                shot_id=int(record.get("shot_id", sample_id)),
                syndrome=syndrome,
                features=features,
                feature_names=feature_names,
                actual_observable=None if actual is None else np.asarray(actual, dtype=np.int8),
                accurate_prediction=None if accurate_prediction is None else np.asarray(accurate_prediction, dtype=np.int8),
                fast_prediction=None if fast_prediction is None else np.asarray(fast_prediction, dtype=np.int8),
                accurate_runtime_us=accurate_runtime_us,
                fast_runtime_us=fast_runtime_us,
                fast_wrong_vs_accurate=fast_wrong,
                fast_logical_fail=fast_fail,
                accurate_logical_fail=accurate_fail,
                hard_runtime=hard_runtime,
                scheduler_risk_label=scheduler_risk_label,
                syndrome_weight_tail=syndrome_weight_tail,
                metadata={
                    **dict(record.get("metadata", {})),
                    "hard_runtime_threshold_us": threshold,
                    "hard_runtime_label_valid": hard_runtime_label_valid,
                    "syndrome_weight_tail": syndrome_weight_tail,
                    "syndrome_weight_tail_threshold": syndrome_tail_threshold,
                    "risk_label_components": component_values,
                    "num_patches": len(patches),
                },
            )
        )
    return samples
