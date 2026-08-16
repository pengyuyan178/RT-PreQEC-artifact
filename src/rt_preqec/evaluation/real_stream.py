"""Paired real-stream evaluation harness over shared Stim shots."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import copy
import heapq
import json

import numpy as np
import pandas as pd
import torch

from rt_preqec.config import ProjectConfig
from rt_preqec.data.dem_parser import (
    filter_local_candidates,
    index_candidates_by_detector,
    parse_dem_error_candidates,
)
from rt_preqec.data.patch_extractor import extract_detector_patches_from_flat_syndrome
from rt_preqec.data.risk_features import (
    combine_feature_blocks,
    extract_patch_aggregate_features,
    extract_syndrome_features,
)
from rt_preqec.data.risk_dataset import (
    hash_indices,
    load_risk_dataset_metadata,
    risk_dataset_split_sidecar_path,
)
from rt_preqec.data.stim_surface_code import generate_surface_code_samples
from rt_preqec.decoders.base import DecodeResult
from rt_preqec.decoders.lookup_decoder import LookupDecoder
from rt_preqec.decoders.pymatching_decoder import (
    PyMatchingDecoder,
    measure_per_shot_decoder_latency,
)
from rt_preqec.decoders.union_find_decoder import UnionFindDecoder
from rt_preqec.metrics.aggregation import (
    compare_modes_summary,
    save_metrics_json,
    save_summary_metrics_csv,
)
from rt_preqec.metrics.predecoder_metrics import (
    abstention_rate,
    accepted_error_rate,
    accept_rate,
    false_accept_rate,
    validation_pass_rate,
)
from rt_preqec.models.risk_profiler import load_risk_profiler_checkpoint, predict_risk_scores
from rt_preqec.models.sequence_builder import build_causal_history_matrix
from rt_preqec.utils import ensure_parent


@dataclass
class RealStreamShotRecord:
    shot_id: int
    syndrome: np.ndarray
    observable: np.ndarray | int
    accurate_prediction: np.ndarray | int
    fast_prediction: np.ndarray | int
    accurate_latency_us: float
    fast_latency_us: float
    features: np.ndarray
    feature_names: list[str]
    risk_label: int
    hard_runtime: int
    fast_wrong_vs_accurate: int
    fast_logical_fail: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModeEvaluationResult:
    mode: str
    metrics: dict[str, Any]
    events: pd.DataFrame
    decisions: pd.DataFrame
    predictions: pd.DataFrame


@dataclass
class OrderedCommitState:
    """Track out-of-order completions and advance only a contiguous commit prefix."""

    completed: list[bool] = field(default_factory=list)
    committed_prefix: int = -1
    pending_completions: list[tuple[float, int]] = field(default_factory=list)

    def advance_to(self, time_us: float) -> int:
        while self.pending_completions and self.pending_completions[0][0] <= float(time_us):
            _, job_id = heapq.heappop(self.pending_completions)
            self.completed[job_id] = True
        while (
            self.committed_prefix + 1 < len(self.completed)
            and self.completed[self.committed_prefix + 1]
        ):
            self.committed_prefix += 1
        return self.committed_prefix

    def lag_at_arrival(self, job_id: int, arrival_time_us: float) -> int:
        if int(job_id) != len(self.completed):
            raise ValueError(
                "ordered-commit evaluation requires sequential job IDs "
                f"0..N-1; expected {len(self.completed)}, got {job_id}"
            )
        self.advance_to(arrival_time_us)
        return int(job_id) - self.committed_prefix - 1

    def schedule_completion(self, job_id: int, completion_time_us: float) -> None:
        if int(job_id) != len(self.completed):
            raise ValueError(
                "ordered-commit completion must be scheduled once in job-ID order; "
                f"expected {len(self.completed)}, got {job_id}"
            )
        self.completed.append(False)
        heapq.heappush(
            self.pending_completions,
            (float(completion_time_us), int(job_id)),
        )


def _decoder_from_name(name: str, bundle: dict[str, Any]) -> Any:
    if name == "pymatching":
        return PyMatchingDecoder.from_detector_error_model(bundle.get("dem"))
    if name == "union_find":
        return UnionFindDecoder()
    return LookupDecoder()


def _observable_array(value: np.ndarray | int) -> np.ndarray:
    return np.asarray(value, dtype=np.int8).reshape(-1)


def _prediction_from_decode_result(result: DecodeResult) -> np.ndarray:
    correction = np.asarray(result.correction, dtype=np.int8).reshape(-1)
    if correction.size == 0:
        return np.zeros((1,), dtype=np.int8)
    if correction.size == 1:
        return correction.astype(np.int8)
    return np.asarray([int(correction.sum() % 2)], dtype=np.int8)


def _build_feature_vector(
    shot: np.ndarray,
    layout: Any,
    candidates_by_detector: dict[int, list[Any]] | None,
    shot_id: int,
) -> tuple[np.ndarray, list[str]]:
    syndrome_features, syndrome_names = extract_syndrome_features(
        shot,
        layout=layout,
        candidates_by_detector=candidates_by_detector,
    )
    patches = (
        extract_detector_patches_from_flat_syndrome(
            shot,
            layout=layout,
            patch_radius=2.5,
            time_radius=1.0,
            active_only=True,
            shot_id=shot_id,
        )
        if layout is not None
        else []
    )
    patch_features, patch_names = extract_patch_aggregate_features(patches)
    return combine_feature_blocks(
        (syndrome_features, syndrome_names), (patch_features, patch_names)
    )


def _jsonable_prediction(value: np.ndarray | int) -> Any:
    array = np.asarray(value, dtype=np.int8).reshape(-1)
    if array.size == 1:
        return int(array[0])
    return array.tolist()


def build_real_stream_records(
    config: ProjectConfig,
) -> tuple[list[RealStreamShotRecord], dict[str, Any]]:
    """Build shared shot records from real Stim data or explicit fallback."""
    bundle = generate_surface_code_samples(config)
    syndrome = np.asarray(bundle["syndrome"], dtype=np.int8)
    observables = np.asarray(bundle["observables"], dtype=np.int8)
    if syndrome.ndim != 2:
        syndrome = syndrome.reshape(syndrome.shape[0], -1)
    if observables.ndim == 1:
        observables = observables.reshape(-1, 1)
    trace_contexts = _generated_trace_contexts(config, len(syndrome))
    layout = bundle.get("layout")
    all_candidates = parse_dem_error_candidates(bundle.get("dem"), layout=layout)
    local_candidates = filter_local_candidates(
        all_candidates,
        max_spatial_diameter=4.0,
        max_time_diameter=2.0,
        allow_observable_flip=False,
    )
    candidates_by_detector = index_candidates_by_detector(local_candidates)
    accurate_decoder = _decoder_from_name(config.decoders.accurate, bundle)
    fast_decoder = _decoder_from_name(config.decoders.fast, bundle)
    accurate_batch_predictions = None
    accurate_batch_metadata: dict[str, Any] = {"timing_mode": "measured_loop_per_shot"}
    if hasattr(accurate_decoder, "decode_batch"):
        try:
            accurate_batch_predictions, accurate_batch_metadata = accurate_decoder.decode_batch(
                syndrome
            )
        except Exception as exc:
            accurate_batch_predictions = None
            accurate_batch_metadata = {
                "placeholder": True,
                "reason": f"decode_batch_failed:{exc}",
                "timing_mode": "measured_loop_per_shot",
            }
    accurate_loop_latencies: list[float] = []
    accurate_predictions: list[np.ndarray] = []
    accurate_result_metadata: list[dict[str, Any]] = []
    for shot in syndrome:
        accurate_result = accurate_decoder.decode(shot)
        accurate_loop_latencies.append(float(accurate_result.latency_us))
        accurate_predictions.append(_prediction_from_decode_result(accurate_result))
        accurate_result_metadata.append(dict(accurate_result.metadata))
    timing_mode = "loop_per_shot"
    hard_runtime_label_valid = True
    accurate_latency_estimate = np.asarray(accurate_loop_latencies, dtype=float)
    timing_metadata: dict[str, Any] = {
        "timing_mode": "loop_per_shot",
        "hard_runtime_label_valid": True,
    }
    if config.timing.use_loop_timing_for_runtime_label:
        accurate_latency_estimate, timing_metadata = measure_per_shot_decoder_latency(
            accurate_decoder,
            syndrome,
            warmup_shots=int(config.timing.warmup_shots),
            repeat_per_shot=int(config.timing.repeat_per_shot),
            max_timing_shots=config.timing.max_timing_shots,
            statistic=str(config.timing.timing_statistic),
        )
        timing_mode = str(timing_metadata.get("timing_mode", "loop_per_shot"))
        hard_runtime_label_valid = bool(timing_metadata.get("hard_runtime_label_valid", True))
    if accurate_batch_predictions is not None and len(syndrome) > 0:
        accurate_batch_array = np.asarray(accurate_batch_predictions, dtype=np.int8)
        accurate_predictions = [
            np.asarray(accurate_batch_array[idx], dtype=np.int8).reshape(-1)
            for idx in range(len(accurate_batch_array))
        ]
    fast_predictions: list[np.ndarray] = []
    fast_latencies: list[float] = []
    fast_metadata: list[dict[str, Any]] = []
    for shot in syndrome:
        fast_result = fast_decoder.decode(shot)
        fast_predictions.append(_prediction_from_decode_result(fast_result))
        fast_latencies.append(float(fast_result.latency_us))
        fast_metadata.append(dict(fast_result.metadata))
    hard_threshold = (
        float(np.percentile(accurate_latency_estimate, config.risk_eval.hard_runtime_percentile))
        if len(accurate_latency_estimate)
        else 0.0
    )
    records: list[RealStreamShotRecord] = []
    for shot_id, shot in enumerate(syndrome):
        trace_context = trace_contexts[shot_id] if shot_id < len(trace_contexts) else {}
        observable = np.asarray(observables[shot_id], dtype=np.int8).reshape(-1)
        accurate_prediction = np.asarray(accurate_predictions[shot_id], dtype=np.int8).reshape(-1)
        fast_prediction = np.asarray(fast_predictions[shot_id], dtype=np.int8).reshape(-1)
        features, feature_names = _build_feature_vector(
            shot, layout, candidates_by_detector, shot_id
        )
        generated_feature_values = {
            key: float(trace_context[key])
            for key in ["burst_context", "residual_or_candidate_complexity", "backlog_proxy"]
            if key in trace_context
        }
        if generated_feature_values:
            features, feature_names = _append_or_replace_features(
                features, feature_names, generated_feature_values
            )
        accurate_latency_us = float(accurate_latency_estimate[shot_id]) * float(
            trace_context.get("accurate_runtime_multiplier", 1.0)
        )
        fast_latency_us = float(fast_latencies[shot_id]) * float(
            trace_context.get("fast_runtime_multiplier", 1.0)
        )
        fast_wrong = int(not np.array_equal(fast_prediction, accurate_prediction))
        fast_fail = int(np.any(fast_prediction != observable))
        accurate_fail = int(np.any(accurate_prediction != observable))
        hard_runtime = int(accurate_latency_us > hard_threshold) if hard_runtime_label_valid else 0
        risk_label = int(bool(fast_wrong or fast_fail or hard_runtime))
        records.append(
            RealStreamShotRecord(
                shot_id=shot_id,
                syndrome=np.asarray(shot, dtype=np.int8).reshape(-1),
                observable=observable,
                accurate_prediction=accurate_prediction,
                fast_prediction=fast_prediction,
                accurate_latency_us=accurate_latency_us,
                fast_latency_us=fast_latency_us,
                features=features.astype(np.float32),
                feature_names=feature_names,
                risk_label=risk_label,
                hard_runtime=hard_runtime,
                fast_wrong_vs_accurate=fast_wrong,
                fast_logical_fail=fast_fail,
                metadata={
                    **bundle.get("metadata", {}),
                    "noise_scenario": trace_context.get(
                        "noise_scenario", bundle.get("metadata", {}).get("noise_scenario")
                    ),
                    "burst_context": trace_context.get("burst_context"),
                    "backlog_proxy": trace_context.get("backlog_proxy"),
                    "residual_or_candidate_complexity": trace_context.get(
                        "residual_or_candidate_complexity"
                    ),
                    "accurate_decoder": getattr(accurate_decoder, "name", config.decoders.accurate),
                    "fast_decoder": getattr(fast_decoder, "name", config.decoders.fast),
                    "accurate_placeholder": bool(
                        accurate_batch_metadata.get("placeholder", False)
                        or accurate_result_metadata[shot_id].get("placeholder", False)
                    ),
                    "fast_placeholder": bool(fast_metadata[shot_id].get("placeholder", False)),
                    "placeholder_fast_decoder": bool(
                        fast_metadata[shot_id].get("placeholder", False)
                    ),
                    "timing_mode": timing_mode,
                    "hard_runtime_label_valid": hard_runtime_label_valid,
                    "timing_metadata": timing_metadata,
                    "accurate_batch_latency_us_total": float(
                        accurate_batch_metadata.get("latency_us", 0.0)
                    ),
                    "accurate_latency_us_estimated_per_shot": accurate_latency_us,
                    "accurate_logical_fail": accurate_fail,
                },
            )
        )
    metadata = {
        **bundle.get("metadata", {}),
        "num_records": len(records),
        "feature_names": records[0].feature_names if records else [],
        "feature_dim": int(records[0].features.shape[0]) if records else 0,
        "timing_mode": timing_mode,
        "hard_runtime_label_valid": hard_runtime_label_valid,
        "hard_runtime_threshold_us": hard_threshold,
        "accurate_batch_latency_us_total": float(accurate_batch_metadata.get("latency_us", 0.0)),
        "accurate_decoder": getattr(accurate_decoder, "name", config.decoders.accurate),
        "fast_decoder": getattr(fast_decoder, "name", config.decoders.fast),
        "placeholder_fast_decoder": any(
            record.metadata.get("placeholder_fast_decoder", False) for record in records
        ),
        "real_qec": not bool(bundle.get("metadata", {}).get("toy", False))
        and bundle.get("dem") is not None,
        "fallback_reason": bundle.get("metadata", {}).get("reason"),
        "num_candidates": len(all_candidates),
        "num_local_candidates": len(local_candidates),
    }
    return records, metadata


def _safe_json_loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return default


def _risk_dataset_syndrome_at(archive: np.lib.npyio.NpzFile, index: int) -> np.ndarray:
    """Return one unpadded syndrome from old or padded risk dataset storage."""
    if "syndromes_padded" in archive.files:
        padded = np.asarray(archive["syndromes_padded"][index], dtype=np.int8).reshape(-1)
        if "syndrome_lengths" in archive.files:
            length = int(archive["syndrome_lengths"][index])
            return padded[:length]
        return padded

    if "syndromes" in archive.files:
        return np.asarray(archive["syndromes"][index], dtype=np.int8).reshape(-1)

    raise ValueError(
        "risk_dataset is missing syndrome storage. Expected 'syndromes' "
        "or 'syndromes_padded' + 'syndrome_lengths'."
    )


def _optional_archive_array(archive: np.lib.npyio.NpzFile, name: str, index: int) -> Any:
    if name not in archive.files:
        return None
    array = archive[name]
    if np.ndim(array) == 0:
        return array.item()
    value = array[int(index)]
    if hasattr(value, "item") and np.asarray(value).shape == ():
        return value.item()
    return value


def load_real_stream_records_from_risk_dataset(
    path: str | Path,
    split: str = "test",
) -> tuple[list[RealStreamShotRecord], dict[str, Any]]:
    """Load held-out real-stream records from a risk dataset split."""
    dataset_path = Path(path)
    archive = np.load(dataset_path, allow_pickle=False)
    required = {
        "features",
        "actual_observables",
        "accurate_predictions",
        "fast_predictions",
        "labels",
        "label_names",
        "runtimes",
        "runtime_names",
        "feature_names",
    }
    missing = sorted(required.difference(set(archive.files)))

    has_old_syndromes = "syndromes" in archive.files
    has_padded_syndromes = (
        "syndromes_padded" in archive.files and "syndrome_lengths" in archive.files
    )
    if not has_old_syndromes and not has_padded_syndromes:
        missing.append("syndromes or syndromes_padded+syndrome_lengths")

    if missing:
        raise ValueError(
            f"risk_dataset is missing {missing}; rebuild dataset with records/features/labels/predictions/runtimes."
        )
    metadata = load_risk_dataset_metadata(dataset_path)
    split_payload = dict(metadata.get("splits", {}))
    for split_path in (
        risk_dataset_split_sidecar_path(dataset_path),
        dataset_path.with_name("risk_dataset_splits.json"),
    ):
        if split_path.exists():
            with split_path.open("r", encoding="utf-8") as handle:
                split_payload.update(json.load(handle))
            break
    split_key = str(split).lower()
    indices = split_payload.get(f"{split_key}_indices")
    if indices is None:
        if split_key == "all":
            indices = list(range(len(archive["features"])))
        else:
            raise ValueError(
                f"risk_dataset has no split '{split_key}'. Expected train/val/test/all."
            )
    indices_array = np.asarray(indices, dtype=np.int64)
    label_names = [str(name) for name in archive["label_names"].tolist()]
    runtime_names = [str(name) for name in archive["runtime_names"].tolist()]
    feature_names = [str(name) for name in archive["feature_names"].tolist()]
    label_index = {name: idx for idx, name in enumerate(label_names)}
    runtime_index = {name: idx for idx, name in enumerate(runtime_names)}
    sample_metadata = list(metadata.get("samples", []))
    setting_ids = np.asarray(archive["setting_ids"]) if "setting_ids" in archive.files else None
    episode_ids = np.asarray(archive["episode_ids"]) if "episode_ids" in archive.files else None
    stream_ids = np.asarray(archive["stream_ids"]) if "stream_ids" in archive.files else None
    arrival_order = (
        np.asarray(archive["arrival_order"]) if "arrival_order" in archive.files else None
    )
    shot_index_within_setting = (
        np.asarray(archive["shot_index_within_setting"])
        if "shot_index_within_setting" in archive.files
        else None
    )
    records: list[RealStreamShotRecord] = []
    for raw_index in indices_array.tolist():
        item_meta = sample_metadata[int(raw_index)] if int(raw_index) < len(sample_metadata) else {}
        record_meta = dict(item_meta.get("metadata", {}))
        if setting_ids is not None:
            record_meta.setdefault("setting_id", int(setting_ids[int(raw_index)]))
        if episode_ids is not None:
            record_meta.setdefault("episode_id", int(episode_ids[int(raw_index)]))
        if stream_ids is not None:
            record_meta.setdefault("stream_id", int(stream_ids[int(raw_index)]))
        if arrival_order is not None:
            record_meta.setdefault("arrival_order", int(arrival_order[int(raw_index)]))
        if shot_index_within_setting is not None:
            record_meta.setdefault(
                "shot_index_within_setting", int(shot_index_within_setting[int(raw_index)])
            )
        for field_name in [
            "difficulty_tier",
            "difficulty_tier_ids",
            "distance",
            "rounds",
            "physical_error_rate",
            "noise_scenario",
            "arrival_order",
            "stream_index",
            "syndrome_weight",
            "syndrome_weight_tail",
            "detector_count",
            "residual_or_candidate_complexity",
            "backlog_proxy",
            "burst_context",
            "hotspot_context",
            "leakage_context",
        ]:
            value = _optional_archive_array(archive, field_name, int(raw_index))
            if value is not None:
                if isinstance(value, np.generic):
                    value = value.item()
                record_meta.setdefault(field_name, value)
        labels = archive["labels"][int(raw_index)]
        runtimes = archive["runtimes"][int(raw_index)]
        records.append(
            RealStreamShotRecord(
                shot_id=int(item_meta.get("shot_id", raw_index)),
                syndrome=_risk_dataset_syndrome_at(archive, int(raw_index)),
                observable=np.asarray(
                    archive["actual_observables"][int(raw_index)], dtype=np.int8
                ).reshape(-1),
                accurate_prediction=np.asarray(
                    archive["accurate_predictions"][int(raw_index)], dtype=np.int8
                ).reshape(-1),
                fast_prediction=np.asarray(
                    archive["fast_predictions"][int(raw_index)], dtype=np.int8
                ).reshape(-1),
                accurate_latency_us=float(runtimes[runtime_index.get("accurate_runtime_us", 0)]),
                fast_latency_us=float(
                    runtimes[runtime_index.get("fast_runtime_us", 1 if len(runtimes) > 1 else 0)]
                ),
                features=np.asarray(archive["features"][int(raw_index)], dtype=np.float32),
                feature_names=feature_names,
                risk_label=int(labels[label_index.get("scheduler_risk_label", 4)]),
                hard_runtime=int(labels[label_index.get("hard_runtime", 3)]),
                fast_wrong_vs_accurate=int(labels[label_index.get("fast_wrong_vs_accurate", 0)]),
                fast_logical_fail=int(labels[label_index.get("fast_logical_fail", 1)]),
                metadata={
                    **record_meta,
                    "raw_dataset_index": int(raw_index),
                    "timing_mode": record_meta.get(
                        "timing_mode", metadata.get("timing_mode", "unknown")
                    ),
                    "hard_runtime_label_valid": bool(
                        record_meta.get(
                            "hard_runtime_label_valid",
                            metadata.get("hard_runtime_label_valid", True),
                        )
                    ),
                },
            )
        )
    meta = {
        **metadata,
        "num_records": len(records),
        "feature_names": feature_names,
        "feature_dim": int(len(feature_names)),
        "eval_source": "risk_dataset",
        "eval_split": split_key,
        "split_policy": str(
            split_payload.get("split_policy", metadata.get("split_policy", "unknown"))
        ),
        "split_boundaries": dict(split_payload.get("split_boundaries", {})),
        "train_indices_hash": split_payload.get(
            "train_indices_hash", hash_indices(split_payload.get("train_indices", []))
        ),
        "test_indices_hash": split_payload.get(
            "test_indices_hash", hash_indices(split_payload.get("test_indices", []))
        ),
        "val_indices_hash": split_payload.get(
            "val_indices_hash", hash_indices(split_payload.get("val_indices", []))
        ),
        "real_qec": (
            not any(bool(record.metadata.get("toy", False)) for record in records)
            if records
            else False
        ),
        "fallback_reason": metadata.get("fallback_reason"),
        "timing_mode": (
            records[0].metadata.get("timing_mode", "unknown")
            if records
            else metadata.get("timing_mode", "unknown")
        ),
        "hard_runtime_label_valid": bool(metadata.get("hard_runtime_label_valid", True)),
    }
    return records, meta


def split_records(
    records: list[RealStreamShotRecord],
    train_fraction: float | None = None,
    val_fraction: float | None = None,
    test_fraction: float = 0.4,
    seed: int = 42,
) -> dict[str, list[RealStreamShotRecord]]:
    """Split records into train/val/test partitions."""
    num_records = len(records)
    rng = np.random.default_rng(seed)
    indices = np.arange(num_records, dtype=np.int64)
    rng.shuffle(indices)
    if train_fraction is None:
        remaining = max(0.0, 1.0 - float(test_fraction))
        val_fraction = 0.0 if val_fraction is None else float(val_fraction)
        train_fraction = max(0.0, remaining - val_fraction)
    if val_fraction is None:
        val_fraction = max(0.0, 1.0 - float(test_fraction) - float(train_fraction))
    test_size = int(round(num_records * float(test_fraction)))
    val_size = int(round(num_records * float(val_fraction)))
    test_size = min(max(test_size, 1 if num_records > 2 else 0), num_records)
    remaining_after_test = max(num_records - test_size, 0)
    val_size = min(max(val_size, 0), remaining_after_test)
    test_indices = indices[:test_size]
    val_indices = indices[test_size : test_size + val_size]
    train_indices = indices[test_size + val_size :]
    return {
        "train": [records[int(idx)] for idx in train_indices.tolist()],
        "val": [records[int(idx)] for idx in val_indices.tolist()],
        "test": [records[int(idx)] for idx in test_indices.tolist()],
        "train_indices": train_indices.astype(np.int64).tolist(),
        "val_indices": val_indices.astype(np.int64).tolist(),
        "test_indices": test_indices.astype(np.int64).tolist(),
    }


def _record_feature_matrix(records: list[RealStreamShotRecord]) -> np.ndarray:
    if not records:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack([np.asarray(record.features, dtype=np.float32) for record in records], axis=0)


def _record_feature_matrix_for_model(
    records: list[RealStreamShotRecord],
    risk_metadata: dict[str, Any],
    normalization: dict[str, np.ndarray] | None,
) -> np.ndarray:
    """Build an inference matrix that matches the checkpoint feature contract."""
    if not records:
        return np.zeros((0, 0), dtype=np.float32)
    checkpoint_names = [str(name) for name in risk_metadata.get("feature_names", [])]
    if checkpoint_names:
        rows = []
        for record in records:
            values = _feature_map(record)
            rows.append([float(values.get(name, 0.0)) for name in checkpoint_names])
        return np.asarray(rows, dtype=np.float32)
    expected_dim = int(np.asarray((normalization or {}).get("mean", []), dtype=np.float32).size)
    matrix = _record_feature_matrix(records)
    if expected_dim and matrix.shape[1] != expected_dim:
        aligned = np.zeros((matrix.shape[0], expected_dim), dtype=np.float32)
        width = min(matrix.shape[1], expected_dim)
        aligned[:, :width] = matrix[:, :width]
        return aligned
    return matrix


def _feature_map(record: RealStreamShotRecord) -> dict[str, float]:
    return {
        str(name): float(record.features[idx])
        for idx, name in enumerate(record.feature_names)
        if idx < len(record.features)
    }


def _feature_value(record: RealStreamShotRecord, name: str, default: float = 0.0) -> float:
    return float(_feature_map(record).get(name, default))


def _append_or_replace_features(
    features: np.ndarray,
    feature_names: list[str],
    values: dict[str, float],
) -> tuple[np.ndarray, list[str]]:
    """Return features with generated stream-context values attached."""
    updated_names = list(feature_names)
    updated_values = np.asarray(features, dtype=np.float32).reshape(-1).tolist()
    name_to_index = {name: idx for idx, name in enumerate(updated_names)}
    for name, value in values.items():
        numeric = float(value)
        if name in name_to_index:
            updated_values[name_to_index[name]] = numeric
        else:
            updated_names.append(name)
            updated_values.append(numeric)
    return np.asarray(updated_values, dtype=np.float32), updated_names


def _generated_trace_contexts(
    config: ProjectConfig, num_records: int
) -> list[dict[str, float | str]]:
    """Build deterministic burst/trace-context overlays for generated Stim shots."""
    if num_records <= 0 or not config.noise_scenarios:
        return [{} for _ in range(max(num_records, 0))]
    scenario = dict(config.noise_scenarios[0])
    scenario_type = str(scenario.get("type", "")).lower()
    scenario_name = str(scenario.get("name", "trace_context")).lower()
    if (
        "burst" not in scenario_type
        and "burst" not in scenario_name
        and not scenario.get("burst_probability")
    ):
        return [{} for _ in range(num_records)]

    rng = np.random.default_rng(int(config.seed) + 104729)
    burst = np.zeros(num_records, dtype=np.float32)
    width = max(int(scenario.get("burst_width", scenario.get("burst_length", 8))), 1)
    probability = scenario.get("burst_probability")
    if probability is not None:
        active = 0
        for idx in range(num_records):
            if active <= 0 and rng.random() < float(probability):
                active = width
            if active > 0:
                burst[idx] = active / float(width)
                active -= 1
    else:
        period = max(int(scenario.get("burst_period", max(width * 4, 32))), width)
        for idx in range(num_records):
            phase = idx % period
            if phase < width:
                burst[idx] = 1.0 - float(phase) / float(width)

    accurate_scale = float(
        scenario.get("accurate_runtime_burst_scale", scenario.get("runtime_burst_scale", 1.0))
    )
    fast_scale = float(scenario.get("fast_runtime_burst_scale", 1.0))
    residual_scale = float(scenario.get("residual_burst_scale", 1.0))
    backlog_scale = float(
        scenario.get("backlog_burst_scale", max(config.runtime.overload_backlog_threshold, 1))
    )

    contexts: list[dict[str, float | str]] = []
    for value in burst.tolist():
        contexts.append(
            {
                "noise_scenario": scenario_name,
                "burst_context": float(value),
                "accurate_runtime_multiplier": 1.0 + float(value) * (accurate_scale - 1.0),
                "fast_runtime_multiplier": 1.0 + float(value) * (fast_scale - 1.0),
                "residual_or_candidate_complexity": float(value) * residual_scale,
                "backlog_proxy": float(value) * backlog_scale,
            }
        )
    return contexts


def _heuristic_risk_scores(records: list[RealStreamShotRecord]) -> np.ndarray:
    if not records:
        return np.zeros((0,), dtype=np.float32)
    scores: list[float] = []
    for record in records:
        feature_values = _feature_map(record)
        if "risk_proxy" in feature_values:
            scores.append(float(np.clip(feature_values["risk_proxy"], 0.0, 1.0)))
            continue
        weight = feature_values.get("syndrome_weight", float(np.asarray(record.syndrome).sum()))
        num_detectors = max(
            feature_values.get("num_detectors", float(np.asarray(record.syndrome).size)), 1.0
        )
        density = feature_values.get("syndrome_density", weight / num_detectors)
        residual = feature_values.get(
            "residual_or_candidate_complexity",
            feature_values.get("mean_candidate_count_active", 0.0),
        )
        backlog = feature_values.get("backlog_proxy", 0.0)
        burst = feature_values.get("burst_context", 0.0)
        leakage = feature_values.get("leakage_context", 0.0)
        syndrome_tail = feature_values.get(
            "syndrome_weight_tail", feature_values.get("syndrome_tail", 0.0)
        )
        residual_norm = residual / (residual + 12.0) if residual > 0.0 else 0.0
        backlog_norm = backlog / (backlog + 16.0) if backlog > 0.0 else 0.0
        score = (
            0.35 * min(max(density * 4.0, 0.0), 1.0)
            + 0.25 * residual_norm
            + 0.20 * backlog_norm
            + 0.10 * min(max(burst, 0.0), 1.0)
            + 0.10 * min(max(leakage, 0.0), 1.0)
            + 0.20 * min(max(syndrome_tail, 0.0), 1.0)
        )
        scores.append(float(np.clip(score, 0.0, 1.0)))
    return np.asarray(scores, dtype=np.float32)


def _predecode_effect(
    record: RealStreamShotRecord,
    config: ProjectConfig,
    *,
    validation_enabled: bool | None = None,
    abstention_enabled: bool | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, float | bool]:
    """Estimate fixed-cost heuristic/CA frontend workload shaping for paired eval."""
    validation = (
        config.predecoder.enable_validation
        if validation_enabled is None
        else bool(validation_enabled)
    )
    abstention = (
        config.predecoder.enable_abstention
        if abstention_enabled is None
        else bool(abstention_enabled)
    )
    overload_mode = bool(
        context and (context.get("overload_mode") or context.get("boundary_drain"))
    )
    confidence_threshold = float(config.predecoder.confidence_threshold)
    predecode_risk_threshold = float(config.predecoder.risk_threshold)
    max_cluster_size = float(config.predecoder.max_cluster_size)
    if overload_mode:
        confidence_threshold = max(confidence_threshold - 0.10, 0.40)
        predecode_risk_threshold = min(predecode_risk_threshold + 0.15, 0.75)
        max_cluster_size += 2.0
    signals = _predecode_signals(record)
    density = float(signals["density"])
    candidate_coverage = float(signals["candidate_coverage"])
    candidates_missing = bool(signals["candidates_missing"])
    confidence = float(np.clip(1.0 - density, 0.0, 1.0))
    risk = _predecode_risk_estimate(record, config)
    validation_pass = _predecode_validation_pass(record, config, max_cluster_size=max_cluster_size)
    feature_values = _feature_map(record)
    mean_patch_density = feature_values.get("mean_patch_density", float(signals["density"]))
    max_patch_density = feature_values.get("max_patch_density", float(signals["density"]))
    mean_candidate_count = feature_values.get("mean_candidate_count_active", 0.0)
    weak_validation_pass = bool(
        float(signals["weight"]) <= max_cluster_size + 4.0
        and float(signals["candidate_coverage"]) >= 0.35
        and float(mean_patch_density) <= 0.35
        and float(max_patch_density) <= 0.65
        and float(mean_candidate_count) <= 32.0
    )
    threshold_pass = confidence >= confidence_threshold and risk <= predecode_risk_threshold
    strong_certificate = validation_pass if validation else True
    weak_certificate = weak_validation_pass if validation else True
    fast_certified = bool(threshold_pass and strong_certificate)
    shape_accepted = bool(threshold_pass and (strong_certificate or weak_certificate))
    if not abstention:
        fast_certified = bool(strong_certificate)
        shape_accepted = bool(strong_certificate or weak_certificate)
    coverage = candidate_coverage if not candidates_missing else max(0.25, 1.0 - density)
    residual_reduction = (
        float(config.predecoder.max_residual_reduction) * float(np.clip(coverage, 0.0, 1.0))
        if shape_accepted
        else 0.0
    )
    frontend_latency = float(config.predecoder.frontend_latency_us)
    validation_latency = float(config.predecoder.validation_latency_us) if validation else 0.0
    return {
        "frontend_latency_us": frontend_latency,
        "validation_latency_us": validation_latency,
        "predecode_latency_us": frontend_latency + validation_latency,
        "estimated_residual_reduction": float(
            np.clip(residual_reduction, 0.0, config.predecoder.max_residual_reduction)
        ),
        "predecode_accept_estimate": bool(shape_accepted),
        "fast_path_certified": bool(fast_certified),
        "predecode_validation_pass_estimate": bool(validation_pass),
        "weak_validation_pass_estimate": bool(weak_validation_pass),
        "predecode_confidence_estimate": confidence,
        "predecode_risk_estimate": risk,
        "effective_predecode_confidence_threshold": confidence_threshold,
        "effective_predecode_risk_threshold": predecode_risk_threshold,
        "effective_max_cluster_size": max_cluster_size,
        "predecode_overload_policy": bool(overload_mode),
        "validation_enabled": bool(validation),
        "abstention_enabled": bool(abstention),
    }


def _predecode_signals(record: RealStreamShotRecord) -> dict[str, float | bool]:
    """Extract the shot-level signals used by the heuristic frontend."""
    feature_values = _feature_map(record)
    weight = feature_values.get("syndrome_weight", float(np.asarray(record.syndrome).sum()))
    num_detectors = max(
        feature_values.get("num_detectors", float(np.asarray(record.syndrome).size)), 1.0
    )
    density = feature_values.get("syndrome_density", weight / num_detectors)
    candidate_coverage = feature_values.get("fraction_active_with_candidate", 0.0)
    candidates_missing = feature_values.get("candidates_missing_flag", 1.0) >= 0.5
    return {
        "weight": float(weight),
        "num_detectors": float(num_detectors),
        "density": float(density),
        "candidate_coverage": float(candidate_coverage),
        "candidates_missing": bool(candidates_missing),
    }


def _predecode_validation_pass(
    record: RealStreamShotRecord,
    config: ProjectConfig,
    *,
    max_cluster_size: float | None = None,
) -> bool:
    """Return the certificate predicate used by the heuristic frontend."""
    signals = _predecode_signals(record)
    feature_values = _feature_map(record)
    distance = int(
        record.metadata.get("distance", config.qec.distances[0] if config.qec.distances else 0) or 0
    )
    mean_patch_density = feature_values.get("mean_patch_density", float(signals["density"]))
    max_patch_density = feature_values.get("max_patch_density", float(signals["density"]))
    mean_candidate_count = feature_values.get("mean_candidate_count_active", 0.0)
    cluster_size = float(
        config.predecoder.max_cluster_size if max_cluster_size is None else max_cluster_size
    )
    return bool(
        (bool(signals["candidates_missing"]) and float(signals["weight"]) <= cluster_size)
        or float(signals["weight"]) <= min(cluster_size, 4.0)
        or (
            float(signals["weight"]) <= cluster_size + 2.0
            and float(signals["candidate_coverage"]) >= 0.60
            and float(mean_patch_density) <= 0.25
            and float(max_patch_density) <= 0.50
            and float(mean_candidate_count) <= 24.0
        )
        or (
            distance >= 9
            and float(signals["density"]) <= 0.030
            and float(signals["candidate_coverage"]) >= 0.15
            and float(max_patch_density) <= 0.60
            and float(mean_candidate_count) <= 16.0
        )
    )


def _predecode_risk_estimate(record: RealStreamShotRecord, config: ProjectConfig) -> float:
    """Return the frontend fast-path risk estimate used for abstention."""
    signals = _predecode_signals(record)
    feature_values = _feature_map(record)
    density = float(signals["density"])
    mean_patch_density = feature_values.get("mean_patch_density", density)
    max_patch_density = feature_values.get("max_patch_density", density)
    candidate_penalty = max(0.0, 0.60 - float(signals["candidate_coverage"]))
    score = (
        float(config.predecoder.density_risk_scale) * density
        + 0.35 * float(mean_patch_density)
        + 0.20 * float(max_patch_density)
        + 0.25 * candidate_penalty
    )
    return float(np.clip(score, 0.0, 1.0))


def _latency_with_predecode(
    backend_latency_us: float, effect: dict[str, float | bool], config: ProjectConfig
) -> float:
    reduction = float(effect.get("estimated_residual_reduction", 0.0))
    min_fraction = float(config.predecoder.min_backend_latency_fraction)
    shaped_backend = max(
        float(backend_latency_us) * min_fraction, float(backend_latency_us) * (1.0 - reduction)
    )
    return float(effect.get("predecode_latency_us", 0.0)) + shaped_backend


def _oracle_predecode_effect(
    record: RealStreamShotRecord, config: ProjectConfig
) -> dict[str, float | bool]:
    effect = _predecode_effect(record, config, validation_enabled=False, abstention_enabled=False)
    reduction = max(
        float(effect.get("estimated_residual_reduction", 0.0)),
        float(config.predecoder.max_residual_reduction),
    )
    effect.update(
        {
            "predecode_accept_estimate": True,
            "fast_path_certified": True,
            "predecode_validation_pass_estimate": True,
            "weak_validation_pass_estimate": True,
            "predecode_confidence_estimate": 1.0,
            "predecode_risk_estimate": 0.0,
            "estimated_residual_reduction": reduction,
            "validation_enabled": False,
            "abstention_enabled": False,
        }
    )
    return effect


def _boundary_context(
    record: RealStreamShotRecord, config: ProjectConfig
) -> tuple[bool, int, bool]:
    interval = max(int(config.runtime.logical_boundary_interval), 1)
    offset = (int(record.shot_id) + 1) % interval
    logical_boundary = offset == 0
    rounds_until_boundary = 0 if logical_boundary else interval - offset
    drain_rounds = max(int(config.risk_eval.boundary_drain_rounds), 0)
    return logical_boundary, rounds_until_boundary, rounds_until_boundary <= drain_rounds


def _queue_context(
    record: RealStreamShotRecord,
    worker_available_times: list[float],
    finish_times: list[float],
    config: ProjectConfig,
    ordered_commit_state: OrderedCommitState | None = None,
) -> dict[str, Any]:
    arrival_time = float(record.shot_id) * float(config.runtime.round_period_us)
    deadline = arrival_time + float(config.runtime.decode_deadline_us)
    unfinished_before_arrival = sum(1 for time_us in finish_times if time_us > arrival_time)
    next_worker_available = min(worker_available_times) if worker_available_times else 0.0
    earliest_start = max(arrival_time, next_worker_available)
    backlog = unfinished_before_arrival + 1
    if ordered_commit_state is None:
        committed_prefix_at_arrival = None
        pauli_frame_lag = unfinished_before_arrival
    else:
        pauli_frame_lag = ordered_commit_state.lag_at_arrival(record.shot_id, arrival_time)
        committed_prefix_at_arrival = ordered_commit_state.committed_prefix
    logical_boundary, rounds_until_boundary, boundary_drain = _boundary_context(record, config)
    drain_backlog_threshold = (
        int(config.risk_eval.rt_qec_drain_backlog_threshold)
        if config.risk_eval.rt_qec_drain_backlog_threshold is not None
        else int(config.runtime.overload_backlog_threshold)
    )
    overload = backlog >= drain_backlog_threshold or pauli_frame_lag >= int(
        config.runtime.max_pauli_frame_lag
    )
    return {
        "arrival_time_us": arrival_time,
        "deadline_us": deadline,
        "earliest_start_us": earliest_start,
        "deadline_slack_us": max(deadline - earliest_start, 0.0),
        "estimated_backlog_before_arrival": backlog,
        "estimated_pauli_frame_lag": pauli_frame_lag,
        "committed_prefix_at_arrival": committed_prefix_at_arrival,
        "logical_boundary": logical_boundary,
        "rounds_until_boundary": rounds_until_boundary,
        "boundary_drain": boundary_drain,
        "overload_mode": overload,
    }


def _without_scheduler_context(context: dict[str, Any]) -> dict[str, Any]:
    """Remove lag/boundary adaptation while keeping per-job deadline feasibility."""
    static_context = dict(context)
    static_context["logical_boundary"] = False
    static_context["boundary_drain"] = False
    static_context["overload_mode"] = False
    return static_context


def _advance_worker_state(
    record: RealStreamShotRecord,
    latency_us: float,
    worker_available_times: list[float],
    finish_times: list[float],
    config: ProjectConfig,
    ordered_commit_state: OrderedCommitState | None = None,
) -> None:
    arrival_time = float(record.shot_id) * float(config.runtime.round_period_us)
    worker_idx = min(
        range(len(worker_available_times)), key=lambda idx: worker_available_times[idx]
    )
    start_time = max(arrival_time, worker_available_times[worker_idx])
    finish_time = start_time + float(latency_us)
    worker_available_times[worker_idx] = finish_time
    finish_times.append(finish_time)
    if ordered_commit_state is not None:
        ordered_commit_state.schedule_completion(record.shot_id, finish_time)


def _choose_rt_qec_decoder(
    record: RealStreamShotRecord,
    risk_score: float,
    shaped_accurate_latency_us: float,
    shaped_fast_latency_us: float,
    context: dict[str, Any],
    config: ProjectConfig,
) -> tuple[str, str]:
    return _choose_rt_qec_decoder_with_contract(
        record,
        risk_score,
        shaped_accurate_latency_us,
        shaped_fast_latency_us,
        context,
        config,
        fast_path_allowed=True,
    )


def _choose_rt_qec_decoder_with_contract(
    record: RealStreamShotRecord,
    risk_score: float,
    shaped_accurate_latency_us: float,
    shaped_fast_latency_us: float,
    context: dict[str, Any],
    config: ProjectConfig,
    *,
    fast_path_allowed: bool,
) -> tuple[str, str]:
    risk_threshold = float(config.risk_eval.ai_risk_threshold)
    if bool(context.get("overload_mode") or context.get("boundary_drain")):
        risk_threshold = max(risk_threshold - 0.05, 0.15)
    high_risk = risk_score >= risk_threshold
    slack_us = float(context["deadline_slack_us"])
    accurate_feasible = shaped_accurate_latency_us <= slack_us
    fast_feasible = shaped_fast_latency_us <= slack_us
    pressure = bool(context["overload_mode"] or context["boundary_drain"])
    if not fast_path_allowed:
        return "accurate", "abstain_or_uncertified_accurate"
    if bool(context["logical_boundary"]) and high_risk and accurate_feasible:
        return "accurate", "boundary_high_risk_accurate"
    if pressure and (not high_risk or not accurate_feasible):
        return "fast", "drain_or_overload_fast"
    if high_risk and (accurate_feasible or not fast_feasible):
        return "accurate", "risk_aware_accurate"
    if fast_feasible:
        return "fast", "low_risk_fast"
    return "accurate", "fast_infeasible_fallback"


def evaluate_mode_on_records(
    records: list[RealStreamShotRecord],
    mode: str,
    config: ProjectConfig,
    risk_model: Any = None,
    normalization: dict[str, np.ndarray] | None = None,
    risk_metadata: dict[str, Any] | None = None,
    thresholds: dict[str, float] | None = None,
    ordered_commit: bool = False,
) -> ModeEvaluationResult:
    """Evaluate one scheduling mode on a paired record set."""
    selected_latencies: list[float] = []
    selected_predictions: list[np.ndarray] = []
    selected_decoders: list[str] = []
    decision_rows: list[dict[str, Any]] = []
    mode_key = str(mode).lower()
    heuristic_scores = _heuristic_risk_scores(records)
    ai_predictions: dict[str, np.ndarray] | None = None
    ai_available = risk_model is not None
    risk_metadata = risk_metadata or {}
    thresholds = thresholds or {}
    risk_threshold = float(thresholds.get("ai_risk_threshold", config.risk_eval.ai_risk_threshold))
    safe_fast_threshold = float(
        thresholds.get("safe_fast_threshold", thresholds.get("ai_safe_fast_threshold", 0.5))
    )
    confidence_threshold = float(
        thresholds.get("ai_confidence_threshold", config.risk_eval.ai_confidence_threshold)
    )
    model_type = str(risk_metadata.get("model_type", "none"))
    model_config = dict(risk_metadata.get("model_config", {}))
    history_length = int(model_config.get("history_length", risk_metadata.get("history_length", 1)))
    history_encoder_type = str(model_config.get("history_encoder_type", "none"))
    pad_mode = str(model_config.get("pad_mode", risk_metadata.get("pad_mode", "edge")))
    requires_history = (
        model_type
        in {
            "risk_tcn",
            "risk_gru",
            "risk_lstm",
            "risk_decomposed_tcn",
            "risk_decomposed_gru",
            "risk_decomposed_lstm",
        }
        or history_length > 1
        or history_encoder_type != "none"
    )
    if (
        mode_key in {"ai_risk", "rt_qec_ai", "rt_qec_learned"}
        and risk_model is not None
        and records
    ):
        feature_matrix = _record_feature_matrix_for_model(records, risk_metadata, normalization)
        history_matrix = (
            build_causal_history_matrix(
                feature_matrix,
                history_length=max(history_length, 1),
                normalization=None,
                pad_mode=pad_mode,
            )
            if requires_history
            else None
        )
        ai_predictions = predict_risk_scores(
            risk_model,
            feature_matrix,
            normalization,
            history_features=history_matrix,
        )
    decision_worker_available_times = [0.0 for _ in range(max(int(config.runtime.num_workers), 1))]
    decision_finish_times: list[float] = []
    decision_commit_state = OrderedCommitState() if ordered_commit else None
    for idx, record in enumerate(records):
        selected_decoder = "accurate"
        selection_reason = mode
        risk_score = float(heuristic_scores[idx]) if idx < len(heuristic_scores) else 0.0
        context = _queue_context(
            record,
            decision_worker_available_times,
            decision_finish_times,
            config,
            decision_commit_state,
        )
        predecode_effect: dict[str, float | bool] = {
            "frontend_latency_us": 0.0,
            "validation_latency_us": 0.0,
            "predecode_latency_us": 0.0,
            "estimated_residual_reduction": 0.0,
            "predecode_accept_estimate": False,
            "fast_path_certified": False,
            "predecode_validation_pass_estimate": False,
            "weak_validation_pass_estimate": False,
            "predecode_confidence_estimate": 0.0,
            "predecode_risk_estimate": 1.0,
            "effective_predecode_confidence_threshold": float(
                config.predecoder.confidence_threshold
            ),
            "effective_predecode_risk_threshold": float(config.predecoder.risk_threshold),
            "effective_max_cluster_size": float(config.predecoder.max_cluster_size),
            "predecode_overload_policy": False,
            "validation_enabled": False,
            "abstention_enabled": False,
        }
        ai_risk_score = None
        ai_runtime_score = None
        ai_hard_runtime_score = None
        ai_runtime_pred = None
        ai_confidence = None
        fast_wrong_prob = None
        fast_logical_fail_prob = None
        hard_runtime_prob = None
        safe_fast_prob = None
        combined_fast_risk = None
        shaped_accurate_latency = float(record.accurate_latency_us)
        shaped_fast_latency = float(record.fast_latency_us)
        if mode_key == "accurate_only":
            selected_decoder = "accurate"
        elif mode_key == "fast_only":
            selected_decoder = "fast"
        elif mode_key == "heuristic_pre_fixed":
            predecode_effect = _predecode_effect(record, config, context=context)
            shaped_accurate_latency = _latency_with_predecode(
                record.accurate_latency_us, predecode_effect, config
            )
            shaped_fast_latency = _latency_with_predecode(
                record.fast_latency_us, predecode_effect, config
            )
            selected_decoder = str(config.risk_eval.heuristic_predecoder_backend).lower()
            if selected_decoder not in {"accurate", "fast"}:
                selected_decoder = "accurate"
            selection_reason = f"heuristic_pre_fixed_{selected_decoder}"
        elif mode_key == "edf":
            accurate_feasible = record.accurate_latency_us <= float(context["deadline_slack_us"])
            selected_decoder = "accurate" if accurate_feasible else "fast"
            selection_reason = (
                "edf_accurate_feasible" if selected_decoder == "accurate" else "edf_fast_fallback"
            )
        elif mode_key == "risk_heuristic":
            selected_decoder = "accurate" if risk_score >= risk_threshold else "fast"
            selection_reason = (
                "heuristic_high_risk" if selected_decoder == "accurate" else "heuristic_low_risk"
            )
        elif mode_key == "oracle_predecoder":
            predecode_effect = _oracle_predecode_effect(record, config)
            shaped_accurate_latency = _latency_with_predecode(
                record.accurate_latency_us, predecode_effect, config
            )
            shaped_fast_latency = _latency_with_predecode(
                record.fast_latency_us, predecode_effect, config
            )
            selected_decoder, selection_reason = _choose_rt_qec_decoder_with_contract(
                record,
                float(record.risk_label),
                shaped_accurate_latency,
                shaped_fast_latency,
                context,
                config,
                fast_path_allowed=bool(
                    predecode_effect.get(
                        "fast_path_certified",
                        predecode_effect.get("predecode_accept_estimate", False),
                    )
                ),
            )
            selection_reason = f"{selection_reason}_oracle_predecode"
        elif mode_key in {"rt_qec", "full_rt_preqec"}:
            predecode_effect = _predecode_effect(record, config, context=context)
            shaped_accurate_latency = _latency_with_predecode(
                record.accurate_latency_us, predecode_effect, config
            )
            shaped_fast_latency = _latency_with_predecode(
                record.fast_latency_us, predecode_effect, config
            )
            selected_decoder, selection_reason = _choose_rt_qec_decoder_with_contract(
                record,
                risk_score,
                shaped_accurate_latency,
                shaped_fast_latency,
                context,
                config,
                fast_path_allowed=bool(
                    predecode_effect.get(
                        "fast_path_certified",
                        predecode_effect.get("predecode_accept_estimate", False),
                    )
                ),
            )
        elif mode_key in {"rt_qec_ai", "rt_qec_learned"}:
            predecode_effect = _predecode_effect(record, config, context=context)
            shaped_accurate_latency = _latency_with_predecode(
                record.accurate_latency_us, predecode_effect, config
            )
            shaped_fast_latency = _latency_with_predecode(
                record.fast_latency_us, predecode_effect, config
            )
            if ai_predictions is None:
                learned_risk = risk_score
                learned_conf = 1.0
                selection_reason_suffix = "heuristic_fallback"
            else:
                ai_risk_score = float(ai_predictions["risk_score"][idx])
                ai_runtime_score = float(ai_predictions["runtime_score"][idx])
                ai_hard_runtime_score = float(
                    ai_predictions.get("hard_runtime_score", ai_predictions["runtime_score"])[idx]
                )
                ai_runtime_pred = float(ai_predictions["runtime_pred"][idx])
                ai_confidence = float(ai_predictions["confidence"][idx])
                if "safe_fast_prob" in ai_predictions:
                    fast_wrong_prob = float(ai_predictions["fast_wrong_prob"][idx])
                    fast_logical_fail_prob = float(ai_predictions["fast_logical_fail_prob"][idx])
                    hard_runtime_values = ai_predictions.get(
                        "hard_runtime_prob", ai_predictions.get("hard_runtime_score")
                    )
                    hard_runtime_prob = (
                        float(hard_runtime_values[idx]) if hard_runtime_values is not None else None
                    )
                    safe_fast_prob = float(ai_predictions["safe_fast_prob"][idx])
                    combined_fast_risk = float(
                        ai_predictions.get("combined_fast_risk", ai_predictions["risk_score"])[idx]
                    )
                    learned_risk = combined_fast_risk
                else:
                    learned_risk = ai_risk_score
                learned_conf = ai_confidence
                selection_reason_suffix = "learned_risk"
                if (
                    config.risk_eval.conservative_on_low_confidence
                    and learned_conf < confidence_threshold
                ):
                    learned_risk = 1.0
                    selection_reason_suffix = "learned_low_conf_conservative"
            selected_decoder, selection_reason = _choose_rt_qec_decoder_with_contract(
                record,
                learned_risk,
                shaped_accurate_latency,
                shaped_fast_latency,
                context,
                config,
                fast_path_allowed=bool(
                    predecode_effect.get(
                        "fast_path_certified",
                        predecode_effect.get("predecode_accept_estimate", False),
                    )
                ),
            )
            selection_reason = f"{selection_reason}_{selection_reason_suffix}"
        elif mode_key in {"rt_qec_without_validation", "rt_qec_no_validation"}:
            predecode_effect = _predecode_effect(
                record, config, validation_enabled=False, context=context
            )
            shaped_accurate_latency = _latency_with_predecode(
                record.accurate_latency_us, predecode_effect, config
            )
            shaped_fast_latency = _latency_with_predecode(
                record.fast_latency_us, predecode_effect, config
            )
            selected_decoder, selection_reason = _choose_rt_qec_decoder_with_contract(
                record,
                risk_score,
                shaped_accurate_latency,
                shaped_fast_latency,
                context,
                config,
                fast_path_allowed=bool(
                    predecode_effect.get(
                        "fast_path_certified",
                        predecode_effect.get("predecode_accept_estimate", False),
                    )
                ),
            )
            selection_reason = f"{selection_reason}_no_validation"
        elif mode_key in {"rt_qec_without_abstention", "rt_qec_no_abstention"}:
            predecode_effect = _predecode_effect(
                record, config, abstention_enabled=False, context=context
            )
            shaped_accurate_latency = _latency_with_predecode(
                record.accurate_latency_us, predecode_effect, config
            )
            shaped_fast_latency = _latency_with_predecode(
                record.fast_latency_us, predecode_effect, config
            )
            selected_decoder, selection_reason = _choose_rt_qec_decoder_with_contract(
                record,
                risk_score,
                shaped_accurate_latency,
                shaped_fast_latency,
                context,
                config,
                fast_path_allowed=bool(
                    predecode_effect.get(
                        "fast_path_certified",
                        predecode_effect.get("predecode_accept_estimate", False),
                    )
                ),
            )
            selection_reason = f"{selection_reason}_no_abstention"
        elif mode_key in {"rt_qec_without_scheduler", "rt_qec_fixed_backend"}:
            predecode_effect = _predecode_effect(record, config, context=context)
            shaped_accurate_latency = _latency_with_predecode(
                record.accurate_latency_us, predecode_effect, config
            )
            shaped_fast_latency = _latency_with_predecode(
                record.fast_latency_us, predecode_effect, config
            )
            selected_decoder, selection_reason = _choose_rt_qec_decoder_with_contract(
                record,
                risk_score,
                shaped_accurate_latency,
                shaped_fast_latency,
                _without_scheduler_context(context),
                config,
                fast_path_allowed=bool(predecode_effect.get("fast_path_certified", False)),
            )
            selection_reason = f"{selection_reason}_no_lag_scheduler"
        elif mode_key == "ai_risk":
            if ai_predictions is None:
                selected_decoder = "accurate"
                selection_reason = "ai_unavailable"
            else:
                ai_risk_score = float(ai_predictions["risk_score"][idx])
                ai_runtime_score = float(ai_predictions["runtime_score"][idx])
                ai_hard_runtime_score = float(
                    ai_predictions.get("hard_runtime_score", ai_predictions["runtime_score"])[idx]
                )
                ai_runtime_pred = float(ai_predictions["runtime_pred"][idx])
                ai_confidence = float(ai_predictions["confidence"][idx])
                if "safe_fast_prob" in ai_predictions:
                    fast_wrong_prob = float(ai_predictions["fast_wrong_prob"][idx])
                    fast_logical_fail_prob = float(ai_predictions["fast_logical_fail_prob"][idx])
                    hard_runtime_values = ai_predictions.get(
                        "hard_runtime_prob", ai_predictions.get("hard_runtime_score")
                    )
                    hard_runtime_prob = (
                        float(hard_runtime_values[idx]) if hard_runtime_values is not None else None
                    )
                    safe_fast_prob = float(ai_predictions["safe_fast_prob"][idx])
                    combined_fast_risk = float(
                        ai_predictions.get("combined_fast_risk", ai_predictions["risk_score"])[idx]
                    )
                    use_fast = (
                        safe_fast_prob >= safe_fast_threshold
                        and combined_fast_risk <= risk_threshold
                        and ai_confidence >= confidence_threshold
                    )
                    selected_decoder = "fast" if use_fast else "accurate"
                    if use_fast:
                        selection_reason = "decomposed_safe_fast"
                    elif safe_fast_prob < safe_fast_threshold:
                        selection_reason = "decomposed_unsafe_fast"
                    elif combined_fast_risk > risk_threshold:
                        selection_reason = "decomposed_fast_risk"
                    else:
                        selection_reason = "low_confidence_conservative"
                else:
                    selected_decoder = "accurate" if ai_risk_score >= risk_threshold else "fast"
                    selection_reason = "ai_threshold"
                    if (
                        config.risk_eval.conservative_on_low_confidence
                        and ai_confidence < confidence_threshold
                    ):
                        selected_decoder = "accurate"
                        selection_reason = "low_confidence_conservative"
        elif mode_key == "oracle_risk":
            selected_decoder = "accurate" if record.risk_label else "fast"
            selection_reason = "oracle_label"
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        selected_latency = (
            shaped_accurate_latency if selected_decoder == "accurate" else shaped_fast_latency
        )
        selected_latencies.append(selected_latency)
        selected_predictions.append(
            np.asarray(
                (
                    record.accurate_prediction
                    if selected_decoder == "accurate"
                    else record.fast_prediction
                ),
                dtype=np.int8,
            ).reshape(-1)
        )
        selected_decoders.append(selected_decoder)
        _advance_worker_state(
            record,
            selected_latency,
            decision_worker_available_times,
            decision_finish_times,
            config,
            decision_commit_state,
        )
        decision_rows.append(
            {
                "shot_id": record.shot_id,
                "mode": mode,
                "selected_decoder": selected_decoder,
                "selection_reason": selection_reason,
                "risk_label": int(record.risk_label),
                "heuristic_risk_score": risk_score,
                "accurate_latency_us": float(record.accurate_latency_us),
                "fast_latency_us": float(record.fast_latency_us),
                "shaped_accurate_latency_us": float(shaped_accurate_latency),
                "shaped_fast_latency_us": float(shaped_fast_latency),
                "predecode_latency_us": float(predecode_effect.get("predecode_latency_us", 0.0)),
                "predecode_accept_estimate": bool(
                    predecode_effect.get("predecode_accept_estimate", False)
                ),
                "predecode_validation_pass_estimate": bool(
                    predecode_effect.get("predecode_validation_pass_estimate", False)
                ),
                "predecode_confidence_estimate": float(
                    predecode_effect.get("predecode_confidence_estimate", 0.0)
                ),
                "predecode_risk_estimate": float(
                    predecode_effect.get("predecode_risk_estimate", 1.0)
                ),
                "estimated_residual_reduction": float(
                    predecode_effect.get("estimated_residual_reduction", 0.0)
                ),
                "validation_enabled": bool(predecode_effect.get("validation_enabled", False)),
                "abstention_enabled": bool(predecode_effect.get("abstention_enabled", False)),
                "estimated_backlog_before_arrival": int(
                    context["estimated_backlog_before_arrival"]
                ),
                "estimated_pauli_frame_lag": int(context["estimated_pauli_frame_lag"]),
                "committed_prefix_at_arrival": context["committed_prefix_at_arrival"],
                "deadline_slack_us": float(context["deadline_slack_us"]),
                "boundary_drain": bool(context["boundary_drain"]),
                "rounds_until_boundary": int(context["rounds_until_boundary"]),
                "overload_mode": bool(context["overload_mode"]),
                "ai_risk_score": ai_risk_score,
                "ai_runtime_score": ai_runtime_score,
                "ai_hard_runtime_score": ai_hard_runtime_score,
                "ai_runtime_pred": ai_runtime_pred,
                "ai_confidence": ai_confidence,
                "fast_wrong_prob": fast_wrong_prob,
                "fast_logical_fail_prob": fast_logical_fail_prob,
                "hard_runtime_prob": hard_runtime_prob,
                "safe_fast_prob": safe_fast_prob,
                "combined_fast_risk": combined_fast_risk,
                "fast_path_certified": bool(
                    predecode_effect.get(
                        "fast_path_certified",
                        predecode_effect.get("predecode_accept_estimate", False),
                    )
                ),
                "weak_validation_pass_estimate": bool(
                    predecode_effect.get("weak_validation_pass_estimate", False)
                ),
                "effective_predecode_confidence_threshold": float(
                    predecode_effect.get(
                        "effective_predecode_confidence_threshold",
                        config.predecoder.confidence_threshold,
                    )
                ),
                "effective_predecode_risk_threshold": float(
                    predecode_effect.get(
                        "effective_predecode_risk_threshold", config.predecoder.risk_threshold
                    )
                ),
                "effective_max_cluster_size": float(
                    predecode_effect.get(
                        "effective_max_cluster_size", config.predecoder.max_cluster_size
                    )
                ),
                "predecode_overload_policy": bool(
                    predecode_effect.get("predecode_overload_policy", False)
                ),
                "ai_model_type": (
                    model_type if mode_key in {"ai_risk", "rt_qec_ai", "rt_qec_learned"} else None
                ),
                "ai_history_length": (
                    history_length
                    if mode_key in {"ai_risk", "rt_qec_ai", "rt_qec_learned"}
                    else None
                ),
                "fast_wrong_vs_accurate": int(record.fast_wrong_vs_accurate),
                "fast_logical_fail": int(record.fast_logical_fail),
                "hard_runtime": int(record.hard_runtime),
            }
        )
    events = simulate_realtime_queue(
        records,
        selected_latencies,
        selected_predictions,
        config,
        selected_decoders,
        ordered_commit=ordered_commit,
    )
    decisions = pd.DataFrame(decision_rows)
    if mode_key in {"ai_risk", "rt_qec_ai", "rt_qec_learned"} and not decisions.empty:
        ai_columns = [
            "shot_id",
            "ai_model_type",
            "ai_history_length",
            "ai_risk_score",
            "ai_hard_runtime_score",
            "ai_runtime_pred",
            "ai_confidence",
            "fast_wrong_prob",
            "fast_logical_fail_prob",
            "hard_runtime_prob",
            "safe_fast_prob",
            "combined_fast_risk",
            "selected_decoder",
        ]
        events = events.drop(columns=["selected_decoder"], errors="ignore").merge(
            decisions[ai_columns], on="shot_id", how="left"
        )
    elif mode_key not in {"ai_risk", "rt_qec_ai", "rt_qec_learned"} and not events.empty:
        events["ai_model_type"] = None
        events["ai_history_length"] = None
        events["ai_risk_score"] = None
        events["ai_hard_runtime_score"] = None
        events["ai_runtime_pred"] = None
        events["ai_confidence"] = None
        events["fast_wrong_prob"] = None
        events["fast_logical_fail_prob"] = None
        events["hard_runtime_prob"] = None
        events["safe_fast_prob"] = None
        events["combined_fast_risk"] = None
    predictions = pd.DataFrame(
        [
            {
                "shot_id": record.shot_id,
                "mode": mode,
                "selected_decoder": selected_decoders[idx],
                "selected_prediction": _jsonable_prediction(selected_predictions[idx]),
                "accurate_prediction": _jsonable_prediction(record.accurate_prediction),
                "fast_prediction": _jsonable_prediction(record.fast_prediction),
                "observable": _jsonable_prediction(record.observable),
            }
            for idx, record in enumerate(records)
        ]
    )
    metrics = compute_mode_metrics(events, predictions, decisions, config)
    metrics.update(
        {
            "mode": mode,
            "ai_risk_available": bool(ai_available),
            "real_qec": bool(records[0].metadata.get("toy", False) is False) if records else False,
            "timing_mode": (
                str(records[0].metadata.get("timing_mode", "unknown")) if records else "unknown"
            ),
            "hard_runtime_label_valid": (
                bool(records[0].metadata.get("hard_runtime_label_valid", True)) if records else True
            ),
            "risk_false_negative_rate": (
                float(
                    np.mean(
                        [
                            int(record.risk_label == 1 and selected_decoders[idx] == "fast")
                            for idx, record in enumerate(records)
                        ]
                    )
                )
                if records
                else 0.0
            ),
            "risk_false_positive_rate": (
                float(
                    np.mean(
                        [
                            int(record.risk_label == 0 and selected_decoders[idx] == "accurate")
                            for idx, record in enumerate(records)
                        ]
                    )
                )
                if records
                else 0.0
            ),
            "logical_error_x_latency": float(
                metrics["logical_error_rate"] * metrics["mean_latency_us"]
            ),
        }
    )
    return ModeEvaluationResult(
        mode=mode, metrics=metrics, events=events, decisions=decisions, predictions=predictions
    )


def simulate_realtime_queue(
    records: list[RealStreamShotRecord],
    selected_latencies: list[float],
    selected_predictions: list[np.ndarray],
    config: ProjectConfig,
    selected_decoders: list[str] | None = None,
    *,
    ordered_commit: bool = False,
) -> pd.DataFrame:
    """Simulate a realtime queue over shared shots and configured workers."""
    selected_decoders = selected_decoders or ["accurate"] * len(records)
    if ordered_commit:
        return _simulate_ordered_commit_queue(
            records,
            selected_latencies,
            selected_predictions,
            config,
            selected_decoders,
        )
    num_workers = max(int(config.runtime.num_workers), 1)
    worker_available_times = [0.0 for _ in range(num_workers)]
    finish_times: list[float] = []
    event_rows: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        arrival_time = float(record.shot_id) * float(config.runtime.round_period_us)
        deadline = arrival_time + float(config.runtime.decode_deadline_us)
        unfinished_before_arrival = sum(1 for time_us in finish_times if time_us > arrival_time)
        worker_idx = min(range(num_workers), key=lambda worker: worker_available_times[worker])
        start_time = max(arrival_time, worker_available_times[worker_idx])
        finish_time = start_time + float(selected_latencies[idx])
        worker_available_times[worker_idx] = finish_time
        finish_times.append(finish_time)
        response_time = finish_time - arrival_time
        deadline_miss = bool(finish_time > deadline)
        backlog = unfinished_before_arrival + 1
        pauli_frame_lag = unfinished_before_arrival
        lag_violation = bool(pauli_frame_lag > int(config.runtime.max_pauli_frame_lag))
        observable = _observable_array(record.observable)
        prediction = np.asarray(selected_predictions[idx], dtype=np.int8).reshape(-1)
        logical_error = bool(np.any(prediction != observable))
        logical_boundary, rounds_until_boundary, boundary_drain = _boundary_context(record, config)
        boundary_commit_success = bool((not deadline_miss) and not lag_violation)
        event_rows.append(
            {
                "shot_id": record.shot_id,
                "arrival_time_us": arrival_time,
                "start_time_us": start_time,
                "finish_time_us": finish_time,
                "worker_id": worker_idx,
                "latency_us": float(selected_latencies[idx]),
                "response_time_us": response_time,
                "deadline_us": deadline,
                "deadline_miss": deadline_miss,
                "backlog": backlog,
                "pauli_frame_lag": pauli_frame_lag,
                "pauli_frame_lag_violation": lag_violation,
                "selected_decoder": selected_decoders[idx],
                "prediction": _jsonable_prediction(prediction),
                "observable": _jsonable_prediction(observable),
                "logical_error": logical_error,
                "logical_boundary": logical_boundary,
                "rounds_until_boundary": rounds_until_boundary,
                "boundary_drain": boundary_drain,
                "boundary_commit_success": boundary_commit_success,
            }
        )
    return pd.DataFrame(event_rows)


def _simulate_ordered_commit_queue(
    records: list[RealStreamShotRecord],
    selected_latencies: list[float],
    selected_predictions: list[np.ndarray],
    config: ProjectConfig,
    selected_decoders: list[str],
) -> pd.DataFrame:
    """Simulate parallel decode completion followed by in-order frame commitment."""
    num_jobs = len(records)
    if not (
        len(selected_latencies) == num_jobs
        and len(selected_predictions) == num_jobs
        and len(selected_decoders) == num_jobs
    ):
        raise ValueError("records, latencies, predictions, and decoders must have equal lengths")
    if num_jobs == 0:
        return pd.DataFrame()

    num_workers = max(int(config.runtime.num_workers), 1)
    worker_available_times = [0.0 for _ in range(num_workers)]
    finish_times: list[float] = []
    commit_state = OrderedCommitState()
    event_rows: list[dict[str, Any]] = []

    for idx, record in enumerate(records):
        job_id = int(record.shot_id)
        arrival_time = float(job_id) * float(config.runtime.round_period_us)
        deadline = arrival_time + float(config.runtime.decode_deadline_us)
        unfinished_before_arrival = sum(1 for time_us in finish_times if time_us > arrival_time)
        pauli_frame_lag = commit_state.lag_at_arrival(job_id, arrival_time)
        committed_prefix_at_arrival = commit_state.committed_prefix
        worker_idx = min(range(num_workers), key=lambda worker: worker_available_times[worker])
        start_time = max(arrival_time, worker_available_times[worker_idx])
        completion_time = start_time + float(selected_latencies[idx])
        worker_available_times[worker_idx] = completion_time
        finish_times.append(completion_time)
        commit_state.schedule_completion(job_id, completion_time)

        response_to_decode = completion_time - arrival_time
        decode_deadline_miss = bool(completion_time > deadline)
        backlog = unfinished_before_arrival + 1
        lag_violation = bool(pauli_frame_lag > int(config.runtime.max_pauli_frame_lag))
        observable = _observable_array(record.observable)
        prediction = np.asarray(selected_predictions[idx], dtype=np.int8).reshape(-1)
        logical_error = bool(np.any(prediction != observable))
        logical_boundary, rounds_until_boundary, boundary_drain = _boundary_context(record, config)
        event_rows.append(
            {
                "shot_id": job_id,
                "arrival_time_us": arrival_time,
                "start_time_us": start_time,
                "finish_time_us": completion_time,
                "completion_time_us": completion_time,
                "worker_id": worker_idx,
                "latency_us": float(selected_latencies[idx]),
                "response_time_us": response_to_decode,
                "response_to_decode_us": response_to_decode,
                "deadline_us": deadline,
                "decode_deadline_miss": decode_deadline_miss,
                "backlog": backlog,
                "committed_prefix_at_arrival": committed_prefix_at_arrival,
                "pauli_frame_lag": pauli_frame_lag,
                "pauli_frame_lag_violation": lag_violation,
                "selected_decoder": selected_decoders[idx],
                "prediction": _jsonable_prediction(prediction),
                "observable": _jsonable_prediction(observable),
                "logical_error": logical_error,
                "logical_boundary": logical_boundary,
                "rounds_until_boundary": rounds_until_boundary,
                "boundary_drain": boundary_drain,
                "ordered_commit_enabled": True,
            }
        )

    events = pd.DataFrame(event_rows)
    completion_times = events["completion_time_us"].to_numpy(dtype=float)
    commit_times = np.maximum.accumulate(completion_times)
    deadlines = events["deadline_us"].to_numpy(dtype=float)
    arrivals = events["arrival_time_us"].to_numpy(dtype=float)
    job_ids = events["shot_id"].to_numpy(dtype=np.int64)
    committed_prefix_at_deadline = np.searchsorted(commit_times, deadlines, side="right") - 1
    commit_deadline_miss = committed_prefix_at_deadline < job_ids

    events["commit_time_us"] = commit_times
    events["response_to_commit_us"] = commit_times - arrivals
    events["committed_prefix_at_deadline"] = committed_prefix_at_deadline.astype(np.int64)
    events["commit_deadline_miss"] = commit_deadline_miss
    # In the ordered-commit model, the externally visible deadline is met only
    # after the complete prefix is committed. Keep the decode outcome explicit.
    events["deadline_miss"] = commit_deadline_miss
    events["boundary_prerequisites_committed"] = ~commit_deadline_miss
    events["boundary_commit_success"] = ~commit_deadline_miss
    return events


def compute_mode_metrics(
    events: pd.DataFrame,
    predictions: pd.DataFrame,
    decisions: pd.DataFrame,
    config: ProjectConfig,
) -> dict[str, Any]:
    """Compute RTSS-style mode metrics."""
    if events.empty:
        return {
            "logical_error_rate": 0.0,
            "mean_latency_us": 0.0,
            "p95_latency_us": 0.0,
            "p99_latency_us": 0.0,
            "p999_latency_us": 0.0,
            "mean_response_time_us": 0.0,
            "p95_response_time_us": 0.0,
            "p99_response_time_us": 0.0,
            "p999_response_time_us": 0.0,
            "deadline_miss_ratio": 0.0,
            "mean_response_to_decode_us": 0.0,
            "p95_response_to_decode_us": 0.0,
            "p99_response_to_decode_us": 0.0,
            "p999_response_to_decode_us": 0.0,
            "max_response_to_decode_us": 0.0,
            "decode_deadline_miss_ratio": 0.0,
            "mean_response_to_commit_us": 0.0,
            "p95_response_to_commit_us": 0.0,
            "p99_response_to_commit_us": 0.0,
            "p999_response_to_commit_us": 0.0,
            "max_response_to_commit_us": 0.0,
            "commit_deadline_miss_ratio": 0.0,
            "mean_backlog": 0.0,
            "p99_backlog": 0.0,
            "max_backlog": 0.0,
            "mean_pauli_frame_lag": 0.0,
            "p99_pauli_frame_lag": 0.0,
            "max_pauli_frame_lag": 0.0,
            "pauli_frame_lag_violation_ratio": 0.0,
            "boundary_commit_success_rate": 0.0,
            "fast_selection_rate": 0.0,
            "accurate_selection_rate": 0.0,
            "accept_rate": 0.0,
            "abstention_rate": 0.0,
            "false_accept_rate": 0.0,
            "accepted_error_rate": 0.0,
            "accepted_fast_logical_fail_rate": 0.0,
            "validation_pass_rate": 0.0,
            "predecode_accept_rate": 0.0,
            "mean_estimated_residual_reduction": 0.0,
            "residual_density_reduction": 0.0,
            "residual_graph_size_reduction": 0.0,
            "num_shots": 0,
            "num_workers": max(int(config.runtime.num_workers), 1),
            "real_qec": False,
            "timing_mode": "unknown",
        }
    latencies = events["latency_us"].to_numpy(dtype=float)
    decode_response_times = events.get(
        "response_to_decode_us", events["response_time_us"]
    ).to_numpy(dtype=float)
    commit_response_times = events.get(
        "response_to_commit_us", events["response_time_us"]
    ).to_numpy(dtype=float)
    response_times = (
        commit_response_times if "response_to_commit_us" in events else decode_response_times
    )
    decode_deadline_misses = events.get("decode_deadline_miss", events["deadline_miss"])
    commit_deadline_misses = events.get("commit_deadline_miss", events["deadline_miss"])
    deadline_misses = (
        commit_deadline_misses if "commit_deadline_miss" in events else decode_deadline_misses
    )
    backlogs = events["backlog"].to_numpy(dtype=float)
    lags = events["pauli_frame_lag"].to_numpy(dtype=float)
    logical_errors = events["logical_error"].to_numpy(dtype=bool)
    fast_rate = (
        float((decisions["selected_decoder"] == "fast").mean()) if not decisions.empty else 0.0
    )
    accurate_rate = (
        float((decisions["selected_decoder"] == "accurate").mean()) if not decisions.empty else 0.0
    )
    accepted_mask = (
        decisions["predecode_accept_estimate"].to_numpy(dtype=bool)
        if "predecode_accept_estimate" in decisions and not decisions.empty
        else np.asarray([], dtype=bool)
    )
    validation_mask = (
        decisions["predecode_validation_pass_estimate"].to_numpy(dtype=bool)
        if "predecode_validation_pass_estimate" in decisions and not decisions.empty
        else np.asarray([], dtype=bool)
    )
    safe_accept = (
        decisions["risk_label"].to_numpy(dtype=int) == 0
        if "risk_label" in decisions and not decisions.empty
        else np.asarray([], dtype=bool)
    )
    accepted_fast_logical_correct = (
        decisions["fast_logical_fail"].to_numpy(dtype=int) == 0
        if "fast_logical_fail" in decisions and not decisions.empty
        else safe_accept
    )
    predecode_accept_rate = (
        float(decisions["predecode_accept_estimate"].astype(bool).mean())
        if "predecode_accept_estimate" in decisions and not decisions.empty
        else 0.0
    )
    mean_residual_reduction = (
        float(decisions["estimated_residual_reduction"].to_numpy(dtype=float).mean())
        if "estimated_residual_reduction" in decisions and not decisions.empty
        else 0.0
    )
    boundary_events = (
        events[events["logical_boundary"].astype(bool)]
        if "logical_boundary" in events
        else pd.DataFrame()
    )
    boundary_commit_success_rate = (
        float(boundary_events["boundary_commit_success"].astype(float).mean())
        if not boundary_events.empty
        else 1.0
    )
    return {
        "logical_error_rate": float(np.mean(logical_errors.astype(float))),
        "mean_latency_us": float(np.mean(latencies)),
        "p95_latency_us": float(np.percentile(latencies, 95)),
        "p99_latency_us": float(np.percentile(latencies, 99)),
        "p999_latency_us": float(np.percentile(latencies, 99.9)),
        "mean_response_time_us": float(np.mean(response_times)),
        "p95_response_time_us": float(np.percentile(response_times, 95)),
        "p99_response_time_us": float(np.percentile(response_times, 99)),
        "p999_response_time_us": float(np.percentile(response_times, 99.9)),
        "deadline_miss_ratio": float(deadline_misses.astype(float).mean()),
        "mean_response_to_decode_us": float(np.mean(decode_response_times)),
        "p95_response_to_decode_us": float(np.percentile(decode_response_times, 95)),
        "p99_response_to_decode_us": float(np.percentile(decode_response_times, 99)),
        "p999_response_to_decode_us": float(np.percentile(decode_response_times, 99.9)),
        "max_response_to_decode_us": float(np.max(decode_response_times)),
        "decode_deadline_miss_ratio": float(decode_deadline_misses.astype(float).mean()),
        "mean_response_to_commit_us": float(np.mean(commit_response_times)),
        "p95_response_to_commit_us": float(np.percentile(commit_response_times, 95)),
        "p99_response_to_commit_us": float(np.percentile(commit_response_times, 99)),
        "p999_response_to_commit_us": float(np.percentile(commit_response_times, 99.9)),
        "max_response_to_commit_us": float(np.max(commit_response_times)),
        "commit_deadline_miss_ratio": float(commit_deadline_misses.astype(float).mean()),
        "mean_backlog": float(np.mean(backlogs)),
        "p99_backlog": float(np.percentile(backlogs, 99)),
        "max_backlog": float(np.max(backlogs)),
        "mean_pauli_frame_lag": float(np.mean(lags)),
        "p99_pauli_frame_lag": float(np.percentile(lags, 99)),
        "max_pauli_frame_lag": float(np.max(lags)),
        "pauli_frame_lag_violation_ratio": float(
            events["pauli_frame_lag_violation"].astype(float).mean()
        ),
        "boundary_commit_success_rate": boundary_commit_success_rate,
        "fast_selection_rate": fast_rate,
        "accurate_selection_rate": accurate_rate,
        "accept_rate": accept_rate(accepted_mask),
        "abstention_rate": abstention_rate(accepted_mask),
        "false_accept_rate": false_accept_rate(accepted_mask, safe_accept),
        "accepted_error_rate": accepted_error_rate(accepted_mask, safe_accept),
        "accepted_fast_logical_fail_rate": accepted_error_rate(
            accepted_mask, accepted_fast_logical_correct
        ),
        "validation_pass_rate": validation_pass_rate(validation_mask),
        "predecode_accept_rate": predecode_accept_rate,
        "mean_estimated_residual_reduction": mean_residual_reduction,
        "residual_density_reduction": mean_residual_reduction,
        "residual_graph_size_reduction": mean_residual_reduction,
        "num_shots": int(len(events)),
        "num_workers": max(int(config.runtime.num_workers), 1),
        "real_qec": True,
        "timing_mode": "per_record",
        "ordered_commit_enabled": bool(
            "ordered_commit_enabled" in events
            and events["ordered_commit_enabled"].astype(bool).all()
        ),
    }


def _metadata_columns_for_grouping(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "setting_id",
        "episode_id",
        "stream_id",
        "distance",
        "physical_error_rate",
        "noise_scenario",
        "difficulty_tier",
    ]
    return [column for column in preferred if column in frame.columns]


def _summarize_event_group(group: pd.DataFrame) -> dict[str, Any]:
    latencies = group["latency_us"].to_numpy(dtype=float)
    decode_response_times = group.get("response_to_decode_us", group["response_time_us"]).to_numpy(
        dtype=float
    )
    commit_response_times = group.get("response_to_commit_us", group["response_time_us"]).to_numpy(
        dtype=float
    )
    response_times = (
        commit_response_times if "response_to_commit_us" in group else decode_response_times
    )
    decode_deadline_misses = group.get("decode_deadline_miss", group["deadline_miss"])
    commit_deadline_misses = group.get("commit_deadline_miss", group["deadline_miss"])
    deadline_misses = (
        commit_deadline_misses if "commit_deadline_miss" in group else decode_deadline_misses
    )
    lags = group["pauli_frame_lag"].to_numpy(dtype=float)
    backlogs = group["backlog"].to_numpy(dtype=float)
    return {
        "num_shots": int(len(group)),
        "logical_error_rate": float(group["logical_error"].astype(float).mean()),
        "deadline_miss_ratio": float(deadline_misses.astype(float).mean()),
        "mean_latency_us": float(np.mean(latencies)),
        "p99_latency_us": float(np.percentile(latencies, 99)),
        "p999_latency_us": float(np.percentile(latencies, 99.9)),
        "mean_response_time_us": float(np.mean(response_times)),
        "p99_response_time_us": float(np.percentile(response_times, 99)),
        "mean_response_to_decode_us": float(np.mean(decode_response_times)),
        "p99_response_to_decode_us": float(np.percentile(decode_response_times, 99)),
        "max_response_to_decode_us": float(np.max(decode_response_times)),
        "decode_deadline_miss_ratio": float(decode_deadline_misses.astype(float).mean()),
        "mean_response_to_commit_us": float(np.mean(commit_response_times)),
        "p99_response_to_commit_us": float(np.percentile(commit_response_times, 99)),
        "max_response_to_commit_us": float(np.max(commit_response_times)),
        "commit_deadline_miss_ratio": float(commit_deadline_misses.astype(float).mean()),
        "mean_backlog": float(np.mean(backlogs)),
        "max_backlog": float(np.max(backlogs)),
        "mean_pauli_frame_lag": float(np.mean(lags)),
        "p99_pauli_frame_lag": float(np.percentile(lags, 99)),
        "max_pauli_frame_lag": float(np.max(lags)),
        "pauli_frame_lag_violation_ratio": float(
            group["pauli_frame_lag_violation"].astype(float).mean()
        ),
        "boundary_commit_success_rate": (
            float(
                group.loc[group["logical_boundary"].astype(bool), "boundary_commit_success"]
                .astype(float)
                .mean()
            )
            if bool(group["logical_boundary"].astype(bool).any())
            else 1.0
        ),
        "fast_selection_rate": float((group["selected_decoder"] == "fast").astype(float).mean()),
        "accurate_selection_rate": float(
            (group["selected_decoder"] == "accurate").astype(float).mean()
        ),
    }


def build_grouped_summary(
    mode_results: list[ModeEvaluationResult],
    records_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Return per-mode, per-setting summaries for paper tables."""
    if not mode_results or records_frame.empty:
        return pd.DataFrame()
    group_columns = _metadata_columns_for_grouping(records_frame)
    if not group_columns:
        return pd.DataFrame()
    metadata = records_frame[["shot_id", *group_columns]].copy()
    rows: list[dict[str, Any]] = []
    for result in mode_results:
        if result.events.empty:
            continue
        merged = result.events.merge(metadata, on="shot_id", how="left", suffixes=("", "_record"))
        grouped = merged.groupby(group_columns, dropna=False, sort=True)
        for group_key, group in grouped:
            values = group_key if isinstance(group_key, tuple) else (group_key,)
            row = {"mode": result.mode}
            row.update({column: value for column, value in zip(group_columns, values)})
            row.update(_summarize_event_group(group))
            rows.append(row)
    return pd.DataFrame(rows)


def run_real_stream_eval(
    config: ProjectConfig,
    risk_checkpoint: str | None = None,
    out_dir: str | Path | None = None,
    risk_dataset_path: str | Path | None = None,
    split: str = "test",
    calibration_path: str | Path | None = None,
    train_seed: int | None = None,
    eval_seed: int | None = None,
) -> dict[str, Any]:
    """Run paired real-stream evaluation across configured modes."""
    eval_source = "risk_dataset" if risk_dataset_path is not None else "generated"
    config = copy.deepcopy(config)
    if eval_source == "risk_dataset":
        records, metadata = load_real_stream_records_from_risk_dataset(
            risk_dataset_path, split=split
        )
        split_info: dict[str, Any] = {
            "train_indices": metadata.get("splits", {}).get("train_indices", []),
            "val_indices": metadata.get("splits", {}).get("val_indices", []),
            "test_indices": metadata.get("splits", {}).get("test_indices", []),
        }
        test_records = sorted(records, key=lambda record: record.shot_id)
    else:
        protocol_train_seed = int(
            train_seed if train_seed is not None else config.data_protocol.train_seed
        )
        protocol_eval_seed = int(
            eval_seed if eval_seed is not None else config.data_protocol.eval_seed
        )
        if (
            bool(config.data_protocol.require_eval_seed_different_from_train_seed)
            and protocol_eval_seed == protocol_train_seed
        ):
            raise ValueError(
                f"eval_seed ({protocol_eval_seed}) must differ from train_seed ({protocol_train_seed}) for held-out seed evaluation."
            )
        config.seed = protocol_eval_seed
        records, metadata = build_real_stream_records(config)
        split_info = split_records(
            records,
            test_fraction=float(config.qec.test_fraction),
            val_fraction=float(config.risk_training.val_fraction),
            seed=int(config.seed),
        )
        split_key = str(split).lower()
        if split_key == "all":
            selected_records = [*split_info["train"], *split_info["val"], *split_info["test"]]
        elif split_key in {"train", "val", "test"}:
            selected_records = split_info[split_key]
        else:
            raise ValueError(f"generated eval split must be train/val/test/all, got {split!r}.")
        test_records = sorted(selected_records, key=lambda record: record.shot_id)
        metadata["train_seed"] = protocol_train_seed
        metadata["eval_seed"] = protocol_eval_seed
    risk_model = None
    normalization = None
    risk_checkpoint_metadata: dict[str, Any] | None = None
    ai_available = False
    if (
        risk_checkpoint is not None
        and str(risk_checkpoint).lower() != "none"
        and Path(risk_checkpoint).exists()
    ):
        risk_model, normalization, risk_checkpoint_metadata = load_risk_profiler_checkpoint(
            risk_checkpoint, device=config.device
        )
        ai_available = True
    thresholds: dict[str, float] = {}
    calibration_source = None
    if calibration_path is not None and str(calibration_path).lower() != "none":
        with Path(calibration_path).open("r", encoding="utf-8") as handle:
            calibration = json.load(handle)
        thresholds = {
            "ai_risk_threshold": float(calibration["selected_ai_risk_threshold"]),
            "ai_confidence_threshold": float(calibration["selected_ai_confidence_threshold"]),
            "safe_fast_threshold": float(
                calibration.get(
                    "selected_safe_fast_threshold",
                    calibration.get("selected_ai_safe_fast_threshold", 0.5),
                )
            ),
        }
        config.risk_eval.ai_risk_threshold = thresholds["ai_risk_threshold"]
        config.risk_eval.ai_confidence_threshold = thresholds["ai_confidence_threshold"]
        calibration_source = str(calibration_path)
    out_root = Path(out_dir or config.output_dir)
    records_frame = pd.DataFrame(
        [
            {
                "shot_id": record.shot_id,
                "syndrome": record.syndrome.tolist(),
                "observable": _jsonable_prediction(record.observable),
                "accurate_prediction": _jsonable_prediction(record.accurate_prediction),
                "fast_prediction": _jsonable_prediction(record.fast_prediction),
                "accurate_latency_us": record.accurate_latency_us,
                "fast_latency_us": record.fast_latency_us,
                "risk_label": record.risk_label,
                "hard_runtime": record.hard_runtime,
                "fast_wrong_vs_accurate": record.fast_wrong_vs_accurate,
                "fast_logical_fail": record.fast_logical_fail,
                "setting_id": record.metadata.get("setting_id"),
                "episode_id": record.metadata.get("episode_id"),
                "stream_id": record.metadata.get("stream_id"),
                "arrival_order": record.metadata.get("arrival_order"),
                "distance": record.metadata.get("distance"),
                "physical_error_rate": record.metadata.get("physical_error_rate"),
                "noise_scenario": record.metadata.get("noise_scenario"),
                "difficulty_tier": record.metadata.get("difficulty_tier"),
                "burst_context": record.metadata.get("burst_context"),
                "backlog_proxy": record.metadata.get("backlog_proxy"),
                "residual_or_candidate_complexity": record.metadata.get(
                    "residual_or_candidate_complexity"
                ),
                "feature_names": record.feature_names,
                "features": record.features.tolist(),
                "metadata": record.metadata,
            }
            for record in test_records
        ]
    )
    records_path = ensure_parent(out_root / "records.csv")
    records_frame.to_csv(records_path, index=False)
    mode_results: list[ModeEvaluationResult] = []
    for mode in config.risk_eval.modes:
        if mode in {"ai_risk", "rt_qec_ai", "rt_qec_learned"} and not ai_available:
            continue
        result = evaluate_mode_on_records(
            test_records,
            mode=mode,
            config=config,
            risk_model=risk_model,
            normalization=normalization,
            risk_metadata=risk_checkpoint_metadata,
            thresholds=thresholds,
        )
        mode_dir = out_root / mode
        if config.outputs.save_events:
            result.events.to_csv(ensure_parent(mode_dir / "events.csv"), index=False)
        if config.outputs.save_decisions:
            result.decisions.to_csv(ensure_parent(mode_dir / "decisions.csv"), index=False)
        if config.outputs.save_predictions:
            result.predictions.to_csv(ensure_parent(mode_dir / "predictions.csv"), index=False)
        mode_results.append(result)
    summary = compare_modes_summary([result.metrics for result in mode_results])
    save_summary_metrics_csv(summary, out_root / "summary_metrics.csv")
    frontend_columns = [
        "mode",
        "logical_error_rate",
        "accept_rate",
        "abstention_rate",
        "false_accept_rate",
        "accepted_error_rate",
        "validation_pass_rate",
        "predecode_accept_rate",
        "mean_estimated_residual_reduction",
        "residual_density_reduction",
        "residual_graph_size_reduction",
        "fast_selection_rate",
        "accurate_selection_rate",
        "boundary_commit_success_rate",
    ]
    summary[[column for column in frontend_columns if column in summary.columns]].to_csv(
        ensure_parent(out_root / "frontend_contract_table.csv"),
        index=False,
    )
    grouped_summary = build_grouped_summary(mode_results, records_frame)
    if not grouped_summary.empty:
        grouped_summary.to_csv(ensure_parent(out_root / "setting_summary.csv"), index=False)
    if bool(config.outputs.save_plots_ready_csv) and mode_results:
        plot_rows: list[dict[str, Any]] = []
        for result in mode_results:
            for event_row in result.events.to_dict(orient="records"):
                event_row["mode"] = result.mode
                plot_rows.append(event_row)
        pd.DataFrame(plot_rows).to_csv(ensure_parent(out_root / "plot_events.csv"), index=False)
        pareto_columns = [
            "mode",
            "logical_error_rate",
            "deadline_miss_ratio",
            "p99_latency_us",
            "p999_latency_us",
            "p99_response_time_us",
            "p999_response_time_us",
            "p99_pauli_frame_lag",
            "max_pauli_frame_lag",
            "pauli_frame_lag_violation_ratio",
            "boundary_commit_success_rate",
            "fast_selection_rate",
            "accurate_selection_rate",
            "accept_rate",
            "abstention_rate",
            "false_accept_rate",
            "accepted_error_rate",
            "validation_pass_rate",
            "predecode_accept_rate",
            "mean_estimated_residual_reduction",
        ]
        summary[[column for column in pareto_columns if column in summary.columns]].to_csv(
            ensure_parent(out_root / "pareto_summary.csv"),
            index=False,
        )
    if config.outputs.save_predictions:
        np.savez_compressed(
            ensure_parent(out_root / "predictions.npz"),
            **{
                result.mode: np.asarray(
                    [
                        np.asarray(record, dtype=object)
                        for record in result.predictions["selected_prediction"].tolist()
                    ],
                    dtype=object,
                )
                for result in mode_results
            },
        )
    warnings: list[str] = []
    checkpoint_train_hash = None
    if risk_checkpoint_metadata:
        train_split_meta = dict(risk_checkpoint_metadata.get("train_split_metadata", {}))
        checkpoint_train_hash = train_split_meta.get(
            "train_indices_hash",
            risk_checkpoint_metadata.get("train_indices_hash"),
        )
    dataset_train_hash = metadata.get("train_indices_hash")
    split_match = True
    if checkpoint_train_hash and dataset_train_hash and checkpoint_train_hash != dataset_train_hash:
        split_match = False
        warnings.append("checkpoint train split hash does not match risk_dataset train split hash")
    metrics_payload = {
        "summary": summary.to_dict(orient="records"),
        "metadata": {
            **metadata,
            "train_indices": split_info.get("train_indices", []),
            "val_indices": split_info.get("val_indices", []),
            "test_indices": split_info.get("test_indices", []),
            "split_seed": int(config.seed),
            "split_policy": metadata.get("split_policy", "generated_random"),
            "eval_source": eval_source,
            "eval_split": str(split),
            "calibration_source": calibration_source,
            "train_indices_hash": metadata.get(
                "train_indices_hash", hash_indices(split_info.get("train_indices", []))
            ),
            "test_indices_hash": metadata.get(
                "test_indices_hash", hash_indices(split_info.get("test_indices", []))
            ),
            "checkpoint_train_split_hash": checkpoint_train_hash,
            "split_match": bool(split_match),
            "warnings": warnings,
            "timing_mode": metadata.get("timing_mode", "unknown"),
            "hard_runtime_label_valid": bool(metadata.get("hard_runtime_label_valid", True)),
            "ai_risk_available": ai_available,
            "risk_checkpoint": None if risk_checkpoint is None else str(risk_checkpoint),
            "risk_checkpoint_metadata": risk_checkpoint_metadata,
            "real_qec": bool(metadata.get("real_qec", False)),
            "fallback_reason": metadata.get("fallback_reason"),
        },
        "real_qec": bool(metadata.get("real_qec", False)),
        "fallback_reason": metadata.get("fallback_reason"),
        "eval_source": eval_source,
        "eval_split": str(split),
        "calibration_source": calibration_source,
        "timing_mode": metadata.get("timing_mode", "unknown"),
        "hard_runtime_label_valid": bool(metadata.get("hard_runtime_label_valid", True)),
        "split_match": bool(split_match),
        "warnings": warnings,
    }
    ai_modes = {"ai_risk", "rt_qec_ai", "rt_qec_learned"}
    if not ai_available and any(mode in ai_modes for mode in config.risk_eval.modes):
        metrics_payload["metadata"]["ai_risk_skipped"] = True
    save_metrics_json(metrics_payload, out_root / "metrics.json")
    return metrics_payload
