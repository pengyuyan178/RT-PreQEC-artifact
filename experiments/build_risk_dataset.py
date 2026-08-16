"""Build a risk-profiler dataset from real or fallback decoding records."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import copy
import json

import numpy as np

from rt_preqec.config import ProjectConfig
from rt_preqec.data.dem_parser import filter_local_candidates, index_candidates_by_detector, parse_dem_error_candidates
from rt_preqec.data.patch_extractor import extract_detector_patches_from_flat_syndrome
from rt_preqec.data.risk_dataset import (
    RiskDatasetSplits,
    build_risk_samples_from_decoding_records,
    create_split_indices,
    hash_indices,
    risk_dataset_split_sidecar_path,
    save_risk_dataset,
    save_risk_dataset_splits,
    splits_from_dict,
)
from rt_preqec.data.stim_surface_code import generate_surface_code_samples
from rt_preqec.decoders.lookup_decoder import LookupDecoder
from rt_preqec.decoders.pymatching_decoder import PyMatchingDecoder, measure_per_shot_decoder_latency
from rt_preqec.decoders.union_find_decoder import UnionFindDecoder
from rt_preqec.metrics.qec_metrics import logical_error_rate
from rt_preqec.utils import dump_json, ensure_parent


RTSS_16SETTINGS_PRESET = "rtss_16settings_480k"


RTSS_16SETTINGS_TABLE: list[dict[str, Any]] = [
    {
        "setting_id": 0,
        "difficulty_tier": "easy",
        "distance": 3,
        "rounds": 3,
        "physical_error_rate": 0.001,
        "noise_scenario": "circuit_depolarizing",
        "purpose": "simple local clusters, high safe_fast rate",
    },
    {
        "setting_id": 1,
        "difficulty_tier": "easy",
        "distance": 3,
        "rounds": 9,
        "physical_error_rate": 0.001,
        "noise_scenario": "circuit_depolarizing",
        "purpose": "easy but longer temporal history",
    },
    {
        "setting_id": 2,
        "difficulty_tier": "easy",
        "distance": 5,
        "rounds": 5,
        "physical_error_rate": 0.001,
        "noise_scenario": "circuit_depolarizing",
        "purpose": "larger code but still simple",
    },
    {
        "setting_id": 3,
        "difficulty_tier": "medium",
        "distance": 5,
        "rounds": 15,
        "physical_error_rate": 0.002,
        "noise_scenario": "circuit_depolarizing",
        "purpose": "medium normal workload",
    },
    {
        "setting_id": 4,
        "difficulty_tier": "medium",
        "distance": 5,
        "rounds": 15,
        "physical_error_rate": 0.005,
        "noise_scenario": "circuit_depolarizing",
        "purpose": "more nontrivial clusters",
    },
    {
        "setting_id": 5,
        "difficulty_tier": "medium",
        "distance": 7,
        "rounds": 7,
        "physical_error_rate": 0.002,
        "noise_scenario": "circuit_depolarizing",
        "purpose": "distance scaling",
    },
    {
        "setting_id": 6,
        "difficulty_tier": "medium",
        "distance": 7,
        "rounds": 21,
        "physical_error_rate": 0.002,
        "noise_scenario": "measurement_heavy",
        "purpose": "temporal measurement-noise patterns",
    },
    {
        "setting_id": 7,
        "difficulty_tier": "hard",
        "distance": 7,
        "rounds": 21,
        "physical_error_rate": 0.005,
        "noise_scenario": "biased_noise",
        "purpose": "biased X/Z patterns and harder fast decoder decisions",
    },
    {
        "setting_id": 8,
        "difficulty_tier": "hard",
        "distance": 7,
        "rounds": 21,
        "physical_error_rate": 0.005,
        "noise_scenario": "local_hotspot",
        "purpose": "localized high-density clusters",
    },
    {
        "setting_id": 9,
        "difficulty_tier": "hard",
        "distance": 7,
        "rounds": 21,
        "physical_error_rate": 0.005,
        "noise_scenario": "burst_noise",
        "purpose": "burst-driven backlog and syndrome_tail positives",
    },
    {
        "setting_id": 10,
        "difficulty_tier": "hard",
        "distance": 9,
        "rounds": 9,
        "physical_error_rate": 0.002,
        "noise_scenario": "circuit_depolarizing",
        "purpose": "larger code normal workload",
    },
    {
        "setting_id": 11,
        "difficulty_tier": "hard",
        "distance": 9,
        "rounds": 27,
        "physical_error_rate": 0.003,
        "noise_scenario": "measurement_heavy",
        "purpose": "long temporal history and runtime variation",
    },
    {
        "setting_id": 12,
        "difficulty_tier": "stress",
        "distance": 9,
        "rounds": 27,
        "physical_error_rate": 0.005,
        "noise_scenario": "local_hotspot",
        "purpose": "high residual graph complexity",
    },
    {
        "setting_id": 13,
        "difficulty_tier": "stress",
        "distance": 9,
        "rounds": 27,
        "physical_error_rate": 0.005,
        "noise_scenario": "burst_noise",
        "purpose": "tail latency / backlog stress",
    },
    {
        "setting_id": 14,
        "difficulty_tier": "stress",
        "distance": 9,
        "rounds": 27,
        "physical_error_rate": 0.0075,
        "noise_scenario": "leakage_like_temporal",
        "purpose": "temporal correlation and hard_runtime positives",
    },
    {
        "setting_id": 15,
        "difficulty_tier": "stress",
        "distance": 9,
        "rounds": 27,
        "physical_error_rate": 0.0075,
        "noise_scenario": "mixed_realistic",
        "purpose": "combined hard case, fast_wrong and logical_fail positives",
    },
]


DIFFICULTY_TIER_IDS = {"easy": 0, "medium": 1, "hard": 2, "stress": 3}


RTSS_LABEL_TARGETS: dict[int, dict[str, float]] = {
    0: {"fast_wrong": 0.035, "fast_fail": 0.020, "hard_runtime": 0.015, "tail": 0.010},
    1: {"fast_wrong": 0.050, "fast_fail": 0.030, "hard_runtime": 0.020, "tail": 0.020},
    2: {"fast_wrong": 0.060, "fast_fail": 0.040, "hard_runtime": 0.020, "tail": 0.025},
    3: {"fast_wrong": 0.100, "fast_fail": 0.070, "hard_runtime": 0.040, "tail": 0.045},
    4: {"fast_wrong": 0.150, "fast_fail": 0.110, "hard_runtime": 0.060, "tail": 0.060},
    5: {"fast_wrong": 0.120, "fast_fail": 0.085, "hard_runtime": 0.040, "tail": 0.050},
    6: {"fast_wrong": 0.135, "fast_fail": 0.090, "hard_runtime": 0.055, "tail": 0.075},
    7: {"fast_wrong": 0.190, "fast_fail": 0.150, "hard_runtime": 0.075, "tail": 0.080},
    8: {"fast_wrong": 0.210, "fast_fail": 0.165, "hard_runtime": 0.090, "tail": 0.130},
    9: {"fast_wrong": 0.220, "fast_fail": 0.170, "hard_runtime": 0.100, "tail": 0.150},
    10: {"fast_wrong": 0.140, "fast_fail": 0.095, "hard_runtime": 0.060, "tail": 0.060},
    11: {"fast_wrong": 0.165, "fast_fail": 0.120, "hard_runtime": 0.085, "tail": 0.100},
    12: {"fast_wrong": 0.245, "fast_fail": 0.205, "hard_runtime": 0.125, "tail": 0.170},
    13: {"fast_wrong": 0.255, "fast_fail": 0.215, "hard_runtime": 0.140, "tail": 0.180},
    14: {"fast_wrong": 0.275, "fast_fail": 0.235, "hard_runtime": 0.155, "tail": 0.170},
    15: {"fast_wrong": 0.285, "fast_fail": 0.245, "hard_runtime": 0.160, "tail": 0.180},
}


def _decoder_from_name(name: str, bundle: dict[str, Any]) -> Any:
    if name == "pymatching":
        return PyMatchingDecoder.from_detector_error_model(bundle.get("dem"))
    if name == "union_find":
        return UnionFindDecoder()
    return LookupDecoder()


def _observable_prediction_from_decode_result(result: Any) -> np.ndarray:
    correction = np.asarray(result.correction, dtype=np.int8).reshape(-1)
    if correction.size == 0:
        return np.zeros((1,), dtype=np.int8)
    return np.asarray([int(correction.sum() % 2)], dtype=np.int8)


def _observable_prediction_from_batch(batch_predictions: np.ndarray) -> np.ndarray:
    array = np.asarray(batch_predictions, dtype=np.int8)
    if array.ndim == 1:
        return array.reshape(1, -1) if array.size else np.zeros((1, 1), dtype=np.int8)
    return array


def _rounds_for_setting(config: ProjectConfig, distance: int, multiplier: int | float) -> int:
    if config.qec_grid.get("rounds_mode") == "proportional":
        return max(1, int(round(float(distance) * float(multiplier))))
    return int(config.qec_grid.get("rounds", config.qec.rounds))


def _settings_from_config(config: ProjectConfig) -> list[dict[str, Any]]:
    if not config.qec_grid:
        return [
            {
                "distance": int(config.qec.distances[0]),
                "rounds": int(config.qec.rounds),
                "physical_error_rate": float(config.qec.physical_error_rates[0]),
                "shots": int(config.risk_dataset.num_shots or config.qec.num_shots),
                "noise_scenario": {"name": config.qec.noise_model, "type": "stim_generated"},
            }
        ]
    distances = [int(value) for value in config.qec_grid.get("distances", config.qec.distances)]
    physical_error_rates = [
        float(value) for value in config.qec_grid.get("physical_error_rates", config.qec.physical_error_rates)
    ]
    multipliers = config.qec_grid.get("rounds_multiplier", [1])
    shots = int(config.qec_grid.get("shots_per_setting", config.risk_dataset.num_shots or config.qec.num_shots))
    explicit_smoke_shots = int(config.risk_dataset.num_shots or config.qec.num_shots)
    if explicit_smoke_shots > 0 and explicit_smoke_shots < shots:
        return [
            {
                "distance": int(config.qec.distances[0]),
                "rounds": int(config.qec.rounds),
                "physical_error_rate": float(config.qec.physical_error_rates[0]),
                "shots": explicit_smoke_shots,
                "noise_scenario": {"name": config.qec.noise_model, "type": "stim_generated"},
            }
        ]
    scenarios = config.noise_scenarios or [{"name": config.qec.noise_model, "type": "stim_generated"}]
    settings: list[dict[str, Any]] = []
    for scenario in scenarios:
        for distance in distances:
            for multiplier in multipliers:
                rounds = _rounds_for_setting(config, distance, multiplier)
                for error_rate in physical_error_rates:
                    settings.append(
                        {
                            "distance": distance,
                            "rounds": rounds,
                            "physical_error_rate": error_rate,
                            "shots": shots,
                            "noise_scenario": dict(scenario),
                        }
                    )
    max_settings = config.qec_grid.get("max_settings")
    if max_settings is not None and int(max_settings) > 0 and len(settings) > int(max_settings):
        positions = np.linspace(0, len(settings) - 1, int(max_settings), dtype=int)
        settings = [settings[int(pos)] for pos in positions.tolist()]
    max_total_shots = config.qec_grid.get("max_total_shots")
    if max_total_shots is not None and int(max_total_shots) > 0 and settings:
        total_budget = int(max_total_shots)
        base = max(total_budget // len(settings), 1)
        remainder = max(total_budget - base * len(settings), 0)
        for idx, setting in enumerate(settings):
            setting["shots"] = base + (1 if idx < remainder else 0)
    return settings


def _postprocess_syndrome(
    syndrome: np.ndarray,
    scenario: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    scenario_type = str(scenario.get("type", "stim_generated"))
    if scenario_type != "postprocess_syndrome":
        return syndrome, {"noise_postprocess": False}
    rng = np.random.default_rng(seed)
    output = np.asarray(syndrome, dtype=np.int8).copy()
    flat = output.reshape(output.shape[0], -1)
    name = str(scenario.get("name", "postprocess"))
    if name == "burst_toy" or "burst_probability" in scenario:
        burst_probability = float(scenario.get("burst_probability", 0.02))
        burst_length = max(int(scenario.get("burst_length", 3)), 1)
        burst_flip_probability = float(scenario.get("burst_flip_probability", 0.2))
        for row in flat:
            if rng.random() < burst_probability and row.size:
                start = int(rng.integers(0, max(row.size - burst_length + 1, 1)))
                end = min(start + burst_length, row.size)
                flips = rng.random(end - start) < burst_flip_probability
                row[start:end] = np.bitwise_xor(row[start:end], flips.astype(np.int8))
    elif name == "hotspot_toy" or "hotspot_fraction" in scenario:
        hotspot_fraction = float(scenario.get("hotspot_fraction", 0.1))
        multiplier = max(float(scenario.get("hotspot_flip_multiplier", 3.0)), 1.0)
        width = max(int(round(flat.shape[1] * hotspot_fraction)), 1) if flat.shape[1] else 0
        if width:
            columns = rng.choice(flat.shape[1], size=min(width, flat.shape[1]), replace=False)
            base_probability = min(0.5, 0.02 * multiplier)
            flips = rng.random((flat.shape[0], len(columns))) < base_probability
            flat[:, columns] = np.bitwise_xor(flat[:, columns], flips.astype(np.int8))
    return flat.reshape(output.shape), {
        "noise_postprocess": True,
        "used_for_risk_training": True,
        "used_for_final_qec_claim": False,
    }


def _configured_timing(config: ProjectConfig) -> dict[str, Any]:
    return {
        "warmup_shots": int(config.timing.warmup_shots),
        "repeat_per_shot": int(config.timing.repeat_per_shot),
        "max_timing_shots": config.timing.max_timing_shots,
        "statistic": str(config.timing.timing_statistic),
    }


def _rtss_setting_table(shots_per_setting: int) -> list[dict[str, Any]]:
    settings: list[dict[str, Any]] = []
    for row in RTSS_16SETTINGS_TABLE:
        setting = dict(row)
        setting["shots"] = int(shots_per_setting)
        setting["noise_scenario_requested"] = str(setting["noise_scenario"])
        setting["noise_scenario_implemented"] = f"compact_rtss_{setting['noise_scenario']}"
        setting["approximation_note"] = (
            "Main-dev synthetic RTSS workload: compact detector syndromes and "
            "Stim-parameter-inspired syndrome/runtime labels are generated "
            "directly for scalable model development. This is not paper_final "
            "held-out evaluation data."
        )
        settings.append(setting)
    return settings


def _scenario_id(name: str) -> int:
    scenario_names = sorted({str(setting["noise_scenario"]) for setting in RTSS_16SETTINGS_TABLE})
    return int(scenario_names.index(str(name)))


def _detector_count(distance: int, rounds: int) -> int:
    return int(max(1, int(rounds) * (int(distance) * int(distance) - 1)))


def _scenario_multipliers(name: str) -> dict[str, float]:
    table = {
        "circuit_depolarizing": {"weight": 1.0, "runtime": 1.0, "candidate": 1.0},
        "measurement_heavy": {"weight": 1.35, "runtime": 1.15, "candidate": 1.1},
        "biased_noise": {"weight": 1.25, "runtime": 1.1, "candidate": 1.25},
        "local_hotspot": {"weight": 1.55, "runtime": 1.25, "candidate": 1.45},
        "burst_noise": {"weight": 1.60, "runtime": 1.35, "candidate": 1.35},
        "leakage_like_temporal": {"weight": 1.75, "runtime": 1.50, "candidate": 1.55},
        "mixed_realistic": {"weight": 1.90, "runtime": 1.60, "candidate": 1.70},
    }
    return table.get(str(name), table["circuit_depolarizing"])


def _temporal_context(name: str, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return burst, hotspot, and leakage context signals in shot order."""
    x = np.arange(int(n), dtype=np.float32)
    burst = np.zeros(int(n), dtype=np.float32)
    hotspot = np.zeros(int(n), dtype=np.float32)
    leakage = np.zeros(int(n), dtype=np.float32)
    scenario = str(name)
    if scenario in {"burst_noise", "mixed_realistic"}:
        period = 1800 if scenario == "mixed_realistic" else 2400
        width = 360 if scenario == "mixed_realistic" else 420
        offset = int(rng.integers(0, period))
        phase = (x + offset) % period
        burst = np.clip(1.0 - phase / max(width, 1), 0.0, 1.0).astype(np.float32)
    if scenario in {"local_hotspot", "mixed_realistic"}:
        period = 2600 if scenario == "mixed_realistic" else 3200
        width = 900 if scenario == "mixed_realistic" else 760
        offset = int(rng.integers(0, period))
        phase = (x + offset) % period
        hotspot = (phase < width).astype(np.float32)
        hotspot *= rng.uniform(0.75, 1.0, size=int(n)).astype(np.float32)
    if scenario in {"leakage_like_temporal", "mixed_realistic"}:
        state = 0.0
        values: list[float] = []
        event_probability = 0.010 if scenario == "mixed_realistic" else 0.014
        decay = 0.985 if scenario == "mixed_realistic" else 0.990
        for _ in range(int(n)):
            if rng.random() < event_probability:
                state += float(rng.uniform(0.6, 1.2))
            state *= decay
            values.append(min(state, 1.5))
        leakage = np.asarray(values, dtype=np.float32)
        if leakage.max(initial=0.0) > 0.0:
            leakage = leakage / float(leakage.max())
    return burst, hotspot, leakage


def _select_top(score: np.ndarray, count: int, eligible: np.ndarray | None = None) -> np.ndarray:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    selected = np.zeros(len(score), dtype=bool)
    count = int(max(count, 0))
    if count <= 0 or len(score) == 0:
        return selected
    eligible_mask = np.ones(len(score), dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool).reshape(-1)
    eligible_indices = np.flatnonzero(eligible_mask)
    if eligible_indices.size == 0:
        return selected
    count = min(count, int(eligible_indices.size))
    eligible_scores = score[eligible_indices]
    chosen_local = np.argpartition(eligible_scores, -count)[-count:]
    selected[eligible_indices[chosen_local]] = True
    return selected


def _split_segments_for_setting(shots_per_setting: int) -> list[tuple[str, int, int]]:
    n = int(shots_per_setting)
    train_end = int(round(n * 0.6))
    val_end = train_end + int(round(n * 0.2))
    return [("train", 0, train_end), ("val", train_end, val_end), ("test", val_end, n)]


def _make_rtss_setting_arrays(
    setting: dict[str, Any],
    shots_per_setting: int,
    seed: int,
) -> dict[str, np.ndarray]:
    setting_id = int(setting["setting_id"])
    rng = np.random.default_rng(int(seed) + 7919 * (setting_id + 1))
    n = int(shots_per_setting)
    local_index = np.arange(n, dtype=np.int32)
    distance = int(setting["distance"])
    rounds = int(setting["rounds"])
    p = float(setting["physical_error_rate"])
    tier = str(setting["difficulty_tier"])
    tier_id = int(DIFFICULTY_TIER_IDS[tier])
    scenario = str(setting["noise_scenario"])
    scenario_id = _scenario_id(scenario)
    detector_count = _detector_count(distance, rounds)
    multipliers = _scenario_multipliers(scenario)
    burst_context, hotspot_context, leakage_context = _temporal_context(scenario, n, rng)
    normalized_time = local_index.astype(np.float32) / max(float(n - 1), 1.0)
    seasonal = np.sin(2.0 * np.pi * (normalized_time * (1.0 + 0.13 * setting_id))).astype(np.float32)
    base_noise = rng.normal(0.0, 1.0, size=n).astype(np.float32)
    complexity_signal = (
        0.55 * base_noise
        + 0.75 * burst_context
        + 0.65 * hotspot_context
        + 0.80 * leakage_context
        + 0.20 * seasonal
        + 0.18 * tier_id
        + 0.10 * np.log1p(detector_count)
    ).astype(np.float32)
    fast_fail_score = complexity_signal + rng.normal(0.0, 0.35, size=n).astype(np.float32)
    fast_wrong_score = (
        complexity_signal
        + 0.35 * hotspot_context
        + 0.20 * leakage_context
        + rng.normal(0.0, 0.40, size=n).astype(np.float32)
    )
    hard_score = (
        complexity_signal
        + 0.55 * burst_context
        + 0.60 * leakage_context
        + 0.0008 * detector_count
        + rng.normal(0.0, 0.30, size=n).astype(np.float32)
    )
    tail_score = (
        complexity_signal
        + 0.70 * hotspot_context
        + 0.70 * burst_context
        + 0.35 * leakage_context
        + rng.normal(0.0, 0.35, size=n).astype(np.float32)
    )
    targets = RTSS_LABEL_TARGETS[setting_id]
    fast_wrong = np.zeros(n, dtype=bool)
    fast_logical_fail = np.zeros(n, dtype=bool)
    hard_runtime = np.zeros(n, dtype=bool)
    syndrome_tail = np.zeros(n, dtype=bool)
    for _, start, end in _split_segments_for_setting(n):
        segment = slice(start, end)
        segment_n = int(end - start)
        fail_count = int(round(segment_n * float(targets["fast_fail"])))
        wrong_count = int(round(segment_n * float(targets["fast_wrong"])))
        hard_count = int(round(segment_n * float(targets["hard_runtime"])))
        tail_count = int(round(segment_n * float(targets["tail"])))
        fail_segment = _select_top(fast_fail_score[segment], fail_count)
        fast_logical_fail[segment] = fail_segment
        wrong_segment = fail_segment.copy()
        extra_wrong = max(wrong_count - int(wrong_segment.sum()), 0)
        if extra_wrong:
            wrong_segment |= _select_top(fast_wrong_score[segment], extra_wrong, eligible=~wrong_segment)
        fast_wrong[segment] = wrong_segment
        hard_runtime[segment] = _select_top(hard_score[segment], hard_count)
        syndrome_tail[segment] = _select_top(tail_score[segment], tail_count)

    accurate_logical_fail = np.logical_xor(fast_wrong, fast_logical_fail)
    safe_for_fast = np.logical_not(np.logical_or(fast_wrong, fast_logical_fail))
    scheduler_risk = np.logical_or.reduce([fast_wrong, fast_logical_fail, hard_runtime, syndrome_tail])
    labels = np.stack(
        [
            fast_wrong.astype(np.int8),
            fast_logical_fail.astype(np.int8),
            accurate_logical_fail.astype(np.int8),
            hard_runtime.astype(np.int8),
            scheduler_risk.astype(np.int8),
            syndrome_tail.astype(np.int8),
            safe_for_fast.astype(np.int8),
            syndrome_tail.astype(np.int8),
        ],
        axis=1,
    )

    base_weight_mean = max(0.15, detector_count * p * 1.35 * float(multipliers["weight"]))
    weight_mean = (
        base_weight_mean
        + 1.10 * fast_wrong.astype(np.float32)
        + 1.80 * fast_logical_fail.astype(np.float32)
        + 2.60 * hard_runtime.astype(np.float32)
        + 5.00 * syndrome_tail.astype(np.float32)
        + 3.00 * burst_context
        + 2.40 * hotspot_context
        + 2.70 * leakage_context
    )
    syndrome_weight = rng.poisson(np.maximum(weight_mean, 0.01)).astype(np.float32)
    syndrome_weight = np.minimum(syndrome_weight, float(detector_count)).astype(np.float32)
    syndrome_density = syndrome_weight / max(float(detector_count), 1.0)
    active_time_bins = np.minimum(rounds, np.maximum(0, np.ceil(syndrome_weight / max(distance, 1)))).astype(np.float32)
    active_time_span = np.where(syndrome_weight > 0, np.minimum(rounds, active_time_bins + 1.0), 0.0).astype(np.float32)
    active_x_span = np.where(syndrome_weight > 0, np.minimum(distance, np.sqrt(syndrome_weight + 1.0)), 0.0).astype(np.float32)
    active_y_span = np.where(syndrome_weight > 0, np.minimum(distance, np.sqrt(syndrome_weight + 1.0) * 0.9), 0.0).astype(np.float32)
    candidate_base = float(multipliers["candidate"]) * (1.0 + 0.28 * tier_id)
    mean_candidate_count = (
        candidate_base
        + 0.04 * syndrome_weight
        + 0.55 * fast_wrong.astype(np.float32)
        + 0.70 * hard_runtime.astype(np.float32)
        + 0.40 * hotspot_context
    ).astype(np.float32)
    max_candidate_count = np.ceil(mean_candidate_count + 1.0 + 2.0 * syndrome_tail.astype(np.float32)).astype(np.float32)
    fraction_active_with_candidate = np.where(syndrome_weight > 0, np.clip(0.45 + 0.08 * tier_id + 0.20 * syndrome_tail, 0.0, 1.0), 0.0)
    residual_complexity = (
        0.30 * syndrome_weight
        + 2.0 * mean_candidate_count
        + 3.0 * hard_runtime.astype(np.float32)
        + 2.5 * syndrome_tail.astype(np.float32)
        + 1.5 * burst_context
        + 1.2 * hotspot_context
    ).astype(np.float32)
    num_patches = np.minimum(syndrome_weight, 32.0).astype(np.float32)
    mean_patch_active = np.where(num_patches > 0, np.maximum(1.0, syndrome_weight / np.maximum(num_patches, 1.0)), 0.0)
    max_patch_active = np.minimum(syndrome_weight, mean_patch_active + 2.0 + 4.0 * syndrome_tail).astype(np.float32)
    mean_patch_size = np.full(n, float(max(distance * distance // 2, 1)), dtype=np.float32)
    max_patch_size = np.full(n, float(max(distance * distance, 1)), dtype=np.float32)
    mean_patch_density = np.where(mean_patch_size > 0, mean_patch_active / mean_patch_size, 0.0).astype(np.float32)
    max_patch_density = np.where(max_patch_size > 0, max_patch_active / max_patch_size, 0.0).astype(np.float32)

    estimated_fast_runtime = (
        0.35
        + 0.010 * syndrome_weight
        + 0.002 * detector_count / max(rounds, 1)
        + 0.12 * fast_wrong.astype(np.float32)
    ).astype(np.float32)
    measured_fast_runtime = np.maximum(
        0.05,
        estimated_fast_runtime + rng.normal(0.0, 0.035, size=n).astype(np.float32),
    ).astype(np.float32)
    accurate_runtime = (
        1.2
        + 0.0065 * detector_count * float(multipliers["runtime"])
        + 0.18 * syndrome_weight
        + 1.35 * residual_complexity
        + 8.0 * hard_runtime.astype(np.float32)
        + 4.0 * burst_context
        + 4.5 * leakage_context
        + rng.normal(0.0, 0.25 + 0.03 * tier_id, size=n).astype(np.float32)
    )
    measured_accurate_runtime = np.maximum(0.1, accurate_runtime).astype(np.float32)
    backlog_proxy = np.zeros(n, dtype=np.float32)
    backlog = 0.0
    for idx in range(n):
        workload = float(measured_accurate_runtime[idx]) if hard_runtime[idx] else float(measured_fast_runtime[idx])
        backlog = max(0.0, backlog + workload - 1.0)
        backlog *= 0.92
        backlog_proxy[idx] = min(backlog, 256.0)

    feature_names = np.asarray(
        [
            "syndrome_weight",
            "syndrome_density",
            "num_detectors",
            "has_any_detection",
            "active_detector_fraction",
            "layout_present",
            "num_active_time_bins",
            "active_time_span",
            "active_x_span",
            "active_y_span",
            "layout_missing_flag",
            "mean_candidate_count_active",
            "max_candidate_count_active",
            "fraction_active_with_candidate",
            "candidates_missing_flag",
            "weight_squared",
            "log1p_weight",
            "num_patches",
            "mean_patch_active",
            "max_patch_active",
            "mean_patch_size",
            "max_patch_size",
            "mean_patch_density",
            "max_patch_density",
            "difficulty_tier_id",
            "distance",
            "rounds",
            "physical_error_rate",
            "noise_scenario_id",
            "shot_phase",
            "burst_context",
            "hotspot_context",
            "leakage_context",
        ],
        dtype="<U128",
    )
    features = np.stack(
        [
            syndrome_weight,
            syndrome_density,
            np.full(n, float(detector_count), dtype=np.float32),
            (syndrome_weight > 0).astype(np.float32),
            syndrome_density,
            np.zeros(n, dtype=np.float32),
            active_time_bins,
            active_time_span,
            active_x_span,
            active_y_span,
            np.ones(n, dtype=np.float32),
            mean_candidate_count,
            max_candidate_count,
            fraction_active_with_candidate.astype(np.float32),
            np.zeros(n, dtype=np.float32),
            syndrome_weight * syndrome_weight,
            np.log1p(syndrome_weight).astype(np.float32),
            num_patches,
            mean_patch_active.astype(np.float32),
            max_patch_active,
            mean_patch_size,
            max_patch_size,
            mean_patch_density,
            max_patch_density,
            np.full(n, float(tier_id), dtype=np.float32),
            np.full(n, float(distance), dtype=np.float32),
            np.full(n, float(rounds), dtype=np.float32),
            np.full(n, float(p), dtype=np.float32),
            np.full(n, float(scenario_id), dtype=np.float32),
            normalized_time.astype(np.float32),
            burst_context,
            hotspot_context,
            leakage_context,
        ],
        axis=1,
    ).astype(np.float32)

    compact_len = 128
    compact_probability = np.clip(
        (0.10 + syndrome_weight) / max(float(compact_len), 1.0),
        0.0,
        0.85,
    ).astype(np.float32)
    syndromes = (rng.random((n, compact_len), dtype=np.float32) < compact_probability[:, None]).astype(np.int8)
    actual_observable = rng.integers(0, 2, size=(n, 1), dtype=np.int8)
    accurate_prediction = np.bitwise_xor(actual_observable, accurate_logical_fail.astype(np.int8).reshape(-1, 1))
    fast_prediction = np.bitwise_xor(actual_observable, fast_logical_fail.astype(np.int8).reshape(-1, 1))
    runtimes = np.stack(
        [
            measured_accurate_runtime,
            measured_fast_runtime,
            estimated_fast_runtime,
            measured_fast_runtime,
            measured_accurate_runtime,
        ],
        axis=1,
    ).astype(np.float32)
    return {
        "features": features,
        "feature_names": feature_names,
        "syndromes": syndromes,
        "actual_observables": actual_observable,
        "accurate_predictions": accurate_prediction,
        "fast_predictions": fast_prediction,
        "labels": labels,
        "runtimes": runtimes,
        "syndrome_weight": syndrome_weight.astype(np.float32),
        "syndrome_weight_tail": syndrome_tail.astype(np.int8),
        "detector_count": np.full(n, detector_count, dtype=np.int32),
        "residual_or_candidate_complexity": residual_complexity.astype(np.float32),
        "estimated_fast_runtime_us": estimated_fast_runtime.astype(np.float32),
        "measured_fast_runtime_us": measured_fast_runtime.astype(np.float32),
        "measured_accurate_runtime_us": measured_accurate_runtime.astype(np.float32),
        "backlog_proxy": backlog_proxy.astype(np.float32),
        "difficulty_tier": np.full(n, tier, dtype="<U16"),
        "difficulty_tier_ids": np.full(n, tier_id, dtype=np.int8),
        "distance": np.full(n, distance, dtype=np.int16),
        "rounds": np.full(n, rounds, dtype=np.int16),
        "physical_error_rate": np.full(n, p, dtype=np.float32),
        "noise_scenario": np.full(n, scenario, dtype="<U32"),
        "seed": np.full(n, int(seed) + setting_id, dtype=np.int64),
        "shot_index_within_setting": local_index.astype(np.int32),
        "burst_context": burst_context.astype(np.float32),
        "hotspot_context": hotspot_context.astype(np.float32),
        "leakage_context": leakage_context.astype(np.float32),
    }


def _concatenate_rtss_settings(settings: list[dict[str, Any]], shots_per_setting: int, seed: int) -> dict[str, Any]:
    chunks = [_make_rtss_setting_arrays(setting, shots_per_setting, seed) for setting in settings]
    total = int(len(settings) * int(shots_per_setting))
    setting_ids = np.concatenate(
        [np.full(int(shots_per_setting), int(setting["setting_id"]), dtype=np.int16) for setting in settings]
    )
    global_index = np.arange(total, dtype=np.int64)
    payload: dict[str, Any] = {
        "features": np.concatenate([chunk["features"] for chunk in chunks], axis=0).astype(np.float32),
        "feature_names": chunks[0]["feature_names"],
        "syndromes": np.concatenate([chunk["syndromes"] for chunk in chunks], axis=0).astype(np.int8),
        "actual_observables": np.concatenate([chunk["actual_observables"] for chunk in chunks], axis=0).astype(np.int8),
        "accurate_predictions": np.concatenate([chunk["accurate_predictions"] for chunk in chunks], axis=0).astype(np.int8),
        "fast_predictions": np.concatenate([chunk["fast_predictions"] for chunk in chunks], axis=0).astype(np.int8),
        "labels": np.concatenate([chunk["labels"] for chunk in chunks], axis=0).astype(np.int8),
        "runtimes": np.concatenate([chunk["runtimes"] for chunk in chunks], axis=0).astype(np.float32),
        "setting_ids": setting_ids,
        "episode_ids": setting_ids.astype(np.int16),
        "stream_ids": setting_ids.astype(np.int16),
        "shot_index_within_setting": np.concatenate(
            [chunk["shot_index_within_setting"] for chunk in chunks], axis=0
        ).astype(np.int32),
        "global_index": global_index,
        "arrival_order": global_index.copy(),
        "stream_index": global_index.copy(),
        "sample_ids": global_index.copy(),
        "shot_ids": global_index.copy(),
    }
    for key in [
        "difficulty_tier",
        "difficulty_tier_ids",
        "distance",
        "rounds",
        "physical_error_rate",
        "noise_scenario",
        "seed",
        "syndrome_weight",
        "syndrome_weight_tail",
        "detector_count",
        "residual_or_candidate_complexity",
        "estimated_fast_runtime_us",
        "measured_fast_runtime_us",
        "measured_accurate_runtime_us",
        "backlog_proxy",
        "burst_context",
        "hotspot_context",
        "leakage_context",
    ]:
        payload[key] = np.concatenate([chunk[key] for chunk in chunks], axis=0)
    return payload


def _rates_for_indices(labels: np.ndarray, label_names: list[str], indices: np.ndarray) -> dict[str, dict[str, float | int]]:
    selected = labels[np.asarray(indices, dtype=np.int64)]
    return {
        name: {
            "rate": float(selected[:, idx].mean()) if len(selected) else 0.0,
            "positives": int(selected[:, idx].sum()) if len(selected) else 0,
        }
        for idx, name in enumerate(label_names)
    }


def _setting_count_dict(setting_ids: np.ndarray, indices: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(np.asarray(setting_ids)[np.asarray(indices, dtype=np.int64)], return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values.tolist(), counts.tolist())}


def _rtss_diagnostics(
    labels: np.ndarray,
    label_names: list[str],
    setting_ids: np.ndarray,
    splits: RiskDatasetSplits,
) -> dict[str, Any]:
    split_indices = {
        "train": np.asarray(splits.train_indices, dtype=np.int64),
        "val": np.asarray(splits.val_indices, dtype=np.int64),
        "test": np.asarray(splits.test_indices, dtype=np.int64),
    }
    global_indices = np.arange(len(labels), dtype=np.int64)
    diagnostics: dict[str, Any] = {
        "global": _rates_for_indices(labels, label_names, global_indices),
        "splits": {
            name: {
                "size": int(len(indices)),
                "setting_counts": _setting_count_dict(setting_ids, indices),
                "labels": _rates_for_indices(labels, label_names, indices),
            }
            for name, indices in split_indices.items()
        },
        "per_setting": {},
        "warnings": [],
    }
    for setting_id in sorted(set(int(value) for value in setting_ids.tolist())):
        indices = np.where(setting_ids == setting_id)[0]
        diagnostics["per_setting"][str(setting_id)] = {
            "size": int(len(indices)),
            "labels": _rates_for_indices(labels, label_names, indices),
        }

    label_aliases = {
        "fast_wrong_vs_accurate": "fast_wrong",
        "fast_logical_fail": "fast_logical_fail",
        "hard_runtime": "hard_runtime",
        "syndrome_weight_tail": "syndrome_tail",
    }
    global_ranges = {
        "fast_wrong_vs_accurate": (0.10, 0.30),
        "fast_logical_fail": (0.08, 0.30),
        "hard_runtime": (0.05, 0.20),
        "syndrome_weight_tail": (0.05, 0.20),
        "safe_for_fast": (0.60, 0.90),
    }
    for label_name, (low, high) in global_ranges.items():
        rate = float(diagnostics["global"].get(label_name, {}).get("rate", 0.0))
        if rate < low or rate > high:
            diagnostics["warnings"].append(
                f"{label_aliases.get(label_name, label_name)} global rate {rate:.4f} outside [{low:.2f}, {high:.2f}]"
            )
    minimums = {
        "train": {
            "fast_wrong_vs_accurate": 3000,
            "fast_logical_fail": 3000,
            "hard_runtime": 2000,
            "syndrome_weight_tail": 2000,
        },
        "val": {
            "fast_wrong_vs_accurate": 1000,
            "fast_logical_fail": 1000,
            "hard_runtime": 500,
            "syndrome_weight_tail": 500,
        },
        "test": {
            "fast_wrong_vs_accurate": 1000,
            "fast_logical_fail": 1000,
            "hard_runtime": 500,
            "syndrome_weight_tail": 500,
        },
    }
    for split_name, split_minimums in minimums.items():
        split_labels = diagnostics["splits"][split_name]["labels"]
        for label_name, minimum in split_minimums.items():
            positives = int(split_labels[label_name]["positives"])
            if positives < int(minimum):
                diagnostics["warnings"].append(
                    f"{split_name} {label_aliases.get(label_name, label_name)} positives {positives} below {minimum}; "
                    "enhance burst_noise/local_hotspot/leakage_like_temporal/mixed_realistic settings."
                )
    for label_name in ["fast_wrong_vs_accurate", "fast_logical_fail", "hard_runtime", "syndrome_weight_tail"]:
        train_rate = float(diagnostics["splits"]["train"]["labels"][label_name]["rate"])
        for split_name in ["val", "test"]:
            rate = float(diagnostics["splits"][split_name]["labels"][label_name]["rate"])
            if abs(train_rate - rate) > 0.05:
                diagnostics["warnings"].append(
                    f"{label_aliases.get(label_name, label_name)} train/{split_name} drift {abs(train_rate - rate):.4f} exceeds 0.05"
                )
    return diagnostics


def _print_rtss_diagnostics(diagnostics: dict[str, Any], label_names: list[str]) -> None:
    print("split size:")
    for split_name in ["train", "val", "test"]:
        print(f"  {split_name}: {diagnostics['splits'][split_name]['size']}")
    print("setting counts:")
    for split_name in ["train", "val", "test"]:
        print(f"  {split_name}: {diagnostics['splits'][split_name]['setting_counts']}")
    print("global label distribution:")
    for name in label_names:
        item = diagnostics["global"][name]
        print(f"  {name}: rate={float(item['rate']):.4f}, positives={int(item['positives'])}")
    for split_name in ["train", "val", "test"]:
        print(f"{split_name} label distribution:")
        for name in label_names:
            item = diagnostics["splits"][split_name]["labels"][name]
            print(f"  {name}: rate={float(item['rate']):.4f}, positives={int(item['positives'])}")
    print("per setting label distribution:")
    for setting_id, payload in diagnostics["per_setting"].items():
        compact = {
            name: round(float(payload["labels"][name]["rate"]), 4)
            for name in label_names
        }
        print(f"  setting {setting_id}: N={payload['size']} {compact}")
    if diagnostics["warnings"]:
        print("warnings:")
        for warning in diagnostics["warnings"]:
            print(f"  WARNING: {warning}")


def _run_rtss_16settings_build(
    config: ProjectConfig,
    out_path: str | Path,
    shots_per_setting: int,
    split_policy: str,
    verbose: bool,
) -> dict[str, Any]:
    if str(split_policy).lower() != "setting_stratified":
        raise ValueError("rtss_16settings_480k preset requires --split-policy setting_stratified")
    settings = _rtss_setting_table(shots_per_setting)
    payload = _concatenate_rtss_settings(settings, shots_per_setting, int(config.seed))
    labels = np.asarray(payload["labels"], dtype=np.int8)
    label_names = [
        "fast_wrong_vs_accurate",
        "fast_logical_fail",
        "accurate_logical_fail",
        "hard_runtime",
        "scheduler_risk_label",
        "syndrome_weight_tail",
        "safe_for_fast",
        "syndrome_tail",
    ]
    runtime_names = [
        "accurate_runtime_us",
        "fast_runtime_us",
        "estimated_fast_runtime_us",
        "measured_fast_runtime_us",
        "measured_accurate_runtime_us",
    ]
    split_payload = create_split_indices(
        num_samples=int(len(labels)),
        split_policy="setting_stratified",
        train_fraction=0.6,
        val_fraction=0.2,
        test_fraction=0.2,
        seed=int(config.seed),
        setting_ids=np.asarray(payload["setting_ids"], dtype=np.int16),
    )
    splits = splits_from_dict(split_payload)
    diagnostics = _rtss_diagnostics(labels, label_names, np.asarray(payload["setting_ids"]), splits)
    target = ensure_parent(out_path)
    split_path = risk_dataset_split_sidecar_path(target)
    metadata = {
        "dataset_role": "main-dev",
        "preset": RTSS_16SETTINGS_PRESET,
        "num_samples": int(len(labels)),
        "num_settings": int(len(settings)),
        "shots_per_setting": int(shots_per_setting),
        "feature_dim": int(payload["features"].shape[1]),
        "feature_names": [str(name) for name in payload["feature_names"].tolist()],
        "label_names": label_names,
        "runtime_names": runtime_names,
        "hard_runtime_label_valid": True,
        "timing_mode": "compact_rtss_synthetic_per_sample",
        "split_policy": splits.split_policy,
        "splits": splits.to_dict(),
        "split_path": str(split_path),
        "settings": settings,
        "difficulty_tier_id_to_name": {str(value): key for key, value in DIFFICULTY_TIER_IDS.items()},
        "sample_metadata_storage": "columnar_npz_arrays",
        "sample_metadata_fields": [
            "setting_ids",
            "difficulty_tier",
            "difficulty_tier_ids",
            "distance",
            "rounds",
            "physical_error_rate",
            "noise_scenario",
            "seed",
            "shot_index_within_setting",
            "global_index",
            "arrival_order",
            "stream_index",
            "estimated_fast_runtime_us",
            "measured_fast_runtime_us",
            "measured_accurate_runtime_us",
            "hard_runtime_label",
            "syndrome_weight",
            "syndrome_weight_tail",
            "detector_count",
            "residual_or_candidate_complexity",
            "fast_selection_oracle",
        ],
        "syndrome_storage": "compact_fixed_length",
        "compact_syndrome_len": int(payload["syndromes"].shape[1]),
        "noise_approximation_policy": (
            "Custom RTSS noise scenarios are approximated by compact syndrome, "
            "temporal context, hotspot/burst/leakage signals, and runtime-label "
            "simulation. Each setting records requested and implemented scenario names."
        ),
        "diagnostics": diagnostics,
    }
    syndrome_lengths = np.full(int(len(labels)), int(payload["syndromes"].shape[1]), dtype=np.int32)
    syndrome_mask = np.ones_like(payload["syndromes"], dtype=bool)
    np.savez_compressed(
        target,
        features=np.asarray(payload["features"], dtype=np.float32),
        syndromes=np.asarray(payload["syndromes"], dtype=np.int8),
        syndromes_padded=np.asarray(payload["syndromes"], dtype=np.int8),
        syndrome_lengths=syndrome_lengths,
        syndrome_mask=syndrome_mask,
        syndrome_storage=np.asarray("compact_fixed_length"),
        actual_observables=np.asarray(payload["actual_observables"], dtype=np.int8),
        accurate_predictions=np.asarray(payload["accurate_predictions"], dtype=np.int8),
        fast_predictions=np.asarray(payload["fast_predictions"], dtype=np.int8),
        labels=labels,
        label_names=np.asarray(label_names, dtype="<U64"),
        runtimes=np.asarray(payload["runtimes"], dtype=np.float32),
        runtime_names=np.asarray(runtime_names, dtype="<U64"),
        feature_names=np.asarray(payload["feature_names"], dtype="<U128"),
        metadata_json=json.dumps(metadata),
        setting_ids=np.asarray(payload["setting_ids"], dtype=np.int16),
        setting_id=np.asarray(payload["setting_ids"], dtype=np.int16),
        episode_ids=np.asarray(payload["episode_ids"], dtype=np.int16),
        stream_ids=np.asarray(payload["stream_ids"], dtype=np.int16),
        shot_index_within_setting=np.asarray(payload["shot_index_within_setting"], dtype=np.int32),
        global_index=np.asarray(payload["global_index"], dtype=np.int64),
        arrival_order=np.asarray(payload["arrival_order"], dtype=np.int64),
        stream_index=np.asarray(payload["stream_index"], dtype=np.int64),
        sample_ids=np.asarray(payload["sample_ids"], dtype=np.int64),
        shot_ids=np.asarray(payload["shot_ids"], dtype=np.int64),
        difficulty_tier=np.asarray(payload["difficulty_tier"], dtype="<U16"),
        difficulty_tier_ids=np.asarray(payload["difficulty_tier_ids"], dtype=np.int8),
        distance=np.asarray(payload["distance"], dtype=np.int16),
        rounds=np.asarray(payload["rounds"], dtype=np.int16),
        physical_error_rate=np.asarray(payload["physical_error_rate"], dtype=np.float32),
        noise_scenario=np.asarray(payload["noise_scenario"], dtype="<U32"),
        seed=np.asarray(payload["seed"], dtype=np.int64),
        syndrome_weight=np.asarray(payload["syndrome_weight"], dtype=np.float32),
        syndrome_weight_tail=np.asarray(payload["syndrome_weight_tail"], dtype=np.int8),
        detector_count=np.asarray(payload["detector_count"], dtype=np.int32),
        residual_or_candidate_complexity=np.asarray(payload["residual_or_candidate_complexity"], dtype=np.float32),
        estimated_fast_runtime_us=np.asarray(payload["estimated_fast_runtime_us"], dtype=np.float32),
        measured_fast_runtime_us=np.asarray(payload["measured_fast_runtime_us"], dtype=np.float32),
        measured_accurate_runtime_us=np.asarray(payload["measured_accurate_runtime_us"], dtype=np.float32),
        hard_runtime_label=labels[:, label_names.index("hard_runtime")].astype(np.int8),
        fast_selection_oracle=labels[:, label_names.index("safe_for_fast")].astype(np.int8),
        confidence_targets=labels[:, label_names.index("safe_for_fast")].astype(np.float32),
        backlog_proxy=np.asarray(payload["backlog_proxy"], dtype=np.float32),
        burst_context=np.asarray(payload["burst_context"], dtype=np.float32),
        hotspot_context=np.asarray(payload["hotspot_context"], dtype=np.float32),
        leakage_context=np.asarray(payload["leakage_context"], dtype=np.float32),
    )
    save_risk_dataset_splits(splits, split_path)
    summary = {
        "total_samples": int(len(labels)),
        "num_shots": int(len(labels)),
        "feature_dim": int(payload["features"].shape[1]),
        "feature_names": [str(name) for name in payload["feature_names"].tolist()],
        "positive_risk_label_rate": float(diagnostics["global"]["scheduler_risk_label"]["rate"]),
        "risk_label_positive_rate": float(diagnostics["global"]["scheduler_risk_label"]["rate"]),
        "hard_runtime_rate": float(diagnostics["global"]["hard_runtime"]["rate"]),
        "fast_wrong_rate": float(diagnostics["global"]["fast_wrong_vs_accurate"]["rate"]),
        "fast_logical_fail_rate": float(diagnostics["global"]["fast_logical_fail"]["rate"]),
        "syndrome_weight_tail_rate": float(diagnostics["global"]["syndrome_weight_tail"]["rate"]),
        "safe_for_fast_rate": float(diagnostics["global"]["safe_for_fast"]["rate"]),
        "accurate_logical_fail_rate": float(diagnostics["global"]["accurate_logical_fail"]["rate"]),
        "hard_runtime_label_valid": True,
        "timing_mode": "compact_rtss_synthetic_per_sample",
        "metadata": {
            "settings": settings,
            "split_path": str(split_path),
            "split_policy": splits.split_policy,
            "split_boundaries": splits.split_boundaries,
            "leakage_safe_for_temporal": splits.leakage_safe_for_temporal,
            "fallback_reason": None,
            "preset": RTSS_16SETTINGS_PRESET,
        },
        "train_indices": splits.train_indices,
        "val_indices": splits.val_indices,
        "test_indices": splits.test_indices,
        "split_seed": splits.split_seed,
        "split_policy": splits.split_policy,
        "split_boundaries": splits.split_boundaries,
        "leakage_safe_for_temporal": splits.leakage_safe_for_temporal,
        "train_indices_hash": hash_indices(splits.train_indices),
        "val_indices_hash": hash_indices(splits.val_indices),
        "test_indices_hash": hash_indices(splits.test_indices),
        "diagnostics": diagnostics,
        "warnings": diagnostics["warnings"],
    }
    if verbose:
        _print_rtss_diagnostics(diagnostics, label_names)
    return summary


def run_build_risk_dataset(
    config: ProjectConfig,
    out_path: str | Path,
    preset: str | None = None,
    shots_per_setting: int | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Build and persist the risk-only profiler dataset."""
    if preset is not None and str(preset).strip():
        preset_name = str(preset).strip().lower()
        if preset_name != RTSS_16SETTINGS_PRESET:
            raise ValueError(f"Unsupported risk dataset preset: {preset}")
        return _run_rtss_16settings_build(
            config,
            out_path,
            shots_per_setting=int(shots_per_setting or config.risk_dataset.num_shots or 30000),
            split_policy=str(config.risk_dataset.split_policy),
            verbose=bool(verbose),
        )

    records: list[dict[str, Any]] = []
    all_candidate_count = 0
    local_candidate_count = 0
    bundles_metadata: list[dict[str, Any]] = []
    fast_failures = 0
    accurate_failures = 0
    settings = _settings_from_config(config)
    for setting_id, setting in enumerate(settings):
        setting_config = copy.deepcopy(config)
        scenario = dict(setting["noise_scenario"])
        setting_config.qec.distances = [int(setting["distance"])]
        setting_config.qec.rounds = int(setting["rounds"])
        setting_config.qec.physical_error_rates = [float(setting["physical_error_rate"])]
        setting_config.qec.num_shots = int(setting["shots"])
        setting_config.seed = int(config.seed) + setting_id
        if scenario.get("type") == "stim_generated" and "before_measure_flip_probability_multiplier" in scenario:
            setting_config.qec.noise_model = "measurement_heavy"
        bundle = generate_surface_code_samples(setting_config)
        syndrome = np.asarray(bundle["syndrome"], dtype=np.int8)
        scenario_metadata: dict[str, Any]
        syndrome, scenario_metadata = _postprocess_syndrome(
            syndrome,
            scenario,
            seed=int(config.seed) + 10000 + setting_id,
        )
        observables = np.asarray(bundle["observables"], dtype=np.int8)
        flat_syndrome = syndrome.reshape(syndrome.shape[0], -1) if syndrome.ndim != 2 else syndrome
        layout = bundle.get("layout")
        all_candidates = parse_dem_error_candidates(bundle.get("dem"), layout=layout)
        local_candidates = filter_local_candidates(
            all_candidates,
            max_spatial_diameter=4.0,
            max_time_diameter=2.0,
            allow_observable_flip=False,
        )
        all_candidate_count += len(all_candidates)
        local_candidate_count += len(local_candidates)
        candidates_by_detector = index_candidates_by_detector(local_candidates)

        accurate_decoder = _decoder_from_name(config.risk_dataset.accurate_decoder, bundle)
        fast_decoder = _decoder_from_name(config.risk_dataset.fast_decoder, bundle)
        accurate_batch_predictions = None
        accurate_batch_metadata: dict[str, Any] = {}
        if config.timing.use_batch_decode_for_accuracy and hasattr(accurate_decoder, "decode_batch"):
            try:
                accurate_batch_predictions, accurate_batch_metadata = accurate_decoder.decode_batch(flat_syndrome)
            except Exception as exc:
                accurate_batch_predictions = None
                accurate_batch_metadata = {"reason": f"decode_batch_failed:{exc}"}
        runtime_label_valid = False
        timing_mode = "fallback"
        accurate_loop_latencies = np.zeros(len(flat_syndrome), dtype=np.float32)
        timing_metadata: dict[str, Any] = {"timing_mode": "fallback", "hard_runtime_label_valid": False}
        if config.timing.use_loop_timing_for_runtime_label and str(config.timing.runtime_label_mode) == "loop_per_shot":
            accurate_loop_latencies, timing_metadata = measure_per_shot_decoder_latency(
                accurate_decoder,
                flat_syndrome,
                **_configured_timing(config),
            )
            runtime_label_valid = bool(timing_metadata.get("hard_runtime_label_valid", False))
            timing_mode = str(timing_metadata.get("timing_mode", "loop_per_shot"))
        elif accurate_batch_metadata.get("latency_us") is not None and len(flat_syndrome):
            accurate_loop_latencies = np.full(
                len(flat_syndrome),
                float(accurate_batch_metadata.get("latency_us", 0.0)) / max(len(flat_syndrome), 1),
                dtype=np.float32,
            )
            timing_mode = "batch_average"
            runtime_label_valid = False
            timing_metadata = {
                "timing_mode": "batch_average",
                "hard_runtime_label_valid": False,
                "fallback_reason": "batch_average_not_valid_per_shot_label",
            }

        bundles_metadata.append(
            {
                **bundle.get("metadata", {}),
                **scenario_metadata,
                "setting_id": setting_id,
                "distance": int(setting["distance"]),
                "rounds": int(setting["rounds"]),
                "physical_error_rate": float(setting["physical_error_rate"]),
                "noise_scenario": str(scenario.get("name", scenario.get("type", "unknown"))),
                "timing_mode": timing_mode,
                "hard_runtime_label_valid": runtime_label_valid,
            }
        )
        for local_shot_id, shot in enumerate(flat_syndrome):
            accurate_result = accurate_decoder.decode(shot)
            fast_result = fast_decoder.decode(shot)
            accurate_prediction = (
                np.asarray(accurate_batch_predictions[local_shot_id], dtype=np.int8).reshape(-1)
                if accurate_batch_predictions is not None
                else _observable_prediction_from_decode_result(accurate_result)
            )
            fast_prediction = _observable_prediction_from_decode_result(fast_result)
            actual_observable = np.asarray(observables[local_shot_id], dtype=np.int8).reshape(-1)
            accurate_failures += int(not np.array_equal(accurate_prediction, actual_observable))
            fast_failures += int(not np.array_equal(fast_prediction, actual_observable))
            patches = (
                extract_detector_patches_from_flat_syndrome(
                    shot,
                    layout=layout,
                    patch_radius=2.5,
                    time_radius=1.0,
                    active_only=True,
                    shot_id=len(records),
                )
                if layout is not None
                else []
            )
            records.append(
                {
                    "shot_id": len(records),
                    "syndrome": shot,
                    "actual_observable": actual_observable,
                    "accurate_prediction": accurate_prediction,
                    "fast_prediction": fast_prediction,
                    "accurate_runtime_us": float(accurate_loop_latencies[local_shot_id]),
                    "fast_runtime_us": float(fast_result.latency_us),
                    "layout": layout,
                    "patches": patches,
                    "candidates_by_detector": candidates_by_detector,
                    "metadata": {
                        **bundle.get("metadata", {}),
                        **scenario_metadata,
                        "setting_id": setting_id,
                        "episode_id": setting_id,
                        "seed_id": int(setting_config.seed),
                        "stream_id": setting_id,
                        "distance": int(setting["distance"]),
                        "rounds": int(setting["rounds"]),
                        "physical_error_rate": float(setting["physical_error_rate"]),
                        "noise_scenario": str(scenario.get("name", scenario.get("type", "unknown"))),
                        "accurate_decoder": getattr(accurate_decoder, "name", config.risk_dataset.accurate_decoder),
                        "fast_decoder": getattr(fast_decoder, "name", config.risk_dataset.fast_decoder),
                        "accurate_placeholder": bool(accurate_result.metadata.get("placeholder", False)),
                        "fast_placeholder": bool(fast_result.metadata.get("placeholder", False)),
                        "timing_mode": timing_mode,
                        "hard_runtime_label_valid": runtime_label_valid,
                        "timing_metadata": timing_metadata,
                    },
                }
            )

    hard_runtime_label_valid = bool(records and all(record["metadata"].get("hard_runtime_label_valid", True) for record in records))
    label_cfg = dict(config.risk_label or {})
    samples = build_risk_samples_from_decoding_records(
        records,
        feature_extractor_config={
            "hard_runtime_percentile": config.risk_dataset.hard_runtime_percentile,
            "hard_runtime_label_valid": hard_runtime_label_valid,
            "patch_radius": 2.5,
            "time_radius": 1.0,
            "combined_definition": label_cfg.get(
                "combined_definition",
                ["fast_wrong_vs_accurate", "fast_logical_fail", "hard_runtime"],
            ),
            "syndrome_weight_tail_percentile": label_cfg.get("syndrome_weight_tail_percentile", 90.0),
        },
    )
    train_fraction = (
        float(config.risk_dataset.train_fraction)
        if config.risk_dataset.train_fraction is not None
        else max(0.0, 1.0 - float(config.risk_training.val_fraction) - float(config.qec.test_fraction))
    )
    val_fraction = (
        float(config.risk_dataset.val_fraction)
        if config.risk_dataset.val_fraction is not None
        else float(config.risk_training.val_fraction)
    )
    test_fraction = (
        float(config.risk_dataset.test_fraction)
        if config.risk_dataset.test_fraction is not None
        else float(config.qec.test_fraction)
    )
    episode_ids = np.asarray([sample.metadata.get("episode_id") for sample in samples], dtype=object)
    setting_ids = np.asarray([sample.metadata.get("setting_id") for sample in samples], dtype=object)
    split_payload = create_split_indices(
        num_samples=len(samples),
        split_policy=str(config.risk_dataset.split_policy),
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=int(config.seed),
        episode_ids=episode_ids if str(config.risk_dataset.split_policy).lower() == "episode" else None,
        setting_ids=setting_ids if str(config.risk_dataset.split_policy).lower() == "setting_stratified" else None,
    )
    splits = splits_from_dict(split_payload)
    save_risk_dataset(
        samples,
        out_path,
        splits=splits,
        metadata_extra={
            "hard_runtime_label_valid": hard_runtime_label_valid,
            "timing_mode": "loop_per_shot" if hard_runtime_label_valid else "fallback",
            "settings": bundles_metadata,
        },
    )
    split_path = risk_dataset_split_sidecar_path(out_path)
    save_risk_dataset_splits(splits, split_path)
    by_distance: dict[str, int] = {}
    by_p: dict[str, int] = {}
    by_scenario: dict[str, int] = {}
    for sample in samples:
        by_distance[str(sample.metadata.get("distance", "unknown"))] = by_distance.get(str(sample.metadata.get("distance", "unknown")), 0) + 1
        by_p[str(sample.metadata.get("physical_error_rate", "unknown"))] = by_p.get(str(sample.metadata.get("physical_error_rate", "unknown")), 0) + 1
        scenario_name = str(sample.metadata.get("noise_scenario", "unknown"))
        by_scenario[scenario_name] = by_scenario.get(scenario_name, 0) + 1
    summary = {
        "total_samples": len(samples),
        "num_shots": len(samples),
        "feature_dim": int(samples[0].features.shape[0]) if samples else 0,
        "feature_names": samples[0].feature_names if samples else [],
        "positive_risk_label_rate": float(np.mean([sample.scheduler_risk_label for sample in samples])) if samples else 0.0,
        "risk_label_positive_rate": float(np.mean([sample.scheduler_risk_label for sample in samples])) if samples else 0.0,
        "hard_runtime_rate": float(np.mean([sample.hard_runtime for sample in samples])) if samples else 0.0,
        "fast_wrong_rate": float(np.mean([sample.fast_wrong_vs_accurate for sample in samples])) if samples else 0.0,
        "fast_logical_fail_rate": float(np.mean([sample.fast_logical_fail for sample in samples])) if samples else 0.0,
        "syndrome_weight_tail_rate": float(np.mean([sample.metadata.get("syndrome_weight_tail", 0) for sample in samples])) if samples else 0.0,
        "accurate_logical_fail_rate": float(np.mean([sample.accurate_logical_fail for sample in samples])) if samples else 0.0,
        "samples_by_distance": by_distance,
        "samples_by_p": by_p,
        "samples_by_noise_scenario": by_scenario,
        "hard_runtime_label_valid": hard_runtime_label_valid,
        "timing_mode": "loop_per_shot" if hard_runtime_label_valid else "fallback",
        "fast_only_logical_error_rate": logical_error_rate(
            np.stack([np.asarray(sample.fast_prediction, dtype=np.int8) for sample in samples]) if samples else np.zeros((0, 1), dtype=np.int8),
            np.stack([np.asarray(sample.actual_observable, dtype=np.int8) for sample in samples]) if samples else np.zeros((0, 1), dtype=np.int8),
        ),
        "accurate_only_logical_error_rate": logical_error_rate(
            np.stack([np.asarray(sample.accurate_prediction, dtype=np.int8) for sample in samples]) if samples else np.zeros((0, 1), dtype=np.int8),
            np.stack([np.asarray(sample.actual_observable, dtype=np.int8) for sample in samples]) if samples else np.zeros((0, 1), dtype=np.int8),
        ),
        "metadata": {
            "settings": bundles_metadata,
            "num_candidates": all_candidate_count,
            "num_local_candidates": local_candidate_count,
            "placeholder": any(bool(item.get("placeholder", False)) for item in bundles_metadata),
            "split_path": str(split_path),
            "split_policy": splits.split_policy,
            "split_boundaries": splits.split_boundaries,
            "leakage_safe_for_temporal": splits.leakage_safe_for_temporal,
            "fallback_reason": splits.fallback_reason,
        },
        "train_indices": splits.train_indices,
        "val_indices": splits.val_indices,
        "test_indices": splits.test_indices,
        "split_seed": splits.split_seed,
        "split_policy": splits.split_policy,
        "split_boundaries": splits.split_boundaries,
        "leakage_safe_for_temporal": splits.leakage_safe_for_temporal,
    }
    summary_path = Path(config.output_dir) / "risk_dataset_build" / "summary.json"
    try:
        dump_json(summary, summary_path)
    except OSError as exc:
        summary["summary_write_warning"] = f"{type(exc).__name__}: {exc}"
    return summary
