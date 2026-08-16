"""Build a patch-level dataset for training the selective predecoder."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
import typer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rt_preqec.data.dataset import save_predecoder_array_dataset
from rt_preqec.data.risk_dataset import hash_indices
from rt_preqec.utils import dump_json, ensure_parent

app = typer.Typer(add_completion=False)


SETTING_TABLE = [
    {"setting_id": 0, "difficulty": "easy", "distance": 3, "rounds": 3, "p": 0.0010, "scenario": "circuit_depolarizing"},
    {"setting_id": 1, "difficulty": "easy", "distance": 3, "rounds": 5, "p": 0.0015, "scenario": "measurement_light"},
    {"setting_id": 2, "difficulty": "easy", "distance": 5, "rounds": 5, "p": 0.0010, "scenario": "circuit_depolarizing"},
    {"setting_id": 3, "difficulty": "medium", "distance": 5, "rounds": 7, "p": 0.0025, "scenario": "measurement_heavy"},
    {"setting_id": 4, "difficulty": "medium", "distance": 5, "rounds": 9, "p": 0.0040, "scenario": "biased_noise"},
    {"setting_id": 5, "difficulty": "medium", "distance": 7, "rounds": 7, "p": 0.0020, "scenario": "circuit_depolarizing"},
    {"setting_id": 6, "difficulty": "hard", "distance": 7, "rounds": 11, "p": 0.0040, "scenario": "local_hotspot"},
    {"setting_id": 7, "difficulty": "hard", "distance": 7, "rounds": 11, "p": 0.0050, "scenario": "burst_noise"},
    {"setting_id": 8, "difficulty": "stress", "distance": 9, "rounds": 13, "p": 0.0060, "scenario": "leakage_like_temporal"},
    {"setting_id": 9, "difficulty": "stress", "distance": 9, "rounds": 13, "p": 0.0075, "scenario": "mixed_realistic"},
]

DIFFICULTY_SCALE = {"easy": 0.75, "medium": 1.0, "hard": 1.25, "stress": 1.45}
SCENARIO_ID = {
    "circuit_depolarizing": 0,
    "measurement_light": 1,
    "measurement_heavy": 2,
    "biased_noise": 3,
    "local_hotspot": 4,
    "burst_noise": 5,
    "leakage_like_temporal": 6,
    "mixed_realistic": 7,
}


def _split_indices_by_setting(setting_ids: np.ndarray) -> dict[str, Any]:
    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    per_setting: dict[str, dict[str, int]] = {}
    for setting_id in sorted(np.unique(setting_ids).tolist()):
        indices = np.flatnonzero(setting_ids == setting_id).astype(np.int64)
        n = int(len(indices))
        train_end = int(round(0.6 * n))
        val_end = train_end + int(round(0.2 * n))
        train_part = indices[:train_end].tolist()
        val_part = indices[train_end:val_end].tolist()
        test_part = indices[val_end:].tolist()
        train.extend(train_part)
        val.extend(val_part)
        test.extend(test_part)
        per_setting[str(int(setting_id))] = {
            "total": n,
            "train": len(train_part),
            "val": len(val_part),
            "test": len(test_part),
        }
    return {
        "train_indices": sorted(train),
        "val_indices": sorted(val),
        "test_indices": sorted(test),
        "split_policy": "setting_stratified",
        "split_boundaries": {"settings": per_setting, "effective_split_policy": "setting_stratified"},
        "leakage_safe_for_temporal": True,
        "train_indices_hash": hash_indices(train),
        "val_indices_hash": hash_indices(val),
        "test_indices_hash": hash_indices(test),
    }


def _add_cluster(mask: np.ndarray, rng: np.random.Generator, *, center: tuple[int, int], size: int) -> None:
    height, width = mask.shape
    i, j = int(center[0]), int(center[1])
    mask[i, j] = 1
    frontier = [(i, j)]
    while int(mask.sum()) < int(size) and frontier:
        base_i, base_j = frontier[int(rng.integers(0, len(frontier)))]
        neighbors = [
            (base_i - 1, base_j),
            (base_i + 1, base_j),
            (base_i, base_j - 1),
            (base_i, base_j + 1),
        ]
        rng.shuffle(neighbors)
        added = False
        for next_i, next_j in neighbors:
            if 0 <= next_i < height and 0 <= next_j < width and mask[next_i, next_j] == 0:
                mask[next_i, next_j] = 1
                frontier.append((next_i, next_j))
                added = True
                break
        if not added:
            frontier.remove((base_i, base_j))


def _scenario_context(
    rng: np.random.Generator,
    scenario: str,
    n: int,
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hotspot = np.zeros((n, patch_size, patch_size), dtype=bool)
    burst = np.zeros(n, dtype=bool)
    leakage = np.zeros(n, dtype=bool)
    if scenario in {"local_hotspot", "mixed_realistic"}:
        center = np.asarray(
            [rng.integers(1, max(patch_size - 1, 2)), rng.integers(1, max(patch_size - 1, 2))],
            dtype=int,
        )
        yy, xx = np.indices((patch_size, patch_size))
        radius = 1.5 if scenario == "local_hotspot" else 2.0
        hotspot = ((yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius**2)[None, :, :]
        hotspot = np.repeat(hotspot, n, axis=0)
    if scenario in {"burst_noise", "mixed_realistic"}:
        burst = rng.random(n) < 0.22
    if scenario in {"leakage_like_temporal", "mixed_realistic"}:
        leakage = rng.random(n) < 0.28
    return hotspot, burst, leakage


def _make_setting_arrays(
    setting: dict[str, Any],
    n: int,
    patch_size: int,
    temporal_window: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed) + int(setting["setting_id"]) * 104729)
    scenario = str(setting["scenario"])
    difficulty = str(setting["difficulty"])
    scale = float(DIFFICULTY_SCALE[difficulty])
    p = float(setting["p"])
    density = np.clip(0.018 + p * 18.0 * scale, 0.018, 0.19)
    hotspot, burst, leakage = _scenario_context(rng, scenario, n, patch_size)

    correction_targets = np.zeros((n, patch_size, patch_size), dtype=np.int8)
    has_correction = rng.random(n) < np.clip(0.58 + 18.0 * p * scale, 0.48, 0.82)
    ambiguous = np.zeros(n, dtype=bool)
    touches_boundary = np.zeros(n, dtype=bool)
    center = patch_size // 2
    for idx in range(n):
        if not bool(has_correction[idx]):
            continue
        max_cluster = 1 if difficulty == "easy" else 2 if difficulty == "medium" else 3 if difficulty == "hard" else 4
        cluster_size = int(rng.integers(1, max_cluster + 1))
        if scenario == "burst_noise" and bool(burst[idx]):
            cluster_size = min(cluster_size + int(rng.integers(1, 3)), 6)
        if scenario == "mixed_realistic" and rng.random() < 0.20:
            cluster_size = min(cluster_size + 2, 7)
        if rng.random() < 0.76:
            start = (
                int(np.clip(center + rng.integers(-1, 2), 0, patch_size - 1)),
                int(np.clip(center + rng.integers(-1, 2), 0, patch_size - 1)),
            )
        else:
            start = (int(rng.integers(0, patch_size)), int(rng.integers(0, patch_size)))
        _add_cluster(correction_targets[idx], rng, center=start, size=cluster_size)
        active = np.argwhere(correction_targets[idx] > 0)
        touches_boundary[idx] = bool(
            len(active)
            and np.any(
                (active[:, 0] == 0)
                | (active[:, 0] == patch_size - 1)
                | (active[:, 1] == 0)
                | (active[:, 1] == patch_size - 1)
            )
        )
        ambiguous[idx] = cluster_size > 3 or bool(touches_boundary[idx])

    patches = rng.random((n, temporal_window, patch_size, patch_size), dtype=np.float32) < density
    patches = patches.astype(np.int8)
    if scenario in {"measurement_heavy", "measurement_light"}:
        measurement_rate = 0.045 if scenario == "measurement_heavy" else 0.020
        patches[:, -1] ^= (rng.random((n, patch_size, patch_size)) < measurement_rate).astype(np.int8)
    if scenario == "biased_noise":
        column = rng.integers(0, patch_size, size=n)
        for idx, col in enumerate(column.tolist()):
            patches[idx, :, :, int(col)] ^= (rng.random((temporal_window, patch_size)) < 0.08).astype(np.int8)
    if scenario in {"local_hotspot", "mixed_realistic"}:
        patches ^= (hotspot[:, None, :, :] & (rng.random((n, temporal_window, patch_size, patch_size)) < 0.08)).astype(np.int8)
    if scenario in {"leakage_like_temporal", "mixed_realistic"}:
        vertical = correction_targets[:, None, :, :] & leakage[:, None, None, None]
        patches ^= np.repeat(vertical.astype(np.int8), temporal_window, axis=1)

    patches[:, -1] ^= correction_targets
    previous_echo = rng.random((n, temporal_window - 1, patch_size, patch_size)) < 0.45
    patches[:, :-1] ^= (correction_targets[:, None, :, :] & previous_echo).astype(np.int8)
    patches = patches.astype(np.float32)

    correction_flat = correction_targets.reshape(n, patch_size * patch_size).astype(np.float32)
    correction_weight = correction_flat.sum(axis=1)
    patch_weight = patches.sum(axis=(1, 2, 3)).astype(np.float32)
    safe = has_correction & (correction_weight <= 3) & (~touches_boundary) & (~burst) & (~ambiguous)
    confidence = np.clip(0.90 - 0.08 * correction_weight - 0.10 * burst - 0.10 * leakage - 0.12 * touches_boundary, 0.05, 0.98)
    confidence = np.where(safe, np.maximum(confidence, 0.72), np.minimum(confidence, 0.45)).astype(np.float32)
    risk = np.clip(1.0 - confidence + 0.02 * patch_weight, 0.02, 0.98).astype(np.float32)
    is_correct = safe.astype(np.float32)
    locations = np.stack(
        [
            rng.integers(temporal_window - 1, max(int(setting["rounds"]), temporal_window), size=n),
            rng.integers(0, int(setting["distance"]), size=n),
            rng.integers(0, int(setting["distance"]), size=n),
        ],
        axis=1,
    ).astype(np.int32)
    return {
        "patches": patches,
        "locations": locations,
        "correction_targets": correction_flat,
        "confidence_targets": confidence,
        "risk_targets": risk,
        "is_correct": is_correct,
        "setting_ids": np.full(n, int(setting["setting_id"]), dtype=np.int16),
        "difficulty_ids": np.full(n, list(DIFFICULTY_SCALE).index(difficulty), dtype=np.int8),
        "scenario_ids": np.full(n, int(SCENARIO_ID[scenario]), dtype=np.int8),
        "distance": np.full(n, int(setting["distance"]), dtype=np.int16),
        "rounds": np.full(n, int(setting["rounds"]), dtype=np.int16),
        "physical_error_rate": np.full(n, p, dtype=np.float32),
        "correction_weight": correction_weight.astype(np.float32),
        "patch_weight": patch_weight.astype(np.float32),
        "touches_boundary": touches_boundary.astype(np.int8),
        "ambiguous": ambiguous.astype(np.int8),
        "burst_context": burst.astype(np.float32),
        "leakage_context": leakage.astype(np.float32),
    }


def _concatenate(chunks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = chunks[0].keys()
    return {key: np.concatenate([chunk[key] for chunk in chunks], axis=0) for key in keys}


def _summary(payload: dict[str, np.ndarray], splits: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    setting_ids = np.asarray(payload["setting_ids"])
    correction_weight = np.asarray(payload["correction_weight"])
    patch_weight = np.asarray(payload["patch_weight"])
    return {
        "num_samples": int(len(setting_ids)),
        "patch_shape": list(payload["patches"].shape[1:]),
        "correction_dim": int(payload["correction_targets"].shape[1]),
        "confidence_mean": float(np.mean(payload["confidence_targets"])),
        "risk_mean": float(np.mean(payload["risk_targets"])),
        "is_correct_rate": float(np.mean(payload["is_correct"])),
        "nonzero_correction_rate": float(np.mean(correction_weight > 0)),
        "mean_correction_weight": float(np.mean(correction_weight)),
        "mean_patch_weight": float(np.mean(patch_weight)),
        "split_sizes": {name: len(splits[f"{name}_indices"]) for name in ["train", "val", "test"]},
        "setting_counts": {str(int(value)): int(np.sum(setting_ids == value)) for value in sorted(np.unique(setting_ids).tolist())},
        "metadata": metadata,
    }


@app.command()
def main(
    out: str = "data/processed/predecoder_dataset_v1_300k.npz",
    summary_out: str = "results/runs/predecoder_dataset_v1_300k_build/summary.json",
    num_samples: int = typer.Option(300000, "--num-samples"),
    patch_size: int = typer.Option(5, "--patch-size"),
    temporal_window: int = typer.Option(3, "--temporal-window"),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    """Generate a standard patch-level predecoder dataset."""
    if int(num_samples) < len(SETTING_TABLE):
        raise ValueError("num_samples must be at least the number of settings")
    base = int(num_samples) // len(SETTING_TABLE)
    remainder = int(num_samples) - base * len(SETTING_TABLE)
    chunks = []
    settings = []
    for idx, setting in enumerate(SETTING_TABLE):
        count = base + (1 if idx < remainder else 0)
        settings.append({**setting, "samples": int(count)})
        chunks.append(_make_setting_arrays(setting, count, int(patch_size), int(temporal_window), int(seed)))
    payload = _concatenate(chunks)
    splits = _split_indices_by_setting(np.asarray(payload["setting_ids"]))
    metadata = {
        "dataset_role": "predecoder-train",
        "schema": "PatchSampleArrayV1",
        "num_samples": int(num_samples),
        "patch_size": int(patch_size),
        "temporal_window": int(temporal_window),
        "correction_dim": int(patch_size) * int(patch_size),
        "split_policy": "setting_stratified",
        "target_policy": "local_correction_mask_generated_from_hidden_local_events",
        "generator": "scripts/build_predecoder_dataset.py",
        "seed": int(seed),
        "settings": settings,
    }
    save_predecoder_array_dataset(
        out,
        patches=payload["patches"].astype(np.float32),
        locations=payload["locations"].astype(np.int32),
        correction_targets=payload["correction_targets"].astype(np.float32),
        confidence_targets=payload["confidence_targets"].astype(np.float32),
        risk_targets=payload["risk_targets"].astype(np.float32),
        is_correct=payload["is_correct"].astype(np.float32),
        metadata={**metadata, "split_path": str(Path(out).with_name(f"{Path(out).stem}_splits.json"))},
        splits=splits,
        extra_arrays={
            "setting_ids": payload["setting_ids"].astype(np.int16),
            "difficulty_ids": payload["difficulty_ids"].astype(np.int8),
            "scenario_ids": payload["scenario_ids"].astype(np.int8),
            "distance": payload["distance"].astype(np.int16),
            "rounds": payload["rounds"].astype(np.int16),
            "physical_error_rate": payload["physical_error_rate"].astype(np.float32),
            "correction_weight": payload["correction_weight"].astype(np.float32),
            "patch_weight": payload["patch_weight"].astype(np.float32),
            "touches_boundary": payload["touches_boundary"].astype(np.int8),
            "ambiguous": payload["ambiguous"].astype(np.int8),
            "burst_context": payload["burst_context"].astype(np.float32),
            "leakage_context": payload["leakage_context"].astype(np.float32),
        },
    )
    summary = _summary(payload, splits, metadata)
    dump_json(summary, ensure_parent(summary_out))


if __name__ == "__main__":
    app()
