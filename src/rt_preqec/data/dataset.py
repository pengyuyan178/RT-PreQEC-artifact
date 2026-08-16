"""Patch dataset utilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from rt_preqec.data.schemas import PatchSample
from rt_preqec.utils import ensure_parent


@dataclass
class DatasetBundle:
    """Serializable dataset bundle."""

    patches: np.ndarray
    locations: np.ndarray
    correction_targets: np.ndarray
    confidence_targets: np.ndarray
    risk_targets: np.ndarray
    is_correct: np.ndarray
    metadata: dict[str, Any]
    syndromes: np.ndarray | None = None
    observables: np.ndarray | None = None


class PredecoderDataset(Dataset[dict[str, torch.Tensor]]):
    """PyTorch dataset for local patch training."""

    def __init__(self, samples: list[PatchSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        patch = torch.tensor(sample.patch[None, ...], dtype=torch.float32)
        correction = torch.tensor(sample.correction_target, dtype=torch.float32)
        return {
            "patch": patch,
            "correction_target": correction,
            "confidence_target": torch.tensor(sample.confidence_target, dtype=torch.float32),
            "risk_target": torch.tensor(sample.risk_target, dtype=torch.float32),
            "is_correct": torch.tensor(float(sample.is_correct), dtype=torch.float32),
        }


def predecoder_dataset_split_sidecar_path(path: str | Path) -> Path:
    """Return the split sidecar path for a predecoder patch dataset."""
    dataset_path = Path(path)
    return dataset_path.with_name(f"{dataset_path.stem}_splits.json")


def _load_predecoder_split_payload(path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    sidecar = predecoder_dataset_split_sidecar_path(path)
    if sidecar.exists():
        with sidecar.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    splits = metadata.get("splits")
    return dict(splits) if isinstance(splits, dict) else {}


class ArrayPredecoderDataset(Dataset[dict[str, torch.Tensor]]):
    """Array-backed dataset for large patch-level predecoder training."""

    def __init__(self, path: str | Path, split: str = "all") -> None:
        self.path = Path(path)
        archive = np.load(self.path, allow_pickle=False)
        self.patches = np.asarray(archive["patches"])
        self.locations = np.asarray(archive["locations"], dtype=np.int32)
        self.correction_targets = np.asarray(archive["correction_targets"], dtype=np.float32)
        self.confidence_targets = np.asarray(archive["confidence_targets"], dtype=np.float32)
        self.risk_targets = np.asarray(archive["risk_targets"], dtype=np.float32)
        self.is_correct = np.asarray(archive["is_correct"], dtype=np.float32)
        self.metadata = json.loads(str(archive["metadata"])) if "metadata" in archive.files else {}
        self.split_payload = _load_predecoder_split_payload(self.path, self.metadata)
        split = str(split).lower()
        self.split = split
        key = f"{split}_indices"
        if split == "all":
            self.indices = np.arange(len(self.patches), dtype=np.int64)
        elif key in self.split_payload:
            self.indices = np.asarray(self.split_payload[key], dtype=np.int64)
        else:
            self.indices = np.arange(len(self.patches), dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        raw_index = int(self.indices[index])
        return {
            "patch": torch.tensor(self.patches[raw_index][None, ...], dtype=torch.float32),
            "correction_target": torch.tensor(self.correction_targets[raw_index], dtype=torch.float32),
            "confidence_target": torch.tensor(float(self.confidence_targets[raw_index]), dtype=torch.float32),
            "risk_target": torch.tensor(float(self.risk_targets[raw_index]), dtype=torch.float32),
            "is_correct": torch.tensor(float(self.is_correct[raw_index]), dtype=torch.float32),
        }


def save_predecoder_array_dataset(
    path: str | Path,
    *,
    patches: np.ndarray,
    locations: np.ndarray,
    correction_targets: np.ndarray,
    confidence_targets: np.ndarray,
    risk_targets: np.ndarray,
    is_correct: np.ndarray,
    metadata: dict[str, Any],
    splits: dict[str, Any] | None = None,
    extra_arrays: dict[str, np.ndarray] | None = None,
) -> None:
    """Save a large patch dataset without materializing PatchSample objects."""
    target = ensure_parent(path)
    payload: dict[str, Any] = {
        "patches": np.asarray(patches),
        "locations": np.asarray(locations, dtype=np.int32),
        "correction_targets": np.asarray(correction_targets),
        "confidence_targets": np.asarray(confidence_targets, dtype=np.float32),
        "risk_targets": np.asarray(risk_targets, dtype=np.float32),
        "is_correct": np.asarray(is_correct, dtype=np.float32),
        "metadata": json.dumps(metadata),
    }
    if extra_arrays:
        payload.update({str(key): value for key, value in extra_arrays.items()})
    np.savez_compressed(target, **payload)
    if splits is not None:
        with predecoder_dataset_split_sidecar_path(target).open("w", encoding="utf-8") as handle:
            json.dump(splits, handle, indent=2)


def save_patch_dataset(samples: list[PatchSample], path: str | Path) -> None:
    """Save patch samples to a compressed NPZ file."""
    target = ensure_parent(path)
    if len(samples) == 0:
        raise ValueError("Cannot save an empty dataset.")
    bundle = DatasetBundle(
        patches=np.stack([sample.patch for sample in samples]),
        locations=np.asarray([sample.location for sample in samples], dtype=np.int32),
        correction_targets=np.stack([sample.correction_target for sample in samples]),
        confidence_targets=np.asarray([sample.confidence_target for sample in samples], dtype=np.float32),
        risk_targets=np.asarray([sample.risk_target for sample in samples], dtype=np.float32),
        is_correct=np.asarray([sample.is_correct for sample in samples], dtype=np.float32),
        metadata={"samples": len(samples), "schema": "PatchSample"},
    )
    np.savez_compressed(
        target,
        patches=bundle.patches,
        locations=bundle.locations,
        correction_targets=bundle.correction_targets,
        confidence_targets=bundle.confidence_targets,
        risk_targets=bundle.risk_targets,
        is_correct=bundle.is_correct,
        metadata=json.dumps(bundle.metadata),
    )


def load_patch_dataset(path: str | Path) -> list[PatchSample]:
    """Load patch samples from NPZ."""
    archive = np.load(Path(path), allow_pickle=False)
    patches = archive["patches"]
    locations = archive["locations"]
    correction_targets = archive["correction_targets"]
    confidence_targets = archive["confidence_targets"]
    risk_targets = archive["risk_targets"]
    is_correct = archive["is_correct"]
    samples: list[PatchSample] = []
    for idx in range(len(patches)):
        samples.append(
            PatchSample(
                patch=np.asarray(patches[idx]),
                location=tuple(int(v) for v in locations[idx]),
                correction_target=np.asarray(correction_targets[idx], dtype=np.float32),
                is_correct=bool(is_correct[idx] > 0.5),
                confidence_target=float(confidence_targets[idx]),
                risk_target=float(risk_targets[idx]),
                metadata={"loaded": True},
            )
        )
    return samples
