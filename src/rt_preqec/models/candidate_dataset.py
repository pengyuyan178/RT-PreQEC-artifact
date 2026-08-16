"""Dataset scaffolding for the selective DEM-candidate predecoder."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def build_candidate_features_from_patch_and_candidates(
    patch: Any,
    candidates: list[Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scaffold for DetectorPatch + LocalErrorCandidate feature extraction.

    Milestone 3B should replace this placeholder with real `DetectorPatch` and
    `LocalErrorCandidate` supervision. The returned tensors follow the model
    contract and intentionally keep candidate outputs tied to DEM local
    candidates rather than arbitrary corrections.
    """
    raise NotImplementedError("Milestone 3B: connect real DetectorPatch + DEM local candidate features")


class CandidatePredecoderDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset for selective candidate predecoding.

    Input source: processed candidate samples when available, or synthetic
    bounded batches for model/loss tests.
    Output item: variable-size detector patch features, candidate features,
    masks, candidate-or-abstain label, accept label, and risk label.
    RT-PreQEC role: trains a predecoder that ranks DEM local candidates or
    abstains; it is never supervised to emit free-form global corrections.
    Realtime fit: padding and masks keep local patch/candidate compute bounded
    and validation/fallback remains mandatory.
    """

    def __init__(
        self,
        samples: list[dict[str, Any]] | None = None,
        synthetic_size: int = 0,
        detector_feature_dim: int = 8,
        candidate_feature_dim: int = 10,
        max_detectors: int = 6,
        max_candidates: int = 5,
        seed: int = 42,
    ) -> None:
        self.samples = samples
        self.synthetic_size = int(synthetic_size)
        self.detector_feature_dim = int(detector_feature_dim)
        self.candidate_feature_dim = int(candidate_feature_dim)
        self.max_detectors = int(max_detectors)
        self.max_candidates = int(max_candidates)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.samples) if self.samples is not None else self.synthetic_size

    def _synthetic_item(self, index: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.seed + int(index))
        p_count = int(rng.integers(1, self.max_detectors + 1))
        c_count = int(rng.integers(1, self.max_candidates + 1))
        detector_features = rng.normal(size=(p_count, self.detector_feature_dim)).astype(np.float32)
        candidate_features = rng.normal(size=(c_count, self.candidate_feature_dim)).astype(np.float32)
        label = int(rng.integers(0, c_count + 1))
        accept = float(label < c_count)
        return {
            "detector_features": torch.tensor(detector_features, dtype=torch.float32),
            "detector_mask": torch.ones(p_count, dtype=torch.bool),
            "candidate_features": torch.tensor(candidate_features, dtype=torch.float32),
            "candidate_mask": torch.ones(c_count, dtype=torch.bool),
            "candidate_label": torch.tensor(label, dtype=torch.long),
            "accept_label": torch.tensor(accept, dtype=torch.float32),
            "risk_label": torch.tensor(float(rng.random() > 0.5), dtype=torch.float32),
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self.samples is None:
            return self._synthetic_item(index)
        sample = self.samples[index]
        return {
            "detector_features": torch.as_tensor(sample["detector_features"], dtype=torch.float32),
            "detector_mask": torch.as_tensor(sample.get("detector_mask", np.ones(len(sample["detector_features"]))), dtype=torch.bool),
            "candidate_features": torch.as_tensor(sample["candidate_features"], dtype=torch.float32),
            "candidate_mask": torch.as_tensor(sample.get("candidate_mask", np.ones(len(sample["candidate_features"]))), dtype=torch.bool),
            "candidate_label": torch.as_tensor(sample["candidate_label"], dtype=torch.long),
            "accept_label": torch.as_tensor(sample.get("accept_label", 1.0), dtype=torch.float32),
            "risk_label": torch.as_tensor(sample.get("risk_label", 0.0), dtype=torch.float32),
        }


def collate_candidate_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad variable detector and candidate counts for candidate predecoder training."""
    batch_size = len(batch)
    max_p = max(int(item["detector_features"].shape[0]) for item in batch)
    max_c = max(int(item["candidate_features"].shape[0]) for item in batch)
    detector_dim = int(batch[0]["detector_features"].shape[-1])
    candidate_dim = int(batch[0]["candidate_features"].shape[-1])
    detector_features = torch.zeros(batch_size, max_p, detector_dim, dtype=torch.float32)
    detector_mask = torch.zeros(batch_size, max_p, dtype=torch.bool)
    candidate_features = torch.zeros(batch_size, max_c, candidate_dim, dtype=torch.float32)
    candidate_mask = torch.zeros(batch_size, max_c, dtype=torch.bool)
    candidate_label = torch.zeros(batch_size, dtype=torch.long)
    accept_label = torch.zeros(batch_size, dtype=torch.float32)
    risk_label = torch.zeros(batch_size, dtype=torch.float32)
    for row, item in enumerate(batch):
        p_count = int(item["detector_features"].shape[0])
        c_count = int(item["candidate_features"].shape[0])
        detector_features[row, :p_count] = item["detector_features"]
        detector_mask[row, :p_count] = item["detector_mask"].bool()
        candidate_features[row, :c_count] = item["candidate_features"]
        candidate_mask[row, :c_count] = item["candidate_mask"].bool()
        label = int(item["candidate_label"].item())
        candidate_label[row] = min(label, c_count)
        accept_label[row] = item["accept_label"].float()
        risk_label[row] = item["risk_label"].float()
    return {
        "detector_features": detector_features,
        "detector_mask": detector_mask,
        "candidate_features": candidate_features,
        "candidate_mask": candidate_mask,
        "candidate_label": candidate_label,
        "accept_label": accept_label,
        "risk_label": risk_label,
    }
