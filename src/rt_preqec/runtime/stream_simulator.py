"""Syndrome stream simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np

from rt_preqec.data.schemas import PatchSample, StreamEvent


@dataclass
class SyndromeStreamSimulator:
    """Yield stream events from dataset samples or synthetic syndrome tensors."""

    round_period_us: float
    deadline_us: float

    def from_dataset(self, samples: list[PatchSample]) -> Iterator[StreamEvent]:
        for idx, sample in enumerate(samples):
            yield StreamEvent(
                event_id=idx,
                syndrome=np.asarray(sample.patch),
                timestamp_us=idx * self.round_period_us,
                deadline_us=idx * self.round_period_us + self.deadline_us,
                logical_boundary=idx % 10 == 0,
                metadata={"source": "dataset", "location": sample.location},
            )

    def from_syndromes(self, syndromes: Iterable[np.ndarray]) -> Iterator[StreamEvent]:
        for idx, syndrome in enumerate(syndromes):
            yield StreamEvent(
                event_id=idx,
                syndrome=np.asarray(syndrome),
                timestamp_us=idx * self.round_period_us,
                deadline_us=idx * self.round_period_us + self.deadline_us,
                logical_boundary=idx % 10 == 0,
                metadata={"source": "synthetic"},
            )

    def from_flat_syndromes(
        self,
        syndromes: Iterable[np.ndarray],
        layout: object | None,
        extra_metadata: dict[str, object] | None = None,
    ) -> Iterator[StreamEvent]:
        metadata = extra_metadata or {}
        for idx, syndrome in enumerate(syndromes):
            yield StreamEvent(
                event_id=idx,
                syndrome=np.asarray(syndrome, dtype=np.int8),
                timestamp_us=idx * self.round_period_us,
                deadline_us=idx * self.round_period_us + self.deadline_us,
                logical_boundary=idx % 10 == 0,
                metadata={"source": "flat_synthetic", "layout": layout, **metadata, "shot_id": idx},
            )
