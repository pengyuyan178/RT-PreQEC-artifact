"""Priority queue for decoding jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from rt_preqec.scheduler.job import DecodingJob


@dataclass
class DecoderJobQueue:
    """Simple in-memory priority queue."""

    items: list[tuple[float, DecodingJob]] = field(default_factory=list)

    def push(self, job: DecodingJob, priority: float) -> None:
        self.items.append((priority, job))
        self.items.sort(key=lambda item: item[0], reverse=True)

    def pop(self) -> DecodingJob:
        return self.items.pop(0)[1]

    def peek(self) -> DecodingJob:
        return self.items[0][1]

    def __len__(self) -> int:
        return len(self.items)

    def update_priorities(self, priority_fn: Callable[[DecodingJob], float]) -> None:
        self.items = [(priority_fn(job), job) for _, job in self.items]
        self.items.sort(key=lambda item: item[0], reverse=True)
