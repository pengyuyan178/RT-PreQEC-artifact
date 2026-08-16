"""Union-find decoder placeholder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rt_preqec.decoders.base import BaseDecoder, DecodeResult
from rt_preqec.runtime.timing import measure_latency_us


@dataclass
class UnionFindDecoder(BaseDecoder):
    """Placeholder union-find decoder."""

    name: str = "union_find"

    def _decode_impl(self, syndrome: np.ndarray) -> DecodeResult:
        correction = np.zeros_like(np.asarray(syndrome), dtype=np.int8)
        return DecodeResult(correction=correction, success=True, latency_us=0.0, metadata={"placeholder": True})

    def decode(self, syndrome: np.ndarray) -> DecodeResult:
        result, latency_us = measure_latency_us(self._decode_impl, syndrome)
        result.latency_us = latency_us
        return result
