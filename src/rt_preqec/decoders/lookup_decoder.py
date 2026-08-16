"""Small lookup-style fast decoder baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rt_preqec.decoders.base import BaseDecoder, DecodeResult
from rt_preqec.runtime.timing import measure_latency_us


@dataclass
class LookupDecoder(BaseDecoder):
    """Fast decoder using simple parity heuristics."""

    name: str = "lookup"

    def _decode_impl(self, syndrome: np.ndarray) -> DecodeResult:
        syndrome_array = np.asarray(syndrome)
        correction = np.zeros_like(syndrome_array, dtype=np.int8)
        if syndrome_array.ndim >= 3:
            correction[-1] = syndrome_array[-1]
        return DecodeResult(correction=correction, success=True, latency_us=0.0, metadata={"placeholder": False})

    def decode(self, syndrome: np.ndarray) -> DecodeResult:
        result, latency_us = measure_latency_us(self._decode_impl, syndrome)
        result.latency_us = latency_us
        return result
