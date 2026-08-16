"""PyMatching-backed accurate decoder with placeholder fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import pymatching
except ImportError:  # pragma: no cover
    pymatching = None

from rt_preqec.data.stim_surface_code import extract_detector_error_model
from rt_preqec.decoders.base import BaseDecoder, DecodeResult
from rt_preqec.runtime.timing import measure_latency_us


def measure_per_shot_decoder_latency(
    decoder: Any,
    syndromes: np.ndarray,
    warmup_shots: int = 100,
    repeat_per_shot: int = 3,
    max_timing_shots: int | None = 2000,
    statistic: str = "median",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Measure decoder latency per shot using repeated single-shot decode calls.

    Batch decoding can provide accurate predictions, but its total latency is
    not a valid per-shot tail label. This helper times individual decodes for
    a bounded subset and fills unmeasured shots with the measured median.
    """
    syndrome_array = np.asarray(syndromes)
    num_shots = int(len(syndrome_array))
    latencies = np.full(num_shots, np.nan, dtype=np.float32)
    if num_shots == 0:
        return latencies, {
            "timing_mode": "loop_per_shot",
            "hard_runtime_label_valid": True,
            "num_timed_shots": 0,
        }
    warmup_count = min(max(int(warmup_shots), 0), num_shots)
    for warmup_idx in range(warmup_count):
        try:
            decoder.decode(syndrome_array[warmup_idx])
        except Exception:
            break
    timing_count = num_shots if max_timing_shots is None else min(num_shots, max(int(max_timing_shots), 0))
    if timing_count <= 0:
        return np.zeros(num_shots, dtype=np.float32), {
            "timing_mode": "fallback",
            "hard_runtime_label_valid": False,
            "num_timed_shots": 0,
            "fallback_reason": "max_timing_shots_zero",
        }
    repeats = max(int(repeat_per_shot), 1)
    selected_indices = np.arange(timing_count, dtype=np.int64)
    for idx in selected_indices.tolist():
        shot_latencies: list[float] = []
        for _ in range(repeats):
            try:
                result = decoder.decode(syndrome_array[int(idx)])
                shot_latencies.append(float(result.latency_us))
            except Exception:
                continue
        if shot_latencies:
            values = np.asarray(shot_latencies, dtype=float)
            latencies[int(idx)] = float(np.mean(values) if statistic == "mean" else np.median(values))
    measured = latencies[np.isfinite(latencies)]
    if measured.size == 0:
        return np.zeros(num_shots, dtype=np.float32), {
            "timing_mode": "fallback",
            "hard_runtime_label_valid": False,
            "num_timed_shots": 0,
            "fallback_reason": "no_successful_loop_timing",
        }
    fallback_value = float(np.median(measured))
    latencies[~np.isfinite(latencies)] = fallback_value
    metadata = {
        "timing_mode": "loop_per_shot",
        "hard_runtime_label_valid": True,
        "num_timed_shots": int(measured.size),
        "max_timing_shots": None if max_timing_shots is None else int(max_timing_shots),
        "repeat_per_shot": repeats,
        "warmup_shots": int(warmup_shots),
        "timing_statistic": statistic,
        "filled_untimed_shots": int(num_shots - measured.size),
        "untimed_fill_strategy": "median_fallback" if measured.size < num_shots else "none",
    }
    return latencies.astype(np.float32), metadata


@dataclass
class PyMatchingDecoder(BaseDecoder):
    """Accurate decoder backed by PyMatching when available."""

    matching: Any | None = None
    name: str = "pymatching"
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_detector_error_model(cls, dem: Any) -> "PyMatchingDecoder":
        """Build decoder directly from a detector error model."""
        if pymatching is None or dem is None:
            return cls(matching=None, metadata={"placeholder": True, "reason": "missing_dependency_or_dem"})
        try:
            matching = pymatching.Matching.from_detector_error_model(dem)
            return cls(matching=matching, metadata={"placeholder": False})
        except Exception as exc:  # pragma: no cover
            return cls(matching=None, metadata={"placeholder": True, "reason": f"matching_build_failed:{exc}"})

    @classmethod
    def from_stim_circuit(cls, circuit: Any) -> "PyMatchingDecoder":
        model = extract_detector_error_model(circuit)
        return cls.from_detector_error_model(model)

    def _decode_impl(self, syndrome: np.ndarray) -> DecodeResult:
        syndrome_array = np.asarray(syndrome)
        if self.matching is None:
            return DecodeResult(
                correction=np.zeros_like(syndrome_array, dtype=np.int8),
                success=True,
                latency_us=0.0,
                metadata={**(self.metadata or {}), "placeholder": True},
            )
        flat = syndrome_array.reshape(-1).astype(np.uint8)
        try:
            correction = np.asarray(self.matching.decode(flat), dtype=np.int8)
            return DecodeResult(
                correction=correction,
                success=True,
                latency_us=0.0,
                metadata={**(self.metadata or {}), "placeholder": False},
            )
        except Exception as exc:  # pragma: no cover
            return DecodeResult(
                correction=np.zeros_like(flat, dtype=np.int8),
                success=False,
                latency_us=0.0,
                metadata={**(self.metadata or {}), "placeholder": True, "decode_failed": True, "reason": str(exc)},
            )

    def decode(self, syndrome: np.ndarray) -> DecodeResult:
        result, latency_us = measure_latency_us(self._decode_impl, syndrome)
        result.latency_us = latency_us
        return result

    def decode_batch(self, syndromes: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Decode a batch of detector syndromes into predicted observables."""
        syndrome_array = np.asarray(syndromes, dtype=np.uint8)
        if self.matching is None:
            batch = np.zeros((len(syndrome_array), 1), dtype=np.int8)
            return batch, {**(self.metadata or {}), "placeholder": True, "latency_us": 0.0}

        def _decode_batch_impl(batch_inputs: np.ndarray) -> np.ndarray:
            if hasattr(self.matching, "decode_batch"):
                return np.asarray(self.matching.decode_batch(batch_inputs), dtype=np.int8)
            outputs = [np.asarray(self.matching.decode(row), dtype=np.int8) for row in batch_inputs]
            return np.stack(outputs)

        try:
            predictions, latency_us = measure_latency_us(_decode_batch_impl, syndrome_array)
            return predictions, {**(self.metadata or {}), "placeholder": False, "latency_us": latency_us}
        except Exception as exc:  # pragma: no cover
            batch = np.zeros((len(syndrome_array), 1), dtype=np.int8)
            return batch, {**(self.metadata or {}), "placeholder": True, "reason": f"decode_batch_failed:{exc}", "latency_us": 0.0}
