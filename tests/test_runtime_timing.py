import numpy as np

from rt_preqec.decoders.base import DecodeResult
from rt_preqec.decoders.pymatching_decoder import measure_per_shot_decoder_latency


class _ToyDecoder:
    def decode(self, syndrome):
        return DecodeResult(
            correction=np.zeros(1, dtype=np.int8),
            success=True,
            latency_us=float(np.asarray(syndrome).sum() + 1.0),
            metadata={},
        )


def test_loop_per_shot_timing_returns_one_latency_per_shot() -> None:
    syndromes = np.asarray([[0, 1], [1, 1], [0, 0]], dtype=np.int8)
    latencies, metadata = measure_per_shot_decoder_latency(
        _ToyDecoder(),
        syndromes,
        warmup_shots=1,
        repeat_per_shot=2,
        max_timing_shots=None,
    )
    assert latencies.shape == (3,)
    assert metadata["timing_mode"] == "loop_per_shot"
    assert metadata["hard_runtime_label_valid"] is True
