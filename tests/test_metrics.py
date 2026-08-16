import numpy as np

from rt_preqec.metrics.predecoder_metrics import accept_rate
from rt_preqec.metrics.realtime_metrics import deadline_miss_ratio, latency_percentiles


def test_deadline_miss_ratio() -> None:
    assert deadline_miss_ratio(np.asarray([1.0, 3.0]), np.asarray([2.0, 2.0])) == 0.5


def test_latency_percentiles() -> None:
    summary = latency_percentiles(np.asarray([1.0, 2.0, 3.0, 4.0]))
    assert summary["p50"] >= 2.0
    assert summary["p99"] >= summary["p95"]


def test_accept_rate() -> None:
    assert accept_rate(np.asarray([True, False, True])) == 2.0 / 3.0
