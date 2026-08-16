import numpy as np

from rt_preqec.config import load_config
from rt_preqec.scheduler.job import DecodingJob
from rt_preqec.scheduler.lag_scheduler import LagBoundedScheduler
from rt_preqec.scheduler.queue import DecoderJobQueue


def test_edf_chooses_earliest_deadline() -> None:
    config = load_config("configs/default.yaml")
    config.scheduler.policy = "edf"
    scheduler = LagBoundedScheduler(config)
    queue = DecoderJobQueue()
    queue.push(DecodingJob(1, np.zeros((3, 5, 5)), 0.0, 5.0, 0.1, 1.0), 0.0)
    queue.push(DecodingJob(2, np.zeros((3, 5, 5)), 0.0, 2.0, 0.1, 1.0), 0.0)
    decision = scheduler.schedule(queue, 0.0, {"lookup": object(), "pymatching": object()}, {"backlog": 2, "pauli_frame_lag": 0})
    assert decision is not None
    assert decision.job.job_id == 2


def test_risk_aware_prioritizes_high_risk_urgent_job() -> None:
    config = load_config("configs/default.yaml")
    scheduler = LagBoundedScheduler(config)
    queue = DecoderJobQueue()
    queue.push(DecodingJob(1, np.zeros((3, 5, 5)), 0.0, 10.0, 0.9, 1.0), 0.0)
    queue.push(DecodingJob(2, np.zeros((3, 5, 5)), 0.0, 20.0, 0.1, 1.0), 0.0)
    decision = scheduler.schedule(queue, 0.0, {"lookup": object(), "pymatching": object()}, {"backlog": 2, "pauli_frame_lag": 0})
    assert decision is not None
    assert decision.job.job_id == 1


def test_overload_mode_triggers() -> None:
    config = load_config("configs/default.yaml")
    scheduler = LagBoundedScheduler(config)
    assert scheduler.should_enter_overload_mode(backlog=64, max_lag=0)
