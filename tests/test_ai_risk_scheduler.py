import numpy as np

from rt_preqec.config import load_config
from rt_preqec.scheduler.job import DecodingJob
from rt_preqec.scheduler.lag_scheduler import LagBoundedScheduler
from rt_preqec.scheduler.queue import DecoderJobQueue


def test_ai_risk_scheduler_prioritizes_high_risk_job() -> None:
    config = load_config("configs/risk_profiler.yaml")
    config.scheduler.use_ai_risk = True
    scheduler = LagBoundedScheduler(config)
    queue = DecoderJobQueue()
    queue.push(
        DecodingJob(1, np.zeros(4), 0.0, 10.0, 0.1, 1.0, ai_risk_score=0.9, ai_runtime_score=20.0, ai_confidence=0.9),
        0.0,
    )
    queue.push(
        DecodingJob(2, np.zeros(4), 0.0, 10.0, 0.1, 1.0, ai_risk_score=0.1, ai_runtime_score=1.0, ai_confidence=0.9),
        0.0,
    )
    decision = scheduler.schedule(queue, 0.0, {"lookup": object(), "pymatching": object()}, {"backlog": 2, "pauli_frame_lag": 0})
    assert decision is not None
    assert decision.job.job_id == 1


def test_low_confidence_conservative_mode_selects_accurate() -> None:
    config = load_config("configs/risk_profiler.yaml")
    config.scheduler.use_ai_risk = True
    config.scheduler.conservative_on_low_confidence = True
    scheduler = LagBoundedScheduler(config)
    job = DecodingJob(
        1,
        np.zeros(4),
        0.0,
        10.0,
        0.1,
        1.0,
        ai_risk_score=0.1,
        ai_runtime_score=1.0,
        ai_confidence=0.1,
    )
    decoder = scheduler.choose_decoder(job, slack_us=5.0, backlog=0, config=config)
    assert decoder == config.decoders.accurate
