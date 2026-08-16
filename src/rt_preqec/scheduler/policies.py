"""Scheduling priority functions."""

from __future__ import annotations

from rt_preqec.scheduler.job import DecodingJob


def fifo_priority(job: DecodingJob, now_us: float) -> float:
    """FIFO priority using creation timestamp."""
    return -job.created_us


def edf_priority(job: DecodingJob, now_us: float) -> float:
    """Earliest deadline first."""
    return -job.deadline_us


def risk_aware_edf_priority(
    job: DecodingJob,
    now_us: float,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    eps: float = 1e-6,
) -> float:
    """Risk-aware EDF score."""
    slack = max(job.deadline_us - now_us, eps)
    urgency = 1.0 / slack
    predicted_runtime_norm = job.predicted_runtime_us / max(job.deadline_us - job.created_us, 1.0)
    boundary_importance = 1.0 if job.logical_boundary else 0.0
    priority = (
        alpha * urgency
        + beta * job.risk_score
        + gamma * predicted_runtime_norm
        + delta * boundary_importance
    )
    return priority
