"""Lag-bounded scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rt_preqec.config import ProjectConfig
from rt_preqec.scheduler.job import DecodingJob
from rt_preqec.scheduler.policies import edf_priority, fifo_priority, risk_aware_edf_priority
from rt_preqec.scheduler.queue import DecoderJobQueue


@dataclass
class ScheduledDecision:
    """Chosen job and decoder."""

    job: DecodingJob
    decoder_name: str
    overload_mode: bool
    priority: float
    metadata: dict[str, Any] = field(default_factory=dict)


class LagBoundedScheduler:
    """Select jobs and decoders under lag constraints."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config

    @staticmethod
    def _combine_score(base_value: float, ai_value: float | None, mode: str = "max") -> float:
        if ai_value is None:
            return float(base_value)
        if mode == "max":
            return float(max(base_value, ai_value))
        if mode == "mean":
            return float(0.5 * (base_value + ai_value))
        return float(ai_value)

    def _effective_risk_runtime(self, job: DecodingJob) -> tuple[float, float, bool]:
        use_ai = bool(self.config.scheduler.use_ai_risk)
        confidence = float(job.ai_confidence) if job.ai_confidence is not None else 0.0
        ai_confident = use_ai and confidence >= self.config.scheduler.ai_confidence_threshold
        if ai_confident:
            risk_score = self._combine_score(job.risk_score, job.ai_risk_score, mode="max")
            predicted_runtime = self._combine_score(job.predicted_runtime_us, job.ai_runtime_score, mode="max")
            return risk_score, predicted_runtime, True
        return float(job.risk_score), float(job.predicted_runtime_us), False

    def should_enter_overload_mode(self, backlog: int, max_lag: int) -> bool:
        """Return whether runtime should route aggressively to faster decoders."""
        return backlog >= self.config.runtime.overload_backlog_threshold or max_lag >= self.config.runtime.max_pauli_frame_lag

    def _priority(self, job: DecodingJob, now_us: float) -> float:
        risk_score, predicted_runtime, ai_used = self._effective_risk_runtime(job)
        effective_job = DecodingJob(
            job_id=job.job_id,
            syndrome=job.syndrome,
            created_us=job.created_us,
            deadline_us=job.deadline_us,
            risk_score=risk_score,
            predicted_runtime_us=predicted_runtime,
            logical_boundary=job.logical_boundary,
            ai_risk_score=job.ai_risk_score,
            ai_runtime_score=job.ai_runtime_score,
            ai_confidence=job.ai_confidence,
            feature_vector=job.feature_vector,
            metadata={**job.metadata, "ai_used_in_priority": ai_used},
        )
        policy = self.config.scheduler.policy
        if policy == "fifo":
            return fifo_priority(effective_job, now_us)
        if policy == "edf":
            return edf_priority(effective_job, now_us)
        return risk_aware_edf_priority(
            effective_job,
            now_us,
            alpha=self.config.scheduler.alpha_urgency,
            beta=self.config.scheduler.beta_risk,
            gamma=self.config.scheduler.gamma_runtime,
            delta=self.config.scheduler.delta_boundary,
        )

    def choose_decoder(self, job: DecodingJob, slack_us: float, backlog: int, config: ProjectConfig) -> str:
        """Choose backend decoder based on slack, risk, and overload."""
        overload = self.should_enter_overload_mode(backlog, int(job.metadata.get("current_lag", 0)))
        effective_risk, effective_runtime, ai_used = self._effective_risk_runtime(job)
        low_confidence = bool(config.scheduler.use_ai_risk) and (
            job.ai_confidence is None or job.ai_confidence < config.scheduler.ai_confidence_threshold
        )
        if low_confidence and config.scheduler.conservative_on_low_confidence:
            return config.decoders.accurate
        if overload and bool(config.scheduler.use_ai_risk) and ai_used:
            if effective_risk >= config.scheduler.ai_risk_threshold:
                return config.decoders.accurate
            return config.decoders.fast
        if overload and slack_us <= effective_runtime * 2:
            return config.decoders.fast
        if effective_risk >= max(config.predecoder.risk_threshold, config.scheduler.ai_risk_threshold if config.scheduler.use_ai_risk else 0.0) or job.logical_boundary:
            return config.decoders.accurate
        return config.decoders.fast

    def schedule(
        self,
        job_queue: DecoderJobQueue,
        now_us: float,
        decoders: dict[str, Any],
        runtime_state: dict[str, Any],
    ) -> ScheduledDecision | None:
        """Pop and route the highest-priority job."""
        if len(job_queue) == 0:
            return None
        job_queue.update_priorities(lambda job: self._priority(job, now_us))
        job = job_queue.pop()
        backlog = int(runtime_state.get("backlog", len(job_queue)))
        max_lag = int(runtime_state.get("pauli_frame_lag", 0))
        overload = self.should_enter_overload_mode(backlog, max_lag)
        slack_us = max(job.deadline_us - now_us, 0.0)
        decoder_name = self.choose_decoder(job, slack_us, backlog, self.config)
        if decoder_name not in decoders:
            decoder_name = self.config.decoders.fallback
        return ScheduledDecision(
            job=job,
            decoder_name=decoder_name,
            overload_mode=overload,
            priority=self._priority(job, now_us),
            metadata={
                "slack_us": slack_us,
                "ai_risk_score": job.ai_risk_score,
                "ai_runtime_score": job.ai_runtime_score,
                "ai_confidence": job.ai_confidence,
            },
        )
