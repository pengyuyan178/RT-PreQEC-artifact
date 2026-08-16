"""Continuous-arrival, ordered-commit simulation of the RT-PreQEC runtime.

This is the event-driven model used for the real-time analysis: jobs arrive strictly
periodically at ``i * round_period_us``, are decoded by ``m`` identical
non-preemptive workers, and are committed to the Pauli frame **in index order**, so a
completed job waits for every lower-indexed job before it counts as committed.

Two decisions are deliberately kept apart:

* **Dispatch order** — which ready job runs next. Supplied by
  :mod:`rt_preqec.evaluation.dispatch_policies`, selectable by name.
* **Routing** — which backend (fast predecoder or accurate decoder) serves the job,
  gated by the eligibility filter. Implemented by :func:`route_job`.

Holding one fixed while varying the other is what makes dispatch order attributable
independently of the gate. It also lets the FIFO/EDF/index-order equivalence for an
equal-deadline periodic stream be asserted rather than assumed
(see :func:`rt_preqec.evaluation.dispatch_policies`).

Relationship to :mod:`rt_preqec.evaluation.real_stream`
-------------------------------------------------------
``real_stream`` evaluates *recorded* shot traces and owns the decoder models, the
predecode effect, and the residual-shaping latency model. This module reuses those
leaf helpers (``_predecode_effect``, ``_latency_with_predecode``) and adds the
continuous-arrival queueing model on top. It does not duplicate decoder logic.

All service times come from recorded backend latencies; no routing decision is
allowed to see a realized service time. :func:`validate_event_trace` enforces that,
along with periodic arrivals, ordered commit, worker non-overlap, and — for a single
worker under an index-order policy — agreement with the exact Lindley recurrence.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rt_preqec.config import ProjectConfig
from rt_preqec.evaluation.dispatch_policies import (
    DISPATCH_POLICY_LABELS,
    get_dispatch_policy,
    select_ready_job,
)
from rt_preqec.evaluation.real_stream import (
    RealStreamShotRecord,
    _latency_with_predecode,
    _predecode_effect,
    _record_feature_matrix_for_model,
)
from rt_preqec.models.risk_profiler import load_risk_profiler_checkpoint, predict_risk_scores
from rt_preqec.models.sequence_builder import build_causal_history_matrix

__all__ = [
    "EPS",
    "PredictionProfiles",
    "SimulationResult",
    "build_prediction_profiles",
    "choose_ready_job",
    "dispatch_context",
    "effective_risk",
    "exact_fifo_waiting_and_response",
    "get_predecode_effect",
    "logical_boundary_context",
    "priority_components",
    "reindex_continuous_records",
    "route_job",
    "shaped_latency",
    "simulate_trace",
    "summarize_trace",
    "validate_event_trace",
]

#: Tolerance for float-time event coincidence. Completions within EPS of each other
#: are treated as simultaneous, which matters for commit-buffer accounting.
EPS = 1e-9

#: Routing behaviours, independent of dispatch order. ``gate`` is the full
#: eligibility-gated selective routing; the others are references.
ROUTING_MODES: dict[str, str] = {
    "accurate_only": "Accurate-only",
    "fast_only": "Fast-only",
    "edf_feasibility": "EDF feasibility routing",
    "gate": "RT-PreQEC gate",
    "gate_without_scheduler": "No-scheduler",
    "gate_without_validation": "No proxy filter",
}


@dataclass
class PredictionProfiles:
    """Causal per-job model outputs used by dispatch and routing.

    Every array is indexed by continuous stream position, and each entry must be
    computable from information available *before* the job is dispatched — that is
    what keeps the simulation causal.
    """

    risk_score: np.ndarray
    confidence: np.ndarray
    predicted_accurate_us: np.ndarray
    predicted_fast_us: np.ndarray
    runtime_score: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, num_jobs: int) -> None:
        for name in [
            "risk_score",
            "confidence",
            "predicted_accurate_us",
            "predicted_fast_us",
            "runtime_score",
        ]:
            values = np.asarray(getattr(self, name))
            if values.shape != (num_jobs,):
                raise ValueError(f"{name} must have shape ({num_jobs},), got {values.shape}")
            if not np.isfinite(values).all():
                raise ValueError(f"{name} contains non-finite values")


@dataclass
class SimulationResult:
    """One (routing mode, dispatch policy, worker count) simulation."""

    mode: str
    dispatch_policy: str
    num_workers: int
    events: pd.DataFrame
    summary: dict[str, Any]
    integrity: dict[str, Any]


def original_shot_id(record: RealStreamShotRecord) -> int:
    """Recorded shot id behind a reindexed stream position."""
    return int(record.metadata.get("original_shot_id", record.shot_id))


def reindex_continuous_records(
    records: Sequence[RealStreamShotRecord],
) -> list[RealStreamShotRecord]:
    """Relabel a held-out split as a contiguous stream, preserving recorded order.

    The evaluation split is a subset of the generated shots, so its ids have gaps.
    The continuous-arrival model needs ``shot_id == position`` to define
    ``arrival = i * T``; the original id is kept in metadata for traceability.
    """
    ordered = sorted(records, key=original_shot_id)
    reindexed: list[RealStreamShotRecord] = []
    for position, record in enumerate(ordered):
        metadata = dict(record.metadata)
        metadata.setdefault("original_shot_id", int(record.shot_id))
        reindexed.append(replace_shot_id(record, position, metadata))
    return reindexed


def replace_shot_id(
    record: RealStreamShotRecord,
    shot_id: int,
    metadata: dict[str, Any],
) -> RealStreamShotRecord:
    """Copy ``record`` with a new stream position and metadata."""
    import dataclasses

    return dataclasses.replace(record, shot_id=int(shot_id), metadata=metadata)


def build_prediction_profiles(
    records: Sequence[RealStreamShotRecord],
    checkpoint_path: Path,
    *,
    device: str,
    fixed_fast_estimate_us: float,
) -> PredictionProfiles:
    """Run the risk profiler over the stream to obtain causal per-job predictions.

    The accurate-latency estimate is ``expm1`` of the model's runtime head, floored
    at 0.05 us. The fast-path estimate is a fixed constant: the predecoder cost is
    input-independent, so a learned estimate would add noise without adding signal.
    """
    record_list = list(records)
    model, normalization, metadata = load_risk_profiler_checkpoint(checkpoint_path, device=device)
    features = _record_feature_matrix_for_model(record_list, metadata, normalization)
    model_config = dict(metadata.get("model_config", {}))
    history_length = int(model_config.get("history_length", metadata.get("history_length", 1)))
    history_encoder = str(model_config.get("history_encoder_type", "none"))
    needs_history = history_length > 1 or history_encoder != "none"
    history = (
        build_causal_history_matrix(
            features,
            history_length=max(history_length, 1),
            normalization=None,
            pad_mode=str(model_config.get("pad_mode", metadata.get("pad_mode", "edge"))),
        )
        if needs_history
        else None
    )
    predictions = predict_risk_scores(
        model,
        features,
        normalization,
        history_features=history,
    )
    risk_values = predictions.get("combined_fast_risk", predictions["risk_score"])
    accurate_estimate = np.maximum(
        np.expm1(np.asarray(predictions["runtime_pred"], dtype=float)), 0.05
    )
    num_jobs = len(record_list)
    profiles = PredictionProfiles(
        risk_score=np.asarray(risk_values, dtype=float).reshape(-1),
        confidence=np.asarray(predictions["confidence"], dtype=float).reshape(-1),
        predicted_accurate_us=accurate_estimate.reshape(-1),
        predicted_fast_us=np.full(num_jobs, float(fixed_fast_estimate_us), dtype=float),
        runtime_score=np.asarray(predictions["runtime_score"], dtype=float).reshape(-1),
        metadata={
            "model_type": str(metadata.get("model_type", "unknown")),
            "history_length": history_length,
            "history_encoder_type": history_encoder,
            "runtime_transform": "expm1(runtime_pred)",
            "fixed_fast_estimate_us": float(fixed_fast_estimate_us),
            "checkpoint": str(checkpoint_path),
        },
    )
    profiles.validate(num_jobs)
    return profiles


def logical_boundary_context(job_id: int, config: ProjectConfig) -> tuple[bool, int, bool]:
    """Return ``(is_boundary, rounds_until_boundary, in_drain_window)`` for a job."""
    interval = max(int(config.runtime.logical_boundary_interval), 1)
    offset = (int(job_id) + 1) % interval
    boundary = offset == 0
    rounds_until = 0 if boundary else interval - offset
    drain = rounds_until <= max(int(config.risk_eval.boundary_drain_rounds), 0)
    return boundary, rounds_until, drain


def effective_risk(job_id: int, profiles: PredictionProfiles, config: ProjectConfig) -> float:
    """Risk score after the low-confidence fallback.

    When the model is not confident enough, the job is treated as maximally risky so
    it takes the accurate path. This is the conservative direction: it costs latency,
    never correctness.
    """
    risk = float(np.clip(profiles.risk_score[job_id], 0.0, 1.0))
    confidence = float(profiles.confidence[job_id])
    if bool(config.risk_eval.conservative_on_low_confidence) and confidence < float(
        config.risk_eval.ai_confidence_threshold
    ):
        return 1.0
    return risk


def _effect_key(job_id: int, pressure: bool, validation_enabled: bool) -> tuple[int, bool, bool]:
    return int(job_id), bool(pressure), bool(validation_enabled)


def get_predecode_effect(
    job_id: int,
    records: Sequence[RealStreamShotRecord],
    config: ProjectConfig,
    cache: dict[tuple[int, bool, bool], dict[str, float | bool]],
    *,
    pressure: bool,
    validation_enabled: bool,
) -> dict[str, float | bool]:
    """Memoised predecode effect. The effect depends only on the cache key triple."""
    key = _effect_key(job_id, pressure, validation_enabled)
    if key not in cache:
        cache[key] = _predecode_effect(
            records[job_id],
            config,
            validation_enabled=validation_enabled,
            context={"overload_mode": bool(pressure), "boundary_drain": False},
        )
    return cache[key]


def shaped_latency(
    backend_us: float,
    effect: dict[str, float | bool],
    config: ProjectConfig,
) -> float:
    """Backend latency after the repository's residual-shaping model."""
    return _latency_with_predecode(float(backend_us), effect, config)


def dispatch_context(
    job_id: int,
    *,
    now_us: float,
    latest_arrived: int,
    committed_prefix: int,
    ready_count: int,
    running_count: int,
    config: ProjectConfig,
    scheduler_enabled: bool,
) -> dict[str, Any]:
    """Timing and causal state visible to the scheduler at a dispatch instant.

    With ``scheduler_enabled=False`` the boundary and overload signals are forced
    off, modelling a runtime that has no QEC-aware scheduling at all.
    """
    boundary, rounds_until, boundary_drain = logical_boundary_context(job_id, config)
    lag = max(int(latest_arrived) - int(committed_prefix) - 1, 0)
    backlog = int(ready_count + running_count)
    drain_threshold = (
        int(config.risk_eval.rt_qec_drain_backlog_threshold)
        if config.risk_eval.rt_qec_drain_backlog_threshold is not None
        else int(config.runtime.overload_backlog_threshold)
    )
    overload = backlog >= drain_threshold or lag >= int(config.runtime.max_pauli_frame_lag)
    if not scheduler_enabled:
        boundary = False
        boundary_drain = False
        overload = False
    arrival = float(job_id) * float(config.runtime.round_period_us)
    deadline = arrival + float(config.runtime.decode_deadline_us)
    return {
        "arrival_time_us": arrival,
        "deadline_us": deadline,
        "deadline_slack_us": max(deadline - float(now_us), 0.0),
        "raw_deadline_slack_us": deadline - float(now_us),
        "logical_boundary": bool(boundary),
        "rounds_until_boundary": int(rounds_until),
        "boundary_drain": bool(boundary_drain),
        "overload_mode": bool(overload),
        "pauli_frame_lag": int(lag),
        "backlog": backlog,
        "committed_prefix": int(committed_prefix),
    }


def priority_components(
    job_id: int,
    *,
    now_us: float,
    context: dict[str, Any],
    predicted_accurate_us: float,
    risk_score: float,
    config: ProjectConfig,
) -> dict[str, float]:
    """Equation 1: weighted urgency, risk, predicted runtime, and boundary terms.

    Note that urgency is ``1/slack``; the Pauli-frame lag is *not* a term here. Lag
    influences dispatch only through the overload signal in
    :func:`dispatch_context`, and through commit-frontier order.
    """
    slack = max(float(context["deadline_us"]) - float(now_us), 1e-6)
    urgency = 1.0 / slack
    runtime_norm = float(predicted_accurate_us) / max(float(config.runtime.decode_deadline_us), 1.0)
    boundary = 1.0 if bool(context["logical_boundary"]) else 0.0
    score = (
        float(config.scheduler.alpha_urgency) * urgency
        + float(config.scheduler.beta_risk) * float(risk_score)
        + float(config.scheduler.gamma_runtime) * runtime_norm
        + float(config.scheduler.delta_boundary) * boundary
    )
    return {
        "priority_score": float(score),
        "priority_urgency": float(urgency),
        "priority_risk": float(risk_score),
        "priority_runtime_norm": float(runtime_norm),
        "priority_boundary": float(boundary),
    }


def choose_ready_job(
    ready: set[int],
    *,
    mode: str,
    dispatch_policy: str,
    now_us: float,
    latest_arrived: int,
    committed_prefix: int,
    running_count: int,
    records: Sequence[RealStreamShotRecord],
    profiles: PredictionProfiles,
    config: ProjectConfig,
    effect_cache: dict[tuple[int, bool, bool], dict[str, float | bool]],
) -> tuple[int, dict[str, Any]]:
    """Pick the next ready job and return it with its dispatch context.

    ``equation_priority`` scores every ready job, so it needs the predecode effect
    and therefore lives here. All other policies are pure order functions and are
    delegated to :mod:`rt_preqec.evaluation.dispatch_policies`.
    """
    if not ready:
        raise ValueError("ready queue is empty")
    # Whether the QEC-aware signals are live is a property of the routing mode alone.
    # A dispatch policy only reorders the ready queue; if it also gated these signals
    # it would change routing too, confounding an order comparison.
    scheduler_enabled = mode not in {"accurate_only", "fast_only", "gate_without_scheduler"}

    if dispatch_policy != "equation_priority":
        # Validate the policy name even when the routing mode ignores ordering, so a
        # typo cannot silently fall through to index order.
        get_dispatch_policy(dispatch_policy)
        job_id = select_ready_job(
            dispatch_policy,
            ready,
            now_us=now_us,
            config=config,
            profiles=profiles,
        )
        context = dispatch_context(
            job_id,
            now_us=now_us,
            latest_arrived=latest_arrived,
            committed_prefix=committed_prefix,
            ready_count=len(ready),
            running_count=running_count,
            config=config,
            scheduler_enabled=scheduler_enabled,
        )
        if not scheduler_enabled:
            return job_id, {**context, "priority_score": -float(job_id)}
        # Report the Equation 1 components for comparability even though they did not
        # drive this choice; the order came from `dispatch_policy`.
        pressure = bool(context["overload_mode"] or context["boundary_drain"])
        effect = get_predecode_effect(
            job_id,
            records,
            config,
            effect_cache,
            pressure=pressure,
            validation_enabled=mode != "gate_without_validation",
        )
        predicted_accurate = shaped_latency(profiles.predicted_accurate_us[job_id], effect, config)
        components = priority_components(
            job_id,
            now_us=now_us,
            context=context,
            predicted_accurate_us=predicted_accurate,
            risk_score=effective_risk(job_id, profiles, config),
            config=config,
        )
        return job_id, {
            **context,
            **components,
            "commit_frontier_distance": int(job_id - committed_prefix - 1),
        }

    validation_enabled = mode != "gate_without_validation"
    ranked: list[tuple[tuple[float, float, int], int, dict[str, Any]]] = []
    for job_id in ready:
        context = dispatch_context(
            job_id,
            now_us=now_us,
            latest_arrived=latest_arrived,
            committed_prefix=committed_prefix,
            ready_count=len(ready),
            running_count=running_count,
            config=config,
            scheduler_enabled=True,
        )
        pressure = bool(context["overload_mode"] or context["boundary_drain"])
        effect = get_predecode_effect(
            job_id,
            records,
            config,
            effect_cache,
            pressure=pressure,
            validation_enabled=validation_enabled,
        )
        predicted_accurate = shaped_latency(profiles.predicted_accurate_us[job_id], effect, config)
        components = priority_components(
            job_id,
            now_us=now_us,
            context=context,
            predicted_accurate_us=predicted_accurate,
            risk_score=effective_risk(job_id, profiles, config),
            config=config,
        )
        # Ties break towards the later deadline then the lower index, so the order is
        # total and reproducible regardless of ready-set iteration order.
        rank_key = (
            float(components["priority_score"]),
            -float(context["deadline_us"]),
            -int(job_id),
        )
        ranked.append((rank_key, int(job_id), {**context, **components}))
    _, selected, selected_context = max(ranked, key=lambda item: item[0])
    return selected, selected_context


def route_job(
    job_id: int,
    mode: str,
    *,
    context: dict[str, Any],
    records: Sequence[RealStreamShotRecord],
    profiles: PredictionProfiles,
    config: ProjectConfig,
    effect_cache: dict[tuple[int, bool, bool], dict[str, float | bool]],
) -> dict[str, Any]:
    """Choose the backend for a dispatched job and report the resulting service.

    The eligibility gate is the safety mechanism: a job the proxy filter rejects
    always takes the accurate path. Routing never reads a realized service time,
    which :func:`validate_event_trace` re-checks from the emitted trace.
    """
    record = records[job_id]
    selected = "accurate"
    reason = mode
    effect: dict[str, float | bool] = {
        "predecode_latency_us": 0.0,
        "estimated_residual_reduction": 0.0,
        "fast_path_certified": False,
        "predecode_accept_estimate": False,
        "predecode_validation_pass_estimate": False,
        "weak_validation_pass_estimate": False,
        "predecode_confidence_estimate": 0.0,
        "predecode_risk_estimate": 1.0,
        "validation_enabled": False,
        "abstention_enabled": False,
        "predecode_overload_policy": False,
    }
    predicted_accurate = float(profiles.predicted_accurate_us[job_id])
    predicted_fast = float(profiles.predicted_fast_us[job_id])
    actual_accurate = float(record.accurate_latency_us)
    actual_fast = float(record.fast_latency_us)

    if mode == "accurate_only":
        selected = "accurate"
    elif mode == "fast_only":
        selected = "fast"
    elif mode == "edf_feasibility":
        selected = (
            "accurate" if predicted_accurate <= float(context["deadline_slack_us"]) else "fast"
        )
        reason = "edf_predicted_accurate_feasible" if selected == "accurate" else "edf_fast"
    else:
        scheduler_enabled = mode != "gate_without_scheduler"
        validation_enabled = mode != "gate_without_validation"
        pressure = bool(
            scheduler_enabled and (context["overload_mode"] or context["boundary_drain"])
        )
        effect = get_predecode_effect(
            job_id,
            records,
            config,
            effect_cache,
            pressure=pressure,
            validation_enabled=validation_enabled,
        )
        predicted_accurate = shaped_latency(predicted_accurate, effect, config)
        predicted_fast = shaped_latency(predicted_fast, effect, config)
        actual_accurate = shaped_latency(actual_accurate, effect, config)
        actual_fast = shaped_latency(actual_fast, effect, config)
        fast_allowed = bool(effect.get("fast_path_certified", False))
        risk = effective_risk(job_id, profiles, config)
        threshold = float(config.risk_eval.ai_risk_threshold)
        if pressure:
            threshold = max(threshold - 0.05, 0.15)
        high_risk = risk >= threshold
        accurate_feasible = predicted_accurate <= float(context["deadline_slack_us"])
        fast_feasible = predicted_fast <= float(context["deadline_slack_us"])
        if not fast_allowed:
            selected, reason = "accurate", "proxy_filter_or_abstention"
        elif bool(context["logical_boundary"]) and high_risk and accurate_feasible:
            selected, reason = "accurate", "boundary_high_risk"
        elif pressure and (not high_risk or not accurate_feasible):
            selected, reason = "fast", "pressure_fast"
        elif high_risk and (accurate_feasible or not fast_feasible):
            selected, reason = "accurate", "risk_aware_accurate"
        elif fast_feasible:
            selected, reason = "fast", "low_risk_fast"
        else:
            selected, reason = "accurate", "fast_predicted_infeasible"

    service = actual_accurate if selected == "accurate" else actual_fast
    predicted_service = predicted_accurate if selected == "accurate" else predicted_fast
    prediction = (
        np.asarray(record.accurate_prediction, dtype=np.int8).reshape(-1)
        if selected == "accurate"
        else np.asarray(record.fast_prediction, dtype=np.int8).reshape(-1)
    )
    observable = np.asarray(record.observable, dtype=np.int8).reshape(-1)
    overhead = float(effect.get("predecode_latency_us", 0.0))
    return {
        "selected_decoder": selected,
        "selection_reason": reason,
        "service_time_us": float(service),
        "predicted_service_time_us": float(predicted_service),
        "backend_service_time_us": float(max(service - overhead, 0.0)),
        "modeled_frontend_validation_us": overhead,
        "logical_error": bool(np.any(prediction != observable)),
        "record_fast_logical_fail": bool(record.fast_logical_fail),
        "record_accurate_logical_fail": bool(
            record.metadata.get(
                "accurate_logical_fail",
                np.any(
                    np.asarray(record.accurate_prediction, dtype=np.int8).reshape(-1) != observable
                ),
            )
        ),
        "ai_risk_score": float(profiles.risk_score[job_id]),
        "effective_risk_score": float(effective_risk(job_id, profiles, config)),
        "ai_confidence": float(profiles.confidence[job_id]),
        "predicted_accurate_service_us": float(predicted_accurate),
        "predicted_fast_service_us": float(predicted_fast),
        "fast_path_certified_proxy": bool(effect.get("fast_path_certified", False)),
        "predecode_accept_proxy": bool(effect.get("predecode_accept_estimate", False)),
        "validation_pass_proxy": bool(effect.get("predecode_validation_pass_estimate", False)),
        "weak_validation_pass_proxy": bool(effect.get("weak_validation_pass_estimate", False)),
        "proxy_filter_enabled": bool(effect.get("validation_enabled", False)),
        "estimated_residual_reduction": float(effect.get("estimated_residual_reduction", 0.0)),
        "decision_uses_realized_service": False,
        "fallback_modeled": False,
        "validation_semantics": "feature_based_pre_dispatch_proxy",
    }


def exact_fifo_waiting_and_response(
    service_times_us: Sequence[float],
    round_period_us: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form single-server Lindley recurrence for a periodic arrival stream.

    Used as an independent cross-check of the event simulator: for one worker under
    an index-order policy the simulated waiting and decode-response times must match
    this recurrence exactly.
    """
    service = np.asarray(service_times_us, dtype=float)
    if service.ndim != 1:
        raise ValueError("service_times_us must be one-dimensional")
    if np.any(~np.isfinite(service)) or np.any(service < 0.0):
        raise ValueError("service times must be finite and non-negative")
    period = float(round_period_us)
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("round_period_us must be finite and positive")
    waiting = np.zeros(service.size, dtype=float)
    response = np.zeros(service.size, dtype=float)
    workload = 0.0
    for index, value in enumerate(service):
        waiting[index] = workload
        response[index] = workload + float(value)
        workload = max(0.0, workload + float(value) - period)
    return waiting, response


def _completion_buffer_occupancy(completion_times: np.ndarray) -> tuple[np.ndarray, int]:
    """Commit-buffer depth: jobs decoded but not yet committable, per completion batch."""
    num_jobs = len(completion_times)
    occupancy = np.zeros(num_jobs, dtype=np.int64)
    completed = np.zeros(num_jobs, dtype=bool)
    committed_prefix = -1
    completed_count = 0
    order = sorted(range(num_jobs), key=lambda job_id: (completion_times[job_id], job_id))
    cursor = 0
    maximum = 0
    while cursor < num_jobs:
        time_us = completion_times[order[cursor]]
        group: list[int] = []
        while cursor < num_jobs and abs(completion_times[order[cursor]] - time_us) <= EPS:
            job_id = order[cursor]
            completed[job_id] = True
            completed_count += 1
            group.append(job_id)
            cursor += 1
        while committed_prefix + 1 < num_jobs and completed[committed_prefix + 1]:
            committed_prefix += 1
        value = completed_count - (committed_prefix + 1)
        maximum = max(maximum, value)
        for job_id in group:
            occupancy[job_id] = value
    return occupancy, int(maximum)


def simulate_trace(
    records: Sequence[RealStreamShotRecord],
    profiles: PredictionProfiles,
    config: ProjectConfig,
    *,
    mode: str,
    num_workers: int,
    dispatch_policy: str = "equation_priority",
) -> SimulationResult:
    """Simulate one continuous-arrival, ordered-commit run.

    ``mode`` selects routing (see :data:`ROUTING_MODES`) and ``dispatch_policy``
    selects queue order (see :mod:`rt_preqec.evaluation.dispatch_policies`). The two
    are orthogonal, which is what allows either to be held fixed while varying the
    other.
    """
    if mode not in ROUTING_MODES:
        raise ValueError(
            f"unsupported routing mode: {mode}; supported: {', '.join(sorted(ROUTING_MODES))}"
        )
    if dispatch_policy != "equation_priority":
        get_dispatch_policy(dispatch_policy)
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    num_jobs = len(records)
    if num_jobs == 0:
        raise ValueError("records are empty")
    profiles.validate(num_jobs)
    expected_ids = list(range(num_jobs))
    if [int(record.shot_id) for record in records] != expected_ids:
        raise ValueError("records must be continuously indexed in stream order")

    period = float(config.runtime.round_period_us)
    relative_deadline = float(config.runtime.decode_deadline_us)
    arrivals = np.arange(num_jobs, dtype=float) * period
    deadlines = arrivals + relative_deadline
    completed = np.zeros(num_jobs, dtype=bool)
    committed_prefix = -1
    latest_arrived = -1
    ready: set[int] = set()
    running: list[tuple[float, int, int]] = []
    idle_workers = list(range(int(num_workers)))
    heapq.heapify(idle_workers)
    effect_cache: dict[tuple[int, bool, bool], dict[str, float | bool]] = {}

    start_times = np.full(num_jobs, np.nan, dtype=float)
    completion_times = np.full(num_jobs, np.nan, dtype=float)
    worker_ids = np.full(num_jobs, -1, dtype=np.int64)
    dispatch_order = np.full(num_jobs, -1, dtype=np.int64)
    completion_order = np.full(num_jobs, -1, dtype=np.int64)
    prefix_at_arrival = np.full(num_jobs, -2, dtype=np.int64)
    lag_at_arrival = np.full(num_jobs, -1, dtype=np.int64)
    ready_at_arrival = np.zeros(num_jobs, dtype=np.int64)
    running_at_arrival = np.zeros(num_jobs, dtype=np.int64)
    backlog_at_arrival = np.zeros(num_jobs, dtype=np.int64)
    completion_prefix = np.full(num_jobs, -2, dtype=np.int64)
    row_data: list[dict[str, Any] | None] = [None] * num_jobs
    max_ready_queue = 0
    dispatch_counter = 0
    completion_counter = 0
    next_arrival = 0
    now = 0.0

    while next_arrival < num_jobs or ready or running:
        if ready and idle_workers:
            next_time = now
        else:
            next_arrival_time = arrivals[next_arrival] if next_arrival < num_jobs else math.inf
            next_completion_time = running[0][0] if running else math.inf
            next_time = min(next_arrival_time, next_completion_time)
            if not np.isfinite(next_time):
                raise RuntimeError("event simulation stalled")
            now = float(next_time)

        completed_now: list[int] = []
        while running and running[0][0] <= now + EPS:
            completion_time, worker_id, job_id = heapq.heappop(running)
            completion_times[job_id] = float(completion_time)
            completed[job_id] = True
            heapq.heappush(idle_workers, int(worker_id))
            completion_order[job_id] = completion_counter
            completion_counter += 1
            completed_now.append(job_id)
        while committed_prefix + 1 < num_jobs and completed[committed_prefix + 1]:
            committed_prefix += 1
        for job_id in completed_now:
            completion_prefix[job_id] = committed_prefix

        while next_arrival < num_jobs and arrivals[next_arrival] <= now + EPS:
            job_id = next_arrival
            prefix_at_arrival[job_id] = committed_prefix
            lag_at_arrival[job_id] = job_id - committed_prefix - 1
            ready_at_arrival[job_id] = len(ready)
            running_at_arrival[job_id] = len(running)
            backlog_at_arrival[job_id] = len(ready) + len(running) + 1
            ready.add(job_id)
            latest_arrived = job_id
            next_arrival += 1
            max_ready_queue = max(max_ready_queue, len(ready))

        while ready and idle_workers:
            worker_id = heapq.heappop(idle_workers)
            job_id, context = choose_ready_job(
                ready,
                mode=mode,
                dispatch_policy=dispatch_policy,
                now_us=now,
                latest_arrived=latest_arrived,
                committed_prefix=committed_prefix,
                running_count=len(running),
                records=records,
                profiles=profiles,
                config=config,
                effect_cache=effect_cache,
            )
            ready.remove(job_id)
            route = route_job(
                job_id,
                mode,
                context=context,
                records=records,
                profiles=profiles,
                config=config,
                effect_cache=effect_cache,
            )
            start_times[job_id] = now
            worker_ids[job_id] = worker_id
            dispatch_order[job_id] = dispatch_counter
            dispatch_counter += 1
            completion_time = now + float(route["service_time_us"])
            heapq.heappush(running, (completion_time, worker_id, job_id))
            boundary, rounds_until, boundary_drain = logical_boundary_context(job_id, config)
            row_data[job_id] = {
                **route,
                "mode": mode,
                "mode_label": ROUTING_MODES[mode],
                "dispatch_policy_name": dispatch_policy,
                "num_workers": int(num_workers),
                "continuous_stream_index": int(job_id),
                "original_shot_id": original_shot_id(records[job_id]),
                "shot_id": int(job_id),
                "arrival_time_us": float(arrivals[job_id]),
                "deadline_us": float(deadlines[job_id]),
                "start_time_us": float(now),
                "worker_id": int(worker_id),
                "dispatch_order": int(dispatch_order[job_id]),
                "ready_queue_length_at_dispatch": int(len(ready) + 1),
                "running_jobs_at_dispatch": int(len(running) - 1),
                "backlog_at_dispatch": int(context["backlog"]),
                "committed_prefix_at_dispatch": int(committed_prefix),
                "pauli_frame_lag_at_dispatch": int(context["pauli_frame_lag"]),
                "deadline_slack_at_dispatch_us": float(context["raw_deadline_slack_us"]),
                "overload_mode_at_dispatch": bool(context["overload_mode"]),
                "logical_boundary": bool(boundary),
                "rounds_until_boundary": int(rounds_until),
                "boundary_drain": bool(boundary_drain),
                "priority_score": float(context.get("priority_score", np.nan)),
                "priority_urgency": float(context.get("priority_urgency", np.nan)),
                "priority_risk": float(context.get("priority_risk", np.nan)),
                "priority_runtime_norm": float(context.get("priority_runtime_norm", np.nan)),
                "priority_boundary": float(context.get("priority_boundary", np.nan)),
            }
            max_ready_queue = max(max_ready_queue, len(ready))

    if any(row is None for row in row_data):
        raise AssertionError("one or more jobs were never dispatched")
    if not np.isfinite(completion_times).all():
        raise AssertionError("one or more jobs never completed")

    commit_times = np.maximum.accumulate(completion_times)
    prefix_at_deadline = np.searchsorted(commit_times, deadlines, side="right") - 1
    decode_miss = completion_times > deadlines
    commit_miss = prefix_at_deadline < np.arange(num_jobs)
    buffer_occupancy, max_buffer = _completion_buffer_occupancy(completion_times)

    rows: list[dict[str, Any]] = []
    for job_id, base_row in enumerate(row_data):
        assert base_row is not None
        commit_response = commit_times[job_id] - arrivals[job_id]
        decode_response = completion_times[job_id] - arrivals[job_id]
        rows.append(
            {
                **base_row,
                "completion_time_us": float(completion_times[job_id]),
                "finish_time_us": float(completion_times[job_id]),
                "commit_time_us": float(commit_times[job_id]),
                "waiting_time_us": float(start_times[job_id] - arrivals[job_id]),
                "response_to_decode_us": float(decode_response),
                "response_to_commit_us": float(commit_response),
                "response_time_us": float(commit_response),
                "decode_to_commit_blocking_us": float(
                    commit_times[job_id] - completion_times[job_id]
                ),
                "decode_deadline_miss": bool(decode_miss[job_id]),
                "commit_deadline_miss": bool(commit_miss[job_id]),
                "deadline_miss": bool(commit_miss[job_id]),
                "committed_prefix_at_arrival": int(prefix_at_arrival[job_id]),
                "pauli_frame_lag": int(lag_at_arrival[job_id]),
                "pauli_frame_lag_violation": bool(
                    lag_at_arrival[job_id] > int(config.runtime.max_pauli_frame_lag)
                ),
                "ready_jobs_before_arrival": int(ready_at_arrival[job_id]),
                "running_jobs_before_arrival": int(running_at_arrival[job_id]),
                "decode_backlog_at_arrival": int(backlog_at_arrival[job_id]),
                "completion_order": int(completion_order[job_id]),
                "committed_prefix_after_completion_batch": int(completion_prefix[job_id]),
                "commit_buffer_occupancy_after_completion_batch": int(buffer_occupancy[job_id]),
                "committed_prefix_at_deadline": int(prefix_at_deadline[job_id]),
                "boundary_prerequisites_committed": bool(not commit_miss[job_id]),
                "boundary_commit_success": bool(not commit_miss[job_id]),
                "dispatch_reordered": bool(dispatch_order[job_id] != job_id),
                "service_estimation_error_us": float(
                    base_row["service_time_us"] - base_row["predicted_service_time_us"]
                ),
            }
        )
    events = pd.DataFrame(rows).sort_values("shot_id").reset_index(drop=True)
    integrity = validate_event_trace(
        events,
        config,
        num_workers=num_workers,
        dispatch_policy=dispatch_policy,
    )
    summary = summarize_trace(
        events,
        config,
        mode=mode,
        num_workers=num_workers,
        dispatch_policy=dispatch_policy,
        max_ready_queue=max_ready_queue,
        max_commit_buffer=max_buffer,
        integrity=integrity,
    )
    return SimulationResult(
        mode=mode,
        dispatch_policy=dispatch_policy,
        num_workers=num_workers,
        events=events,
        summary=summary,
        integrity=integrity,
    )


def validate_event_trace(
    events: pd.DataFrame,
    config: ProjectConfig,
    *,
    num_workers: int,
    dispatch_policy: str,
    tolerance_us: float = 1e-6,
) -> dict[str, Any]:
    """Re-derive every timing quantity from the trace and assert it matches.

    This is an independent recomputation, not a restatement of the simulator: it
    rebuilds commits, lag, deadline misses, and boundary success from the raw
    arrival/start/service columns. It also asserts no routing decision consumed a
    realized service time, and for a single worker under index-order dispatch it
    cross-checks against the exact Lindley recurrence.
    """
    num_jobs = len(events)
    indices = np.arange(num_jobs, dtype=np.int64)
    period = float(config.runtime.round_period_us)
    arrivals = events["arrival_time_us"].to_numpy(dtype=float)
    starts = events["start_time_us"].to_numpy(dtype=float)
    completions = events["completion_time_us"].to_numpy(dtype=float)
    commits = events["commit_time_us"].to_numpy(dtype=float)
    deadlines = events["deadline_us"].to_numpy(dtype=float)
    np.testing.assert_array_equal(events["shot_id"].to_numpy(dtype=np.int64), indices)
    np.testing.assert_allclose(arrivals, indices * period, rtol=0.0, atol=tolerance_us)
    if num_jobs > 1:
        np.testing.assert_allclose(np.diff(arrivals), period, rtol=0.0, atol=tolerance_us)
    if np.any(starts + tolerance_us < arrivals):
        raise AssertionError("a job started before arrival")
    np.testing.assert_allclose(
        completions,
        starts + events["service_time_us"].to_numpy(dtype=float),
        rtol=0.0,
        atol=tolerance_us,
    )
    expected_commits = np.maximum.accumulate(completions)
    np.testing.assert_allclose(commits, expected_commits, rtol=0.0, atol=tolerance_us)
    expected_prefix_arrival = np.searchsorted(expected_commits, arrivals, side="right") - 1
    np.testing.assert_array_equal(
        events["committed_prefix_at_arrival"].to_numpy(dtype=np.int64),
        expected_prefix_arrival,
    )
    expected_lag = indices - expected_prefix_arrival - 1
    np.testing.assert_array_equal(events["pauli_frame_lag"].to_numpy(dtype=np.int64), expected_lag)
    expected_prefix_deadline = np.searchsorted(expected_commits, deadlines, side="right") - 1
    np.testing.assert_array_equal(
        events["committed_prefix_at_deadline"].to_numpy(dtype=np.int64),
        expected_prefix_deadline,
    )
    expected_commit_miss = expected_prefix_deadline < indices
    np.testing.assert_array_equal(
        events["commit_deadline_miss"].to_numpy(dtype=bool), expected_commit_miss
    )
    boundary = events["logical_boundary"].to_numpy(dtype=bool)
    np.testing.assert_array_equal(
        events.loc[boundary, "boundary_commit_success"].to_numpy(dtype=bool),
        (~expected_commit_miss)[boundary],
    )
    for worker_id in range(num_workers):
        worker = events.loc[events["worker_id"] == worker_id].sort_values("start_time_us")
        if len(worker) > 1:
            previous_completion = worker["completion_time_us"].to_numpy(dtype=float)[:-1]
            next_start = worker["start_time_us"].to_numpy(dtype=float)[1:]
            if np.any(next_start + tolerance_us < previous_completion):
                raise AssertionError(f"worker {worker_id} has overlapping jobs")
    recurrence_checked = False
    max_waiting_discrepancy = math.nan
    max_response_discrepancy = math.nan
    # The Lindley recurrence assumes the server takes jobs in index order, which for
    # this stream holds for every order-only policy but not for equation priority.
    if num_workers == 1 and dispatch_policy in {"index_order", "commit_frontier", "fifo", "edf"}:
        service = events["service_time_us"].to_numpy(dtype=float)
        waiting, response = exact_fifo_waiting_and_response(service, period)
        actual_waiting = events["waiting_time_us"].to_numpy(dtype=float)
        actual_response = events["response_to_decode_us"].to_numpy(dtype=float)
        np.testing.assert_allclose(actual_waiting, waiting, rtol=0.0, atol=tolerance_us)
        np.testing.assert_allclose(actual_response, response, rtol=0.0, atol=tolerance_us)
        recurrence_checked = True
        max_waiting_discrepancy = float(np.max(np.abs(actual_waiting - waiting)))
        max_response_discrepancy = float(np.max(np.abs(actual_response - response)))
    if bool(events["decision_uses_realized_service"].astype(bool).any()):
        raise AssertionError("a routing decision used realized service time")
    return {
        "integrity_passed": True,
        "num_jobs": num_jobs,
        "continuous_arrivals_verified": True,
        "num_nonperiodic_arrival_gaps": int(
            np.count_nonzero(np.abs(np.diff(arrivals) - period) > tolerance_us)
        ),
        "ordered_commit_verified": True,
        "boundary_prefix_deadline_verified": True,
        "worker_nonoverlap_verified": True,
        "causal_service_decision_verified": True,
        "fifo_recurrence_checked": recurrence_checked,
        "max_abs_waiting_discrepancy_us": max_waiting_discrepancy,
        "max_abs_decode_response_discrepancy_us": max_response_discrepancy,
    }


def summarize_trace(
    events: pd.DataFrame,
    config: ProjectConfig,
    *,
    mode: str,
    num_workers: int,
    dispatch_policy: str,
    max_ready_queue: int,
    max_commit_buffer: int,
    integrity: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate one trace into the summary row used by every downstream table."""
    service = events["service_time_us"].to_numpy(dtype=float)
    decode_response = events["response_to_decode_us"].to_numpy(dtype=float)
    commit_response = events["response_to_commit_us"].to_numpy(dtype=float)
    boundary_events = events.loc[events["logical_boundary"].astype(bool)]
    commit_miss_ratio = float(events["commit_deadline_miss"].astype(float).mean())
    lag_violation_ratio = float(events["pauli_frame_lag_violation"].astype(float).mean())
    boundary_success = (
        float(boundary_events["boundary_commit_success"].astype(float).mean())
        if not boundary_events.empty
        else 1.0
    )
    maximum_lag = int(events["pauli_frame_lag"].max())
    lag_within_budget = maximum_lag <= int(config.runtime.max_pauli_frame_lag)
    return {
        "mode": mode,
        "mode_label": ROUTING_MODES[mode],
        "num_workers": int(num_workers),
        "num_jobs": int(len(events)),
        "round_period_us": float(config.runtime.round_period_us),
        "relative_deadline_us": float(config.runtime.decode_deadline_us),
        "lag_budget_jobs": int(config.runtime.max_pauli_frame_lag),
        "boundary_interval_jobs": int(config.runtime.logical_boundary_interval),
        "num_boundaries": int(len(boundary_events)),
        "dispatch_policy": DISPATCH_POLICY_LABELS.get(dispatch_policy, dispatch_policy),
        "dispatch_policy_name": dispatch_policy,
        "queue_model": "global_nonpreemptive_identical_workers",
        "commit_model": "parallel_decode_in_order_commit",
        "primary_response_definition": "response_to_commit_us",
        "mean_service_time_us": float(np.mean(service)),
        "p99_service_time_us": float(np.percentile(service, 99)),
        "observed_max_service_time_us": float(np.max(service)),
        "offered_load_rho": float(np.sum(service))
        / (float(len(service)) * float(num_workers) * float(config.runtime.round_period_us)),
        "mean_response_to_decode_us": float(np.mean(decode_response)),
        "p99_response_to_decode_us": float(np.percentile(decode_response, 99)),
        "observed_max_response_to_decode_us": float(np.max(decode_response)),
        "mean_response_to_commit_us": float(np.mean(commit_response)),
        "p99_response_to_commit_us": float(np.percentile(commit_response, 99)),
        "observed_max_response_to_commit_us": float(np.max(commit_response)),
        "mean_response_time_us": float(np.mean(commit_response)),
        "p99_response_time_us": float(np.percentile(commit_response, 99)),
        "observed_max_response_time_us": float(np.max(commit_response)),
        "decode_deadline_miss_ratio": float(events["decode_deadline_miss"].astype(float).mean()),
        "commit_deadline_miss_ratio": commit_miss_ratio,
        "deadline_miss_ratio": commit_miss_ratio,
        "maximum_decode_backlog": int(events["decode_backlog_at_arrival"].max()),
        "maximum_ready_queue": int(max_ready_queue),
        "maximum_commit_buffer_occupancy": int(max_commit_buffer),
        "maximum_pauli_frame_lag": maximum_lag,
        "lag_violation_ratio": lag_violation_ratio,
        "boundary_commit_success_rate": boundary_success,
        "logical_error_rate": float(events["logical_error"].astype(float).mean()),
        "fast_selection_rate": float((events["selected_decoder"] == "fast").astype(float).mean()),
        "proxy_filter_pass_rate": float(events["fast_path_certified_proxy"].astype(float).mean()),
        "mean_decode_to_commit_blocking_us": float(events["decode_to_commit_blocking_us"].mean()),
        "observed_max_decode_to_commit_blocking_us": float(
            events["decode_to_commit_blocking_us"].max()
        ),
        "dispatch_reordering_ratio": float(events["dispatch_reordered"].mean()),
        "mean_abs_service_estimation_error_us": float(
            events["service_estimation_error_us"].abs().mean()
        ),
        "service_underprediction_ratio": float(
            (events["service_estimation_error_us"] > 0.0).mean()
        ),
        "trace_zero_commit_deadline_miss": bool(commit_miss_ratio == 0.0),
        "trace_lag_within_budget": bool(lag_within_budget),
        "trace_all_boundaries_successful": bool(boundary_success == 1.0),
        "trace_zero_violation": bool(
            commit_miss_ratio == 0.0 and lag_within_budget and boundary_success == 1.0
        ),
        "trace_near_zero_violation": bool(
            commit_miss_ratio <= 0.001
            and lag_violation_ratio <= 0.001
            and boundary_success >= 0.999
        ),
        **integrity,
    }
