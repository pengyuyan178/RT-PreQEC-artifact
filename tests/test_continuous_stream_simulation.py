"""Behavioural tests for the continuous-arrival, ordered-commit simulator.

Ported from the RTSS 2026 review-response experiments, retargeted at
:mod:`rt_preqec.evaluation.continuous_stream` so the properties they established are
enforced by the main test suite rather than by a one-off script.

The cases cover, in order: agreement with the exact Lindley recurrence, ordered-commit
blocking, causality of routing decisions, the dispatch-order equivalence, and the
runtime guard.
"""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


def _find_repo_root(path: Path) -> Path:
    for candidate in path.resolve().parents:
        if (candidate / "src" / "rt_preqec").is_dir():
            return candidate
    raise RuntimeError(f"could not find repository root above {path}")


ROOT = _find_repo_root(Path(__file__))
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from rt_preqec.config import ProjectConfig
from rt_preqec.evaluation.continuous_stream import (
    ROUTING_MODES,
    PredictionProfiles,
    exact_fifo_waiting_and_response,
    reindex_continuous_records,
    simulate_trace,
)
from rt_preqec.evaluation.real_stream import RealStreamShotRecord
from rt_preqec.evaluation.runtime_guard import (
    RuntimeMargins,
    apply_runtime_guard,
    calibrate_runtime_margins,
    guard_metadata,
)

#: Dispatch policies that reduce to index order on an equal-deadline periodic stream.
ORDER_EQUIVALENT_POLICIES = ["index_order", "commit_frontier", "fifo", "edf"]


def _record(job_id: int, accurate_us: float, fast_us: float = 1.0) -> RealStreamShotRecord:
    return RealStreamShotRecord(
        shot_id=job_id,
        syndrome=np.asarray([0], dtype=np.int8),
        observable=np.asarray([0], dtype=np.int8),
        accurate_prediction=np.asarray([0], dtype=np.int8),
        fast_prediction=np.asarray([1], dtype=np.int8),
        accurate_latency_us=float(accurate_us),
        fast_latency_us=float(fast_us),
        features=np.asarray([0.0], dtype=np.float32),
        feature_names=["syndrome_weight"],
        risk_label=1,
        hard_runtime=0,
        fast_wrong_vs_accurate=1,
        fast_logical_fail=1,
        metadata={"accurate_logical_fail": 0},
    )


def _profiles(
    num_jobs: int,
    *,
    risk: float | list[float] = 1.0,
    predicted_accurate: list[float] | None = None,
    predicted_fast: list[float] | None = None,
) -> PredictionProfiles:
    accurate = (
        np.asarray(predicted_accurate, dtype=float)
        if predicted_accurate is not None
        else np.ones(num_jobs, dtype=float)
    )
    fast = (
        np.asarray(predicted_fast, dtype=float)
        if predicted_fast is not None
        else np.ones(num_jobs, dtype=float)
    )
    risk_score = (
        np.asarray(risk, dtype=float)
        if isinstance(risk, list)
        else np.full(num_jobs, float(risk), dtype=float)
    )
    return PredictionProfiles(
        risk_score=risk_score,
        confidence=np.ones(num_jobs, dtype=float),
        predicted_accurate_us=accurate,
        predicted_fast_us=fast,
        runtime_score=np.zeros(num_jobs, dtype=float),
    )


def _config(
    *, period: float = 10.0, deadline: float = 10.0, boundary_interval: int = 100
) -> ProjectConfig:
    config = ProjectConfig()
    config.runtime.round_period_us = period
    config.runtime.decode_deadline_us = deadline
    config.runtime.logical_boundary_interval = boundary_interval
    config.runtime.max_pauli_frame_lag = 4
    config.runtime.overload_backlog_threshold = 100
    config.risk_eval.rt_qec_drain_backlog_threshold = 100
    config.risk_eval.boundary_drain_rounds = 0
    config.risk_eval.ai_risk_threshold = 0.3
    config.risk_eval.ai_confidence_threshold = 0.5
    config.predecoder.frontend_latency_us = 0.0
    config.predecoder.validation_latency_us = 0.0
    config.predecoder.max_residual_reduction = 0.0
    return config


def _routing_config(**kwargs: float | int) -> ProjectConfig:
    """A config whose proxy filter can actually certify the fast path.

    The default ``predecoder.risk_threshold`` of 0.05 is below the frontend's own risk
    estimate for these synthetic records, so every job would be forced onto the
    accurate path and any test of routing behaviour would pass vacuously.
    """
    config = _config(**kwargs)  # type: ignore[arg-type]
    config.predecoder.risk_threshold = 0.5
    return config


#: A backlogged trace: mean service far exceeds the period even after fast routing, so
#: several jobs are ready at once for every worker count under test. Without that, only
#: one job is ever ready, every policy trivially agrees, and the ordering tests below
#: prove nothing. :func:`_assert_dispatch_is_contended` enforces the property.
_CONTENDED_SERVICE = [55.0, 50.0, 42.0, 48.0, 38.0, 45.0, 60.0, 40.0, 52.0, 35.0, 44.0, 58.0]
_CONTENDED_DEADLINE = 90.0


def _contended_trace() -> tuple[list[RealStreamShotRecord], PredictionProfiles, ProjectConfig]:
    records = [
        _record(index, value, fast_us=value * 0.6)
        for index, value in enumerate(_CONTENDED_SERVICE)
    ]
    # Risk alternating across `ai_risk_threshold` (0.3) is what splits traffic between
    # the two backends; a uniform risk would send every job the same way.
    profiles = _profiles(
        len(records),
        predicted_accurate=[25.0] * len(_CONTENDED_SERVICE),
        predicted_fast=[15.0] * len(_CONTENDED_SERVICE),
        risk=[0.9, 0.1] * (len(_CONTENDED_SERVICE) // 2),
    )
    return records, profiles, _routing_config(deadline=_CONTENDED_DEADLINE)


def _assert_dispatch_is_contended(result: object) -> None:
    """Fail loudly if the trace degenerated, rather than passing an empty comparison."""
    summary = result.summary  # type: ignore[attr-defined]
    decoders = result.events["selected_decoder"].tolist()  # type: ignore[attr-defined]
    assert int(summary["maximum_ready_queue"]) >= 2, (
        "only one job was ever ready, so dispatch order could not matter: "
        f"maximum_ready_queue={summary['maximum_ready_queue']}"
    )
    assert 0 < decoders.count("fast") < len(decoders), (
        f"the gate did not split traffic across both backends: {decoders}"
    )


# --------------------------------------------------------------------------- #
# Lindley recurrence cross-check
# --------------------------------------------------------------------------- #


def test_underloaded_trace_has_zero_waiting_and_matches_the_recurrence() -> None:
    service = [4.0, 4.0, 4.0]
    waiting, response = exact_fifo_waiting_and_response(service, 10.0)
    np.testing.assert_allclose(waiting, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(response, service)

    records = [_record(index, value) for index, value in enumerate(service)]
    result = simulate_trace(
        records,
        _profiles(len(records), predicted_accurate=service),
        _config(),
        mode="accurate_only",
        num_workers=1,
        dispatch_policy="index_order",
    )
    np.testing.assert_allclose(result.events["waiting_time_us"], waiting)
    np.testing.assert_allclose(result.events["response_to_decode_us"], response)
    assert not bool(result.events["commit_deadline_miss"].any())


def test_overloaded_trace_matches_the_manually_computed_recurrence() -> None:
    service = [15.0, 15.0, 2.0]
    waiting, response = exact_fifo_waiting_and_response(service, 10.0)
    np.testing.assert_allclose(waiting, [0.0, 5.0, 10.0])
    np.testing.assert_allclose(response, [15.0, 20.0, 12.0])

    records = [_record(index, value) for index, value in enumerate(service)]
    result = simulate_trace(
        records,
        _profiles(len(records), predicted_accurate=service),
        _config(),
        mode="accurate_only",
        num_workers=1,
        dispatch_policy="index_order",
    )
    np.testing.assert_allclose(result.events["waiting_time_us"], waiting)
    np.testing.assert_allclose(result.events["response_to_decode_us"], response)
    assert result.integrity["fifo_recurrence_checked"]


def test_recurrence_rejects_a_nonpositive_period() -> None:
    with pytest.raises(ValueError, match="round_period_us"):
        exact_fifo_waiting_and_response([1.0, 2.0], 0.0)


def test_recurrence_rejects_negative_service_times() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        exact_fifo_waiting_and_response([1.0, -2.0], 10.0)


# --------------------------------------------------------------------------- #
# Ordered commit
# --------------------------------------------------------------------------- #


def test_out_of_order_completion_does_not_advance_the_prefix_or_boundary() -> None:
    """A job finished early still waits for every lower index before it counts."""
    records = [_record(0, 30.0), _record(1, 1.0), _record(2, 1.0)]
    result = simulate_trace(
        records,
        _profiles(3, predicted_accurate=[30.0, 1.0, 1.0]),
        _config(deadline=15.0, boundary_interval=2),
        mode="accurate_only",
        num_workers=2,
        dispatch_policy="index_order",
    )
    events = result.events
    np.testing.assert_allclose(events["completion_time_us"], [30.0, 11.0, 21.0])
    np.testing.assert_allclose(events["commit_time_us"], [30.0, 30.0, 30.0])
    assert events["committed_prefix_at_arrival"].tolist() == [-1, -1, -1]
    assert events["pauli_frame_lag"].tolist() == [0, 1, 2]
    # Job 1 decoded well inside its deadline yet still misses on commit.
    assert bool(events.loc[1, "decode_deadline_miss"]) is False
    assert bool(events.loc[1, "commit_deadline_miss"]) is True
    assert bool(events.loc[1, "boundary_commit_success"]) is False


def test_completion_and_commit_precede_a_same_time_arrival() -> None:
    """At an exact tie, the completing job commits before the arriving job observes it."""
    records = [_record(0, 10.0), _record(1, 1.0)]
    result = simulate_trace(
        records,
        _profiles(2, predicted_accurate=[10.0, 1.0]),
        _config(period=10.0),
        mode="accurate_only",
        num_workers=1,
        dispatch_policy="index_order",
    )
    second = result.events.iloc[1]
    assert int(second["committed_prefix_at_arrival"]) == 0
    assert int(second["pauli_frame_lag"]) == 0
    assert float(second["waiting_time_us"]) == 0.0


def test_commit_frontier_dispatch_preserves_stream_order_under_overload() -> None:
    records = [_record(0, 30.0), _record(1, 5.0), _record(2, 5.0)]
    result = simulate_trace(
        records,
        _profiles(3, predicted_accurate=[30.0, 5.0, 5.0]),
        _config(deadline=100.0, boundary_interval=3),
        mode="gate",
        num_workers=1,
        dispatch_policy="commit_frontier",
    )
    events = result.events
    assert events.sort_values("dispatch_order")["shot_id"].tolist() == [0, 1, 2]
    np.testing.assert_allclose(events["completion_time_us"], [30.0, 35.0, 40.0])
    np.testing.assert_allclose(events["commit_time_us"], [30.0, 35.0, 40.0])
    assert result.integrity["fifo_recurrence_checked"]


def test_equation_priority_reorders_ready_jobs_and_commit_waits_for_the_prefix() -> None:
    """Equation-1 priority can serve job 2 before job 1; the commit order does not change."""
    records = [_record(0, 30.0), _record(1, 5.0), _record(2, 5.0)]
    result = simulate_trace(
        records,
        _profiles(3, predicted_accurate=[30.0, 5.0, 5.0]),
        _config(deadline=100.0, boundary_interval=3),
        mode="gate",
        num_workers=1,
        dispatch_policy="equation_priority",
    )
    events = result.events
    assert events.sort_values("dispatch_order")["shot_id"].tolist() == [0, 2, 1]
    np.testing.assert_allclose(events["completion_time_us"], [30.0, 40.0, 35.0])
    np.testing.assert_allclose(events["commit_time_us"], [30.0, 40.0, 40.0])
    # Job 2 finished first but sat in the commit buffer waiting for job 1.
    assert int(events.loc[2, "commit_buffer_occupancy_after_completion_batch"]) == 1
    assert float(events.loc[2, "decode_to_commit_blocking_us"]) == 5.0


# --------------------------------------------------------------------------- #
# Causality
# --------------------------------------------------------------------------- #


def test_routing_uses_predicted_not_realized_service_time() -> None:
    """A badly under-predicted job is admitted and then misses; no oracle is available."""
    records = [_record(0, 100.0, fast_us=1.0)]
    result = simulate_trace(
        records,
        _profiles(1, predicted_accurate=[1.0]),
        _config(deadline=10.0),
        mode="edf_feasibility",
        num_workers=1,
        dispatch_policy="edf",
    )
    event = result.events.iloc[0]
    assert event["selected_decoder"] == "accurate"
    assert float(event["predicted_service_time_us"]) == 1.0
    assert float(event["service_time_us"]) == 100.0
    assert bool(event["commit_deadline_miss"])
    assert not bool(event["decision_uses_realized_service"])


@pytest.mark.parametrize("mode", sorted(ROUTING_MODES))
def test_no_routing_mode_consumes_a_realized_service_time(mode: str) -> None:
    service = [12.0, 3.0, 8.0, 2.0]
    records = [_record(index, value) for index, value in enumerate(service)]
    result = simulate_trace(
        records,
        _profiles(len(records), predicted_accurate=[4.0] * len(service)),
        _config(deadline=30.0),
        mode=mode,
        num_workers=2,
        dispatch_policy="index_order",
    )
    assert not bool(result.events["decision_uses_realized_service"].any())
    assert result.integrity["causal_service_decision_verified"]
    assert result.summary["integrity_passed"]


# --------------------------------------------------------------------------- #
# Reindexing
# --------------------------------------------------------------------------- #


def test_reindexing_controls_arrivals_and_the_boundary_index() -> None:
    """Boundaries follow the continuous stream position, not the original shot id."""
    continuous = reindex_continuous_records([_record(9, 1.0), _record(0, 1.0), _record(3, 1.0)])
    assert [item.shot_id for item in continuous] == [0, 1, 2]
    assert [item.metadata["original_shot_id"] for item in continuous] == [0, 3, 9]

    result = simulate_trace(
        continuous,
        _profiles(3),
        _config(boundary_interval=3),
        mode="accurate_only",
        num_workers=1,
        dispatch_policy="index_order",
    )
    np.testing.assert_allclose(np.diff(result.events["arrival_time_us"]), [10.0, 10.0])
    assert result.events["logical_boundary"].tolist() == [False, False, True]
    assert result.events.loc[2, "original_shot_id"] == 9


def test_non_continuous_records_are_rejected() -> None:
    """Arrivals are derived from the index, so a gap would silently shift the timeline."""
    records = [_record(0, 1.0), _record(2, 1.0)]
    with pytest.raises(ValueError, match="continuously indexed"):
        simulate_trace(
            records,
            _profiles(2),
            _config(),
            mode="accurate_only",
            num_workers=1,
            dispatch_policy="index_order",
        )


# --------------------------------------------------------------------------- #
# Dispatch order is orthogonal to routing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("policy", ORDER_EQUIVALENT_POLICIES[1:])
@pytest.mark.parametrize("num_workers", [1, 2, 3])
def test_fifo_edf_and_commit_frontier_are_the_same_schedule(
    policy: str, num_workers: int
) -> None:
    """The identity the fixed-routing baselines rely on, asserted rather than assumed.

    With one relative deadline shared by every job, arrival order, deadline order, and
    index order coincide, so these policies must produce a bit-identical schedule.
    """
    records, profiles, config = _contended_trace()
    reference = simulate_trace(
        records,
        profiles,
        config,
        mode="gate",
        num_workers=num_workers,
        dispatch_policy="index_order",
    )
    _assert_dispatch_is_contended(reference)
    candidate = simulate_trace(
        records,
        profiles,
        config,
        mode="gate",
        num_workers=num_workers,
        dispatch_policy=policy,
    )
    for column in ("start_time_us", "completion_time_us", "commit_time_us", "service_time_us"):
        np.testing.assert_allclose(
            candidate.events[column], reference.events[column], rtol=0.0, atol=1e-9
        )
    assert (
        candidate.events["dispatch_order"].tolist()
        == reference.events["dispatch_order"].tolist()
    )
    assert candidate.events["selected_decoder"].tolist() == (
        reference.events["selected_decoder"].tolist()
    )


@pytest.mark.parametrize("num_workers", [1, 2, 3])
def test_equation_priority_actually_departs_from_index_order(num_workers: int) -> None:
    """Counterpart to the identity test: not every policy collapses to index order.

    Without this, the identity above could be satisfied by a simulator that ignored the
    dispatch policy entirely.
    """
    records, profiles, config = _contended_trace()
    reference = simulate_trace(
        records,
        profiles,
        config,
        mode="gate",
        num_workers=num_workers,
        dispatch_policy="index_order",
    )
    reordered = simulate_trace(
        records,
        profiles,
        config,
        mode="gate",
        num_workers=num_workers,
        dispatch_policy="equation_priority",
    )
    assert (
        reordered.events["dispatch_order"].tolist() != reference.events["dispatch_order"].tolist()
    )
    # Commit order is a property of the model, not the policy, so it must not move.
    assert reordered.events["shot_id"].tolist() == reference.events["shot_id"].tolist()
    np.testing.assert_array_equal(
        np.diff(reordered.events["commit_time_us"].to_numpy(dtype=float)) >= -1e-9,
        np.ones(len(records) - 1, dtype=bool),
    )


def test_only_the_routing_mode_can_disable_the_qec_aware_signals() -> None:
    """Regression guard: a dispatch policy must not suppress the overload signal.

    An earlier version of the policy table marked ``fifo`` as scheduler-disabling. That
    also switched routing off, so the FIFO/index-order identity broke by thousands of
    microseconds and a "FIFO baseline" was silently a no-scheduler baseline. Only the
    ``gate_without_scheduler`` mode may do that.
    """
    records, profiles, config = _contended_trace()
    # Make overload and boundary drain reachable, so there is a signal to suppress.
    config.runtime.overload_backlog_threshold = 2
    config.risk_eval.rt_qec_drain_backlog_threshold = 2
    config.runtime.logical_boundary_interval = 4
    config.risk_eval.boundary_drain_rounds = 1

    for policy in ("index_order", "fifo", "edf", "lst", "equation_priority"):
        events = simulate_trace(
            records,
            profiles,
            config,
            mode="gate",
            num_workers=1,
            dispatch_policy=policy,
        ).events
        assert bool(events["overload_mode_at_dispatch"].any()), (
            f"policy {policy!r} suppressed the overload signal, which is a routing input"
        )
        assert "pressure_fast" in set(events["selection_reason"]), (
            f"policy {policy!r} lost the overload-driven routing decision"
        )

    disabled = simulate_trace(
        records,
        profiles,
        config,
        mode="gate_without_scheduler",
        num_workers=1,
        dispatch_policy="index_order",
    ).events
    assert not bool(disabled["overload_mode_at_dispatch"].any())
    assert "pressure_fast" not in set(disabled["selection_reason"])


def test_changing_the_dispatch_policy_does_not_change_routing() -> None:
    """Guards the confound the fixed-routing study exists to rule out.

    If a policy also altered which backend served a job, a difference attributed to
    queue order would really be a routing difference.
    """
    records, profiles, config = _contended_trace()
    decoders = {}
    for policy in ("index_order", "fifo", "edf", "lst"):
        result = simulate_trace(
            records,
            profiles,
            config,
            mode="gate",
            num_workers=1,
            dispatch_policy=policy,
        )
        _assert_dispatch_is_contended(result)
        decoders[policy] = result.events.sort_values("shot_id")["selected_decoder"].tolist()
    reference = decoders["index_order"]
    for policy, selected in decoders.items():
        assert selected == reference, f"{policy} changed routing: {selected} != {reference}"


def test_summary_records_which_policy_produced_the_row() -> None:
    records = [_record(index, 4.0) for index in range(4)]
    result = simulate_trace(
        records,
        _profiles(4),
        _config(),
        mode="gate",
        num_workers=1,
        dispatch_policy="lst",
    )
    assert result.summary["dispatch_policy_name"] == "lst"
    assert result.summary["dispatch_policy"] == "least_laxity_first"
    assert result.dispatch_policy == "lst"


def test_unknown_dispatch_policy_and_routing_mode_are_rejected() -> None:
    records = [_record(0, 1.0)]
    with pytest.raises(ValueError, match="unknown dispatch policy"):
        simulate_trace(
            records,
            _profiles(1),
            _config(),
            mode="gate",
            num_workers=1,
            dispatch_policy="round_robin",
        )
    with pytest.raises(ValueError, match="unsupported routing mode"):
        simulate_trace(
            records,
            _profiles(1),
            _config(),
            mode="oracle",
            num_workers=1,
            dispatch_policy="index_order",
        )


# --------------------------------------------------------------------------- #
# Proposition 1'
# --------------------------------------------------------------------------- #


def test_bounded_service_gives_the_predicted_response_and_lag_bound() -> None:
    """Proposition 1': if C <= m*T then response-to-commit <= C and lag <= ceil(C/T)-1.

    Checked on a trace whose per-job service is exactly at the bound, since that is
    where the inequality is tight and an off-by-one would show.
    """
    period, num_workers = 10.0, 3
    bound = period * num_workers  # C = m*T, the boundary case
    service = [bound, bound, bound, bound, bound, bound, bound, bound, bound]
    records = [_record(index, value) for index, value in enumerate(service)]
    result = simulate_trace(
        records,
        _profiles(len(records), predicted_accurate=service),
        _config(period=period, deadline=bound),
        mode="accurate_only",
        num_workers=num_workers,
        dispatch_policy="index_order",
    )
    events = result.events
    assert float(events["response_to_commit_us"].max()) <= bound + 1e-9
    expected_lag_bound = int(np.ceil(bound / period)) - 1
    assert int(events["pauli_frame_lag"].max()) <= expected_lag_bound
    assert not bool(events["commit_deadline_miss"].any())


# --------------------------------------------------------------------------- #
# Runtime guard
# --------------------------------------------------------------------------- #


def test_higher_quantile_margin_and_reported_coverage() -> None:
    """The margin lands on an observed residual, so coverage is never interpolated away."""
    records = [_record(0, 5.0, 1.0), _record(1, 10.0, 2.0), _record(2, 20.0, 4.0)]
    margin = calibrate_runtime_margins(records, _profiles(3, predicted_accurate=[4.0] * 3), [0.5])[
        0
    ]
    assert margin.accurate_margin_us == 6.0
    assert margin.fast_margin_us == 1.0
    assert margin.accurate_coverage == 2.0 / 3.0
    assert margin.fast_coverage == 2.0 / 3.0
    assert margin.quantile_method == "higher"


def test_margins_are_clamped_at_zero_when_the_model_over_predicts() -> None:
    """An over-predicting model needs no guard; a negative margin would be optimistic."""
    records = [_record(index, 1.0, 0.5) for index in range(3)]
    margin = calibrate_runtime_margins(
        records, _profiles(3, predicted_accurate=[50.0] * 3, predicted_fast=[50.0] * 3), [0.9]
    )[0]
    assert margin.accurate_margin_us == 0.0
    assert margin.fast_margin_us == 0.0


def test_guard_adds_locked_margins_without_changing_risk() -> None:
    source = _profiles(2, predicted_accurate=[2.0, 3.0], predicted_fast=[1.0, 1.5])
    margin = RuntimeMargins(
        quantile=0.95,
        accurate_margin_us=7.0,
        fast_margin_us=2.0,
        accurate_coverage=0.95,
        fast_coverage=0.95,
        num_calibration_jobs=100,
    )
    guarded = apply_runtime_guard(source, margin)
    np.testing.assert_allclose(guarded.predicted_accurate_us, [9.0, 10.0])
    np.testing.assert_allclose(guarded.predicted_fast_us, [3.0, 3.5])
    np.testing.assert_allclose(guarded.risk_score, source.risk_score)
    np.testing.assert_allclose(guarded.confidence, source.confidence)
    assert guarded.metadata["runtime_guard_quantile"] == 0.95
    # The source must be untouched, or a later unguarded run would inherit the margin.
    np.testing.assert_allclose(source.predicted_accurate_us, [2.0, 3.0])


def test_guarded_run_stays_causal_and_reports_its_margins() -> None:
    records = [_record(0, 30.0), _record(1, 5.0)]
    margin = RuntimeMargins(
        quantile=0.95,
        accurate_margin_us=10.0,
        fast_margin_us=1.0,
        accurate_coverage=1.0,
        fast_coverage=1.0,
        num_calibration_jobs=2,
    )
    guarded = apply_runtime_guard(_profiles(2, predicted_accurate=[5.0, 5.0]), margin)
    result = simulate_trace(
        records,
        guarded,
        _config(deadline=100.0, boundary_interval=3),
        mode="gate",
        num_workers=1,
        dispatch_policy="commit_frontier",
    )
    assert not bool(result.events["decision_uses_realized_service"].any())
    metadata = guard_metadata(guarded)
    assert metadata["runtime_guard_enabled"]
    assert metadata["runtime_guard_quantile"] == 0.95
    assert metadata["runtime_guard_accurate_margin_us"] == 10.0


def test_guard_metadata_is_present_and_neutral_when_no_guard_is_applied() -> None:
    """Summaries always carry the columns, so a guarded run is never confused for a plain one."""
    metadata = guard_metadata(_profiles(2))
    assert metadata["runtime_guard_enabled"] is False
    assert np.isnan(metadata["runtime_guard_quantile"])
    assert metadata["runtime_guard_accurate_margin_us"] == 0.0
    assert metadata["runtime_guard_fast_margin_us"] == 0.0
