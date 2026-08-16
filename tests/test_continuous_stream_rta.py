# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _find_repo_root(path: Path) -> Path:
    for candidate in path.resolve().parents:
        if (candidate / "src" / "rt_preqec").is_dir():
            return candidate
    raise RuntimeError(f"could not find repository root above {path}")


ROOT = _find_repo_root(Path(__file__))
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rt_preqec.config import ProjectConfig
from rt_preqec.evaluation.real_stream import (
    RealStreamShotRecord,
    evaluate_mode_on_records,
    simulate_realtime_queue,
)
from scripts.evaluate_continuous_stream_rta import (
    exact_fifo_waiting_and_response_times,
    reindex_continuous_records,
)


def _record(shot_id: int) -> RealStreamShotRecord:
    return RealStreamShotRecord(
        shot_id=shot_id,
        syndrome=np.asarray([0], dtype=np.int8),
        observable=np.asarray([0], dtype=np.int8),
        accurate_prediction=np.asarray([0], dtype=np.int8),
        fast_prediction=np.asarray([0], dtype=np.int8),
        accurate_latency_us=1.0,
        fast_latency_us=1.0,
        features=np.asarray([0.0], dtype=np.float32),
        feature_names=["risk_proxy"],
        risk_label=0,
        hard_runtime=0,
        fast_wrong_vs_accurate=0,
        fast_logical_fail=0,
        metadata={},
    )


def _config(
    *, period: float = 10.0, deadline: float = 10.0, boundary_interval: int = 100
) -> ProjectConfig:
    config = ProjectConfig()
    config.runtime.round_period_us = period
    config.runtime.decode_deadline_us = deadline
    config.runtime.logical_boundary_interval = boundary_interval
    config.runtime.num_workers = 1
    return config


def _simulate(service_times: list[float], *, deadline: float = 10.0):
    records = [_record(index) for index in range(len(service_times))]
    return simulate_realtime_queue(
        records,
        selected_latencies=service_times,
        selected_predictions=[record.accurate_prediction for record in records],
        config=_config(deadline=deadline),
        selected_decoders=["accurate"] * len(records),
        ordered_commit=True,
    )


def test_case_a_underloaded_trace_has_zero_waiting_and_no_misses() -> None:
    service = [4.0, 4.0, 4.0]
    waiting, response = exact_fifo_waiting_and_response_times(service, 10.0)
    np.testing.assert_allclose(waiting, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(response, [4.0, 4.0, 4.0])

    events = _simulate(service)
    np.testing.assert_allclose(events["start_time_us"] - events["arrival_time_us"], waiting)
    np.testing.assert_allclose(events["response_time_us"], response)
    assert not events["deadline_miss"].any()


def test_case_b_overloaded_trace_matches_manual_recurrence() -> None:
    service = [15.0, 15.0, 2.0]
    waiting, response = exact_fifo_waiting_and_response_times(service, 10.0)
    np.testing.assert_allclose(waiting, [0.0, 5.0, 10.0])
    np.testing.assert_allclose(response, [15.0, 20.0, 12.0])

    events = _simulate(service)
    np.testing.assert_allclose(events["start_time_us"] - events["arrival_time_us"], waiting)
    np.testing.assert_allclose(events["response_time_us"], response)
    assert events["deadline_miss"].tolist() == [True, True, True]


def test_case_c_reindexing_makes_arrival_differences_periodic() -> None:
    records = [_record(9), _record(0), _record(3)]
    continuous = reindex_continuous_records(records)
    assert [record.shot_id for record in continuous] == [0, 1, 2]
    assert [record.metadata["original_shot_id"] for record in continuous] == [0, 3, 9]

    events = simulate_realtime_queue(
        continuous,
        selected_latencies=[1.0, 1.0, 1.0],
        selected_predictions=[record.accurate_prediction for record in continuous],
        config=_config(),
        ordered_commit=True,
    )
    np.testing.assert_allclose(np.diff(events["arrival_time_us"]), [10.0, 10.0])


def test_case_d_boundaries_use_continuous_index_not_original_shot_id() -> None:
    continuous = reindex_continuous_records([_record(0), _record(3), _record(9)])
    events = simulate_realtime_queue(
        continuous,
        selected_latencies=[1.0, 1.0, 1.0],
        selected_predictions=[record.accurate_prediction for record in continuous],
        config=_config(boundary_interval=3),
        ordered_commit=True,
    )
    assert events["logical_boundary"].tolist() == [False, False, True]
    assert events.loc[events["logical_boundary"], "shot_id"].tolist() == [2]
    assert continuous[2].metadata["original_shot_id"] == 9


def test_case_e_ordered_commit_blocks_later_completions_and_boundary() -> None:
    records = [_record(index) for index in range(3)]
    config = _config(period=10.0, deadline=15.0, boundary_interval=2)
    config.runtime.num_workers = 2
    events = simulate_realtime_queue(
        records,
        selected_latencies=[30.0, 1.0, 1.0],
        selected_predictions=[record.accurate_prediction for record in records],
        config=config,
        selected_decoders=["accurate"] * len(records),
        ordered_commit=True,
    )

    np.testing.assert_allclose(events["completion_time_us"], [30.0, 11.0, 21.0])
    np.testing.assert_allclose(events["commit_time_us"], [30.0, 30.0, 30.0])
    np.testing.assert_allclose(events["response_to_decode_us"], [30.0, 1.0, 1.0])
    np.testing.assert_allclose(events["response_to_commit_us"], [30.0, 20.0, 10.0])
    assert events["committed_prefix_at_arrival"].tolist() == [-1, -1, -1]
    assert events["pauli_frame_lag"].tolist() == [0, 1, 2]
    assert events["decode_deadline_miss"].tolist() == [True, False, False]
    assert events["commit_deadline_miss"].tolist() == [True, True, False]
    assert events["deadline_miss"].tolist() == [True, True, False]
    assert events["logical_boundary"].tolist() == [False, True, False]
    assert not bool(events.loc[1, "boundary_commit_success"])
    assert int(events.loc[1, "committed_prefix_at_deadline"]) == -1


def test_case_f_routing_context_uses_ordered_commit_lag() -> None:
    records = [_record(index) for index in range(3)]
    for record, service_time in zip(records, [30.0, 1.0, 1.0], strict=True):
        record.accurate_latency_us = service_time
    config = _config(period=10.0, deadline=15.0, boundary_interval=2)
    config.runtime.num_workers = 2

    result = evaluate_mode_on_records(
        records,
        mode="accurate_only",
        config=config,
        ordered_commit=True,
    )

    assert result.decisions["committed_prefix_at_arrival"].tolist() == [-1, -1, -1]
    assert result.decisions["estimated_pauli_frame_lag"].tolist() == [0, 1, 2]
    assert result.events["pauli_frame_lag"].tolist() == [0, 1, 2]
    assert result.metrics["decode_deadline_miss_ratio"] == 1.0 / 3.0
    assert result.metrics["commit_deadline_miss_ratio"] == 2.0 / 3.0
    assert result.metrics["deadline_miss_ratio"] == 2.0 / 3.0
    assert result.metrics["mean_response_time_us"] == 20.0
    assert result.metrics["mean_response_to_decode_us"] == 32.0 / 3.0
    assert result.metrics["mean_response_to_commit_us"] == 20.0
    assert result.metrics["boundary_commit_success_rate"] == 0.0
