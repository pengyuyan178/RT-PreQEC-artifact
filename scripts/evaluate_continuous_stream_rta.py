"""Finite-trace response-time analysis with truly periodic arrivals.

This script is intentionally separate from the legacy real-stream evaluation.
It reloads the saved held-out records, reindexes them as a continuous stream,
and reruns every scheduling decision under the resulting queue state.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

from rt_preqec.config import ProjectConfig, load_config
from rt_preqec.evaluation.real_stream import (
    RealStreamShotRecord,
    evaluate_mode_on_records,
    split_records,
)
from rt_preqec.models.risk_profiler import load_risk_profiler_checkpoint
from scripts.run_paper_experiment_suite import _load_records_csv

DEFAULT_MODES = [
    "accurate_only",
    "fast_only",
    "edf",
    "rt_qec_ai",
    "rt_qec_without_scheduler",
]
DEFAULT_WORKERS = [1, 2, 3, 4]
WINDOW_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
MODE_LABELS = {
    "accurate_only": "Accurate-only",
    "fast_only": "Fast-only",
    "edf": "EDF",
    "rt_qec_ai": "RT-PreQEC",
    "rt_qec_without_scheduler": "No-scheduler",
}
OUTPUT_FILENAMES = [
    "code/evaluate_continuous_stream_rta.py",
    "code/real_stream.py",
    "code/test_continuous_stream_rta.py",
    "code/real_stream_eval_main_ai_selected.yaml",
    "code/environment.yml",
    "arrival_gap_diagnostic.csv",
    "arrival_gap_diagnostic.json",
    "continuous_stream_summary.csv",
    "continuous_stream_events.csv",
    "window_demand_excess.csv",
    "minimum_capacity_summary.csv",
    "fig_a1_worker_capacity.pdf",
    "fig_a1_worker_capacity.png",
    "fig_a1_window_excess.pdf",
    "fig_a1_window_excess.png",
    "README.md",
]


def _original_shot_id(record: RealStreamShotRecord) -> int:
    return int(record.metadata.get("original_shot_id", record.shot_id))


def _snapshot_rebuttal_code(out_dir: Path, config_path: Path) -> None:
    """Archive the exact experiment entry point, tests, and environment inputs."""
    code_dir = out_dir / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    sources = [
        (Path(__file__).resolve(), code_dir / "evaluate_continuous_stream_rta.py"),
        (
            ROOT / "src" / "rt_preqec" / "evaluation" / "real_stream.py",
            code_dir / "real_stream.py",
        ),
        (
            ROOT / "tests" / "test_continuous_stream_rta.py",
            code_dir / "test_continuous_stream_rta.py",
        ),
        (config_path.resolve(), code_dir / config_path.name),
        (ROOT / "environment.yml", code_dir / "environment.yml"),
    ]
    for source, destination in sources:
        if not source.is_file():
            raise FileNotFoundError(f"rebuttal code snapshot source not found: {source}")
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)


def reindex_continuous_records(
    records: Sequence[RealStreamShotRecord],
) -> list[RealStreamShotRecord]:
    """Return deep-copied records indexed as one job every stream period."""
    ordered = sorted(records, key=_original_shot_id)
    continuous: list[RealStreamShotRecord] = []
    for stream_index, source in enumerate(ordered):
        record = copy.deepcopy(source)
        record.metadata = dict(record.metadata)
        record.metadata["original_shot_id"] = _original_shot_id(source)
        record.shot_id = stream_index
        continuous.append(record)
    return continuous


def exact_fifo_waiting_and_response_times(
    service_times_us: Sequence[float] | np.ndarray,
    round_period_us: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Lindley's recurrence to one finite FIFO trace."""
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
    for index, service_time in enumerate(service):
        waiting[index] = workload
        response[index] = workload + service_time
        workload = max(0.0, workload + service_time - period)
    return waiting, response


def arrival_gap_diagnostic(
    records: Sequence[RealStreamShotRecord],
    round_period_us: float,
    *,
    records_path: str | Path | None = None,
    config_path: str | Path | None = None,
    expected_random_test_ids: Sequence[int] | None = None,
    split_seed: int | None = None,
) -> dict[str, Any]:
    """Summarize the spacing induced by the legacy shot-ID arrival formula."""
    original_ids = np.asarray(
        sorted(_original_shot_id(record) for record in records), dtype=np.int64
    )
    gaps = np.diff(original_ids)
    if original_ids.size == 0:
        raise ValueError("records.csv contains no jobs")
    if gaps.size:
        mean_gap = float(np.mean(gaps))
        gap_fields: dict[str, Any] = {
            "min_original_shot_id_gap": int(np.min(gaps)),
            "max_original_shot_id_gap": int(np.max(gaps)),
            "mean_original_shot_id_gap": mean_gap,
            "median_original_shot_id_gap": float(np.median(gaps)),
            "p95_original_shot_id_gap": float(np.percentile(gaps, 95)),
            "p99_original_shot_id_gap": float(np.percentile(gaps, 99)),
            "fraction_gaps_equal_one": float(np.mean(gaps == 1)),
            "effective_mean_interarrival_time_us_existing_code": mean_gap * float(round_period_us),
        }
    else:
        gap_fields = {
            "min_original_shot_id_gap": None,
            "max_original_shot_id_gap": None,
            "mean_original_shot_id_gap": None,
            "median_original_shot_id_gap": None,
            "p95_original_shot_id_gap": None,
            "p99_original_shot_id_gap": None,
            "fraction_gaps_equal_one": None,
            "effective_mean_interarrival_time_us_existing_code": None,
        }
    expected_ids = (
        np.asarray(sorted(int(value) for value in expected_random_test_ids), dtype=np.int64)
        if expected_random_test_ids is not None
        else None
    )
    split_reproduced = (
        bool(np.array_equal(original_ids, expected_ids)) if expected_ids is not None else None
    )
    diagnostic: dict[str, Any] = {
        "num_test_jobs": int(original_ids.size),
        "min_original_shot_id": int(original_ids.min()),
        "max_original_shot_id": int(original_ids.max()),
        "num_consecutive_original_shot_id_gaps": int(gaps.size),
        **gap_fields,
        "nominal_round_period_us": float(round_period_us),
        "legacy_arrival_formula": "arrival_time_us = original_shot_id * round_period_us",
        "continuous_arrival_formula": "arrival_time_us = continuous_stream_index * round_period_us",
        "random_split_then_sorted_by_original_shot_id": (
            split_reproduced if split_reproduced is not None else True
        ),
        "configured_random_test_split_reproduced": split_reproduced,
        "configured_random_split_seed": int(split_seed) if split_seed is not None else None,
        "configured_random_test_split_size": (
            int(expected_ids.size) if expected_ids is not None else None
        ),
        "split_verification_method": (
            "existing split_records helper" if expected_ids is not None else None
        ),
        "arrival_gap_issue_verified": bool(gaps.size and np.any(gaps != 1)),
        "records_path": str(records_path) if records_path is not None else None,
        "config_path": str(config_path) if config_path is not None else None,
        "percentile_method": "numpy linear percentile",
    }
    effective = diagnostic["effective_mean_interarrival_time_us_existing_code"]
    diagnostic["effective_to_nominal_interarrival_ratio"] = (
        float(effective) / float(round_period_us) if effective is not None else None
    )
    return diagnostic


def window_demand_excess_rows(
    service_times_us: Sequence[float] | np.ndarray,
    round_period_us: float,
    mode: str,
    window_sizes: Iterable[int] = WINDOW_SIZES,
) -> list[dict[str, Any]]:
    """Compute maximum finite-trace service demand above k*T."""
    service = np.asarray(service_times_us, dtype=float)
    prefix = np.concatenate(([0.0], np.cumsum(service, dtype=float)))
    rows: list[dict[str, Any]] = []
    for raw_size in window_sizes:
        size = int(raw_size)
        if size <= 0 or size > service.size:
            continue
        demands = prefix[size:] - prefix[:-size]
        start = int(np.argmax(demands))
        maximum_demand = float(demands[start])
        capacity = float(size * round_period_us)
        rows.append(
            {
                "mode": mode,
                "mode_label": MODE_LABELS.get(mode, mode),
                "num_workers": 1,
                "window_size_jobs": size,
                "maximum_window_service_demand_us": maximum_demand,
                "window_capacity_us": capacity,
                "demand_excess_us": maximum_demand - capacity,
                "window_start_index": start,
                "window_end_index_inclusive": start + size - 1,
            }
        )
    return rows


def _assert_continuous_event_semantics(events: pd.DataFrame, config: ProjectConfig) -> None:
    indices = np.arange(len(events), dtype=np.int64)
    np.testing.assert_array_equal(events["shot_id"].to_numpy(dtype=np.int64), indices)
    expected_arrivals = indices.astype(float) * float(config.runtime.round_period_us)
    np.testing.assert_allclose(
        events["arrival_time_us"].to_numpy(dtype=float), expected_arrivals, rtol=0.0, atol=1e-10
    )
    interval = max(int(config.runtime.logical_boundary_interval), 1)
    expected_boundaries = ((indices + 1) % interval) == 0
    np.testing.assert_array_equal(
        events["logical_boundary"].to_numpy(dtype=bool), expected_boundaries
    )
    if not events["ordered_commit_enabled"].astype(bool).all():
        raise AssertionError("continuous-stream events must use ordered commit semantics")
    expected_lag = indices - events["committed_prefix_at_arrival"].to_numpy(dtype=np.int64) - 1
    np.testing.assert_array_equal(events["pauli_frame_lag"].to_numpy(dtype=np.int64), expected_lag)
    arrivals = events["arrival_time_us"].to_numpy(dtype=float)
    deadlines = events["deadline_us"].to_numpy(dtype=float)
    completions = events["completion_time_us"].to_numpy(dtype=float)
    commits = events["commit_time_us"].to_numpy(dtype=float)
    expected_commits = np.maximum.accumulate(completions)
    np.testing.assert_allclose(commits, expected_commits, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(
        events["response_to_decode_us"].to_numpy(dtype=float),
        completions - arrivals,
        rtol=0.0,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        events["response_to_commit_us"].to_numpy(dtype=float),
        commits - arrivals,
        rtol=0.0,
        atol=1e-10,
    )
    expected_prefix_at_deadline = np.searchsorted(expected_commits, deadlines, side="right") - 1
    np.testing.assert_array_equal(
        events["committed_prefix_at_deadline"].to_numpy(dtype=np.int64),
        expected_prefix_at_deadline,
    )
    decode_deadline_miss = completions > deadlines
    commit_deadline_miss = expected_prefix_at_deadline < indices
    np.testing.assert_array_equal(
        events["decode_deadline_miss"].to_numpy(dtype=bool), decode_deadline_miss
    )
    np.testing.assert_array_equal(
        events["commit_deadline_miss"].to_numpy(dtype=bool), commit_deadline_miss
    )
    np.testing.assert_array_equal(
        events["deadline_miss"].to_numpy(dtype=bool), commit_deadline_miss
    )
    boundary_success = expected_prefix_at_deadline >= indices
    np.testing.assert_array_equal(
        events["boundary_prerequisites_committed"].to_numpy(dtype=bool), boundary_success
    )
    np.testing.assert_array_equal(
        events.loc[expected_boundaries, "boundary_commit_success"].to_numpy(dtype=bool),
        boundary_success[expected_boundaries],
    )


def _augment_events(
    result: Any,
    records: Sequence[RealStreamShotRecord],
    config: ProjectConfig,
    mode: str,
    num_workers: int,
    *,
    tolerance_us: float,
) -> tuple[pd.DataFrame, dict[str, float | bool]]:
    events = result.events.copy()
    _assert_continuous_event_semantics(events, config)
    events.insert(0, "mode", mode)
    events.insert(1, "mode_label", MODE_LABELS.get(mode, mode))
    events.insert(2, "num_workers", int(num_workers))
    events.insert(3, "continuous_stream_index", events["shot_id"].to_numpy(dtype=np.int64))
    events.insert(4, "original_shot_id", [_original_shot_id(record) for record in records])
    events["service_time_us"] = events["latency_us"].to_numpy(dtype=float)
    events["waiting_time_us"] = events["start_time_us"].to_numpy(dtype=float) - events[
        "arrival_time_us"
    ].to_numpy(dtype=float)
    events["lag_budget_jobs"] = int(config.runtime.max_pauli_frame_lag)

    decision_columns = [
        "shot_id",
        "selection_reason",
        "estimated_backlog_before_arrival",
        "estimated_pauli_frame_lag",
        "committed_prefix_at_arrival",
        "deadline_slack_us",
        "overload_mode",
        "fast_path_certified",
        "ai_risk_score",
        "ai_confidence",
    ]
    available = [column for column in decision_columns if column in result.decisions.columns]
    if available:
        decisions = result.decisions[available].copy()
        decisions = decisions.rename(
            columns={column: f"decision_{column}" for column in available if column != "shot_id"}
        )
        events = events.merge(decisions, on="shot_id", how="left", validate="one_to_one")
        if "decision_committed_prefix_at_arrival" in events:
            np.testing.assert_array_equal(
                events["decision_committed_prefix_at_arrival"].to_numpy(dtype=np.int64),
                events["committed_prefix_at_arrival"].to_numpy(dtype=np.int64),
            )
        if "decision_estimated_pauli_frame_lag" in events:
            np.testing.assert_array_equal(
                events["decision_estimated_pauli_frame_lag"].to_numpy(dtype=np.int64),
                events["pauli_frame_lag"].to_numpy(dtype=np.int64),
            )

    recurrence: dict[str, float | bool] = {
        "finite_trace_recurrence_checked": False,
        "exact_finite_trace_max_waiting_time_us": np.nan,
        "exact_finite_trace_max_response_time_us": np.nan,
        "max_abs_waiting_time_discrepancy_us": np.nan,
        "max_abs_response_time_discrepancy_us": np.nan,
    }
    events["exact_fifo_waiting_time_us"] = np.nan
    events["exact_fifo_response_time_us"] = np.nan
    events["waiting_time_discrepancy_us"] = np.nan
    events["response_time_discrepancy_us"] = np.nan
    if int(num_workers) == 1:
        exact_waiting, exact_response = exact_fifo_waiting_and_response_times(
            events["service_time_us"].to_numpy(dtype=float),
            float(config.runtime.round_period_us),
        )
        simulator_waiting = events["waiting_time_us"].to_numpy(dtype=float)
        simulator_response = events["response_time_us"].to_numpy(dtype=float)
        waiting_difference = simulator_waiting - exact_waiting
        response_difference = simulator_response - exact_response
        max_waiting_difference = float(np.max(np.abs(waiting_difference))) if len(events) else 0.0
        max_response_difference = float(np.max(np.abs(response_difference))) if len(events) else 0.0
        np.testing.assert_allclose(simulator_waiting, exact_waiting, rtol=0.0, atol=tolerance_us)
        np.testing.assert_allclose(simulator_response, exact_response, rtol=0.0, atol=tolerance_us)
        events["exact_fifo_waiting_time_us"] = exact_waiting
        events["exact_fifo_response_time_us"] = exact_response
        events["waiting_time_discrepancy_us"] = waiting_difference
        events["response_time_discrepancy_us"] = response_difference
        recurrence = {
            "finite_trace_recurrence_checked": True,
            "exact_finite_trace_max_waiting_time_us": (
                float(np.max(exact_waiting)) if len(events) else 0.0
            ),
            "exact_finite_trace_max_response_time_us": (
                float(np.max(exact_response)) if len(events) else 0.0
            ),
            "max_abs_waiting_time_discrepancy_us": max_waiting_difference,
            "max_abs_response_time_discrepancy_us": max_response_difference,
        }
    return events, recurrence


def _summary_row(
    mode: str,
    num_workers: int,
    events: pd.DataFrame,
    result: Any,
    config: ProjectConfig,
    recurrence: dict[str, float | bool],
) -> dict[str, Any]:
    service = events["service_time_us"].to_numpy(dtype=float)
    response_to_decode = events["response_to_decode_us"].to_numpy(dtype=float)
    response_to_commit = events["response_to_commit_us"].to_numpy(dtype=float)
    decode_deadline_miss_ratio = float(events["decode_deadline_miss"].astype(float).mean())
    commit_deadline_miss_ratio = float(events["commit_deadline_miss"].astype(float).mean())
    lag_violation_ratio = float(events["pauli_frame_lag_violation"].astype(float).mean())
    maximum_lag = int(events["pauli_frame_lag"].max())
    boundary_events = events.loc[events["logical_boundary"].astype(bool)]
    boundary_success_rate = (
        float(boundary_events["boundary_commit_success"].astype(float).mean())
        if not boundary_events.empty
        else 1.0
    )
    zero_decode_deadline_miss = decode_deadline_miss_ratio == 0.0
    zero_commit_deadline_miss = commit_deadline_miss_ratio == 0.0
    lag_within_budget = maximum_lag <= int(config.runtime.max_pauli_frame_lag)
    all_boundaries_successful = boundary_success_rate == 1.0
    return {
        "mode": mode,
        "mode_label": MODE_LABELS.get(mode, mode),
        "num_workers": int(num_workers),
        "num_jobs": int(len(events)),
        "round_period_us": float(config.runtime.round_period_us),
        "relative_deadline_us": float(config.runtime.decode_deadline_us),
        "pauli_frame_lag_budget": int(config.runtime.max_pauli_frame_lag),
        "logical_boundary_interval_jobs": int(config.runtime.logical_boundary_interval),
        "num_logical_boundaries": int(len(boundary_events)),
        "queue_model": "parallel_decode_with_in_order_commit_buffer",
        "primary_response_definition": "response_to_commit_us",
        "primary_deadline_definition": "commit_time_us <= deadline_us",
        "lag_definition": "stream_index - committed_prefix_at_arrival - 1",
        "boundary_success_definition": "committed_prefix_at_deadline >= boundary_index",
        "mean_service_time_us": float(np.mean(service)),
        "p99_service_time_us": float(np.percentile(service, 99)),
        "observed_max_service_time_us": float(np.max(service)),
        "offered_load_rho": float(np.sum(service))
        / (float(len(service)) * float(num_workers) * float(config.runtime.round_period_us)),
        "mean_response_time_us": float(np.mean(response_to_commit)),
        "p99_response_time_us": float(np.percentile(response_to_commit, 99)),
        "observed_max_response_time_us": float(np.max(response_to_commit)),
        "mean_response_to_decode_us": float(np.mean(response_to_decode)),
        "p99_response_to_decode_us": float(np.percentile(response_to_decode, 99)),
        "observed_max_response_to_decode_us": float(np.max(response_to_decode)),
        "mean_response_to_commit_us": float(np.mean(response_to_commit)),
        "p99_response_to_commit_us": float(np.percentile(response_to_commit, 99)),
        "observed_max_response_to_commit_us": float(np.max(response_to_commit)),
        "deadline_miss_ratio": commit_deadline_miss_ratio,
        "decode_deadline_miss_ratio": decode_deadline_miss_ratio,
        "commit_deadline_miss_ratio": commit_deadline_miss_ratio,
        "maximum_backlog": int(events["backlog"].max()),
        "maximum_pauli_frame_lag": maximum_lag,
        "lag_violation_ratio": lag_violation_ratio,
        "boundary_commit_success_rate": boundary_success_rate,
        "logical_error_rate": float(events["logical_error"].astype(float).mean()),
        "fast_selection_rate": float(result.metrics["fast_selection_rate"]),
        "trace_zero_deadline_miss": bool(zero_commit_deadline_miss),
        "trace_zero_decode_deadline_miss": bool(zero_decode_deadline_miss),
        "trace_zero_commit_deadline_miss": bool(zero_commit_deadline_miss),
        "trace_lag_within_budget": bool(lag_within_budget),
        "trace_all_boundaries_successful": bool(all_boundaries_successful),
        "trace_zero_violation": bool(
            zero_commit_deadline_miss and lag_within_budget and all_boundaries_successful
        ),
        **recurrence,
    }


def _minimum_capacity_summary(summary: pd.DataFrame, modes: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for mode in modes:
        mode_rows = summary.loc[summary["mode"] == mode].sort_values("num_workers")
        passing = mode_rows.loc[mode_rows["trace_zero_violation"].astype(bool)]
        minimum = int(passing.iloc[0]["num_workers"]) if not passing.empty else None
        rows.append(
            {
                "mode": mode,
                "mode_label": MODE_LABELS.get(mode, mode),
                "minimum_zero_violation_worker_count": minimum,
                "trace_zero_violation_found": minimum is not None,
                "evaluated_worker_counts": ",".join(
                    str(int(value)) for value in mode_rows["num_workers"]
                ),
                "criterion": (
                    "zero commit-deadline misses; ordered-prefix lag <= Lmax; "
                    "all boundaries committed by deadline"
                ),
            }
        )
    frame = pd.DataFrame(rows)
    frame["minimum_zero_violation_worker_count"] = frame[
        "minimum_zero_violation_worker_count"
    ].astype("Int64")
    return frame


def _plot_worker_capacity(summary: pd.DataFrame, out_dir: Path) -> None:
    plotted_modes = ["accurate_only", "edf", "rt_qec_ai", "rt_qec_without_scheduler"]
    colors = {
        "accurate_only": "#1f77b4",
        "edf": "#d62728",
        "rt_qec_ai": "#2ca02c",
        "rt_qec_without_scheduler": "#7f7f7f",
    }
    markers = {"accurate_only": "o", "edf": "s", "rt_qec_ai": "^", "rt_qec_without_scheduler": "D"}
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.6), constrained_layout=True)
    panels = [
        ("commit_deadline_miss_ratio", "Commit-deadline miss ratio"),
        ("maximum_pauli_frame_lag", "Maximum Pauli-frame lag"),
        ("boundary_commit_success_rate", "Boundary-commit success"),
    ]
    for mode in plotted_modes:
        subset = summary.loc[summary["mode"] == mode].sort_values("num_workers")
        if subset.empty:
            continue
        for axis, (column, _) in zip(axes, panels, strict=True):
            axis.plot(
                subset["num_workers"],
                subset[column],
                color=colors[mode],
                marker=markers[mode],
                linewidth=1.7,
                markersize=5,
                label=MODE_LABELS[mode],
            )
    worker_values = sorted(int(value) for value in summary["num_workers"].unique())
    for axis, (_, ylabel) in zip(axes, panels, strict=True):
        axis.set_xlabel("Worker count")
        axis.set_ylabel(ylabel)
        axis.set_xticks(worker_values)
        axis.grid(True, axis="y", color="#d9d9d9", linewidth=0.7)
    axes[0].set_ylim(-0.02, 1.02)
    axes[2].set_ylim(-0.02, 1.02)
    lag_budget = int(summary["pauli_frame_lag_budget"].iloc[0])
    axes[1].axhline(
        lag_budget,
        color="#444444",
        linestyle="--",
        linewidth=1.0,
        label=f"Lag budget ({lag_budget})",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=4, frameon=False)
    figure.savefig(out_dir / "fig_a1_worker_capacity.pdf", bbox_inches="tight")
    figure.savefig(out_dir / "fig_a1_worker_capacity.png", dpi=240, bbox_inches="tight")
    plt.close(figure)


def _plot_window_excess(window_frame: pd.DataFrame, out_dir: Path) -> None:
    colors = ["#1f77b4", "#ff7f0e", "#d62728", "#2ca02c", "#7f7f7f"]
    markers = ["o", "v", "s", "^", "D"]
    window_sizes = sorted(int(value) for value in window_frame["window_size_jobs"].unique())
    positions = np.arange(len(window_sizes), dtype=float)
    figure, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    plotted_modes = [str(mode) for mode in window_frame["mode"].drop_duplicates()]
    for index, mode in enumerate(plotted_modes):
        color = colors[index % len(colors)]
        marker = markers[index % len(markers)]
        subset = window_frame.loc[window_frame["mode"] == mode].sort_values("window_size_jobs")
        if subset.empty:
            continue
        mode_positions = [window_sizes.index(int(value)) for value in subset["window_size_jobs"]]
        axis.plot(
            mode_positions,
            subset["demand_excess_us"],
            color=color,
            marker=marker,
            linewidth=1.6,
            markersize=4.5,
            label=MODE_LABELS.get(mode, mode),
        )
    axis.axhline(0.0, color="#222222", linewidth=1.0)
    axis.set_xticks(positions, [str(value) for value in window_sizes])
    axis.set_xlabel("Window size k (jobs)")
    axis.set_ylabel("Maximum finite-trace demand excess (us)")
    axis.grid(True, axis="y", color="#d9d9d9", linewidth=0.7)
    axis.legend(frameon=False, ncol=2)
    figure.savefig(out_dir / "fig_a1_window_excess.pdf", bbox_inches="tight")
    figure.savefig(out_dir / "fig_a1_window_excess.png", dpi=240, bbox_inches="tight")
    plt.close(figure)


def _write_readme(
    out_dir: Path,
    diagnostic: dict[str, Any],
    minimum_capacity: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    config_path: Path,
    records_path: Path,
    checkpoint_path: Path,
) -> None:
    minimum_lines = []
    for row in minimum_capacity.itertuples(index=False):
        value = row.minimum_zero_violation_worker_count
        rendered = "null" if pd.isna(value) else str(int(value))
        minimum_lines.append(f"| {row.mode_label} | {rendered} |")
    checked = summary.loc[summary["finite_trace_recurrence_checked"].astype(bool)]
    max_waiting_discrepancy = (
        float(checked["max_abs_waiting_time_discrepancy_us"].max()) if not checked.empty else 0.0
    )
    max_response_discrepancy = (
        float(checked["max_abs_response_time_discrepancy_us"].max()) if not checked.empty else 0.0
    )
    split_seed = diagnostic["configured_random_split_seed"]
    num_jobs = diagnostic["num_test_jobs"]
    min_id = diagnostic["min_original_shot_id"]
    max_id = diagnostic["max_original_shot_id"]
    mean_gap = diagnostic["mean_original_shot_id_gap"]
    median_gap = diagnostic["median_original_shot_id_gap"]
    p95_gap = diagnostic["p95_original_shot_id_gap"]
    p99_gap = diagnostic["p99_original_shot_id_gap"]
    fraction_unit_gap = diagnostic["fraction_gaps_equal_one"]
    period_us = diagnostic["nominal_round_period_us"]
    effective_interarrival_us = diagnostic["effective_mean_interarrival_time_us_existing_code"]
    modes_arg = ",".join(summary["mode"].drop_duplicates())
    workers_arg = ",".join(str(int(value)) for value in sorted(summary["num_workers"].unique()))
    reproduction_command = (
        "conda run -n RT-preqec python "
        f"{(out_dir / 'code' / 'evaluate_continuous_stream_rta.py').as_posix()} "
        f"--config {(out_dir / 'code' / config_path.name).as_posix()} "
        f"--records {records_path.as_posix()} "
        f"--risk-checkpoint {checkpoint_path.as_posix()} --modes {modes_arg} "
        f"--workers {workers_arg} --out {out_dir.as_posix()}"
    )
    text = f"""# Reviewer A1: continuous-stream finite-trace analysis

## Scope

This rebuttal analysis uses truly periodic, continuous arrivals: after sorting by original shot
ID, job `i` is assigned continuous stream index `i`, arrives at `i * T`, and uses that index
for logical boundaries. The original ID remains in
`record.metadata[\"original_shot_id\"]` and in the event table. Every mode and worker count is
reevaluated with `evaluate_mode_on_records`; saved decisions are not replayed.

Parallel decoder completions enter an in-order commit buffer. At arrival `t`, lag is
`t - committed_prefix - 1`. Each event records decode completion, ordered commit,
response-to-decode, and response-to-commit times. A boundary succeeds only when its complete
prefix has committed by the boundary deadline.

The generic response-time and deadline-miss columns in the continuous-stream summary use
response-to-commit semantics. Explicit response-to-decode and decode-deadline columns are
retained so decode completion and externally visible ordered commitment cannot be conflated.

The result is exact for the evaluated finite trace under the repository's queue model. It is
not a formal all-workloads schedulability guarantee. The observed maximum service (execution)
time is not a WCET. The worker sweep separates scheduling behavior from service-capacity
provisioning.

## Arrival-gap finding

The legacy code maps arrival time to `record.shot_id * round_period_us`, while the generated
test records were randomly selected and then sorted by original shot ID. Reapplying the
repository's `split_records` helper with configured evaluation seed {split_seed} exactly
reproduces the saved test IDs. In this trace there are {num_jobs:,} jobs with original IDs
{min_id} through {max_id}. Consecutive original-ID gaps have mean {mean_gap:.6f}, median
{median_gap:.6f}, p95 {p95_gap:.6f}, and p99 {p99_gap:.6f}; {fraction_unit_gap:.6%} equal one.
With nominal `T = {period_us:.6f} us`, the legacy formula produces an effective mean
interarrival time of {effective_interarrival_us:.6f} us.

## Exact one-worker check

For each one-worker mode, waiting and response times are independently recomputed with
`W_0 = 0`, `W_(i+1) = max(0, W_i + C_i - T)`, and `R_i = W_i + C_i`. These values are asserted
event by event against the queue simulator. The maximum absolute discrepancies are
{max_waiting_discrepancy:.3e} us for waiting time and {max_response_discrepancy:.3e} us for
response time. Reported maxima use the terms observed maximum service time, exact finite-trace
maximum waiting time, and exact finite-trace maximum response time.

The window-demand table reports `max_j(sum(C_i) - k*T)` only for windows present in this finite
trace. It is a burst-demand diagnostic, not a universal network-calculus bound or WCET
guarantee.

## Minimum evaluated capacity

`trace_zero_violation` requires zero response-to-commit deadline misses, maximum ordered-prefix
Pauli-frame lag no greater than `Lmax`, and commitment of every boundary prefix by its deadline.

| Mode | Minimum zero-violation worker count |
|---|---:|
{chr(10).join(minimum_lines)}

## Inputs and reproduction

- Config: `{config_path.as_posix()}`
- Records: `{records_path.as_posix()}`
- Risk checkpoint: `{checkpoint_path.as_posix()}`
- Nominal period: `{diagnostic['nominal_round_period_us']} us`
- Lag budget: `{int(summary['pauli_frame_lag_budget'].iloc[0])}` jobs
- Logical boundary interval: `{int(summary['logical_boundary_interval_jobs'].iloc[0])}` jobs

Run from the repository root:

```powershell
{reproduction_command}
```

All files in this directory are separate rebuttal artifacts. No legacy result is overwritten.
The exact experiment script, focused tests, selected configuration, and environment
specification used for this run are archived under `code/` in this directory.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def run_continuous_stream_analysis(
    *,
    config_path: str | Path,
    records_path: str | Path,
    risk_checkpoint_path: str | Path,
    modes: Sequence[str] = DEFAULT_MODES,
    workers: Sequence[int] = DEFAULT_WORKERS,
    out_dir: str | Path,
    recurrence_tolerance_us: float = 1e-8,
) -> dict[str, Any]:
    config_path = Path(config_path)
    records_path = Path(records_path)
    checkpoint_path = Path(risk_checkpoint_path)
    out_dir = Path(out_dir)
    for path, label in (
        (config_path, "config"),
        (records_path, "records"),
        (checkpoint_path, "risk checkpoint"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    selected_modes = [str(mode).strip() for mode in modes if str(mode).strip()]
    selected_workers = sorted(set(int(value) for value in workers))
    if not selected_modes:
        raise ValueError("at least one mode is required")
    if not selected_workers or any(value <= 0 for value in selected_workers):
        raise ValueError("worker counts must be positive integers")
    if 1 not in selected_workers:
        raise ValueError("worker counts must include 1 for the exact finite-trace recurrence check")

    base_config = load_config(config_path)
    if base_config.timing.set_torch_num_threads is not None:
        import torch

        torch.set_num_threads(int(base_config.timing.set_torch_num_threads))
    loaded_records = _load_records_csv(records_path)
    configured_split = split_records(
        list(range(int(base_config.qec.num_shots))),
        test_fraction=float(base_config.qec.test_fraction),
        val_fraction=float(base_config.risk_training.val_fraction),
        seed=int(base_config.data_protocol.eval_seed),
    )
    diagnostic = arrival_gap_diagnostic(
        loaded_records,
        float(base_config.runtime.round_period_us),
        records_path=records_path,
        config_path=config_path,
        expected_random_test_ids=configured_split["test"],
        split_seed=int(base_config.data_protocol.eval_seed),
    )
    records = reindex_continuous_records(loaded_records)
    if [record.shot_id for record in records] != list(range(len(records))):
        raise AssertionError("continuous stream reindexing failed")

    risk_model, normalization, risk_metadata = load_risk_profiler_checkpoint(
        checkpoint_path,
        device=base_config.device,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    _snapshot_rebuttal_code(out_dir, config_path)
    pd.DataFrame([diagnostic]).to_csv(out_dir / "arrival_gap_diagnostic.csv", index=False)
    with (out_dir / "arrival_gap_diagnostic.json").open("w", encoding="utf-8") as handle:
        json.dump(diagnostic, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")

    summary_rows: list[dict[str, Any]] = []
    event_frames: list[pd.DataFrame] = []
    window_rows: list[dict[str, Any]] = []
    for mode in selected_modes:
        for num_workers in selected_workers:
            config = copy.deepcopy(base_config)
            config.runtime.num_workers = int(num_workers)
            result = evaluate_mode_on_records(
                records,
                mode=mode,
                config=config,
                risk_model=risk_model,
                normalization=normalization,
                risk_metadata=risk_metadata,
                ordered_commit=True,
            )
            events, recurrence = _augment_events(
                result,
                records,
                config,
                mode,
                num_workers,
                tolerance_us=recurrence_tolerance_us,
            )
            summary_rows.append(_summary_row(mode, num_workers, events, result, config, recurrence))
            event_frames.append(events)
            if num_workers == 1:
                window_rows.extend(
                    window_demand_excess_rows(
                        events["service_time_us"].to_numpy(dtype=float),
                        float(config.runtime.round_period_us),
                        mode,
                    )
                )

    summary = pd.DataFrame(summary_rows)
    events = pd.concat(
        [frame.dropna(axis="columns", how="all") for frame in event_frames],
        ignore_index=True,
        sort=False,
    )
    window_frame = pd.DataFrame(window_rows)
    minimum_capacity = _minimum_capacity_summary(summary, selected_modes)
    summary.to_csv(out_dir / "continuous_stream_summary.csv", index=False)
    events.to_csv(out_dir / "continuous_stream_events.csv", index=False)
    window_frame.to_csv(out_dir / "window_demand_excess.csv", index=False)
    minimum_capacity.to_csv(out_dir / "minimum_capacity_summary.csv", index=False)
    _plot_worker_capacity(summary, out_dir)
    _plot_window_excess(window_frame, out_dir)
    _write_readme(
        out_dir,
        diagnostic,
        minimum_capacity,
        summary,
        config_path=config_path,
        records_path=records_path,
        checkpoint_path=checkpoint_path,
    )
    generated_paths = [out_dir / name for name in OUTPUT_FILENAMES]
    missing = [path for path in generated_paths if not path.is_file()]
    if missing:
        raise AssertionError(f"expected outputs were not generated: {missing}")
    return {
        "diagnostic": diagnostic,
        "summary": summary,
        "events": events,
        "window_demand_excess": window_frame,
        "minimum_capacity": minimum_capacity,
        "generated_paths": generated_paths,
    }


def _comma_separated_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _comma_separated_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _print_results(result: dict[str, Any]) -> None:
    print("Generated files:")
    for path in result["generated_paths"]:
        print(f"  {path}")
    summary = result["summary"]
    compact_columns = [
        "mode_label",
        "num_workers",
        "offered_load_rho",
        "p99_response_to_decode_us",
        "p99_response_to_commit_us",
        "decode_deadline_miss_ratio",
        "commit_deadline_miss_ratio",
        "maximum_pauli_frame_lag",
        "boundary_commit_success_rate",
        "trace_zero_violation",
    ]
    print("\nCompact metrics:")
    print(
        summary[compact_columns].to_string(index=False, float_format=lambda value: f"{value:.6g}")
    )
    print("\nMinimum zero-violation worker count:")
    for row in result["minimum_capacity"].itertuples(index=False):
        value = (
            "null"
            if pd.isna(row.minimum_zero_violation_worker_count)
            else str(int(row.minimum_zero_violation_worker_count))
        )
        print(f"  {row.mode_label}: {value}")
    checked = summary.loc[summary["finite_trace_recurrence_checked"].astype(bool)]
    waiting = (
        float(checked["max_abs_waiting_time_discrepancy_us"].max()) if not checked.empty else 0.0
    )
    response = (
        float(checked["max_abs_response_time_discrepancy_us"].max()) if not checked.empty else 0.0
    )
    print("\nLindley recurrence versus queue simulator:")
    print(f"  maximum absolute waiting-time discrepancy: {waiting:.3e} us")
    print(f"  maximum absolute response-time discrepancy: {response:.3e} us")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/real_stream_eval_main_ai_selected.yaml")
    parser.add_argument(
        "--records",
        default="results/runs/paper_suite_d7_rtqec_ai_selected/main/records.csv",
    )
    parser.add_argument("--risk-checkpoint", default="checkpoints/risk_lstm_v2_smoke_30.pt")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--workers", default=",".join(str(value) for value in DEFAULT_WORKERS))
    parser.add_argument("--out", default="rebuttal")
    args = parser.parse_args()
    result = run_continuous_stream_analysis(
        config_path=args.config,
        records_path=args.records,
        risk_checkpoint_path=args.risk_checkpoint,
        modes=_comma_separated_strings(args.modes),
        workers=_comma_separated_ints(args.workers),
        out_dir=args.out,
    )
    _print_results(result)


if __name__ == "__main__":
    main()
