"""Unit tests for the named dispatch policies.

These exercise the selectors directly, without the simulator, so a change in queue
order shows up here rather than only as a shifted metric downstream.
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
from rt_preqec.evaluation.dispatch_policies import (
    DISPATCH_POLICIES,
    DISPATCH_POLICY_LABELS,
    SCHEDULER_DISABLED_POLICIES,
    describe_policy,
    get_dispatch_policy,
    select_ready_job,
)


def _config(*, period: float = 10.0, deadline: float = 25.0) -> ProjectConfig:
    config = ProjectConfig()
    config.runtime.round_period_us = period
    config.runtime.decode_deadline_us = deadline
    return config


class _Profiles:
    """Minimal stand-in exposing only what the selectors read."""

    def __init__(self, predicted_accurate_us: list[float]) -> None:
        self.predicted_accurate_us = np.asarray(predicted_accurate_us, dtype=float)


@pytest.mark.parametrize("policy", sorted(DISPATCH_POLICIES))
def test_every_policy_picks_a_member_of_the_ready_set(policy: str) -> None:
    ready = {2, 5, 7}
    chosen = select_ready_job(
        policy,
        ready,
        now_us=50.0,
        config=_config(),
        profiles=_Profiles([1.0] * 8),
    )
    assert chosen in ready


@pytest.mark.parametrize("policy", sorted(DISPATCH_POLICIES))
def test_order_only_policies_agree_on_an_equal_deadline_periodic_stream(policy: str) -> None:
    """FIFO, EDF, LST, and index order coincide when arrival and deadline share an order.

    Arrival ``i*T`` and absolute deadline ``i*T + D`` are both increasing in the index,
    and with equal predicted service the laxity is too, so all four reduce to
    ``min(ready)``. Asserting it here is what lets the sweep report them as baselines
    rather than as distinct schedulers.
    """
    ready = {4, 1, 9, 3}
    chosen = select_ready_job(
        policy,
        ready,
        now_us=90.0,
        config=_config(),
        profiles=_Profiles([2.0] * 10),
    )
    assert chosen == 1


def test_lst_prefers_the_job_whose_predicted_service_leaves_least_laxity() -> None:
    """LST diverges from index order once predicted service dominates the deadline gap."""
    ready = {0, 1}
    # Job 1's deadline is one period later, but its predicted service is far longer,
    # so its laxity is the smaller of the two.
    profiles = _Profiles([1.0, 100.0])
    assert (
        select_ready_job("lst", ready, now_us=10.0, config=_config(), profiles=profiles) == 1
    )
    assert (
        select_ready_job("index_order", ready, now_us=10.0, config=_config(), profiles=profiles)
        == 0
    )


def test_edf_breaks_ties_by_index_so_dispatch_is_deterministic() -> None:
    """A zero relative deadline makes deadlines equal to arrivals; ties must be stable."""
    ready = {3, 1, 2}
    chosen = select_ready_job(
        "edf",
        ready,
        now_us=0.0,
        config=_config(period=0.0, deadline=5.0),
        profiles=_Profiles([1.0] * 4),
    )
    assert chosen == 1


def test_commit_frontier_is_an_alias_of_index_order() -> None:
    assert DISPATCH_POLICIES["commit_frontier"] is DISPATCH_POLICIES["index_order"]
    assert (
        DISPATCH_POLICY_LABELS["commit_frontier"] == DISPATCH_POLICY_LABELS["index_order"]
    )


def test_no_dispatch_policy_disables_the_scheduler() -> None:
    """A policy may reorder the queue but must not touch routing.

    Suppressing the boundary or overload signals would change which backend serves a
    job, so an "ordering" comparison would silently also be a routing comparison.
    Disabling the scheduler is a routing mode (``gate_without_scheduler``) instead.
    """
    assert SCHEDULER_DISABLED_POLICIES == frozenset()


def test_unknown_policy_names_the_supported_set() -> None:
    with pytest.raises(ValueError, match="unknown dispatch policy"):
        get_dispatch_policy("shortest_job_first")
    with pytest.raises(ValueError, match="equation_priority"):
        get_dispatch_policy("shortest_job_first")


def test_equation_priority_is_not_selectable_here() -> None:
    """It needs the predecode effect and dispatch context, which only the simulator has."""
    assert "equation_priority" not in DISPATCH_POLICIES
    assert "equation_priority" in DISPATCH_POLICY_LABELS


def test_empty_ready_set_raises_rather_than_failing_inside_min() -> None:
    with pytest.raises(ValueError, match="ready queue is empty"):
        select_ready_job(
            "index_order", set(), now_us=0.0, config=_config(), profiles=_Profiles([])
        )


def test_describe_policy_carries_both_label_and_name() -> None:
    assert describe_policy("commit_frontier") == {
        "dispatch_policy": "earliest_uncommitted_ready_job",
        "dispatch_policy_name": "commit_frontier",
    }
