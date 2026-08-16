"""First-class ready-queue dispatch policies for the continuous-arrival runtime.

Every policy answers one question: given the set of ready job indices, which one
runs next? Backend selection (the eligibility gate and fast/accurate routing) is a
separate decision, made afterwards in :mod:`rt_preqec.evaluation.continuous_stream`.
Keeping the two apart is what lets an experiment vary queue order while holding
routing fixed, which is how dispatch order is attributed independently of the gate.

The policies were originally implemented as ``contextmanager`` monkeypatches over a
frozen simulator during the RTSS 2026 review response. They are ordinary named
strategies here, selected through :func:`get_dispatch_policy`.

Policy semantics
----------------
``index_order``
    Lowest ready index first. For a periodic stream with one constant relative
    deadline this is simultaneously FIFO, EDF, and commit-frontier order, because
    both arrival ``i*T`` and absolute deadline ``i*T + D`` increase with the index.
    It is also the order Proposition 1' assumes.
``commit_frontier``
    Alias of ``index_order``, named for the property that matters under overload:
    the lowest ready index is the job nearest the committed prefix, so serving it
    first is what shortens ordered-commit recovery.
``equation_priority``
    The paper's Equation 1 weighted score over urgency, learned risk, predicted
    runtime, and boundary context. Highest score wins.
``fifo`` / ``edf`` / ``lst``
    Textbook real-time baselines, kept distinct from ``index_order`` so an
    experiment can assert the equivalence rather than assume it. ``lst`` breaks the
    tie by least laxity using the predicted accurate latency as a WCET-style
    service proxy, computed before routing so the order is well defined.
"""

from __future__ import annotations

from collections.abc import Callable, Set
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rt_preqec.config import ProjectConfig
    from rt_preqec.evaluation.continuous_stream import PredictionProfiles

__all__ = [
    "DISPATCH_POLICIES",
    "DISPATCH_POLICY_LABELS",
    "SCHEDULER_DISABLED_POLICIES",
    "get_dispatch_policy",
    "select_ready_job",
]

# Human-readable provenance strings. These land in the `dispatch_policy` column of
# every summary, so a reader can tell which queue order produced a row.
DISPATCH_POLICY_LABELS: dict[str, str] = {
    "index_order": "earliest_uncommitted_ready_job",
    "commit_frontier": "earliest_uncommitted_ready_job",
    "equation_priority": "equation_1_dynamic_priority",
    "fifo": "fifo",
    "edf": "earliest_absolute_deadline",
    "lst": "least_laxity_first",
}

# A dispatch policy chooses ORDER ONLY. None of them may suppress the boundary or
# overload signals, because those feed routing, and changing routing here would
# confound an order comparison with a routing change -- exactly the confound the
# fixed-routing study exists to rule out. Disabling the scheduler is a property of
# the *routing mode* (`gate_without_scheduler`), not of a dispatch policy.
SCHEDULER_DISABLED_POLICIES: frozenset[str] = frozenset()


def _select_index_order(
    ready: Set[int],
    *,
    now_us: float,
    config: ProjectConfig,
    profiles: PredictionProfiles,
) -> int:
    del now_us, config, profiles
    return min(ready)


def _select_fifo(
    ready: Set[int],
    *,
    now_us: float,
    config: ProjectConfig,
    profiles: PredictionProfiles,
) -> int:
    del now_us, profiles
    period = float(config.runtime.round_period_us)
    return min(ready, key=lambda job_id: (float(job_id) * period, int(job_id)))


def _select_edf(
    ready: Set[int],
    *,
    now_us: float,
    config: ProjectConfig,
    profiles: PredictionProfiles,
) -> int:
    del now_us, profiles
    period = float(config.runtime.round_period_us)
    relative_deadline = float(config.runtime.decode_deadline_us)
    return min(
        ready,
        key=lambda job_id: (float(job_id) * period + relative_deadline, int(job_id)),
    )


def _select_lst(
    ready: Set[int],
    *,
    now_us: float,
    config: ProjectConfig,
    profiles: PredictionProfiles,
) -> int:
    period = float(config.runtime.round_period_us)
    relative_deadline = float(config.runtime.decode_deadline_us)
    return min(
        ready,
        key=lambda job_id: (
            float(job_id) * period
            + relative_deadline
            - float(now_us)
            - float(profiles.predicted_accurate_us[job_id]),
            int(job_id),
        ),
    )


# `equation_priority` is absent here on purpose: scoring a job needs the predecode
# effect and the dispatch context, which only the simulator holds. It is applied in
# `continuous_stream.choose_ready_job`, which checks for it before consulting this
# table.
DISPATCH_POLICIES: dict[str, Callable[..., int]] = {
    "index_order": _select_index_order,
    "commit_frontier": _select_index_order,
    "fifo": _select_fifo,
    "edf": _select_edf,
    "lst": _select_lst,
}

#: Policies whose ready-job choice needs no scoring, i.e. everything except
#: ``equation_priority``.
ORDER_ONLY_POLICIES = frozenset(DISPATCH_POLICIES)


def get_dispatch_policy(name: str) -> Callable[..., int]:
    """Return the selector for ``name``, or raise with the supported set listed."""
    try:
        return DISPATCH_POLICIES[name]
    except KeyError:
        supported = ", ".join(sorted(set(DISPATCH_POLICIES) | {"equation_priority"}))
        raise ValueError(f"unknown dispatch policy {name!r}; supported: {supported}") from None


def select_ready_job(
    policy: str,
    ready: Set[int],
    *,
    now_us: float,
    config: ProjectConfig,
    profiles: PredictionProfiles,
) -> int:
    """Pick the next ready job under ``policy``.

    Raises ``ValueError`` on an empty ready set, so a stalled simulator surfaces
    here rather than as a confusing ``min()`` error deeper in.
    """
    if not ready:
        raise ValueError("ready queue is empty")
    selector = get_dispatch_policy(policy)
    return int(selector(ready, now_us=now_us, config=config, profiles=profiles))


def describe_policy(policy: str) -> dict[str, Any]:
    """Provenance record for a policy, for embedding in summaries and metadata."""
    return {
        "dispatch_policy": DISPATCH_POLICY_LABELS.get(policy, policy),
        "dispatch_policy_name": policy,
    }
