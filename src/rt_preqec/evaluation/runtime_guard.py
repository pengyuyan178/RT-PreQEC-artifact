"""Runtime margin guard for the continuous-arrival runtime.

The scheduler dispatches on *predicted* service time, so a systematic
under-prediction can make an infeasible job look feasible. This module calibrates a
one-sided additive margin per backend and inflates the predictions by it, turning a
point estimate into a conservative upper estimate.

Calibration is deliberately split from use: margins are fitted on the exploratory
split only, then applied unchanged to the confirmation split. Fitting on the data a
result is later claimed from would leak the outcome into the policy.

The margin only makes the runtime *more* conservative — it can push a job onto the
accurate path but never onto the fast path — so it trades latency for safety and
never the other way round.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from rt_preqec.evaluation.continuous_stream import PredictionProfiles
from rt_preqec.evaluation.real_stream import RealStreamShotRecord

__all__ = [
    "RuntimeMargins",
    "apply_runtime_guard",
    "calibrate_runtime_margins",
    "guard_metadata",
]


@dataclass(frozen=True)
class RuntimeMargins:
    """A calibrated pair of additive service-time margins, with their coverage."""

    quantile: float
    accurate_margin_us: float
    fast_margin_us: float
    accurate_coverage: float
    fast_coverage: float
    num_calibration_jobs: int
    quantile_method: str = "higher"


def _higher_quantile(values: np.ndarray, quantile: float) -> float:
    """Quantile with the ``higher`` interpolation, i.e. never below the empirical point.

    Interpolating would place the margin between two observed residuals and quietly
    lose coverage; ``higher`` keeps the guarantee one-sided.
    """
    return float(np.quantile(values, float(quantile), method="higher"))


def calibrate_runtime_margins(
    records: Sequence[RealStreamShotRecord],
    profiles: PredictionProfiles,
    quantiles: Sequence[float],
) -> list[RuntimeMargins]:
    """Fit one-sided additive margins from prediction residuals.

    Must be called on a calibration split, never on the split a result is reported
    from. Margins are clamped at zero: a model that over-predicts needs no guard, and
    a negative margin would make the runtime optimistic.
    """
    record_list = list(records)
    profiles.validate(len(record_list))
    actual_accurate = np.asarray(
        [record.accurate_latency_us for record in record_list], dtype=float
    )
    actual_fast = np.asarray([record.fast_latency_us for record in record_list], dtype=float)
    predicted_accurate = np.asarray(profiles.predicted_accurate_us, dtype=float)
    predicted_fast = np.asarray(profiles.predicted_fast_us, dtype=float)
    accurate_residual = actual_accurate - predicted_accurate
    fast_residual = actual_fast - predicted_fast
    rows: list[RuntimeMargins] = []
    for quantile in quantiles:
        accurate_margin = max(0.0, _higher_quantile(accurate_residual, quantile))
        fast_margin = max(0.0, _higher_quantile(fast_residual, quantile))
        rows.append(
            RuntimeMargins(
                quantile=float(quantile),
                accurate_margin_us=float(accurate_margin),
                fast_margin_us=float(fast_margin),
                accurate_coverage=float(
                    np.mean(predicted_accurate + accurate_margin >= actual_accurate)
                ),
                fast_coverage=float(np.mean(predicted_fast + fast_margin >= actual_fast)),
                num_calibration_jobs=len(record_list),
            )
        )
    return rows


def apply_runtime_guard(
    profiles: PredictionProfiles,
    margins: RuntimeMargins,
) -> PredictionProfiles:
    """Return a copy of ``profiles`` with predictions inflated by ``margins``.

    Risk and confidence are untouched: the guard corrects timing estimates only, so
    the eligibility decision is unchanged.
    """
    metadata = dict(profiles.metadata)
    metadata.update(
        {
            "runtime_guard_enabled": True,
            "runtime_guard_quantile": float(margins.quantile),
            "accurate_margin_us": float(margins.accurate_margin_us),
            "fast_margin_us": float(margins.fast_margin_us),
            "runtime_guard_quantile_method": margins.quantile_method,
        }
    )
    guarded = PredictionProfiles(
        risk_score=np.asarray(profiles.risk_score, dtype=float).copy(),
        confidence=np.asarray(profiles.confidence, dtype=float).copy(),
        predicted_accurate_us=(
            np.asarray(profiles.predicted_accurate_us, dtype=float)
            + float(margins.accurate_margin_us)
        ),
        predicted_fast_us=(
            np.asarray(profiles.predicted_fast_us, dtype=float) + float(margins.fast_margin_us)
        ),
        runtime_score=np.asarray(profiles.runtime_score, dtype=float).copy(),
        metadata=metadata,
    )
    guarded.validate(len(profiles.risk_score))
    return guarded


def guard_metadata(profiles: PredictionProfiles) -> dict[str, Any]:
    """Guard provenance columns for a summary row, whether or not a guard is active."""
    enabled = bool(profiles.metadata.get("runtime_guard_enabled", False))
    return {
        "runtime_guard_enabled": enabled,
        "runtime_guard_quantile": (
            float(profiles.metadata["runtime_guard_quantile"]) if enabled else np.nan
        ),
        "runtime_guard_accurate_margin_us": (
            float(profiles.metadata["accurate_margin_us"]) if enabled else 0.0
        ),
        "runtime_guard_fast_margin_us": (
            float(profiles.metadata["fast_margin_us"]) if enabled else 0.0
        ),
    }
