"""Verify a dispatch sweep reproduces a pinned reference summary.

An artifact reviewer needs one command that answers "do I get the published numbers on
this machine?". This re-runs the sweep from a trace and diffs every metric against a
reference ``summary.csv``, failing on any deviation beyond tolerance.

Only variants present in *both* frames are compared, and the count is reported, so a
reference that covers fewer variants cannot make the check pass by comparing nothing.

Example
-------
    python scripts/verify_reproduction.py \
        --reference results/tables/dispatch_sweep_d7_reference.csv \
        --config configs/dispatch_sweep_d7.yaml \
        --records results/runs/dispatch_sweep_traces/d7/records.csv \
        --risk-checkpoint checkpoints/risk_lstm_v2_smoke_30.pt \
        --locked-policy configs/policies/runtime_guard_q95_d7.json \
        --workers 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _find_repo_root(path: Path) -> Path:
    for candidate in (path.resolve(), *path.resolve().parents):
        if (candidate / "src" / "rt_preqec").is_dir():
            return candidate
    raise RuntimeError(f"could not find repository root above {path}")


ROOT = _find_repo_root(Path(__file__))
for _import_path in (ROOT, ROOT / "src"):
    if str(_import_path) not in sys.path:
        sys.path.insert(0, str(_import_path))

import pandas as pd  # noqa: E402

from rt_preqec.config import load_config  # noqa: E402
from rt_preqec.evaluation.continuous_stream import (  # noqa: E402
    build_prediction_profiles,
    reindex_continuous_records,
)
from rt_preqec.evaluation.runtime_guard import RuntimeMargins  # noqa: E402
from scripts.run_dispatch_sweep import (  # noqa: E402
    VARIANTS,
    parse_ints,
    parse_strings,
    run_sweep,
    sha256,
)
from scripts.run_paper_experiment_suite import _load_records_csv  # noqa: E402

#: Metrics that must reproduce. Deliberately includes the safety metrics (lag, boundary,
#: LER) alongside the timing ones, so a run cannot pass by matching latency alone.
VERIFIED_METRICS = [
    "commit_deadline_miss_ratio",
    "p99_response_to_commit_us",
    "mean_response_to_commit_us",
    "maximum_pauli_frame_lag",
    "lag_violation_ratio",
    "boundary_commit_success_rate",
    "logical_error_rate",
    "fast_selection_rate",
    "offered_load_rho",
]


def compare(
    reference: pd.DataFrame,
    actual: pd.DataFrame,
    *,
    tolerance: float,
) -> tuple[pd.DataFrame, float]:
    """Diff every (variant, worker count) pair the two frames share."""
    keys = ["variant", "num_workers"]
    metrics = [m for m in VERIFIED_METRICS if m in reference.columns and m in actual.columns]
    left = reference.set_index(keys)
    right = actual.set_index(keys)
    shared = sorted(set(left.index) & set(right.index))
    if not shared:
        raise SystemExit(
            "reference and rerun share no (variant, num_workers) pair; nothing to verify"
        )
    rows: list[dict[str, Any]] = []
    for key in shared:
        for metric in metrics:
            expected = float(left.loc[key, metric])
            got = float(right.loc[key, metric])
            absolute = abs(expected - got)
            relative = absolute / max(abs(expected), 1e-12)
            rows.append(
                {
                    "variant": key[0],
                    "num_workers": key[1],
                    "metric": metric,
                    "reference": expected,
                    "reproduced": got,
                    "abs_diff": absolute,
                    "rel_diff": relative,
                    "within_tolerance": absolute <= tolerance or relative <= tolerance,
                }
            )
    frame = pd.DataFrame(rows)
    worst = float(frame[["abs_diff", "rel_diff"]].min(axis=1).max())
    return frame, worst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="pinned summary.csv to diff against")
    parser.add_argument("--config", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--risk-checkpoint", required=True)
    parser.add_argument("--workers", default="2")
    parser.add_argument("--variants", default=None, help="default: every variant in --reference")
    parser.add_argument("--locked-policy", default=None)
    parser.add_argument("--fixed-fast-estimate-us", type=float, default=4.8)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--out", default=None, help="optional directory for the diff report")
    args = parser.parse_args()

    reference = pd.read_csv(args.reference)
    config = load_config(Path(args.config))
    checkpoint = Path(args.risk_checkpoint)
    records = reindex_continuous_records(_load_records_csv(Path(args.records)))
    profiles = build_prediction_profiles(
        records,
        checkpoint,
        device=config.device,
        fixed_fast_estimate_us=float(args.fixed_fast_estimate_us),
    )

    margins_by_variant: dict[str, RuntimeMargins] = {}
    if args.locked_policy:
        locked = json.loads(Path(args.locked_policy).read_text(encoding="utf-8"))
        expected_hash = locked.get("checkpoint_sha256")
        if expected_hash and sha256(checkpoint) != expected_hash:
            raise SystemExit(
                "risk checkpoint does not match the locked policy's checkpoint_sha256"
            )
        margins = RuntimeMargins(**dict(locked["runtime_guard"]))
        name = f"commit_frontier_guard_q{int(round(margins.quantile * 100))}"
        VARIANTS[name] = ("gate", "commit_frontier")
        margins_by_variant[name] = margins

    if args.variants:
        variants = parse_strings(args.variants)
    else:
        # Default to whatever the reference covers, so the check is as wide as the claim.
        variants = [v for v in reference["variant"].unique().tolist() if v in VARIANTS]
        skipped = sorted(set(reference["variant"]) - set(variants))
        if skipped:
            print(f"NOTE: reference variants not reproducible here, skipped: {skipped}")

    print(f"{len(records)} jobs; verifying {len(variants)} variants")
    summary, _events, _integrity = run_sweep(
        records,
        profiles,
        config,
        variants=variants,
        workers=parse_ints(args.workers),
        margins_by_variant=margins_by_variant,
        split_name="verification",
    )

    frame, worst = compare(reference, summary, tolerance=float(args.tolerance))
    failures = frame.loc[~frame["within_tolerance"]]
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out_dir / "reproduction_diff.csv", index=False)
        summary.to_csv(out_dir / "summary.csv", index=False)

    pairs = frame[["variant", "num_workers"]].drop_duplicates()
    print(
        f"\ncompared {len(frame)} values over {len(pairs)} (variant, worker) pairs "
        f"x {frame['metric'].nunique()} metrics"
    )
    print(f"worst deviation: {worst:.3e} (tolerance {args.tolerance:.1e})")
    if not summary["integrity_passed"].all():
        raise SystemExit("integrity checks FAILED during the rerun; results are not sound")
    if failures.empty:
        print("*** REPRODUCED: every metric matches the reference ***")
        return
    print("*** MISMATCH ***")
    print(failures.to_string(index=False))
    raise SystemExit(1)


if __name__ == "__main__":
    main()
