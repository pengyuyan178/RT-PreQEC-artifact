"""Aggregate dispatch sweeps across regimes and seeds into paired policy comparisons.

Each sweep run (one regime, one seed) writes its own ``summary.csv`` via
``scripts/run_dispatch_sweep.py``. This collects them and reports, per
(regime, worker count), the *within-trace* delta between two variants.

Pairing within a trace is the point: regime and seed both move the metrics far more
than dispatch order does, so an unpaired mean across seeds would drown the effect.
Sign consistency across seeds is reported alongside the mean, because a mean delta
with mixed signs is not evidence of an ordering.

Example
-------
    python scripts/aggregate_dispatch_sweeps.py \
        --root results/runs/multi_regime \
        --baseline equation_priority --treatment commit_frontier
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

METRICS = [
    "commit_deadline_miss_ratio",
    "p99_response_to_commit_us",
    "lag_violation_ratio",
    "maximum_pauli_frame_lag",
    "boundary_commit_success_rate",
    "logical_error_rate",
]

#: Metrics reported as percentage points rather than raw units.
PERCENTAGE_METRICS = frozenset(
    {
        "commit_deadline_miss_ratio",
        "lag_violation_ratio",
        "boundary_commit_success_rate",
        "logical_error_rate",
    }
)


def load_summaries(root: Path, pattern: str) -> pd.DataFrame:
    paths = sorted(root.glob(pattern))
    if not paths:
        raise SystemExit(f"no summaries matching {root}/{pattern}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True, sort=False)
    print(f"loaded {len(frame)} summary rows from {len(paths)} runs")
    return frame


def paired_deltas(
    frame: pd.DataFrame,
    *,
    baseline: str,
    treatment: str,
    keys: list[str],
) -> pd.DataFrame:
    """Join the two variants on ``keys`` so each delta compares the same trace."""
    for variant in (baseline, treatment):
        if variant not in set(frame["variant"]):
            raise SystemExit(
                f"variant {variant!r} not present; available: "
                f"{', '.join(sorted(set(frame['variant'])))}"
            )
    metrics = [metric for metric in METRICS if metric in frame.columns]
    left = frame[frame["variant"] == baseline].set_index(keys)
    right = frame[frame["variant"] == treatment].set_index(keys)
    if left.index.has_duplicates or right.index.has_duplicates:
        raise SystemExit(
            f"keys {keys} do not identify a unique run; each (regime, seed, W) must "
            "appear once per variant or the pairing is ambiguous"
        )
    joined = left[metrics].join(right[metrics], lsuffix="_base", rsuffix="_treat", how="inner")
    if joined.empty:
        raise SystemExit(f"no traces have both {baseline!r} and {treatment!r}")
    for metric in metrics:
        joined[f"delta_{metric}"] = joined[f"{metric}_base"] - joined[f"{metric}_treat"]
    return joined.reset_index()


def write_report(
    out_path: Path,
    frame: pd.DataFrame,
    deltas: pd.DataFrame,
    *,
    baseline: str,
    treatment: str,
    group_keys: list[str],
) -> None:
    metrics = [metric for metric in METRICS if f"delta_{metric}" in deltas.columns]
    lines = [
        "# Aggregate dispatch sweep results",
        "",
        f"Paired within-trace deltas: **{baseline} minus {treatment}**. A positive delta",
        f"means `{treatment}` scored lower on that metric; for miss ratio, p99, lag, and",
        "LER, lower is better, so positive favours the treatment.",
        "",
    ]
    if "regime" in frame.columns and "eval_seed" in frame.columns:
        per_regime = ", ".join(
            f"{regime}={sorted(group['eval_seed'].unique().tolist())}"
            for regime, group in frame.groupby("regime")
        )
        lines.extend([f"Seeds per regime: {per_regime}", ""])
    lines.extend([f"## Paired deltas by {', '.join(group_keys)}", ""])
    header = "| " + " | ".join([*group_keys, "n", *(f"d({m})" for m in metrics)]) + " |"
    lines.extend([header, "|" + "|".join("---:" for _ in header.split("|")[1:-1]) + "|"])
    for key_values, group in deltas.groupby(group_keys, sort=True):
        keys_tuple = key_values if isinstance(key_values, tuple) else (key_values,)
        cells = [str(value) for value in keys_tuple]
        cells.append(str(len(group)))
        for metric in metrics:
            values = group[f"delta_{metric}"]
            scale, unit = (100.0, "pp") if metric in PERCENTAGE_METRICS else (1.0, "")
            cells.append(
                f"{values.mean() * scale:+.3f}{unit} "
                f"[{values.min() * scale:+.3f}, {values.max() * scale:+.3f}]"
            )
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Sign consistency (commit deadline miss ratio)",
            "",
            f"How often `{treatment}` beat, tied, or lost to `{baseline}` per group. Mixed",
            "signs mean the mean delta above should not be read as an ordering.",
            "",
        ]
    )
    for key_values, group in deltas.groupby(group_keys, sort=True):
        keys_tuple = key_values if isinstance(key_values, tuple) else (key_values,)
        label = ", ".join(
            f"{key}={value}" for key, value in zip(group_keys, keys_tuple, strict=True)
        )
        values = group["delta_commit_deadline_miss_ratio"]
        lines.append(
            f"- {label}: treatment better {int((values > 0).sum())}, "
            f"equal {int((values == 0).sum())}, worse {int((values < 0).sum())}"
        )
    lines.append("")
    if "integrity_passed" in frame.columns:
        failed = frame.loc[~frame["integrity_passed"].astype(bool)]
        lines.extend(
            [
                "## Integrity",
                "",
                (
                    f"All {len(frame)} runs passed."
                    if failed.empty
                    else f"**{len(failed)} of {len(frame)} runs FAILED integrity checks.**"
                ),
                "",
            ]
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="directory containing the sweep runs")
    parser.add_argument("--pattern", default="*/summary.csv", help="glob relative to --root")
    parser.add_argument("--baseline", default="equation_priority")
    parser.add_argument("--treatment", default="commit_frontier")
    parser.add_argument(
        "--pair-keys",
        default="regime,eval_seed,num_workers",
        help="columns identifying one trace; deltas are computed within these",
    )
    parser.add_argument("--group-keys", default="regime,num_workers")
    args = parser.parse_args()

    root = Path(args.root)
    frame = load_summaries(root, args.pattern)
    frame.to_csv(root / "all_summaries.csv", index=False)

    pair_keys = [key for key in args.pair_keys.split(",") if key.strip() in frame.columns]
    group_keys = [key for key in args.group_keys.split(",") if key.strip() in frame.columns]
    if not pair_keys or not group_keys:
        raise SystemExit(
            f"none of the requested keys are present; columns: {', '.join(frame.columns)}"
        )

    metrics = [metric for metric in METRICS if metric in frame.columns]
    aggregate = (
        frame.groupby([*group_keys, "variant"])[metrics].agg(["mean", "min", "max"]).reset_index()
    )
    aggregate.columns = ["_".join(col).rstrip("_") for col in aggregate.columns]
    aggregate.to_csv(root / "aggregate_by_variant.csv", index=False)

    deltas = paired_deltas(
        frame, baseline=args.baseline, treatment=args.treatment, keys=pair_keys
    )
    deltas.to_csv(root / "paired_deltas.csv", index=False)
    write_report(
        root / "AGGREGATE_RESULTS.md",
        frame,
        deltas,
        baseline=args.baseline,
        treatment=args.treatment,
        group_keys=group_keys,
    )
    print(f"wrote {root / 'AGGREGATE_RESULTS.md'} ({len(deltas)} paired comparisons)")


if __name__ == "__main__":
    main()
