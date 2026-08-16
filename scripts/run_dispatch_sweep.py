"""Continuous-arrival dispatch sweep over routing modes, dispatch policies, and workers.

Consolidates the three RTSS 2026 review-response experiments into one entry point on
top of :mod:`rt_preqec.evaluation.continuous_stream`:

* **commit-aware capacity sweep** — vary the dispatch policy at fixed routing to
  isolate the effect of queue order (``--variants equation_priority,commit_frontier``).
* **fixed-routing RT baselines** — FIFO / EDF / LST dispatch with the eligibility gate
  held fixed (``--variants gate_fifo,gate_edf,gate_lst``).
* **multi-regime scaling** — the same sweep across worker counts and traces.

A "variant" pairs a routing mode with a dispatch policy, since those are the two
independent decisions. ``--locked-policy`` replays a pre-registered runtime guard,
verifying the checkpoint hash so a locked policy cannot silently drift.

Example
-------
    python scripts/run_dispatch_sweep.py \
        --config configs/real_stream_eval_main_ai_selected.yaml \
        --records results/runs/paper_suite_d7_rtqec_ai_selected/main/records.csv \
        --risk-checkpoint checkpoints/risk_lstm_v2_smoke_30.pt \
        --workers 1,2,3,4,5,6 \
        --out results/runs/dispatch_sweep_d7
"""

from __future__ import annotations

import argparse
import hashlib
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

from rt_preqec.config import ProjectConfig, load_config  # noqa: E402
from rt_preqec.evaluation.continuous_stream import (  # noqa: E402
    PredictionProfiles,
    build_prediction_profiles,
    reindex_continuous_records,
    simulate_trace,
)
from rt_preqec.evaluation.runtime_guard import (  # noqa: E402
    RuntimeMargins,
    apply_runtime_guard,
    calibrate_runtime_margins,
    guard_metadata,
)
from scripts.run_paper_experiment_suite import _load_records_csv  # noqa: E402

#: A variant is (routing mode, dispatch policy). Routing and order are orthogonal, so
#: naming the pair is what makes a comparison interpretable.
VARIANTS: dict[str, tuple[str, str]] = {
    # references
    "accurate_only": ("accurate_only", "index_order"),
    "fast_only": ("fast_only", "index_order"),
    "edf": ("edf_feasibility", "edf"),
    "no_scheduler": ("gate_without_scheduler", "index_order"),
    "no_validation": ("gate_without_validation", "equation_priority"),
    # the ordering comparison at fixed (gate) routing
    "equation_priority": ("gate", "equation_priority"),
    "commit_frontier": ("gate", "commit_frontier"),
    # textbook RT dispatch baselines, gate routing held fixed
    "gate_fifo": ("gate", "fifo"),
    "gate_edf": ("gate", "edf"),
    "gate_lst": ("gate", "lst"),
}

DEFAULT_VARIANTS = [
    "accurate_only",
    "fast_only",
    "edf",
    "no_scheduler",
    "equation_priority",
    "commit_frontier",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in str(value).split(",") if item.strip()]


def parse_strings(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def run_sweep(
    records: list[Any],
    profiles: PredictionProfiles,
    config: ProjectConfig,
    *,
    variants: list[str],
    workers: list[int],
    margins_by_variant: dict[str, RuntimeMargins] | None = None,
    split_name: str = "sweep",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simulate every (variant, worker count) pair and collect summaries and events."""
    margins_by_variant = margins_by_variant or {}
    summaries: list[dict[str, Any]] = []
    integrity_rows: list[dict[str, Any]] = []
    event_frames: list[pd.DataFrame] = []
    for variant in variants:
        if variant not in VARIANTS:
            raise ValueError(
                f"unknown variant {variant!r}; supported: {', '.join(sorted(VARIANTS))}"
            )
        mode, policy = VARIANTS[variant]
        used = profiles
        if variant in margins_by_variant:
            used = apply_runtime_guard(profiles, margins_by_variant[variant])
        for num_workers in workers:
            result = simulate_trace(
                records,
                used,
                config,
                mode=mode,
                num_workers=num_workers,
                dispatch_policy=policy,
            )
            row = {
                **result.summary,
                **guard_metadata(used),
                "variant": variant,
                "split_name": split_name,
            }
            summaries.append(row)
            integrity_rows.append(
                {
                    "variant": variant,
                    "mode": mode,
                    "dispatch_policy_name": policy,
                    "num_workers": num_workers,
                    **result.integrity,
                }
            )
            events = result.events.copy()
            events["variant"] = variant
            event_frames.append(events)
            print(
                f"  {variant:26} W={num_workers} "
                f"miss={row['commit_deadline_miss_ratio']:.5f} "
                f"p99={row['p99_response_to_commit_us']:.2f}us "
                f"maxlag={row['maximum_pauli_frame_lag']:<5} "
                f"LER={row['logical_error_rate']:.5f} "
                f"integrity={row['integrity_passed']}"
            )
    return (
        pd.DataFrame(summaries),
        pd.concat(event_frames, ignore_index=True),
        pd.DataFrame(integrity_rows),
    )


REPORT_COLUMNS = [
    "variant",
    "num_workers",
    "commit_deadline_miss_ratio",
    "p99_response_to_commit_us",
    "maximum_pauli_frame_lag",
    "lag_violation_ratio",
    "boundary_commit_success_rate",
    "logical_error_rate",
    "fast_selection_rate",
    "offered_load_rho",
    "integrity_passed",
]


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [column for column in REPORT_COLUMNS if column in frame.columns]
    view = frame[columns].sort_values(["variant", "num_workers"])
    header = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join("---:" for _ in columns) + "|"
    rows = []
    for row in view.itertuples(index=False):
        cells = [f"{value:.6g}" if isinstance(value, float) else str(value) for value in row]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, rule, *rows])


def write_report(
    out_path: Path,
    *,
    summary: pd.DataFrame,
    integrity: pd.DataFrame,
    calibration: list[RuntimeMargins],
    metadata: dict[str, Any],
) -> None:
    """Write the human-readable companion to summary.csv."""
    passed = bool(integrity["integrity_passed"].all())
    lines = [
        "# Dispatch sweep results",
        "",
        f"Integrity: **{'PASS' if passed else 'FAIL'}** over {len(integrity)} runs.",
        "",
        f"Regime `{metadata['regime']}`, {metadata['num_jobs']} jobs, "
        f"records `{metadata['records']}`.",
        "",
        "Exact finite-trace evidence under the simulator in",
        "`rt_preqec.evaluation.continuous_stream`, not a formal schedulability result.",
        "Reported maxima are observed trace maxima, not WCET.",
        "",
        "## Variants",
        "",
        "| Variant | Routing mode | Dispatch policy |",
        "|---|---|---|",
    ]
    for variant in metadata["variants"]:
        mode, policy = VARIANTS[variant]
        lines.append(f"| {variant} | {mode} | {policy} |")
    if calibration:
        lines.extend(
            [
                "",
                "## Runtime-margin calibration",
                "",
                "| q | Accurate margin (us) | Fast margin (us) | Accurate cov. | Fast cov. | N |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in calibration:
            lines.append(
                f"| {row.quantile:.2f} | {row.accurate_margin_us:.6g} | "
                f"{row.fast_margin_us:.6g} | {row.accurate_coverage:.3%} | "
                f"{row.fast_coverage:.3%} | {row.num_calibration_jobs} |"
            )
    lines.extend(["", "## Capacity table", "", _markdown_table(summary), ""])

    ordering = [v for v in ("equation_priority", "commit_frontier") if v in set(summary["variant"])]
    if len(ordering) == 2:
        lines.extend(
            [
                "## Dispatch order at fixed gate routing",
                "",
                "Both rows below use `mode=gate`, so any difference is attributable to",
                "queue order alone.",
                "",
            ]
        )
        for worker_count in sorted(summary["num_workers"].unique()):
            worker = summary.loc[summary["num_workers"] == worker_count].set_index("variant")
            parts = " / ".join(
                f"{worker.loc[v, 'commit_deadline_miss_ratio']:.3%}" for v in ordering
            )
            lags = " / ".join(f"{int(worker.loc[v, 'maximum_pauli_frame_lag'])}" for v in ordering)
            lers = " / ".join(f"{worker.loc[v, 'logical_error_rate']:.3%}" for v in ordering)
            lines.append(
                f"- W={int(worker_count)} ({' / '.join(ordering)}): "
                f"miss {parts}; max lag {lags}; LER {lers}."
            )
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--risk-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help=f"comma-separated; available: {', '.join(sorted(VARIANTS))}",
    )
    parser.add_argument("--workers", default="1,2,3,4,5,6")
    parser.add_argument("--fixed-fast-estimate-us", type=float, default=4.8)
    parser.add_argument(
        "--locked-policy",
        default=None,
        help="pre-registered runtime guard JSON; adds a guarded commit_frontier variant",
    )
    parser.add_argument(
        "--calibrate-guard-quantiles",
        default=None,
        help="fit runtime-guard margins on THIS trace and report them (exploratory use only)",
    )
    parser.add_argument("--regime", default="unspecified")
    parser.add_argument("--eval-seed", type=int, default=-1)
    parser.add_argument("--save-events", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(Path(args.config))
    checkpoint = Path(args.risk_checkpoint)
    records = reindex_continuous_records(_load_records_csv(Path(args.records)))
    print(f"{len(records)} continuous jobs from {args.records}")
    profiles = build_prediction_profiles(
        records,
        checkpoint,
        device=config.device,
        fixed_fast_estimate_us=float(args.fixed_fast_estimate_us),
    )

    variants = parse_strings(args.variants)
    margins_by_variant: dict[str, RuntimeMargins] = {}
    calibration: list[RuntimeMargins] = []

    if args.calibrate_guard_quantiles:
        quantiles = [float(q) for q in parse_strings(args.calibrate_guard_quantiles)]
        calibration = calibrate_runtime_margins(records, profiles, quantiles)
        pd.DataFrame([m.__dict__ for m in calibration]).to_csv(
            out_dir / "runtime_margins.csv", index=False
        )
        print(f"calibrated {len(calibration)} runtime-guard margin sets on this trace")

    if args.locked_policy:
        locked = json.loads(Path(args.locked_policy).read_text(encoding="utf-8"))
        expected = locked.get("checkpoint_sha256")
        if expected and sha256(checkpoint) != expected:
            raise ValueError(
                "risk checkpoint does not match the locked policy's checkpoint_sha256; "
                "the pre-registered guard was calibrated against a different model"
            )
        margins = RuntimeMargins(**dict(locked["runtime_guard"]))
        name = f"commit_frontier_guard_q{int(round(margins.quantile * 100))}"
        VARIANTS[name] = ("gate", "commit_frontier")
        variants.append(name)
        margins_by_variant[name] = margins
        print(f"locked policy replayed as variant {name!r} (quantile={margins.quantile})")

    split_name = (
        f"{args.regime}_seed_{args.eval_seed}" if args.eval_seed >= 0 else str(args.regime)
    )
    summary, events, integrity = run_sweep(
        records,
        profiles,
        config,
        variants=variants,
        workers=parse_ints(args.workers),
        margins_by_variant=margins_by_variant,
        split_name=split_name,
    )
    summary["regime"] = args.regime
    if args.eval_seed >= 0:
        summary["eval_seed"] = int(args.eval_seed)
    summary.to_csv(out_dir / "summary.csv", index=False)
    integrity.to_csv(out_dir / "integrity.csv", index=False)
    if args.save_events:
        events.to_csv(out_dir / "events.csv.gz", index=False, compression="gzip")

    metadata = {
        "regime": args.regime,
        "eval_seed": int(args.eval_seed),
        "config": str(args.config),
        "config_sha256": sha256(Path(args.config)),
        "records": str(args.records),
        "records_sha256": sha256(Path(args.records)),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "variants": variants,
        "workers": parse_ints(args.workers),
        "locked_policy": args.locked_policy,
        "all_integrity_checks_passed": bool(summary["integrity_passed"].all()),
        "num_jobs": int(len(records)),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(
        out_dir / "RESULTS.md",
        summary=summary,
        integrity=integrity,
        calibration=calibration,
        metadata=metadata,
    )

    if not summary["integrity_passed"].all():
        failed = summary.loc[~summary["integrity_passed"], ["variant", "num_workers"]]
        raise SystemExit(f"integrity checks FAILED for:\n{failed.to_string(index=False)}")
    print(f"\nwrote {out_dir}/summary.csv  ({len(summary)} rows, all integrity checks passed)")


if __name__ == "__main__":
    main()
