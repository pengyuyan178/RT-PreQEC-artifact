"""Robustness of the runtime to front-end signal inaccuracy.

Answers: how much does routing quality depend on the front end being accurate, and
does the eligibility gate keep logical error bounded when the signals are wrong?

The trace is held fixed — true service demands and true logical outcomes never move —
and graded noise is injected *only* into the estimates the scheduler consumes. So any
degradation is attributable to the front end, not to a different workload.

Noise model, one monotone knob ``eps``
--------------------------------------
* ``predicted_accurate_us``, ``predicted_fast_us`` are multiplied by ``exp(N(0, eps))``,
  which is multiplicative and stays positive, so a latency estimate can never go
  negative or flip sign.
* ``risk_score``, ``confidence`` get additive ``N(0, eps)`` clipped to ``[0, 1]``, the
  range the gate defines them on.

``eps=0`` reproduces the unperturbed run exactly and serves as the sanity anchor;
``eps=1.6`` is close to random signals. Each ``eps>0`` is repeated over several noise
seeds, because a single draw says nothing about the spread.

Example
-------
    python scripts/run_frontend_robustness.py \
        --config configs/dispatch_sweep_d7.yaml \
        --records results/runs/dispatch_sweep_traces/d7/records.csv \
        --risk-checkpoint checkpoints/risk_lstm_v2_smoke_30.pt \
        --out results/runs/frontend_robustness_d7
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _find_repo_root(path: Path) -> Path:
    for candidate in (path.resolve(), *path.resolve().parents):
        if (candidate / "src" / "rt_preqec").is_dir():
            return candidate
    raise RuntimeError(f"could not find repository root above {path}")


ROOT = _find_repo_root(Path(__file__))
for _import_path in (ROOT, ROOT / "src"):
    if str(_import_path) not in sys.path:
        sys.path.insert(0, str(_import_path))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from rt_preqec.config import load_config  # noqa: E402
from rt_preqec.evaluation.continuous_stream import (  # noqa: E402
    PredictionProfiles,
    build_prediction_profiles,
    reindex_continuous_records,
    simulate_trace,
)
from scripts.run_dispatch_sweep import VARIANTS, parse_ints, parse_strings, sha256  # noqa: E402
from scripts.run_paper_experiment_suite import _load_records_csv  # noqa: E402

#: Offset keeping the noise seeds clear of the trace-generation seeds, so a robustness
#: replicate can never accidentally reuse a data seed.
SEED_BASE = 10_000


def perturb(
    profiles: PredictionProfiles, eps: float, rng: np.random.Generator
) -> PredictionProfiles:
    """Inject front-end noise at level ``eps``, leaving the true trace untouched."""
    num_jobs = len(profiles.risk_score)

    def multiplicative(values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * np.exp(rng.normal(0.0, eps, num_jobs))

    def additive_clipped(values: np.ndarray) -> np.ndarray:
        noisy = np.asarray(values, dtype=float) + rng.normal(0.0, eps, num_jobs)
        return np.clip(noisy, 0.0, 1.0)

    perturbed = PredictionProfiles(
        risk_score=additive_clipped(profiles.risk_score),
        confidence=additive_clipped(profiles.confidence),
        predicted_accurate_us=multiplicative(profiles.predicted_accurate_us),
        predicted_fast_us=multiplicative(profiles.predicted_fast_us),
        runtime_score=np.asarray(profiles.runtime_score, dtype=float).copy(),
        metadata={**profiles.metadata, "frontend_noise_eps": float(eps)},
    )
    perturbed.validate(num_jobs)
    return perturbed


def _plot(aggregate: pd.DataFrame, workers: list[int], out_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for num_workers in workers:
        rows = aggregate[aggregate["num_workers"] == num_workers].sort_values("eps")
        label = f"W={num_workers}"
        axes[0].plot(rows["eps"], rows["logical_error_rate_mean"], marker="o", label=label)
        axes[0].fill_between(
            rows["eps"],
            rows["logical_error_rate_min"],
            rows["logical_error_rate_max"],
            alpha=0.15,
        )
        axes[1].plot(
            rows["eps"], rows["commit_deadline_miss_ratio_mean"], marker="o", label=label
        )
    axes[0].set(
        xlabel="front-end noise $\\epsilon$",
        ylabel="logical error rate",
        title="LER vs front-end noise",
    )
    axes[1].set(
        xlabel="front-end noise $\\epsilon$",
        ylabel="commit-deadline miss ratio",
        title="Miss ratio vs front-end noise",
    )
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(out_dir / "fig_frontend_robustness.png", dpi=200)
    figure.savefig(out_dir / "fig_frontend_robustness.pdf")
    plt.close(figure)


def _write_report(
    out_path: Path, aggregate: pd.DataFrame, metadata: dict[str, Any]
) -> None:
    lines = [
        "# Front-end signal robustness",
        "",
        f"Base trace `{metadata['records']}` ({metadata['num_jobs']} jobs), variant "
        f"`{metadata['variant']}`, {metadata['noise_seeds']} noise seeds per "
        "$\\epsilon>0$.",
        "",
        "Noise is injected only into the front-end estimates the scheduler consumes.",
        "True service demands and true logical outcomes are held fixed, so the",
        "degradation below is attributable to front-end inaccuracy alone.",
        "$\\epsilon=0$ is the unperturbed anchor.",
        "",
        "| W | eps | LER mean [min,max] | miss mean (max) | p99 commit us | max lag | fast-sel |",
        "|---:|---:|---|---|---:|---:|---:|",
    ]
    for _, row in aggregate.iterrows():
        lines.append(
            f"| {int(row['num_workers'])} | {row['eps']:.2f} | "
            f"{row['logical_error_rate_mean']:.4f} "
            f"[{row['logical_error_rate_min']:.4f}, {row['logical_error_rate_max']:.4f}] | "
            f"{row['commit_deadline_miss_ratio_mean']:.4f} "
            f"({row['commit_deadline_miss_ratio_max']:.4f}) | "
            f"{row['p99_response_to_commit_us_mean']:.2f} | "
            f"{row['maximum_pauli_frame_lag_max']:.0f} | "
            f"{row['fast_selection_rate_mean']:.3f} |"
        )
    lines.append("")
    if not metadata["all_integrity_checks_passed"]:
        lines.extend(["**Some runs FAILED integrity checks; treat the table as unsound.**", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--risk-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", default="2,4")
    parser.add_argument("--eps-grid", default="0,0.1,0.2,0.4,0.8,1.6")
    parser.add_argument("--noise-seeds", type=int, default=8)
    parser.add_argument(
        "--variant",
        default="commit_frontier",
        help=f"routing/dispatch pair to hold fixed; available: {', '.join(sorted(VARIANTS))}",
    )
    parser.add_argument("--fixed-fast-estimate-us", type=float, default=4.8)
    args = parser.parse_args()

    if args.variant not in VARIANTS:
        raise SystemExit(f"unknown variant {args.variant!r}; see --help")
    mode, policy = VARIANTS[args.variant]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(Path(args.config))
    records = reindex_continuous_records(_load_records_csv(Path(args.records)))
    base_profiles = build_prediction_profiles(
        records,
        Path(args.risk_checkpoint),
        device=config.device,
        fixed_fast_estimate_us=float(args.fixed_fast_estimate_us),
    )
    workers = parse_ints(args.workers)
    eps_grid = [float(value) for value in parse_strings(args.eps_grid)]
    print(f"{len(records)} jobs, variant {args.variant} (mode={mode}, policy={policy})")

    metrics = [
        "commit_deadline_miss_ratio",
        "p99_response_to_commit_us",
        "maximum_pauli_frame_lag",
        "lag_violation_ratio",
        "boundary_commit_success_rate",
        "logical_error_rate",
        "fast_selection_rate",
    ]
    rows: list[dict[str, Any]] = []
    for num_workers in workers:
        for eps in eps_grid:
            # eps=0 is deterministic, so repeating it would only duplicate one row.
            replicates = 1 if eps == 0.0 else int(args.noise_seeds)
            for replicate in range(replicates):
                rng = np.random.default_rng(SEED_BASE + int(round(eps * 1000)) + replicate)
                profiles = (
                    base_profiles if eps == 0.0 else perturb(base_profiles, eps, rng)
                )
                result = simulate_trace(
                    records,
                    profiles,
                    config,
                    mode=mode,
                    num_workers=num_workers,
                    dispatch_policy=policy,
                )
                rows.append(
                    {
                        "num_workers": num_workers,
                        "eps": eps,
                        "noise_seed": replicate,
                        **{metric: float(result.summary[metric]) for metric in metrics},
                        "integrity_passed": bool(result.summary["integrity_passed"]),
                    }
                )
                print(
                    f"  W={num_workers} eps={eps:<5} k={replicate} "
                    f"LER={result.summary['logical_error_rate']:.4f} "
                    f"miss={result.summary['commit_deadline_miss_ratio']:.4f} "
                    f"p99={result.summary['p99_response_to_commit_us']:.2f}us",
                    flush=True,
                )

    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "robustness_raw.csv", index=False)
    grouped = raw.groupby(["num_workers", "eps"])
    aggregate = grouped[metrics].agg(["mean", "min", "max"])
    aggregate.columns = ["_".join(column) for column in aggregate.columns]
    aggregate["num_replicates"] = grouped.size()
    aggregate = aggregate.reset_index()
    aggregate.to_csv(out_dir / "robustness_aggregate.csv", index=False)

    metadata = {
        "records": str(args.records),
        "records_sha256": sha256(Path(args.records)),
        "config": str(args.config),
        "checkpoint": str(args.risk_checkpoint),
        "checkpoint_sha256": sha256(Path(args.risk_checkpoint)),
        "variant": args.variant,
        "mode": mode,
        "dispatch_policy": policy,
        "workers": workers,
        "eps_grid": eps_grid,
        "noise_seeds": int(args.noise_seeds),
        "num_jobs": int(len(records)),
        "noise_model": {
            "predicted_latencies": "multiplied by exp(N(0, eps))",
            "risk_and_confidence": "plus N(0, eps), clipped to [0, 1]",
        },
        "all_integrity_checks_passed": bool(raw["integrity_passed"].all()),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    _plot(aggregate, workers, out_dir)
    _write_report(out_dir / "ROBUSTNESS_RESULTS.md", aggregate, metadata)

    if not raw["integrity_passed"].all():
        failed = raw.loc[~raw["integrity_passed"], ["num_workers", "eps", "noise_seed"]]
        raise SystemExit(f"integrity checks FAILED for:\n{failed.to_string(index=False)}")
    print(f"\nwrote {out_dir} ({len(raw)} runs, all integrity checks passed)")


if __name__ == "__main__":
    main()
