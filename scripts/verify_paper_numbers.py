"""Replay every regime from the committed traces and diff against the published summary.

An artifact reviewer needs one command that answers "do the numbers in the paper come out
of this repository?". The suite regenerates syndromes with an unseeded stim sampler, so a
fresh run cannot reproduce published values; the committed records.csv carries the
syndromes together with the per-shot latency labels, so replaying it exercises the queue
simulation and the metric aggregation against pinned inputs.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _find_repo_root(path: Path) -> Path:
    for candidate in (path.resolve(), *path.resolve().parents):
        if (candidate / "src" / "rt_preqec").is_dir():
            return candidate
    raise RuntimeError(f"could not find repository root above {path}")


ROOT = _find_repo_root(Path(__file__))
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import argparse  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rt_preqec.config import load_config  # noqa: E402
from rt_preqec.evaluation.real_stream import simulate_realtime_queue  # noqa: E402
from scripts.run_paper_experiment_suite import _load_records_csv  # noqa: E402

#: run directory under table/figure_data/ -> config that produced it
REGIMES = {
    "d7_main": "configs/real_stream_eval_main_ai_selected.yaml",
    "d11_scaling": "configs/real_stream_eval_scaling.yaml",
    "burst": "configs/real_stream_eval_burst.yaml",
    "burst_2w": "configs/real_stream_eval_burst_2w.yaml",
}

def _boundary_success(events: pd.DataFrame, budget: int) -> float:
    """Boundary-commit success is averaged over boundary rounds only, not all jobs."""
    boundary = events.loc[events["logical_boundary"].astype(bool)]
    if boundary.empty:
        return 0.0
    return float(boundary["boundary_commit_success"].astype(float).mean())


#: summary column -> how to recompute it from the replayed event frame
CHECKS = {
    "p99_response_time_us": lambda e, budget: float(np.percentile(e["response_time_us"], 99)),
    "p999_response_time_us": lambda e, budget: float(np.percentile(e["response_time_us"], 99.9)),
    "deadline_miss_ratio": lambda e, budget: float(e["deadline_miss"].mean()),
    "pauli_frame_lag_violation_ratio": lambda e, budget: float(
        (e["pauli_frame_lag"] > budget).mean()
    ),
    "p99_pauli_frame_lag": lambda e, budget: float(np.percentile(e["pauli_frame_lag"], 99)),
    "boundary_commit_success_rate": _boundary_success,
    "logical_error_rate": lambda e, budget: float(e["logical_error"].mean()),
}


def replay_mode(records_by_shot: dict, config, published: pd.DataFrame) -> pd.DataFrame:
    """Re-simulate one mode, reusing its archived service times and routing decisions."""
    ordered, latencies, predictions, decoders = [], [], [], []
    for row in published.itertuples():
        record = records_by_shot.get(int(row.shot_id))
        if record is None:
            raise SystemExit(f"shot {row.shot_id} absent from records.csv")
        ordered.append(record)
        latencies.append(float(row.latency_us))
        decoders.append(str(getattr(row, "selected_decoder", "accurate")))
        predictions.append(np.atleast_1d(np.asarray(row.prediction, dtype=np.int8)))
    return simulate_realtime_queue(ordered, latencies, predictions, config, decoders)


def check_regime(run_dir: Path, config_path: Path, rtol: float) -> tuple[int, int]:
    config = load_config(str(config_path))
    budget = int(config.runtime.max_pauli_frame_lag)
    records = {int(r.shot_id): r for r in _load_records_csv(run_dir / "records.csv")}
    summary = pd.read_csv(run_dir / "summary_metrics.csv").set_index("mode")

    passed = failed = 0
    for mode in summary.index:
        events_path = run_dir / str(mode) / "events.csv"
        if not events_path.exists():
            continue
        events = replay_mode(records, config, pd.read_csv(events_path))
        deltas = []
        for column, recompute in CHECKS.items():
            if column not in summary.columns:
                continue
            want = float(summary.loc[mode, column])
            got = recompute(events, budget)
            if np.isclose(got, want, rtol=rtol, atol=1e-12):
                passed += 1
            else:
                failed += 1
                deltas.append(f"{column}: published={want:.6g} replay={got:.6g}")
        status = "OK" if not deltas else "DIFF"
        print(f"  {mode:28s} {status}")
        for line in deltas:
            print(f"      {line}")
    return passed, failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(ROOT / "table" / "figure_data"))
    parser.add_argument("--regime", action="append", choices=sorted(REGIMES))
    parser.add_argument("--rtol", type=float, default=1e-6)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    total_pass = total_fail = 0
    for regime in args.regime or sorted(REGIMES):
        run_dir = data_dir / regime
        if not (run_dir / "records.csv").exists():
            print(f"{regime}: no records.csv, skipping")
            continue
        print(f"\n=== {regime} ===")
        p, f = check_regime(run_dir, ROOT / REGIMES[regime], args.rtol)
        total_pass += p
        total_fail += f

    print(f"\n{total_pass} metric checks reproduce, {total_fail} deviate (rtol={args.rtol:g})")
    sys.exit(1 if total_fail else 0)


if __name__ == "__main__":
    main()
