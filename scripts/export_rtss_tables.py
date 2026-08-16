"""Export RTSS paper tables from a real-stream suite run."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import typer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rt_preqec.utils import ensure_parent

app = typer.Typer(add_completion=False)

MAIN_MODES = [
    "accurate_only",
    "fast_only",
    "edf",
    "rt_qec_ai",
    "oracle_predecoder",
    "oracle_risk",
]

ABLATION_MODES = [
    "rt_qec",
    "rt_qec_without_validation",
    "rt_qec_without_abstention",
    "rt_qec_without_scheduler",
    "heuristic_pre_fixed",
    "risk_heuristic",
    "ai_risk",
]

MODE_LABELS = {
    "accurate_only": "Accurate-only",
    "fast_only": "Fast-only",
    "edf": "EDF",
    "heuristic_pre_fixed": "Front-end only",
    "risk_heuristic": "Rule risk",
    "rt_qec": "Heuristic runtime",
    "rt_qec_ai": "RT-PreQEC",
    "ai_risk": "Learned risk only",
    "rt_qec_without_validation": "No validation",
    "rt_qec_without_abstention": "No abstention",
    "rt_qec_without_scheduler": "No scheduler",
    "oracle_predecoder": "Oracle front-end",
    "oracle_risk": "Oracle risk",
}

MODE_ORDER = [
    *MAIN_MODES,
    *ABLATION_MODES,
]

MAIN_COLUMNS = [
    "mode",
    "logical_error_rate",
    "p99_response_time_us",
    "p999_response_time_us",
    "deadline_miss_ratio",
    "p99_pauli_frame_lag",
    "pauli_frame_lag_violation_ratio",
    "boundary_commit_success_rate",
    "fast_selection_rate",
    "accurate_selection_rate",
    "accept_rate",
    "abstention_rate",
    "false_accept_rate",
    "accepted_error_rate",
    "validation_pass_rate",
    "predecode_accept_rate",
    "mean_estimated_residual_reduction",
]

DELTA_COLUMNS = [
    "logical_error_rate",
    "p99_response_time_us",
    "deadline_miss_ratio",
    "p99_pauli_frame_lag",
    "pauli_frame_lag_violation_ratio",
    "boundary_commit_success_rate",
]


def _select_modes(frame: pd.DataFrame, modes: list[str]) -> pd.DataFrame:
    selected = frame[frame["mode"].isin(modes)].copy()
    selected["mode"] = pd.Categorical(selected["mode"], categories=modes, ordered=True)
    selected = selected.sort_values("mode")
    selected["mode"] = selected["mode"].astype(str)
    return selected


def _with_display_mode(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "mode" in out.columns:
        out.insert(1, "display_mode", out["mode"].map(MODE_LABELS).fillna(out["mode"]))
    return out


@app.command()
def main(
    run_dir: str = typer.Option("results/runs/paper_suite_d7_rtqec_ai_selected/main", "--run-dir"),
    out: str = typer.Option("results/tables", "--out"),
    burst_1w_run_dir: str | None = typer.Option(None, "--burst-1w-run-dir"),
    burst_2w_run_dir: str | None = typer.Option(None, "--burst-2w-run-dir"),
) -> None:
    """Write main comparison, safety-contract, and scheduler delta tables."""
    run_path = Path(run_dir)
    out_path = Path(out)
    summary_path = run_path / "summary_metrics.csv"
    if not summary_path.exists():
        raise typer.BadParameter(f"missing summary_metrics.csv under {run_path}")

    frame = pd.read_csv(summary_path)
    main_selected = _with_display_mode(_select_modes(frame, MAIN_MODES)[[c for c in MAIN_COLUMNS if c in frame.columns]])
    main_selected.to_csv(ensure_parent(out_path / "rtss_main_table.csv"), index=False)

    ablation_modes = ["rt_qec_ai", *ABLATION_MODES]
    ablation = _with_display_mode(_select_modes(frame, ablation_modes)[[c for c in MAIN_COLUMNS if c in frame.columns]])
    ablation.to_csv(ensure_parent(out_path / "rtss_main_ablation_table.csv"), index=False)

    safety_modes = ["rt_qec_ai", "rt_qec", "rt_qec_without_validation", "rt_qec_without_abstention"]
    safety = _with_display_mode(_select_modes(frame, safety_modes)[[c for c in MAIN_COLUMNS if c in frame.columns]])
    safety.to_csv(ensure_parent(out_path / "rtss_safety_contract_table.csv"), index=False)

    ai_modes = ["rt_qec_ai", "rt_qec", "ai_risk", "risk_heuristic", "oracle_risk"]
    ai = _with_display_mode(_select_modes(frame, ai_modes)[[c for c in MAIN_COLUMNS if c in frame.columns]])
    ai.to_csv(ensure_parent(out_path / "rtss_ai_risk_table.csv"), index=False)

    if "rt_qec_ai" not in set(frame["mode"]):
        return
    indexed = frame.set_index("mode")
    base = indexed.loc["rt_qec_ai"]
    rows: list[dict[str, object]] = []
    for mode in [
        "rt_qec_without_validation",
        "rt_qec_without_abstention",
        "rt_qec_without_scheduler",
        "rt_qec",
        "edf",
        "heuristic_pre_fixed",
    ]:
        if mode not in indexed.index:
            continue
        row: dict[str, object] = {"mode": mode}
        for column in DELTA_COLUMNS:
            if column in indexed.columns:
                row[f"delta_{column}"] = float(indexed.loc[mode, column] - base[column])
        rows.append(row)
    pd.DataFrame(rows).to_csv(ensure_parent(out_path / "rtss_ablation_delta_vs_rt_qec.csv"), index=False)

    if burst_1w_run_dir and burst_2w_run_dir:
        capacity_rows: list[pd.DataFrame] = []
        for label, burst_dir in [("1w", burst_1w_run_dir), ("2w", burst_2w_run_dir)]:
            burst_path = Path(burst_dir) / "summary_metrics.csv"
            if not burst_path.exists():
                continue
            burst = pd.read_csv(burst_path)
            burst = _select_modes(burst, ["accurate_only", "rt_qec_ai", "rt_qec", "oracle_predecoder"])
            burst["capacity_setting"] = label
            capacity_rows.append(burst)
        if capacity_rows:
            capacity = pd.concat(capacity_rows, ignore_index=True)
            capacity = _with_display_mode(capacity)
            capacity_columns = ["capacity_setting", *[c for c in MAIN_COLUMNS if c in capacity.columns], "num_workers"]
            capacity[[c for c in capacity_columns if c in capacity.columns]].to_csv(
                ensure_parent(out_path / "rtss_burst_capacity_table.csv"),
                index=False,
            )


if __name__ == "__main__":
    app()
