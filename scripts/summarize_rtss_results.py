"""Export concise RTSS regime and frontend-contract summary tables."""

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

MODE_LABELS = {
    "accurate_only": "Accurate-only",
    "fast_only": "Fast-only",
    "edf": "EDF",
    "heuristic_pre_fixed": "Front-end only",
    "rt_qec": "Heuristic runtime",
    "rt_qec_ai": "RT-PreQEC",
    "rt_qec_without_validation": "No validation",
    "rt_qec_without_abstention": "No abstention",
    "oracle_predecoder": "Oracle front-end",
}


def _summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@app.command()
def main(
    d7: str = typer.Option("results/runs/paper_suite_d7_rtqec_ai_selected/main/summary_metrics.csv", "--d7"),
    d11: str = typer.Option("results/runs/paper_suite_d11_rtqec_ai/main/summary_metrics.csv", "--d11"),
    burst: str = typer.Option("results/runs/paper_suite_burst_rtqec_ai/main/summary_metrics.csv", "--burst"),
    out_dir: str = typer.Option("results/tables", "--out-dir"),
) -> None:
    """Write Phase 1/2 paper-ready summary tables."""
    out = Path(out_dir)
    regime_rows: list[dict[str, object]] = []
    for regime, path in [("d7_main", d7), ("d11_scaling", d11), ("burst", burst)]:
        frame = _summary(Path(path))
        if frame.empty:
            continue
        for mode in ["accurate_only", "fast_only", "edf", "rt_qec_ai"]:
            subset = frame[frame["mode"] == mode]
            if subset.empty:
                continue
            row = subset.iloc[0].to_dict()
            regime_rows.append(
                {
                    "regime": regime,
                    "mode": mode,
                    "display_mode": MODE_LABELS.get(mode, mode),
                    "logical_error_rate": row.get("logical_error_rate"),
                    "p99_response_time_us": row.get("p99_response_time_us"),
                    "p999_response_time_us": row.get("p999_response_time_us"),
                    "pauli_frame_lag_violation_ratio": row.get("pauli_frame_lag_violation_ratio"),
                    "boundary_commit_success_rate": row.get("boundary_commit_success_rate"),
                    "fast_selection_rate": row.get("fast_selection_rate"),
                    "accept_rate": row.get("accept_rate"),
                    "timing_mode": row.get("timing_mode"),
                    "real_qec": row.get("real_qec"),
                }
            )
    pd.DataFrame(regime_rows).to_csv(ensure_parent(out / "regime_summary_table.csv"), index=False)

    d7_frame = _summary(Path(d7))
    contract_modes = [
        "rt_qec_ai",
        "heuristic_pre_fixed",
        "rt_qec",
        "rt_qec_without_validation",
        "rt_qec_without_abstention",
        "oracle_predecoder",
    ]
    contract_columns = [
        "mode",
        "display_mode",
        "logical_error_rate",
        "accept_rate",
        "abstention_rate",
        "false_accept_rate",
        "accepted_error_rate",
        "validation_pass_rate",
        "predecode_accept_rate",
        "mean_estimated_residual_reduction",
        "fast_selection_rate",
        "accurate_selection_rate",
        "pauli_frame_lag_violation_ratio",
        "boundary_commit_success_rate",
    ]
    if not d7_frame.empty:
        contract = d7_frame[d7_frame["mode"].isin(contract_modes)].copy()
        contract["mode"] = pd.Categorical(contract["mode"], categories=contract_modes, ordered=True)
        contract = contract.sort_values("mode")
        contract["mode"] = contract["mode"].astype(str)
        contract["display_mode"] = contract["mode"].map(MODE_LABELS).fillna(contract["mode"])
        contract[[column for column in contract_columns if column in contract.columns]].to_csv(
            ensure_parent(out / "frontend_contract_table.csv"),
            index=False,
        )


if __name__ == "__main__":
    app()
