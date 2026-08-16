from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from scripts.export_rtss_tables import app as export_tables_app
from scripts.make_rtss_plots import THRESHOLD_FRONTIER_DISPLAY_THRESHOLDS
from scripts.make_rtss_plots import app as make_rtss_plots_app
from scripts.run_paper_experiment_suite import PAPER_SCHEDULER_RISK_THRESHOLDS, _parse_grid


def _write_summary(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [
            {
                "mode": "rt_qec",
                "logical_error_rate": 0.01,
                "p99_response_time_us": 4.0,
                "p999_response_time_us": 5.0,
                "deadline_miss_ratio": 0.02,
                "p99_pauli_frame_lag": 2.0,
                "pauli_frame_lag_violation_ratio": 0.01,
                "boundary_commit_success_rate": 0.98,
                "fast_selection_rate": 0.5,
                "accurate_selection_rate": 0.5,
                "accept_rate": 0.4,
                "abstention_rate": 0.6,
                "false_accept_rate": 0.0,
                "accepted_error_rate": 0.0,
                "validation_pass_rate": 0.8,
                "predecode_accept_rate": 0.4,
                "mean_estimated_residual_reduction": 0.1,
            },
            {
                "mode": "rt_qec_without_scheduler",
                "logical_error_rate": 0.02,
                "p99_response_time_us": 7.0,
                "p999_response_time_us": 8.0,
                "deadline_miss_ratio": 0.05,
                "p99_pauli_frame_lag": 4.0,
                "pauli_frame_lag_violation_ratio": 0.03,
                "boundary_commit_success_rate": 0.92,
                "fast_selection_rate": 0.3,
                "accurate_selection_rate": 0.7,
                "accept_rate": 0.4,
                "abstention_rate": 0.6,
                "false_accept_rate": 0.1,
                "accepted_error_rate": 0.1,
                "validation_pass_rate": 0.8,
                "predecode_accept_rate": 0.4,
                "mean_estimated_residual_reduction": 0.1,
            },
        ]
    )
    frame.to_csv(path / "summary_metrics.csv", index=False)


def _write_events(path: Path, mode: str) -> None:
    mode_path = path / mode
    mode_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "shot_id": [0, 1, 2],
            "pauli_frame_lag": [0, 1, 2],
            "backlog": [1, 2, 3],
            "response_time_us": [1.0, 2.0, 3.0],
        }
    ).to_csv(mode_path / "events.csv", index=False)


def test_export_rtss_tables_writes_expected_csvs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "tables"
    _write_summary(run_dir)
    result = CliRunner().invoke(export_tables_app, ["--run-dir", str(run_dir), "--out", str(out_dir)])
    assert result.exit_code == 0, result.output
    assert (out_dir / "rtss_main_ablation_table.csv").exists()
    assert (out_dir / "rtss_safety_contract_table.csv").exists()
    assert (out_dir / "rtss_ablation_delta_vs_rt_qec.csv").exists()


def test_make_rtss_plots_writes_core_figures(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    burst_run_dir = tmp_path / "burst"
    out_dir = tmp_path / "figures"
    _write_summary(run_dir)
    _write_summary(burst_run_dir)
    _write_events(run_dir, "rt_qec")
    _write_events(run_dir, "rt_qec_without_scheduler")
    _write_events(burst_run_dir, "rt_qec")
    _write_events(burst_run_dir, "rt_qec_without_scheduler")
    result = CliRunner().invoke(
        make_rtss_plots_app,
        ["--run-dir", str(run_dir), "--out", str(out_dir), "--burst-run-dir", str(burst_run_dir)],
    )
    assert result.exit_code == 0, result.output
    assert (out_dir / "logical_error_rate_vs_pauli_frame_lag_violation.png").exists()
    assert (out_dir / "logical_error_rate_vs_p99_response_time.png").exists()
    assert (out_dir / "boundary_commit_success_vs_logical_error.png").exists()
    assert (out_dir / "pauli_frame_lag_over_time_by_mode.png").exists()
    assert (out_dir / "response_time_cdf_by_mode.png").exists()
    assert (out_dir / "burst_lag_backlog_trace.png").exists()


def test_swept_scheduler_thresholds_are_exactly_the_plotted_ones() -> None:
    """The sweep must run every point the figure shows, and no point it does not.

    These two lists live in different modules -- one drives the experiment, the other the
    figure -- and nothing else couples them. If they drift, the sweep silently either wastes
    runs on configurations that are filtered out before plotting, or the figure interpolates
    across a threshold that was never measured. Both failures are invisible in the output.
    """
    swept = [round(value, 2) for value in _parse_grid(PAPER_SCHEDULER_RISK_THRESHOLDS)]
    plotted = [round(value, 2) for value in THRESHOLD_FRONTIER_DISPLAY_THRESHOLDS]
    assert swept == plotted, (
        "scheduler sweep grid and plotted thresholds disagree; "
        f"swept={swept} plotted={plotted}"
    )
    assert swept == sorted(swept), "thresholds must ascend so the plotted line is monotone in x"
    assert len(set(swept)) == len(swept), "duplicate threshold would be averaged into one point"
