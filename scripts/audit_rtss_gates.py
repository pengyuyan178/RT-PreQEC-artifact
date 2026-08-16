"""Audit RTSS experiment artifacts against the execution-plan gates."""

from __future__ import annotations

from pathlib import Path
import json
import sys

import pandas as pd
import typer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rt_preqec.utils import ensure_parent

app = typer.Typer(add_completion=False)


def _gate_row(phase: str, item: str, status: str, evidence: str, detail: str = "") -> dict[str, str]:
    return {"phase": phase, "item": item, "status": status, "evidence": evidence, "detail": detail}


def _status(condition: bool, weak: bool = False) -> str:
    if condition:
        return "pass"
    return "weak" if weak else "fail"


def _load_summary(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "main" / "summary_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path).set_index("mode")


def _audit_regime(name: str, run_dir: Path, phase: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    summary = _load_summary(run_dir)
    evidence = str(run_dir / "main" / "summary_metrics.csv")
    if summary.empty:
        return [_gate_row(phase, f"{name}: summary exists", "fail", evidence, "missing or empty")]
    required = {"accurate_only", "fast_only", "edf", "rt_qec", "rt_qec_without_scheduler", "rt_qec_without_validation"}
    missing = sorted(required.difference(set(summary.index)))
    rows.append(_gate_row(phase, f"{name}: required modes", _status(not missing), evidence, ",".join(missing)))
    if missing:
        return rows
    rt = summary.loc["rt_qec"]
    acc = summary.loc["accurate_only"]
    fast = summary.loc["fast_only"]
    edf = summary.loc["edf"]
    no_sched = summary.loc["rt_qec_without_scheduler"]
    no_val = summary.loc["rt_qec_without_validation"]
    certified = summary.loc["rt_qec_ai"] if "rt_qec_ai" in summary.index else rt
    certified_name = "rt_qec_ai" if "rt_qec_ai" in summary.index else "rt_qec"
    rows.extend(
        [
            _gate_row(
                phase,
                f"{name}: fast timing stable",
                _status(float(fast["pauli_frame_lag_violation_ratio"]) <= 0.02, weak=True),
                evidence,
                f"fast lag violation={float(fast['pauli_frame_lag_violation_ratio']):.6g}",
            ),
            _gate_row(
                phase,
                f"{name}: accurate lower LER than fast",
                _status(float(acc["logical_error_rate"]) < float(fast["logical_error_rate"])),
                evidence,
                f"accurate={float(acc['logical_error_rate']):.6g}, fast={float(fast['logical_error_rate']):.6g}",
            ),
            _gate_row(
                phase,
                f"{name}: rt_qec preserves logical reliability",
                _status(
                    float(rt["logical_error_rate"]) <= max(
                        0.25 * float(fast["logical_error_rate"]),
                        4.0 * float(acc["logical_error_rate"]),
                        0.002,
                    ),
                    weak=True,
                ),
                evidence,
                (
                    f"rt={float(rt['logical_error_rate']):.6g}, "
                    f"accurate={float(acc['logical_error_rate']):.6g}, "
                    f"fast={float(fast['logical_error_rate']):.6g}"
                ),
            ),
            _gate_row(
                phase,
                f"{name}: rt_qec beats accurate on lag or boundary",
                _status(
                    float(rt["pauli_frame_lag_violation_ratio"]) < float(acc["pauli_frame_lag_violation_ratio"])
                    or float(rt["boundary_commit_success_rate"]) > float(acc["boundary_commit_success_rate"]),
                    weak=True,
                ),
                evidence,
                (
                    f"lag rt={float(rt['pauli_frame_lag_violation_ratio']):.6g}, "
                    f"acc={float(acc['pauli_frame_lag_violation_ratio']):.6g}; "
                    f"boundary rt={float(rt['boundary_commit_success_rate']):.6g}, "
                    f"acc={float(acc['boundary_commit_success_rate']):.6g}"
                ),
            ),
            _gate_row(
                phase,
                f"{name}: certified method beats edf on LER",
                _status(float(certified["logical_error_rate"]) < float(edf["logical_error_rate"])),
                evidence,
                (
                    f"{certified_name}={float(certified['logical_error_rate']):.6g}, "
                    f"rt={float(rt['logical_error_rate']):.6g}, "
                    f"edf={float(edf['logical_error_rate']):.6g}"
                ),
            ),
            _gate_row(
                phase,
                f"{name}: scheduler ablation contributes",
                _status(
                    float(rt["pauli_frame_lag_violation_ratio"]) < float(no_sched["pauli_frame_lag_violation_ratio"])
                    or float(rt["p99_response_time_us"]) < float(no_sched["p99_response_time_us"])
                    or float(rt["boundary_commit_success_rate"]) > float(no_sched["boundary_commit_success_rate"]),
                    weak=True,
                ),
                evidence,
                (
                    f"lag rt={float(rt['pauli_frame_lag_violation_ratio']):.6g}, "
                    f"no_sched={float(no_sched['pauli_frame_lag_violation_ratio']):.6g}; "
                    f"p99 rt={float(rt['p99_response_time_us']):.6g}, "
                    f"no_sched={float(no_sched['p99_response_time_us']):.6g}"
                ),
            ),
            _gate_row(
                phase,
                f"{name}: validation ablation is unsafe",
                _status(float(no_val["logical_error_rate"]) > float(rt["logical_error_rate"])),
                evidence,
                f"no_validation={float(no_val['logical_error_rate']):.6g}, rt={float(rt['logical_error_rate']):.6g}",
            ),
        ]
    )
    return rows


@app.command()
def main(
    phase0: str = typer.Option("results/runs/_phase0_smoke", "--phase0"),
    d7: str = typer.Option("results/runs/paper_suite_d7_rtqec_ai_selected", "--d7"),
    d11: str = typer.Option("results/runs/paper_suite_d11_rtqec_ai", "--d11"),
    burst: str = typer.Option("results/runs/paper_suite_burst_rtqec_ai", "--burst"),
    overhead: str = typer.Option("results/runs/wcet_overheads_d7", "--overhead"),
    out: str = typer.Option("results/tables/rtss_gate_audit.csv", "--out"),
) -> None:
    """Write a CSV gate audit for paper-experiment readiness."""
    rows: list[dict[str, str]] = []

    phase0_dir = Path(phase0)
    p0_summary_path = phase0_dir / "summary_metrics.csv"
    if p0_summary_path.exists():
        p0 = pd.read_csv(p0_summary_path)
        rows.append(_gate_row("P0", "real_qec true", _status(bool(p0["real_qec"].astype(bool).all())), str(p0_summary_path)))
        rows.append(
            _gate_row(
                "P0",
                "loop/per-record timing",
                _status(set(p0["timing_mode"].astype(str)) == {"loop_per_shot"}),
                str(p0_summary_path),
                ",".join(sorted(set(p0["timing_mode"].astype(str)))),
            )
        )
    else:
        rows.append(_gate_row("P0", "summary exists", "fail", str(p0_summary_path), "missing"))

    rows.extend(_audit_regime("d7", Path(d7), "P1/P3/P4"))
    rows.extend(_audit_regime("burst", Path(burst), "P3-burst"))
    rows.extend(_audit_regime("d11", Path(d11), "P3-scaling"))

    overhead_path = Path(overhead) / "overhead_summary.csv"
    if overhead_path.exists():
        overhead_frame = pd.read_csv(overhead_path)
        components = set(overhead_frame["component"].astype(str))
        required_components = {"frontend", "validation", "scheduler", "fast_backend", "accurate_backend"}
        rows.append(
            _gate_row(
                "P7",
                "overhead components measured",
                _status(required_components.issubset(components)),
                str(overhead_path),
                ",".join(sorted(components)),
            )
        )
    else:
        rows.append(_gate_row("P7", "overhead summary exists", "fail", str(overhead_path), "missing"))

    for run_name, run_path in [("d7", Path(d7)), ("burst", Path(burst)), ("d11", Path(d11))]:
        manifest_path = run_path / "suite_manifest.json"
        status = "fail"
        detail = "missing"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            status = _status(bool(manifest.get("git_commit_hash")) and bool(manifest.get("seed_list")) and bool(manifest.get("software_versions")))
            detail = f"git={manifest.get('git_commit_hash')}"
        rows.append(_gate_row("P8", f"{run_name}: manifest frozen metadata", status, str(manifest_path), detail))

    pd.DataFrame(rows).to_csv(ensure_parent(out), index=False)


if __name__ == "__main__":
    app()
