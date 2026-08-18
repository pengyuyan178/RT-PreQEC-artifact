"""Run the focused RT-QEC paper experiment suite."""

from __future__ import annotations

from pathlib import Path
import ast
import copy
import itertools
import json
import platform
import subprocess
import sys

import pandas as pd
import typer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rt_preqec.config import load_config
from rt_preqec.evaluation.real_stream import RealStreamShotRecord, evaluate_mode_on_records, run_real_stream_eval
from rt_preqec.utils import ensure_parent

app = typer.Typer(add_completion=False)


def _parse_grid(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _write_table(rows: list[dict[str, object]], path: Path) -> None:
    pd.DataFrame(rows).to_csv(ensure_parent(path), index=False)


def _load_completed_threshold_summary(run_dir: Path) -> dict[str, object] | None:
    summary_path = run_dir / "summary_metrics.csv"
    if not summary_path.exists():
        return None
    summary_frame = pd.read_csv(summary_path)
    if summary_frame.empty:
        return None
    return dict(summary_frame.iloc[0].to_dict())


def _literal(value: object, default: object) -> object:
    if value is None or pd.isna(value):
        return default
    if isinstance(value, (list, dict, int, float)):
        return value
    try:
        return ast.literal_eval(str(value))
    except Exception:
        return default


def _array_literal(value: object) -> list[int]:
    parsed = _literal(value, [])
    if isinstance(parsed, list):
        return [int(item) for item in parsed]
    return [int(parsed)]


def _load_records_csv(path: Path) -> list[RealStreamShotRecord]:
    frame = pd.read_csv(path)
    records: list[RealStreamShotRecord] = []
    for _, row in frame.iterrows():
        metadata = _literal(row.get("metadata"), {})
        if not isinstance(metadata, dict):
            metadata = {}
        feature_names = _literal(row.get("feature_names"), [])
        features = _literal(row.get("features"), [])
        records.append(
            RealStreamShotRecord(
                shot_id=int(row["shot_id"]),
                syndrome=pd.Series(_array_literal(row["syndrome"]), dtype="int8").to_numpy(),
                observable=pd.Series(_array_literal(row["observable"]), dtype="int8").to_numpy(),
                accurate_prediction=pd.Series(_array_literal(row["accurate_prediction"]), dtype="int8").to_numpy(),
                fast_prediction=pd.Series(_array_literal(row["fast_prediction"]), dtype="int8").to_numpy(),
                accurate_latency_us=float(row["accurate_latency_us"]),
                fast_latency_us=float(row["fast_latency_us"]),
                features=pd.Series(features, dtype="float32").to_numpy(),
                feature_names=[str(name) for name in feature_names],
                risk_label=int(row["risk_label"]),
                hard_runtime=int(row["hard_runtime"]),
                fast_wrong_vs_accurate=int(row["fast_wrong_vs_accurate"]),
                fast_logical_fail=int(row["fast_logical_fail"]),
                metadata=metadata,
            )
        )
    return sorted(records, key=lambda record: record.shot_id)


def _compact_metadata(payload: dict[str, object]) -> dict[str, object]:
    metadata = dict(payload.get("metadata", {}))
    keys = [
        "eval_source",
        "eval_split",
        "split_policy",
        "split_boundaries",
        "train_indices_hash",
        "val_indices_hash",
        "test_indices_hash",
        "checkpoint_train_split_hash",
        "split_match",
        "warnings",
        "timing_mode",
        "hard_runtime_label_valid",
        "ai_risk_available",
        "real_qec",
        "fallback_reason",
        "num_records",
        "num_samples",
        "feature_dim",
        "preset",
    ]
    return {key: metadata.get(key, payload.get(key)) for key in keys if metadata.get(key, payload.get(key)) is not None}


def _command_output(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    return completed.stdout.strip() or None


def _git_head_from_files() -> str | None:
    git_dir = ROOT / ".git"
    head_path = git_dir / "HEAD"
    if not head_path.exists():
        return None
    head = head_path.read_text(encoding="utf-8").strip()
    if not head.startswith("ref:"):
        return head or None
    ref_name = head.split(":", 1)[1].strip()
    ref_path = git_dir / ref_name
    if ref_path.exists():
        value = ref_path.read_text(encoding="utf-8").strip()
        return value or None
    packed_refs = git_dir / "packed-refs"
    if packed_refs.exists():
        for line in packed_refs.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line or line.startswith("#") or line.startswith("^"):
                continue
            parts = line.split()
            if len(parts) == 2 and parts[1] == ref_name:
                return parts[0]
    return None


def _git_commit_hash() -> str | None:
    return _command_output(["git", "rev-parse", "HEAD"]) or _git_head_from_files()


def _software_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in ["numpy", "pandas", "torch", "stim", "pymatching", "matplotlib"]:
        try:
            module = __import__(package)
            versions[package] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            versions[package] = None
    return versions


#: Scheduler risk thresholds swept for the decoupled threshold figure. These are exactly the
#: points the paper reports, so the sweep runs no configuration the figure does not show.
#: Changing this list changes the LER / p99 ranges quoted in the decoupled-sweep section, and
#: must stay in sync with ``THRESHOLD_FRONTIER_DISPLAY_THRESHOLDS`` in ``make_rtss_plots.py``.
PAPER_SCHEDULER_RISK_THRESHOLDS = "0.10,0.15,0.25,0.30,0.35,0.40,0.50"


@app.command()
def main(
    config: str = "configs/real_stream_eval.yaml",
    risk_dataset: str | None = typer.Option(None, "--risk-dataset"),
    records: str | None = typer.Option(
        None,
        "--records",
        help="Replay a saved records.csv instead of sampling fresh syndromes. This is "
        "what reproduces the published numbers exactly.",
    ),
    split: str = typer.Option("test", "--split"),
    threshold_split: str = typer.Option("val", "--threshold-split"),
    risk_checkpoint: str = typer.Option("none", "--risk-checkpoint"),
    calibration: str | None = typer.Option(None, "--calibration"),
    out: str = typer.Option("results/runs/paper_suite", "--out"),
    include_ai: bool = typer.Option(False, "--include-ai/--no-include-ai"),
    reuse_main: bool = typer.Option(
        False,
        "--reuse-main/--no-reuse-main",
        help="Reuse an existing main run under --out instead of rerunning it.",
    ),
    run_threshold_sweep: bool = typer.Option(True, "--threshold-sweep/--no-threshold-sweep"),
    risk_thresholds: str = typer.Option("0.25,0.35,0.50,0.65,0.80", "--risk-thresholds"),
    predecode_risk_thresholds: str | None = typer.Option(None, "--predecode-risk-thresholds"),
    scheduler_risk_thresholds: str | None = typer.Option(
        PAPER_SCHEDULER_RISK_THRESHOLDS,
        "--scheduler-risk-thresholds",
        help="Scheduler risk thresholds to sweep. Defaults to the points the paper plots.",
    ),
    confidence_thresholds: str = typer.Option("0.50,0.70,0.85,0.95", "--confidence-thresholds"),
    reuse_completed_thresholds: bool = typer.Option(
        True,
        "--reuse-completed-thresholds/--no-reuse-completed-thresholds",
        help="Reuse completed threshold-sweep subruns when resuming an interrupted suite.",
    ),
) -> None:
    """Run baselines, focused ablations, and optional Pareto threshold sweep."""
    base_cfg = load_config(config)
    run_root = Path(out)
    replay_records = _load_records_csv(Path(records)) if records is not None else None
    main_modes = [
        "accurate_only",
        "fast_only",
        "heuristic_pre_fixed",
        "edf",
        "risk_heuristic",
        "rt_qec",
        "rt_qec_without_validation",
        "rt_qec_without_abstention",
        "rt_qec_without_scheduler",
        "oracle_predecoder",
        "oracle_risk",
    ]
    if include_ai:
        main_modes.insert(-1, "ai_risk")
        main_modes.insert(-1, "rt_qec_ai")
    main_cfg = copy.deepcopy(base_cfg)
    main_cfg.risk_eval.modes = main_modes
    main_metrics_path = run_root / "main" / "metrics.json"
    if reuse_main and main_metrics_path.exists():
        with main_metrics_path.open("r", encoding="utf-8") as handle:
            main_payload = json.load(handle)
    else:
        main_payload = run_real_stream_eval(
            main_cfg,
            risk_checkpoint=None if str(risk_checkpoint).lower() == "none" else risk_checkpoint,
            out_dir=run_root / "main",
            risk_dataset_path=risk_dataset,
            split=split,
            calibration_path=calibration,
            preloaded_records=replay_records,
        )

    sweep_rows: list[dict[str, object]] = []
    if run_threshold_sweep:
        main_records: list[RealStreamShotRecord] | None = replay_records
        main_records_path = run_root / "main" / "records.csv"
        if main_records is not None:
            pass
        elif main_records_path.exists() and str(threshold_split).lower() == str(split).lower():
            main_records = _load_records_csv(main_records_path)
        elif risk_dataset is None:
            threshold_records_path = run_root / f"_threshold_{str(threshold_split).lower()}_records" / "records.csv"
            if threshold_records_path.exists() and reuse_completed_thresholds:
                main_records = _load_records_csv(threshold_records_path)
            else:
                threshold_cfg = copy.deepcopy(base_cfg)
                threshold_cfg.risk_eval.modes = ["rt_qec"]
                threshold_cfg.outputs.save_events = False
                threshold_cfg.outputs.save_decisions = False
                threshold_cfg.outputs.save_predictions = False
                threshold_cfg.outputs.save_plots_ready_csv = False
                run_real_stream_eval(
                    threshold_cfg,
                    risk_checkpoint=None if str(risk_checkpoint).lower() == "none" else risk_checkpoint,
                    out_dir=threshold_records_path.parent,
                    risk_dataset_path=None,
                    split=threshold_split,
                    calibration_path=calibration,
                )
                main_records = _load_records_csv(threshold_records_path)
        predecode_grid = _parse_grid(predecode_risk_thresholds or risk_thresholds)
        scheduler_grid = _parse_grid(scheduler_risk_thresholds or risk_thresholds)
        sweep_modes = ["rt_qec", "rt_qec_ai"] if include_ai else ["rt_qec"]
        for sweep_mode, predecode_risk_threshold, scheduler_risk_threshold, confidence_threshold in itertools.product(
            sweep_modes,
            predecode_grid,
            scheduler_grid,
            _parse_grid(confidence_thresholds),
        ):
            cfg = copy.deepcopy(base_cfg)
            cfg.risk_eval.modes = [sweep_mode]
            cfg.risk_eval.ai_risk_threshold = float(scheduler_risk_threshold)
            cfg.risk_eval.ai_confidence_threshold = float(confidence_threshold)
            cfg.predecoder.risk_threshold = float(predecode_risk_threshold)
            cfg.predecoder.confidence_threshold = float(confidence_threshold)
            cfg.outputs.save_events = False
            cfg.outputs.save_decisions = False
            cfg.outputs.save_predictions = False
            cfg.outputs.save_plots_ready_csv = False
            run_dir = (
                run_root
                / "threshold_sweep"
                / f"{sweep_mode}_pre_{predecode_risk_threshold:.2f}_sched_{scheduler_risk_threshold:.2f}_conf_{confidence_threshold:.2f}"
            )
            summary = _load_completed_threshold_summary(run_dir) if reuse_completed_thresholds else None
            if summary is None:
                if main_records is not None and risk_dataset is None and sweep_mode == "rt_qec":
                    result = evaluate_mode_on_records(main_records, sweep_mode, cfg)
                    summary = dict(result.metrics)
                    pd.DataFrame([summary]).to_csv(ensure_parent(run_dir / "summary_metrics.csv"), index=False)
                else:
                    payload = run_real_stream_eval(
                        cfg,
                        risk_checkpoint=None if str(risk_checkpoint).lower() == "none" else risk_checkpoint,
                        out_dir=run_dir,
                        risk_dataset_path=risk_dataset,
                        split=threshold_split,
                        calibration_path=calibration,
                        preloaded_records=replay_records,
                    )
                    summary = dict(payload.get("summary", [{}])[0])
            sweep_rows.append(
                {
                    "mode": sweep_mode,
                    "predecode_risk_threshold": float(predecode_risk_threshold),
                    "scheduler_risk_threshold": float(scheduler_risk_threshold),
                    "risk_threshold": float(scheduler_risk_threshold),
                    "confidence_threshold": float(confidence_threshold),
                    **summary,
                    "run_dir": str(run_dir),
                }
            )
        _write_table(sweep_rows, run_root / "threshold_sweep.csv")

    manifest = {
        "config": config,
        "risk_dataset": risk_dataset,
        "split": split,
        "threshold_split": threshold_split,
        "risk_checkpoint": None if str(risk_checkpoint).lower() == "none" else risk_checkpoint,
        "calibration": calibration,
        "include_ai": bool(include_ai),
        "reuse_main": bool(reuse_main),
        "main_modes": main_modes,
        "main_summary_path": str(run_root / "main" / "summary_metrics.csv"),
        "main_pareto_path": str(run_root / "main" / "pareto_summary.csv"),
        "main_setting_summary_path": str(run_root / "main" / "setting_summary.csv"),
        "threshold_sweep_path": str(run_root / "threshold_sweep.csv") if run_threshold_sweep else None,
        "num_threshold_runs": len(sweep_rows),
        "seed_list": {
            "config_seed": int(base_cfg.seed),
            "train_seed": int(base_cfg.data_protocol.train_seed),
            "eval_seed": int(base_cfg.data_protocol.eval_seed),
        },
        "software_versions": _software_versions(),
        "git_commit_hash": _git_commit_hash(),
        "git_status_short": _command_output(["git", "status", "--short"]),
        "main_metadata": _compact_metadata(main_payload),
    }
    with ensure_parent(run_root / "suite_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


if __name__ == "__main__":
    app()
