"""Measure RTSS frontend, validation, scheduler, and backend overheads."""

from __future__ import annotations

from pathlib import Path
import json
import os
import platform
import statistics
import sys
import time

import numpy as np
import pandas as pd
import typer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rt_preqec.config import load_config
from rt_preqec.evaluation.real_stream import (
    _choose_rt_qec_decoder,
    _heuristic_risk_scores,
    _latency_with_predecode,
    _predecode_effect,
    _predecode_validation_pass,
    _queue_context,
    build_real_stream_records,
)
from rt_preqec.utils import ensure_parent

app = typer.Typer(add_completion=False)


def _pin_cpu(cpu: int | None) -> bool:
    if cpu is None:
        return False
    try:
        os.sched_setaffinity(0, {int(cpu)})  # type: ignore[attr-defined]
        return True
    except Exception:
        return False


def _set_thread_env() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _measure_call(fn, repeats: int = 1) -> float:
    start = time.perf_counter_ns()
    for _ in range(max(int(repeats), 1)):
        fn()
    end = time.perf_counter_ns()
    return float(end - start) / 1000.0 / float(max(int(repeats), 1))


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean_us": 0.0, "median_us": 0.0, "p99_us": 0.0, "p999_us": 0.0, "max_us": 0.0}
    array = np.asarray(values, dtype=float)
    return {
        "mean_us": float(np.mean(array)),
        "median_us": float(statistics.median(array.tolist())),
        "p99_us": float(np.percentile(array, 99)),
        "p999_us": float(np.percentile(array, 99.9)),
        "max_us": float(np.max(array)),
    }


@app.command()
def main(
    config: str = typer.Option("configs/real_stream_eval_main.yaml", "--config"),
    num_shots: int = typer.Option(20000, "--num-shots"),
    warmup_shots: int = typer.Option(1000, "--warmup-shots"),
    pin_cpu: int | None = typer.Option(None, "--pin-cpu"),
    out: str = typer.Option("results/runs/wcet_overheads_d7", "--out"),
) -> None:
    """Measure isolated Python overheads and write raw traces plus summary tables."""
    _set_thread_env()
    pinned = _pin_cpu(pin_cpu)
    cfg = load_config(config)
    cfg.qec.num_shots = max(int(num_shots), 1)
    cfg.timing.warmup_shots = min(max(int(warmup_shots), 0), max(int(num_shots) - 1, 0))
    cfg.timing.max_timing_shots = cfg.qec.num_shots

    records, metadata = build_real_stream_records(cfg)
    eval_records = records[cfg.timing.warmup_shots :]
    heuristic_scores = _heuristic_risk_scores(eval_records)

    frontend_values: list[float] = []
    validation_values: list[float] = []
    scheduler_values: list[float] = []
    fast_backend_values: list[float] = []
    accurate_backend_values: list[float] = []
    raw_rows: list[dict[str, float | int]] = []

    worker_available_times = [0.0 for _ in range(max(int(cfg.runtime.num_workers), 1))]
    finish_times: list[float] = []
    effects = []
    contexts = []
    for idx, record in enumerate(eval_records):
        frontend_us = _measure_call(lambda record=record: _predecode_effect(record, cfg))
        effect = _predecode_effect(record, cfg)
        validation_us = _measure_call(lambda record=record: _predecode_validation_pass(record, cfg))
        context = _queue_context(record, worker_available_times, finish_times, cfg)
        shaped_accurate = _latency_with_predecode(record.accurate_latency_us, effect, cfg)
        shaped_fast = _latency_with_predecode(record.fast_latency_us, effect, cfg)
        score = float(heuristic_scores[idx]) if idx < len(heuristic_scores) else 0.0
        scheduler_us = _measure_call(
            lambda record=record, score=score, shaped_accurate=shaped_accurate, shaped_fast=shaped_fast, context=context: _choose_rt_qec_decoder(
                record,
                score,
                shaped_accurate,
                shaped_fast,
                context,
                cfg,
            )
        )
        selected_decoder, _ = _choose_rt_qec_decoder(record, score, shaped_accurate, shaped_fast, context, cfg)
        selected_latency = shaped_accurate if selected_decoder == "accurate" else shaped_fast
        worker_idx = min(range(len(worker_available_times)), key=lambda value: worker_available_times[value])
        arrival_time = float(record.shot_id) * float(cfg.runtime.round_period_us)
        start_time = max(arrival_time, worker_available_times[worker_idx])
        finish_time = start_time + selected_latency
        worker_available_times[worker_idx] = finish_time
        finish_times.append(finish_time)

        frontend_values.append(frontend_us)
        validation_values.append(validation_us)
        scheduler_values.append(scheduler_us)
        fast_backend_values.append(float(record.fast_latency_us))
        accurate_backend_values.append(float(record.accurate_latency_us))
        effects.append(effect)
        contexts.append(context)
        raw_rows.append(
            {
                "shot_id": int(record.shot_id),
                "frontend_us": frontend_us,
                "validation_us": validation_us,
                "scheduler_us": scheduler_us,
                "fast_backend_us": float(record.fast_latency_us),
                "accurate_backend_us": float(record.accurate_latency_us),
                "predecode_latency_model_us": float(effect.get("predecode_latency_us", 0.0)),
                "selected_backend_latency_model_us": float(selected_latency),
            }
        )

    summary_rows = []
    for component, values in [
        ("frontend", frontend_values),
        ("validation", validation_values),
        ("scheduler", scheduler_values),
        ("fast_backend", fast_backend_values),
        ("accurate_backend", accurate_backend_values),
    ]:
        row = {"component": component, **_summary(values)}
        summary_rows.append(row)

    out_path = Path(out)
    pd.DataFrame(raw_rows).to_csv(ensure_parent(out_path / "overhead_trace.csv"), index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(ensure_parent(out_path / "overhead_summary.csv"), index=False)

    protocol = {
        "config": config,
        "num_shots_requested": int(num_shots),
        "warmup_shots": int(warmup_shots),
        "num_measured_shots": len(eval_records),
        "pin_cpu": pin_cpu,
        "cpu_pinned": bool(pinned),
        "thread_env": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "timing_mode": metadata.get("timing_mode"),
        "real_qec": metadata.get("real_qec"),
        "hard_runtime_label_valid": metadata.get("hard_runtime_label_valid"),
        "notes": [
            "frontend/validation/scheduler columns are isolated Python microbenchmarks",
            "backend columns are empirical per-record decoder timing labels",
        ],
    }
    with ensure_parent(out_path / "measurement_protocol.json").open("w", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2)


if __name__ == "__main__":
    app()
