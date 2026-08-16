import numpy as np
import json
import pandas as pd
import torch
import pytest
from pathlib import Path

from rt_preqec.config import ProjectConfig
from rt_preqec.data.risk_dataset import create_split_indices, save_risk_dataset_splits, splits_from_dict
from rt_preqec.evaluation.real_stream import (
    RealStreamShotRecord,
    build_grouped_summary,
    compute_mode_metrics,
    evaluate_mode_on_records,
    _predecode_effect,
    _record_feature_matrix_for_model,
    _predecode_validation_pass,
    run_real_stream_eval,
    simulate_realtime_queue,
)
from rt_preqec.models.risk_profiler import TinyRiskProfiler


def _config() -> ProjectConfig:
    cfg = ProjectConfig()
    cfg.runtime.round_period_us = 1.0
    cfg.runtime.decode_deadline_us = 2.0
    cfg.runtime.max_pauli_frame_lag = 8
    cfg.runtime.logical_boundary_interval = 2
    cfg.risk_eval.ai_risk_threshold = 0.5
    cfg.risk_eval.ai_confidence_threshold = 0.5
    cfg.predecoder.confidence_threshold = 0.7
    cfg.predecoder.risk_threshold = 0.4
    return cfg


def _record(
    shot_id: int,
    risk_label: int,
    accurate_prediction: int,
    fast_prediction: int,
    observable: int,
    accurate_latency_us: float,
    fast_latency_us: float,
) -> RealStreamShotRecord:
    return RealStreamShotRecord(
        shot_id=shot_id,
        syndrome=np.asarray([shot_id % 2, 1], dtype=np.int8),
        observable=np.asarray([observable], dtype=np.int8),
        accurate_prediction=np.asarray([accurate_prediction], dtype=np.int8),
        fast_prediction=np.asarray([fast_prediction], dtype=np.int8),
        accurate_latency_us=accurate_latency_us,
        fast_latency_us=fast_latency_us,
        features=np.asarray([float(shot_id), float(risk_label)], dtype=np.float32),
        feature_names=["weight", "risk_proxy"],
        risk_label=risk_label,
        hard_runtime=int(accurate_latency_us > 2.5),
        fast_wrong_vs_accurate=int(accurate_prediction != fast_prediction),
        fast_logical_fail=int(fast_prediction != observable),
        metadata={"timing_mode": "unit_test"},
    )


def test_modes_select_expected_decoders() -> None:
    cfg = _config()
    records = [
        _record(0, 1, 0, 1, 0, 3.0, 1.0),
        _record(1, 0, 1, 1, 1, 3.5, 0.5),
    ]
    accurate = evaluate_mode_on_records(records, "accurate_only", cfg)
    fast = evaluate_mode_on_records(records, "fast_only", cfg)
    oracle = evaluate_mode_on_records(records, "oracle_risk", cfg)
    oracle_pre = evaluate_mode_on_records(records, "oracle_predecoder", cfg)
    assert set(accurate.decisions["selected_decoder"]) == {"accurate"}
    assert set(fast.decisions["selected_decoder"]) == {"fast"}
    assert oracle.decisions["selected_decoder"].tolist() == ["accurate", "fast"]
    assert oracle_pre.metrics["mean_estimated_residual_reduction"] >= 0.0


def test_rt_qec_mode_reports_predecode_and_lag_fields() -> None:
    cfg = _config()
    records = [
        _record(0, 1, 0, 1, 0, 3.0, 1.0),
        _record(1, 0, 1, 1, 1, 3.5, 0.5),
    ]
    result = evaluate_mode_on_records(records, "rt_qec", cfg)
    assert "predecode_accept_rate" in result.metrics
    assert "accept_rate" in result.metrics
    assert "abstention_rate" in result.metrics
    assert "false_accept_rate" in result.metrics
    assert "accepted_error_rate" in result.metrics
    assert "validation_pass_rate" in result.metrics
    assert "pauli_frame_lag_violation_ratio" in result.metrics
    assert "estimated_residual_reduction" in result.decisions
    assert "boundary_drain" in result.events


def test_heuristic_pre_fixed_reduces_selected_latency() -> None:
    cfg = _config()
    cfg.risk_eval.heuristic_predecoder_backend = "accurate"
    cfg.predecoder.confidence_threshold = 0.4
    cfg.predecoder.risk_threshold = 0.6
    cfg.predecoder.density_risk_scale = 0.0
    record = _record(0, 0, 0, 0, 0, 10.0, 1.0)
    result = evaluate_mode_on_records([record], "heuristic_pre_fixed", cfg)
    selected = float(result.events["latency_us"].iloc[0])
    assert selected < record.accurate_latency_us
    assert result.decisions["predecode_accept_estimate"].iloc[0]


def test_predecode_accept_error_is_separate_from_fast_backend_failure() -> None:
    cfg = _config()
    cfg.risk_eval.heuristic_predecoder_backend = "accurate"
    cfg.runtime.logical_boundary_interval = 100
    cfg.risk_eval.boundary_drain_rounds = 0
    cfg.predecoder.confidence_threshold = 0.1
    cfg.predecoder.risk_threshold = 1.0
    cfg.predecoder.max_cluster_size = 6
    record = _record(0, 0, 0, 1, 0, 10.0, 1.0)
    record.features = np.asarray([1.0, 0.02, 64.0, 0.80, 0.02, 0.10, 2.0], dtype=np.float32)
    record.feature_names = [
        "syndrome_weight",
        "syndrome_density",
        "num_detectors",
        "fraction_active_with_candidate",
        "mean_patch_density",
        "max_patch_density",
        "mean_candidate_count_active",
    ]
    result = evaluate_mode_on_records([record], "heuristic_pre_fixed", cfg)
    assert result.decisions["predecode_accept_estimate"].iloc[0]
    assert result.metrics["accepted_error_rate"] == 0.0
    assert result.metrics["accepted_fast_logical_fail_rate"] == 1.0


def test_without_scheduler_disables_lag_pressure_fast_routing() -> None:
    cfg = _config()
    cfg.runtime.round_period_us = 0.1
    cfg.runtime.decode_deadline_us = 2.0
    cfg.runtime.max_pauli_frame_lag = 1
    cfg.runtime.logical_boundary_interval = 100
    cfg.runtime.overload_backlog_threshold = 2
    cfg.risk_eval.rt_qec_drain_backlog_threshold = 2
    cfg.risk_eval.boundary_drain_rounds = 0
    cfg.risk_eval.ai_risk_threshold = 0.5
    cfg.predecoder.confidence_threshold = 0.1
    cfg.predecoder.risk_threshold = 1.0
    cfg.predecoder.max_cluster_size = 6
    cfg.predecoder.density_risk_scale = 0.0
    records = [_record(shot_id, 1, 0, 0, 0, 5.0, 0.5) for shot_id in range(4)]

    rt_qec = evaluate_mode_on_records(records, "rt_qec", cfg)
    without_scheduler = evaluate_mode_on_records(records, "rt_qec_without_scheduler", cfg)

    assert rt_qec.metrics["fast_selection_rate"] > without_scheduler.metrics["fast_selection_rate"]
    assert "drain_or_overload_fast" in set(rt_qec.decisions["selection_reason"])
    assert "drain_or_overload_fast_no_lag_scheduler" not in set(without_scheduler.decisions["selection_reason"])


def test_edf_prefers_accurate_when_deadline_feasible() -> None:
    cfg = _config()
    cfg.runtime.decode_deadline_us = 10.0
    record = _record(0, 0, 0, 1, 0, 3.0, 1.0)
    result = evaluate_mode_on_records([record], "edf", cfg)
    assert result.decisions["selected_decoder"].iloc[0] == "accurate"
    assert result.metrics["logical_error_rate"] == 0.0


def test_distance_aware_certificate_accepts_sparse_large_distance_record() -> None:
    cfg = _config()
    cfg.qec.distances = [11]
    cfg.predecoder.max_cluster_size = 12
    record = _record(0, 0, 0, 0, 0, 10.0, 1.0)
    record.metadata["distance"] = 11
    record.features = np.asarray([36.0, 0.027, 1320.0, 0.20, 0.20, 0.50, 12.0], dtype=np.float32)
    record.feature_names = [
        "syndrome_weight",
        "syndrome_density",
        "num_detectors",
        "fraction_active_with_candidate",
        "mean_patch_density",
        "max_patch_density",
        "mean_candidate_count_active",
    ]
    assert _predecode_validation_pass(record, cfg)


def test_weak_certificate_shapes_without_fast_certification() -> None:
    cfg = _config()
    cfg.predecoder.confidence_threshold = 0.10
    cfg.predecoder.risk_threshold = 1.0
    cfg.predecoder.max_cluster_size = 6
    record = _record(0, 0, 0, 1, 0, 10.0, 1.0)
    record.features = np.asarray([9.0, 0.09, 100.0, 0.40, 0.30, 0.60, 30.0], dtype=np.float32)
    record.feature_names = [
        "syndrome_weight",
        "syndrome_density",
        "num_detectors",
        "fraction_active_with_candidate",
        "mean_patch_density",
        "max_patch_density",
        "mean_candidate_count_active",
    ]
    effect = _predecode_effect(record, cfg)
    assert effect["weak_validation_pass_estimate"]
    assert effect["predecode_accept_estimate"]
    assert not effect["fast_path_certified"]

    result = evaluate_mode_on_records([record], "rt_qec", cfg)
    decision = result.decisions.iloc[0]
    assert bool(decision["predecode_accept_estimate"])
    assert not bool(decision["fast_path_certified"])
    assert decision["selected_decoder"] == "accurate"
    assert decision["shaped_accurate_latency_us"] < record.accurate_latency_us


def test_model_feature_matrix_aligns_checkpoint_names() -> None:
    record = _record(0, 0, 0, 0, 0, 1.0, 1.0)
    record.features = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    record.feature_names = ["a", "extra", "b"]
    matrix = _record_feature_matrix_for_model(
        [record],
        {"feature_names": ["b", "missing", "a"]},
        {"mean": np.zeros(3, dtype=np.float32), "std": np.ones(3, dtype=np.float32)},
    )
    np.testing.assert_allclose(matrix, np.asarray([[3.0, 0.0, 1.0]], dtype=np.float32))


def test_simulate_realtime_queue_computes_deadlines_and_backlog() -> None:
    cfg = _config()
    records = [
        _record(0, 1, 0, 1, 0, 3.0, 1.0),
        _record(1, 0, 1, 1, 1, 3.5, 2.5),
        _record(2, 0, 0, 0, 0, 0.5, 0.5),
    ]
    events = simulate_realtime_queue(
        records,
        selected_latencies=[3.0, 2.5, 0.5],
        selected_predictions=[record.accurate_prediction for record in records],
        config=cfg,
        selected_decoders=["accurate", "fast", "fast"],
    )
    assert events["deadline_miss"].any()
    assert events["backlog"].max() >= 2
    assert "pauli_frame_lag_violation" in events


def test_simulate_realtime_queue_uses_multiple_workers() -> None:
    cfg = _config()
    cfg.runtime.num_workers = 2
    records = [
        _record(0, 1, 0, 1, 0, 4.0, 1.0),
        _record(1, 0, 1, 1, 1, 4.0, 1.0),
    ]
    events = simulate_realtime_queue(
        records,
        selected_latencies=[4.0, 4.0],
        selected_predictions=[record.accurate_prediction for record in records],
        config=cfg,
        selected_decoders=["accurate", "accurate"],
    )
    assert set(events["worker_id"].tolist()) == {0, 1}


def test_compute_mode_metrics_has_required_fields() -> None:
    cfg = _config()
    result = evaluate_mode_on_records(
        [
            _record(0, 1, 0, 1, 0, 3.0, 1.0),
            _record(1, 0, 1, 1, 1, 3.5, 0.5),
        ],
        "fast_only",
        cfg,
    )
    metrics = compute_mode_metrics(result.events, result.predictions, result.decisions, cfg)
    assert "logical_error_rate" in metrics
    assert "p99_latency_us" in metrics
    assert "accept_rate" in metrics
    assert "accepted_error_rate" in metrics
    assert "validation_pass_rate" in metrics
    assert metrics["fast_selection_rate"] == 1.0


def test_grouped_summary_uses_record_metadata() -> None:
    cfg = _config()
    records = [
        _record(0, 1, 0, 1, 0, 3.0, 1.0),
        _record(1, 0, 1, 1, 1, 3.5, 0.5),
    ]
    for idx, record in enumerate(records):
        record.metadata["setting_id"] = idx % 2
        record.metadata["noise_scenario"] = "unit"
    result = evaluate_mode_on_records(records, "rt_qec", cfg)
    records_frame = pd.DataFrame(
        [
            {
                "shot_id": record.shot_id,
                "setting_id": record.metadata["setting_id"],
                "noise_scenario": record.metadata["noise_scenario"],
            }
            for record in records
        ]
    )
    summary = build_grouped_summary([result], records_frame)
    assert set(summary["setting_id"].tolist()) == {0, 1}
    assert "p99_response_time_us" in summary


def _write_eval_dataset(path: Path, n: int = 10, f: int = 2) -> None:
    features = np.random.default_rng(0).normal(size=(n, f)).astype(np.float32)
    labels = np.zeros((n, 5), dtype=np.int8)
    labels[:, 4] = np.arange(n) % 2
    labels[:, 3] = labels[:, 4]
    metadata = {
        "num_samples": n,
        "hard_runtime_label_valid": True,
        "samples": [
            {
                "sample_id": idx,
                "shot_id": idx,
                "metadata": {"timing_mode": "loop_per_shot", "hard_runtime_label_valid": True, "toy": True},
            }
            for idx in range(n)
        ],
    }
    np.savez_compressed(
        path,
        features=features,
        syndromes=np.zeros((n, 2), dtype=np.int8),
        actual_observables=np.zeros((n, 1), dtype=np.int8),
        accurate_predictions=np.zeros((n, 1), dtype=np.int8),
        fast_predictions=np.ones((n, 1), dtype=np.int8),
        labels=labels,
        label_names=np.asarray(
            [
                "fast_wrong_vs_accurate",
                "fast_logical_fail",
                "accurate_logical_fail",
                "hard_runtime",
                "scheduler_risk_label",
            ],
            dtype="<U64",
        ),
        runtimes=np.stack([np.full(n, 3.0), np.full(n, 1.0)], axis=1).astype(np.float32),
        runtime_names=np.asarray(["accurate_runtime_us", "fast_runtime_us"], dtype="<U64"),
        feature_names=np.asarray(["f0", "f1"], dtype="<U64"),
        metadata_json=json.dumps(metadata),
    )
    splits = splits_from_dict(create_split_indices(n, "stream_block", 0.5, 0.2, 0.3, seed=42))
    save_risk_dataset_splits(splits, path.with_name("risk_dataset_splits.json"))


def _write_eval_checkpoint(path: Path) -> None:
    model = TinyRiskProfiler(input_dim=2, hidden_dim=4, num_layers=1)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": 2,
            "feature_names": ["f0", "f1"],
            "normalization": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
            "model_hparams": {"hidden_dim": 4, "num_layers": 1, "dropout": 0.0},
            "train_indices_hash": "different",
        },
        path,
    )


def test_real_stream_eval_loads_risk_dataset_test_split(tmp_path: Path) -> None:
    data = tmp_path / "risk_dataset.npz"
    ckpt = tmp_path / "risk.pt"
    _write_eval_dataset(data)
    _write_eval_checkpoint(ckpt)
    cfg = _config()
    cfg.risk_eval.modes = ["accurate_only", "ai_risk"]
    payload = run_real_stream_eval(cfg, risk_checkpoint=str(ckpt), out_dir=tmp_path / "out", risk_dataset_path=data, split="test")
    assert payload["eval_source"] == "risk_dataset"
    assert payload["eval_split"] == "test"
    assert payload["metadata"]["split_policy"] == "stream_block"
    assert payload["metadata"]["split_match"] is False
    assert payload["warnings"]


def test_generated_eval_requires_different_seed(tmp_path: Path) -> None:
    cfg = _config()
    cfg.qec.num_shots = 4
    cfg.risk_eval.modes = ["accurate_only"]
    with pytest.raises(ValueError):
        run_real_stream_eval(cfg, out_dir=tmp_path / "out", train_seed=42, eval_seed=42)


def test_real_stream_eval_uses_calibration_thresholds(tmp_path: Path) -> None:
    data = tmp_path / "risk_dataset.npz"
    ckpt = tmp_path / "risk.pt"
    cal = tmp_path / "cal.json"
    _write_eval_dataset(data)
    _write_eval_checkpoint(ckpt)
    cal.write_text(
        json.dumps({"selected_ai_risk_threshold": 0.25, "selected_ai_confidence_threshold": 0.75}),
        encoding="utf-8",
    )
    cfg = _config()
    cfg.risk_eval.modes = ["ai_risk"]
    payload = run_real_stream_eval(
        cfg,
        risk_checkpoint=str(ckpt),
        out_dir=tmp_path / "out_cal",
        risk_dataset_path=data,
        split="test",
        calibration_path=cal,
    )
    assert payload["metadata"]["calibration_source"] == str(cal)
