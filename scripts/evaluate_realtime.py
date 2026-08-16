"""Evaluate realtime scheduling modes, including AI risk scheduling."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
import typer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rt_preqec.config import load_config
from rt_preqec.data.dataset import ArrayPredecoderDataset, load_patch_dataset
from rt_preqec.data.dem_parser import index_candidates_by_detector, parse_dem_error_candidates
from rt_preqec.data.risk_dataset import load_risk_dataset
from rt_preqec.data.stim_surface_code import generate_surface_code_samples
from rt_preqec.decoders.lookup_decoder import LookupDecoder
from rt_preqec.decoders.pymatching_decoder import PyMatchingDecoder
from rt_preqec.decoders.union_find_decoder import UnionFindDecoder
from rt_preqec.logging_utils import configure_logging, get_logger
from rt_preqec.metrics.aggregation import save_metrics_csv, save_metrics_json
from rt_preqec.models.predecoder import TinyNeuralPredecoder
from rt_preqec.models.risk_profiler import TinyRiskProfiler
from rt_preqec.predecode.selective_predecoder import SelectivePredecoder
from rt_preqec.runtime.pipeline import RTPreQECPipeline
from rt_preqec.runtime.stream_simulator import SyndromeStreamSimulator
from rt_preqec.scheduler.lag_scheduler import LagBoundedScheduler
from rt_preqec.utils import ensure_parent

app = typer.Typer(add_completion=False)
logger = get_logger(__name__)


def _load_predecoder(cfg: Any, checkpoint: str) -> Any | None:
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        return None
    payload = torch.load(checkpoint_path, map_location=cfg.device)
    checkpoint_cfg = payload.get("config", {})
    predecoder_cfg = checkpoint_cfg.get("predecoder", {}) if isinstance(checkpoint_cfg, dict) else {}
    model_cfg = checkpoint_cfg.get("model", {}) if isinstance(checkpoint_cfg, dict) else {}
    model = TinyNeuralPredecoder(
        temporal_window=int(predecoder_cfg.get("temporal_window", cfg.predecoder.temporal_window)),
        patch_size=int(predecoder_cfg.get("patch_size", cfg.predecoder.patch_size)),
        hidden_channels=int(model_cfg.get("hidden_channels", cfg.model.hidden_channels)),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(cfg.device)
    return model


def _load_risk_profiler(cfg: Any, checkpoint: str | None) -> tuple[Any | None, dict[str, Any] | None]:
    if checkpoint is None:
        return None, None
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        return None, None
    payload = torch.load(checkpoint_path, map_location=cfg.device)
    model = TinyRiskProfiler(
        input_dim=int(payload.get("input_dim", len(payload.get("feature_names", [])))),
        hidden_dim=cfg.risk_model.hidden_dim,
        num_layers=cfg.risk_model.num_layers,
        dropout=cfg.risk_model.dropout,
    )
    model.load_state_dict(payload["state_dict"])
    model.to(cfg.device)
    return model, payload.get("normalization")


def _build_decoders(bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "lookup": LookupDecoder(),
        "union_find": UnionFindDecoder(),
        "pymatching": PyMatchingDecoder.from_detector_error_model(bundle.get("dem")),
    }


def _risk_feature_extractor(syndrome: np.ndarray, **kwargs: Any) -> tuple[np.ndarray, list[str]]:
    from rt_preqec.data.risk_features import combine_feature_blocks, extract_patch_aggregate_features, extract_syndrome_features

    patches = kwargs.get("patches", [])
    layout = kwargs.get("layout")
    candidates_by_detector = kwargs.get("candidates_by_detector")
    syndrome_features, syndrome_names = extract_syndrome_features(
        syndrome,
        layout=layout,
        candidates_by_detector=candidates_by_detector,
    )
    patch_features, patch_names = extract_patch_aggregate_features(
        [patch for patch in patches if hasattr(patch, "detector_ids")]
    )
    return combine_feature_blocks((syndrome_features, syndrome_names), (patch_features, patch_names))


@app.command()
def main(
    config: str = "configs/eval_realtime.yaml",
    data: str = "data/processed/predecoder_dataset_v1_300k.npz",
    checkpoint: str = "checkpoints/predecoder_v1_300k.pt",
    risk_checkpoint: str | None = None,
    mode: str = "full",
    out: str = "results/runs/smoke_eval",
) -> None:
    """Run realtime evaluation for baseline and AI risk modes."""
    cfg = load_config(config)
    configure_logging(cfg.log_level)
    bundle = generate_surface_code_samples(cfg)
    decoders = _build_decoders(bundle)
    predecoder_model = _load_predecoder(cfg, checkpoint)
    risk_model, risk_normalization = _load_risk_profiler(cfg, risk_checkpoint)

    if mode == "accurate_only":
        cfg.decoders.fast = cfg.decoders.accurate
        cfg.scheduler.policy = "fifo"
        cfg.scheduler.use_ai_risk = False
    elif mode == "fast_only":
        cfg.decoders.accurate = cfg.decoders.fast
        cfg.scheduler.policy = "fifo"
        cfg.scheduler.use_ai_risk = False
    elif mode == "edf":
        cfg.scheduler.policy = "edf"
        cfg.scheduler.use_ai_risk = False
    elif mode == "risk_heuristic":
        cfg.scheduler.policy = "risk_aware_edf"
        cfg.scheduler.use_ai_risk = False
    elif mode == "ai_risk":
        cfg.scheduler.policy = "risk_aware_edf"
        cfg.scheduler.use_ai_risk = True
    elif mode == "full":
        cfg.scheduler.policy = "risk_aware_edf"
    else:
        raise typer.BadParameter(f"Unsupported mode: {mode}")

    predecoder_mode = "risk_only" if mode == "ai_risk" else ("candidate" if bundle.get("layout") is not None else "toy")
    predecoder = SelectivePredecoder(
        model=predecoder_model if mode == "full" else None,
        confidence_threshold=cfg.predecoder.confidence_threshold,
        risk_threshold=cfg.predecoder.risk_threshold,
        correction_threshold=cfg.predecoder.correction_threshold,
        enable_validation=cfg.predecoder.enable_validation,
        enable_abstention=cfg.predecoder.enable_abstention,
        device=cfg.device,
        layout=bundle.get("layout"),
        candidates_by_detector=index_candidates_by_detector(parse_dem_error_candidates(bundle.get("dem"), layout=bundle.get("layout"))),
        mode=predecoder_mode,
    )
    scheduler = LagBoundedScheduler(cfg)
    pipeline = RTPreQECPipeline(
        cfg,
        predecoder,
        scheduler,
        decoders,
        risk_profiler_model=risk_model if mode == "ai_risk" else None,
        risk_feature_extractor=_risk_feature_extractor,
        risk_normalization=risk_normalization,
    )

    simulator = SyndromeStreamSimulator(cfg.runtime.round_period_us, cfg.runtime.decode_deadline_us)
    stream = None
    data_path = Path(data)
    if data_path.exists() and data_path.suffix == ".npz":
        try:
            patch_dataset = ArrayPredecoderDataset(data_path, split="test")
            sample_count = min(len(patch_dataset), 64)

            def _patch_stream() -> Any:
                for idx in range(sample_count):
                    yield np.asarray(patch_dataset[idx]["patch"].numpy().squeeze(0), dtype=np.int8)

            stream = simulator.from_syndromes(_patch_stream())
        except Exception:
            try:
                risk_samples = load_risk_dataset(data_path)

                def _risk_stream() -> Any:
                    for sample, event in zip(
                        risk_samples,
                        simulator.from_flat_syndromes(
                            [sample.syndrome for sample in risk_samples],
                            layout=bundle.get("layout"),
                            extra_metadata={
                                "patch_radius": 2.5,
                                "time_radius": 1.0,
                                "candidates_by_detector": predecoder.candidates_by_detector,
                            },
                        ),
                    ):
                        event.metadata["actual_observable"] = sample.actual_observable
                        yield event

                stream = _risk_stream()
            except Exception:
                patch_samples = load_patch_dataset(data_path)
                stream = simulator.from_dataset(patch_samples[: min(len(patch_samples), 64)])
    if stream is None:
        syndrome = np.asarray(bundle["syndrome"], dtype=np.int8)
        if syndrome.ndim == 2:
            observables = np.asarray(bundle["observables"], dtype=np.int8)
            def _bundle_stream() -> Any:
                for idx, event in enumerate(
                    simulator.from_flat_syndromes(
                        syndrome,
                        layout=bundle.get("layout"),
                        extra_metadata={
                            "patch_radius": 2.5,
                            "time_radius": 1.0,
                            "candidates_by_detector": predecoder.candidates_by_detector,
                        },
                    )
                ):
                    event.metadata["actual_observable"] = observables[idx]
                    yield event
            stream = _bundle_stream()
        else:
            stream = simulator.from_syndromes(syndrome[: min(len(syndrome), 64)])

    metrics = pipeline.run_stream(stream)
    lat_summary = pipeline.latency_stats.summary()
    metrics_flat = {
        "logical_error_rate_placeholder": metrics.get("logical_error_rate", 0.0),
        "logical_error_rate": metrics.get("logical_error_rate", 0.0),
        "mean_latency_us": float(np.mean(pipeline.state.latencies_us)) if pipeline.state.latencies_us else 0.0,
        "p95_latency_us": lat_summary["p95"],
        "p99_latency_us": lat_summary["p99"],
        "p999_latency_us": lat_summary["p999"],
        "deadline_miss_ratio": metrics["deadline_miss_ratio"],
        "max_pauli_frame_lag": metrics["max_pauli_frame_lag"],
        "average_pauli_frame_lag": metrics["average_pauli_frame_lag"],
        "mean_backlog": metrics["backlog"]["mean"] if isinstance(metrics.get("backlog"), dict) else 0.0,
        "max_backlog": metrics["backlog"]["max"] if isinstance(metrics.get("backlog"), dict) else 0.0,
        "fast_selection_rate": metrics.get("fast_selection_rate", 0.0),
        "accurate_selection_rate": metrics.get("accurate_selection_rate", 0.0),
        "ai_risk_mean": metrics.get("ai_risk_mean", 0.0),
        "ai_confidence_mean": metrics.get("ai_confidence_mean", 0.0),
        "mode": mode,
    }
    run_dir = Path(out)
    save_metrics_json(metrics_flat, run_dir / "metrics.json")
    save_metrics_csv(metrics_flat, run_dir / "metrics.csv")
    pipeline.profiler.save_csv(run_dir / "events.csv")
    pd.DataFrame([{"latency_us": value} for value in pipeline.state.latencies_us]).to_csv(
        ensure_parent(run_dir / "latency.csv"),
        index=False,
    )
    pd.DataFrame(pipeline.profiler.events).to_csv(ensure_parent(run_dir / "scheduler_decisions.csv"), index=False)
    logger.info("saved evaluation outputs to %s", out)


if __name__ == "__main__":
    app()
