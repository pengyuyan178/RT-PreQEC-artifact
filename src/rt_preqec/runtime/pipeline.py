"""End-to-end RT-PreQEC runtime pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from rt_preqec.config import ProjectConfig
from rt_preqec.data.patch_extractor import extract_detector_patches_from_flat_syndrome, extract_local_patches
from rt_preqec.data.risk_features import combine_feature_blocks, extract_patch_aggregate_features, extract_syndrome_features
from rt_preqec.data.schemas import StreamEvent
from rt_preqec.decoders.base import BaseDecoder
from rt_preqec.metrics.aggregation import aggregate_run_metrics
from rt_preqec.metrics.qec_metrics import logical_error_rate
from rt_preqec.metrics.predecoder_metrics import (
    accept_rate,
    residual_density_reduction,
    residual_graph_size_reduction,
    validation_pass_rate,
)
from rt_preqec.metrics.realtime_metrics import (
    average_pauli_frame_lag,
    backlog_stats,
    deadline_miss_ratio,
    latency_percentiles,
    max_pauli_frame_lag,
)
from rt_preqec.predecode.residual import (
    apply_candidates_to_flat_syndrome,
    compute_flat_residual_stats,
    compute_residual_density,
    remove_accepted_clusters,
)
from rt_preqec.predecode.selective_predecoder import SelectivePredecoder
from rt_preqec.runtime.profiler import RuntimeProfiler
from rt_preqec.runtime.timing import RunningLatencyStats
from rt_preqec.scheduler.job import DecodingJob
from rt_preqec.scheduler.lag_scheduler import LagBoundedScheduler
from rt_preqec.scheduler.policies import risk_aware_edf_priority
from rt_preqec.scheduler.queue import DecoderJobQueue


@dataclass
class PipelineState:
    """Mutable runtime state."""

    pauli_frame_lag: int = 0
    backlog_history: list[int] = field(default_factory=list)
    lag_history: list[int] = field(default_factory=list)
    latencies_us: list[float] = field(default_factory=list)
    deadlines_us: list[float] = field(default_factory=list)
    accepted_masks: list[np.ndarray] = field(default_factory=list)
    validation_masks: list[np.ndarray] = field(default_factory=list)
    predicted_observables: list[np.ndarray] = field(default_factory=list)
    actual_observables: list[np.ndarray] = field(default_factory=list)
    selected_decoders: list[str] = field(default_factory=list)
    ai_risk_scores: list[float] = field(default_factory=list)
    ai_confidences: list[float] = field(default_factory=list)


class RTPreQECPipeline:
    """Runnable RT-PreQEC pipeline with toy and layout-aware paths."""

    def __init__(
        self,
        config: ProjectConfig,
        predecoder: SelectivePredecoder,
        scheduler: LagBoundedScheduler,
        decoders: dict[str, BaseDecoder],
        risk_profiler_model: Any | None = None,
        risk_feature_extractor: Any | None = None,
        risk_normalization: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.predecoder = predecoder
        self.scheduler = scheduler
        self.decoders = decoders
        self.risk_profiler_model = risk_profiler_model
        self.risk_feature_extractor = risk_feature_extractor
        self.risk_normalization = risk_normalization or {}
        self.profiler = RuntimeProfiler()
        self.latency_stats = RunningLatencyStats()
        self.job_queue = DecoderJobQueue()
        self.state = PipelineState()

    def _extract_risk_features(
        self,
        syndrome: np.ndarray,
        layout: Any | None,
        patches: list[Any],
        candidates_by_detector: dict[int, Any] | None,
    ) -> tuple[np.ndarray, list[str]]:
        if callable(self.risk_feature_extractor):
            return self.risk_feature_extractor(syndrome, layout=layout, patches=patches, candidates_by_detector=candidates_by_detector)
        syndrome_features, syndrome_names = extract_syndrome_features(
            syndrome,
            layout=layout,
            candidates_by_detector=candidates_by_detector,
        )
        patch_features, patch_names = extract_patch_aggregate_features(
            [patch for patch in patches if hasattr(patch, "detector_ids")]
        )
        return combine_feature_blocks((syndrome_features, syndrome_names), (patch_features, patch_names))

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.risk_normalization.get("mean", []), dtype=np.float32)
        std = np.asarray(self.risk_normalization.get("std", []), dtype=np.float32)
        if mean.size == 0 or std.size == 0 or mean.shape != features.shape or std.shape != features.shape:
            return features.astype(np.float32)
        return ((features.astype(np.float32) - mean) / np.where(std > 1e-6, std, 1.0)).astype(np.float32)

    def _infer_risk_scores(self, features: np.ndarray) -> tuple[float | None, float | None, float | None]:
        if self.risk_profiler_model is None:
            return None, None, None
        try:
            import torch
        except ImportError:  # pragma: no cover
            return None, None, None
        self.risk_profiler_model.eval()
        with torch.no_grad():
            batch = torch.tensor(self._normalize_features(features)[None, :], dtype=torch.float32)
            outputs = self.risk_profiler_model.predict_proba(batch)
        risk_score = float(outputs["risk_score"].cpu().numpy().reshape(-1)[0])
        confidence = float(outputs["confidence"].cpu().numpy().reshape(-1)[0])
        runtime_score = float(np.expm1(outputs["runtime_pred"].cpu().numpy().reshape(-1)[0]))
        return risk_score, runtime_score, confidence

    def _has_layout_path(self, event: StreamEvent) -> bool:
        return (
            event.metadata.get("layout") is not None
            and np.asarray(event.syndrome).ndim == 1
            and self.predecoder.mode in {"candidate", "risk_only"}
        )

    def process_event(self, event: StreamEvent) -> dict[str, Any]:
        """Process one stream event."""
        layout = event.metadata.get("layout")
        candidates_by_detector = event.metadata.get("candidates_by_detector")
        if self._has_layout_path(event):
            layout = event.metadata["layout"]
            patches = extract_detector_patches_from_flat_syndrome(
                np.asarray(event.syndrome, dtype=np.int8),
                layout=layout,
                patch_radius=float(event.metadata.get("patch_radius", 2.5)),
                time_radius=event.metadata.get("time_radius"),
                active_only=True,
                max_patches=event.metadata.get("max_patches"),
                shot_id=event.metadata.get("shot_id"),
            )
            predecode_result = self.predecoder.run(patches)
            accepted_candidates = predecode_result.accepted_candidates or []
            residual = (
                apply_candidates_to_flat_syndrome(np.asarray(event.syndrome, dtype=np.int8), accepted_candidates)
                if len(accepted_candidates) > 0
                else np.asarray(event.syndrome, dtype=np.int8).copy()
            )
            residual_stats = compute_flat_residual_stats(np.asarray(event.syndrome, dtype=np.int8), residual)
            residual_density = residual_stats["residual_weight"] / max(residual_stats["original_weight"], 1.0)
        else:
            patches = extract_local_patches(
                event.syndrome,
                patch_size=self.config.predecoder.patch_size,
                temporal_window=self.config.predecoder.temporal_window,
            )
            predecode_result = self.predecoder.run(patches)
            locations = [tuple(item["location"]) for item in patches]
            residual = remove_accepted_clusters(
                event.syndrome,
                list(predecode_result.local_corrections),
                locations,
                predecode_result.accepted_mask,
            )
            residual_density = compute_residual_density(event.syndrome, residual)
        feature_vector, feature_names = self._extract_risk_features(
            np.asarray(residual, dtype=np.int8).reshape(-1),
            layout=layout,
            patches=patches,
            candidates_by_detector=candidates_by_detector,
        )
        ai_risk_score, ai_runtime_score, ai_confidence = self._infer_risk_scores(feature_vector)

        job = DecodingJob(
            job_id=event.event_id,
            syndrome=residual,
            created_us=event.timestamp_us,
            deadline_us=event.deadline_us,
            risk_score=float(np.mean(predecode_result.risk)) if len(predecode_result.risk) else 0.0,
            predicted_runtime_us=float(np.count_nonzero(residual) + 1),
            logical_boundary=event.logical_boundary,
            ai_risk_score=ai_risk_score,
            ai_runtime_score=ai_runtime_score,
            ai_confidence=ai_confidence,
            feature_vector=feature_vector,
            metadata={
                "current_lag": self.state.pauli_frame_lag,
                "feature_names": feature_names,
            },
        )
        priority = risk_aware_edf_priority(
            job,
            now_us=event.timestamp_us,
            alpha=self.config.scheduler.alpha_urgency,
            beta=self.config.scheduler.beta_risk,
            gamma=self.config.scheduler.gamma_runtime,
            delta=self.config.scheduler.delta_boundary,
        )
        self.job_queue.push(job, priority)
        decision = self.scheduler.schedule(
            self.job_queue,
            now_us=event.timestamp_us,
            decoders=self.decoders,
            runtime_state={"backlog": len(self.job_queue), "pauli_frame_lag": self.state.pauli_frame_lag},
        )
        if decision is None:
            raise RuntimeError("Scheduler returned no decision for a non-empty queue.")
        decoder = self.decoders[decision.decoder_name]
        decode_result = decoder.decode(decision.job.syndrome)
        self.latency_stats.update(decode_result.latency_us)
        self.state.latencies_us.append(decode_result.latency_us)
        self.state.deadlines_us.append(max(event.deadline_us - event.timestamp_us, 0.0))
        self.state.pauli_frame_lag = max(
            0,
            self.state.pauli_frame_lag + int(decode_result.latency_us > self.config.runtime.decode_deadline_us) - 1,
        )
        self.state.backlog_history.append(len(self.job_queue))
        self.state.lag_history.append(self.state.pauli_frame_lag)
        self.state.accepted_masks.append(predecode_result.accepted_mask)
        self.state.validation_masks.append(predecode_result.validation_pass)
        self.state.selected_decoders.append(decision.decoder_name)
        self.state.ai_risk_scores.append(float(ai_risk_score or 0.0))
        self.state.ai_confidences.append(float(ai_confidence or 0.0))
        predicted_observable = np.asarray(decode_result.metadata.get("predicted_observable", decode_result.correction), dtype=np.int8).reshape(-1)
        actual_observable = np.asarray(event.metadata.get("actual_observable", np.zeros_like(predicted_observable)), dtype=np.int8).reshape(-1)
        self.state.predicted_observables.append(predicted_observable)
        self.state.actual_observables.append(actual_observable)
        deadline_miss = bool(decode_result.latency_us > self.state.deadlines_us[-1])
        event_record = {
            "event_id": event.event_id,
            "decoder": decision.decoder_name,
            "selected_decoder": decision.decoder_name,
            "latency_us": decode_result.latency_us,
            "deadline_us": self.state.deadlines_us[-1],
            "deadline_miss": deadline_miss,
            "backlog": len(self.job_queue),
            "pauli_frame_lag": self.state.pauli_frame_lag,
            "accept_rate": accept_rate(predecode_result.accepted_mask),
            "validation_pass_rate": validation_pass_rate(predecode_result.validation_pass),
            "residual_density": residual_density,
            "logical_boundary": event.logical_boundary,
            "predecode_mode": self.predecoder.mode,
            "ai_risk_score": ai_risk_score,
            "ai_runtime_score": ai_runtime_score,
            "ai_confidence": ai_confidence,
        }
        self.profiler.record_event(event_record)
        return event_record

    def run_stream(self, stream: Iterable[StreamEvent]) -> dict[str, Any]:
        """Run a full stream and aggregate metrics."""
        for event in stream:
            self.process_event(event)
        accepted_concat = (
            np.concatenate(self.state.accepted_masks) if self.state.accepted_masks else np.asarray([], dtype=bool)
        )
        validation_concat = (
            np.concatenate(self.state.validation_masks) if self.state.validation_masks else np.asarray([], dtype=bool)
        )
        realtime = {
            "deadline_miss_ratio": deadline_miss_ratio(np.asarray(self.state.latencies_us), np.asarray(self.state.deadlines_us)),
            "latency": latency_percentiles(np.asarray(self.state.latencies_us)),
            "backlog": backlog_stats(np.asarray(self.state.backlog_history)),
            "max_pauli_frame_lag": max_pauli_frame_lag(np.asarray(self.state.lag_history)),
            "average_pauli_frame_lag": average_pauli_frame_lag(np.asarray(self.state.lag_history)),
        }
        qec_metrics = {
            "logical_error_rate": logical_error_rate(
                np.stack(self.state.predicted_observables) if self.state.predicted_observables else np.zeros((0, 1), dtype=np.int8),
                np.stack(self.state.actual_observables) if self.state.actual_observables else np.zeros((0, 1), dtype=np.int8),
            ),
            "fast_selection_rate": float(np.mean([name == self.config.decoders.fast for name in self.state.selected_decoders])) if self.state.selected_decoders else 0.0,
            "accurate_selection_rate": float(np.mean([name == self.config.decoders.accurate for name in self.state.selected_decoders])) if self.state.selected_decoders else 0.0,
            "ai_risk_mean": float(np.mean(self.state.ai_risk_scores)) if self.state.ai_risk_scores else 0.0,
            "ai_confidence_mean": float(np.mean(self.state.ai_confidences)) if self.state.ai_confidences else 0.0,
        }
        predecoder = {
            "accept_rate": accept_rate(accepted_concat),
            "validation_pass_rate": validation_pass_rate(validation_concat),
            "residual_density_reduction": residual_density_reduction(1.0, float(np.mean([r["residual_density"] for r in self.profiler.events])) if self.profiler.events else 1.0),
            "residual_graph_size_reduction": residual_graph_size_reduction(
                len(self.profiler.events) * max(self.config.predecoder.patch_size ** 2, 1),
                int(sum(event["residual_density"] * self.config.predecoder.patch_size ** 2 for event in self.profiler.events)),
            ),
        }
        return aggregate_run_metrics([realtime, predecoder, qec_metrics])
