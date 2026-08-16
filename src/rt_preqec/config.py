"""Configuration utilities for RT-PreQEC."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping at {path}, got {type(data)!r}")
    return data


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries without mutating the inputs."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass
class QECConfig:
    """QEC data generation and evaluation settings."""

    code: str = "rotated_surface_code"
    basis: str = "memory_x"
    distances: list[int] = field(default_factory=lambda: [3])
    rounds: int = 3
    physical_error_rates: list[float] = field(default_factory=lambda: [0.001])
    noise_model: str = "circuit_level_depolarizing"
    num_shots: int = 32
    test_fraction: float = 0.4


@dataclass
class PredecoderConfig:
    """Selective predecoder settings."""

    patch_size: int = 5
    temporal_window: int = 3
    confidence_threshold: float = 0.95
    risk_threshold: float = 0.05
    correction_threshold: float = 0.90
    confidence_threshold_grid: list[float] | None = None
    max_cluster_size: int = 6
    min_boundary_distance: int = 2
    enable_validation: bool = True
    enable_abstention: bool = True
    frontend_latency_us: float = 0.25
    validation_latency_us: float = 0.05
    max_residual_reduction: float = 0.60
    min_backend_latency_fraction: float = 0.35
    density_risk_scale: float = 1.0


@dataclass
class ModelConfig:
    """Neural model settings."""

    type: str = "tiny_cnn"
    hidden_channels: int = 16
    hidden_dim: int = 64
    num_layers: int = 3
    feature_layers: int = 2
    dropout: float = 0.0
    quantized: bool = False
    history_encoder_type: str = "none"
    history_length: int = 1
    history_hidden_dim: int = 64
    bidirectional: bool = False
    use_layer_norm: bool = True
    combination_weights: dict[str, float] = field(default_factory=dict)
    detector_feature_dim: int = 8
    candidate_feature_dim: int = 10
    scorer: str = "bilinear"
    pooling: str = "mean_max"


@dataclass
class TrainingConfig:
    """Training loop settings."""

    batch_size: int = 16
    epochs: int = 1
    lr: float = 1e-3
    weight_decay: float = 0.0
    num_workers: int = 0
    weighted_sampler: bool = False
    grad_clip_norm: float | None = None
    deterministic: bool = False


@dataclass
class RuntimeConfig:
    """Runtime and deadline settings."""

    round_period_us: float = 1.0
    decode_deadline_us: float = 10.0
    max_pauli_frame_lag: int = 8
    logical_boundary_interval: int = 100
    num_workers: int = 1
    overload_backlog_threshold: int = 32
    torch_num_threads: int | None = None


@dataclass
class TimingConfig:
    """Decoder timing protocol settings."""

    accuracy_decode_mode: str = "batch"
    runtime_label_mode: str = "loop_per_shot"
    warmup_shots: int = 100
    repeat_per_shot: int = 3
    max_timing_shots: int | None = 2000
    timing_statistic: str = "median"
    use_batch_decode_for_accuracy: bool = True
    use_loop_timing_for_runtime_label: bool = True
    set_torch_num_threads: int | None = 1
    set_numpy_num_threads: int | None = 1


@dataclass
class SchedulerConfig:
    """Scheduler settings."""

    policy: str = "risk_aware_edf"
    alpha_urgency: float = 1.0
    beta_risk: float = 1.0
    gamma_runtime: float = 0.5
    delta_boundary: float = 2.0
    use_ai_risk: bool = False
    ai_risk_threshold: float = 0.5
    ai_confidence_threshold: float = 0.5
    conservative_on_low_confidence: bool = True


@dataclass
class DecoderConfig:
    """Decoder selection settings."""

    fast: str = "lookup"
    accurate: str = "pymatching"
    fallback: str = "pymatching"


@dataclass
class RiskDatasetConfig:
    """Risk-profiler dataset generation settings."""

    num_shots: int = 5000
    hard_runtime_percentile: float = 90.0
    fast_decoder: str = "lookup"
    accurate_decoder: str = "pymatching"
    feature_mode: str = "syndrome_patch_aggregate"
    split_policy: str = "stream_block"
    train_fraction: float | None = None
    val_fraction: float | None = None
    test_fraction: float | None = None


@dataclass
class RiskModelConfig:
    """Risk-profiler model settings."""

    type: str = "tiny_mlp"
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.0


@dataclass
class RiskTrainingConfig:
    """Risk-profiler training settings."""

    batch_size: int = 256
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    val_fraction: float = 0.2
    num_workers: int = 0


@dataclass
class DataProtocolConfig:
    """Evaluation data protocol settings."""

    eval_source: str = "generated"
    train_seed: int = 42
    eval_seed: int = 1001
    require_eval_seed_different_from_train_seed: bool = True


@dataclass
class RiskEvalConfig:
    """Real-stream risk evaluation settings."""

    modes: list[str] = field(
        default_factory=lambda: [
            "accurate_only",
            "fast_only",
            "heuristic_pre_fixed",
            "edf",
            "rt_qec",
            "risk_heuristic",
            "ai_risk",
            "oracle_predecoder",
            "oracle_risk",
        ]
    )
    hard_runtime_percentile: float = 90.0
    ai_risk_threshold: float = 0.5
    ai_confidence_threshold: float = 0.5
    conservative_on_low_confidence: bool = True
    use_same_shots_for_all_modes: bool = True
    heuristic_predecoder_backend: str = "accurate"
    boundary_drain_rounds: int = 2
    rt_qec_drain_backlog_threshold: int | None = None


@dataclass
class OutputConfig:
    """Output toggles for experiment artifacts."""

    save_events: bool = True
    save_decisions: bool = True
    save_predictions: bool = True
    save_plots_ready_csv: bool = True


@dataclass
class ProjectConfig:
    """Top-level project configuration."""

    seed: int = 42
    device: str = "cpu"
    output_dir: str = "results/runs/default"
    log_level: str = "INFO"
    qec: QECConfig = field(default_factory=QECConfig)
    predecoder: PredecoderConfig = field(default_factory=PredecoderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    decoders: DecoderConfig = field(default_factory=DecoderConfig)
    risk_dataset: RiskDatasetConfig = field(default_factory=RiskDatasetConfig)
    risk_model: RiskModelConfig = field(default_factory=RiskModelConfig)
    risk_training: RiskTrainingConfig = field(default_factory=RiskTrainingConfig)
    data_protocol: DataProtocolConfig = field(default_factory=DataProtocolConfig)
    risk_eval: RiskEvalConfig = field(default_factory=RiskEvalConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)
    loss: dict[str, float] = field(default_factory=dict)
    qec_grid: dict[str, Any] = field(default_factory=dict)
    noise_scenarios: list[dict[str, Any]] = field(default_factory=list)
    risk_label: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to a plain dictionary."""
        return asdict(self)


def _build_project_config(config_dict: dict[str, Any]) -> ProjectConfig:
    return ProjectConfig(
        seed=config_dict.get("seed", 42),
        device=config_dict.get("device", "cpu"),
        output_dir=config_dict.get("output_dir", "results/runs/default"),
        log_level=config_dict.get("log_level", "INFO"),
        qec=QECConfig(**config_dict.get("qec", {})),
        predecoder=PredecoderConfig(**config_dict.get("predecoder", {})),
        model=ModelConfig(**config_dict.get("model", {})),
        training=TrainingConfig(**config_dict.get("training", {})),
        runtime=RuntimeConfig(**config_dict.get("runtime", {})),
        timing=TimingConfig(**config_dict.get("timing", {})),
        scheduler=SchedulerConfig(**config_dict.get("scheduler", {})),
        decoders=DecoderConfig(**config_dict.get("decoders", {})),
        risk_dataset=RiskDatasetConfig(**config_dict.get("risk_dataset", {})),
        risk_model=RiskModelConfig(**config_dict.get("risk_model", {})),
        risk_training=RiskTrainingConfig(**config_dict.get("risk_training", {})),
        data_protocol=DataProtocolConfig(**config_dict.get("data_protocol", {})),
        risk_eval=RiskEvalConfig(**config_dict.get("risk_eval", {})),
        outputs=OutputConfig(**config_dict.get("outputs", {})),
        loss=dict(config_dict.get("loss", {})),
        qec_grid=dict(config_dict.get("qec_grid", {})),
        noise_scenarios=list(config_dict.get("noise_scenarios", [])),
        risk_label=dict(config_dict.get("risk_label", {})),
    )


def load_config(default_path: str | Path, override_path: str | Path | None = None) -> ProjectConfig:
    """Load config from default YAML plus an optional override YAML."""
    base = load_yaml(default_path)
    if override_path is not None:
        override = load_yaml(override_path)
        base = merge_dicts(base, override)
    config = _build_project_config(base)
    seed_everything(config.seed)
    return config


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and Torch where available."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True


def make_torch_generator(seed: int):
    """Return a seeded torch generator when torch is available."""
    if torch is None:  # pragma: no cover
        return None
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def seed_worker(worker_id: int) -> None:
    """Seed DataLoader workers from torch's worker seed."""
    del worker_id
    if torch is None:  # pragma: no cover
        return
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
