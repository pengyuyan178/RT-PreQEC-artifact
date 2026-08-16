"""Stim surface-code helpers with toy fallback paths."""

from __future__ import annotations

from typing import Any

import numpy as np

from rt_preqec.config import ProjectConfig
from rt_preqec.data.layout import DetectorLayout, build_detector_layout_from_dem

try:
    import stim
except ImportError:  # pragma: no cover
    stim = None


def build_surface_code_circuit(
    distance: int,
    rounds: int,
    basis: str,
    noise_params: dict[str, Any] | None = None,
) -> Any:
    """Build a rotated surface-code memory circuit or return a placeholder spec."""
    noise_params = noise_params or {}
    basis_map = {
        "memory_x": "surface_code:rotated_memory_x",
        "memory_z": "surface_code:rotated_memory_z",
        "rotated_memory_x": "surface_code:rotated_memory_x",
        "rotated_memory_z": "surface_code:rotated_memory_z",
    }
    basis_key = basis_map.get(basis, "surface_code:rotated_memory_x")
    if stim is None:
        return {
            "toy": True,
            "distance": distance,
            "rounds": rounds,
            "basis": basis,
            "noise_params": noise_params,
            "reason": "stim_unavailable",
        }
    try:
        return stim.Circuit.generated(
            basis_key,
            distance=distance,
            rounds=rounds,
            after_clifford_depolarization=noise_params.get("after_clifford_depolarization", 0.001),
            before_round_data_depolarization=noise_params.get("before_round_data_depolarization", 0.0),
            before_measure_flip_probability=noise_params.get("before_measure_flip_probability", 0.0),
            after_reset_flip_probability=noise_params.get("after_reset_flip_probability", 0.0),
        )
    except Exception as exc:  # pragma: no cover
        return {
            "toy": True,
            "distance": distance,
            "rounds": rounds,
            "basis": basis,
            "noise_params": noise_params,
            "reason": f"stim_generation_failed:{exc}",
            "todo": "Adapt to the installed Stim generated-circuit signature.",
        }


def extract_detector_error_model(circuit: Any) -> Any:
    """Extract a Stim detector error model when possible."""
    if stim is None or isinstance(circuit, dict):
        return None
    try:
        return circuit.detector_error_model(decompose_errors=True)
    except Exception:  # pragma: no cover
        return None


def _toy_sample_syndromes(
    distance: int,
    rounds: int,
    num_shots: int,
    seed: int | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    syndrome = rng.integers(0, 2, size=(num_shots, rounds, distance, distance), dtype=np.int8)
    observables = rng.integers(0, 2, size=(num_shots, 1), dtype=np.int8)
    return {
        "detection_events": syndrome,
        "observables": observables,
        "metadata": {"toy": True, "reason": "fallback_random_syndrome"},
    }


def sample_syndromes(circuit: Any, num_shots: int, seed: int | None = None) -> dict[str, Any]:
    """Sample detector events and observables from a circuit or toy fallback."""
    if isinstance(circuit, dict):
        return _toy_sample_syndromes(circuit["distance"], circuit["rounds"], num_shots, seed)

    if stim is None:  # pragma: no cover
        return _toy_sample_syndromes(3, 3, num_shots, seed)

    try:
        sampler = circuit.compile_detector_sampler()
        dets, obs = sampler.sample(num_shots, separate_observables=True)
        dets = np.asarray(dets, dtype=np.int8)
        obs = np.asarray(obs, dtype=np.int8)
        return {
            "detection_events": dets,
            "observables": obs,
            "metadata": {"toy": False, "num_detectors": int(dets.shape[1]), "flat_detector_vector": True},
        }
    except Exception as exc:  # pragma: no cover
        return {
            **_toy_sample_syndromes(3, 3, num_shots, seed),
            "metadata": {"toy": True, "reason": f"stim_sampling_failed:{exc}"},
        }


def _reshape_detector_events_for_toy_pipeline(
    detection_events: np.ndarray,
    rounds: int,
    distance: int,
) -> np.ndarray:
    """Pad flat detector events into [shots, rounds, H, W] for the existing toy pipeline."""
    shots, num_detectors = detection_events.shape
    side = max(distance, int(np.ceil(np.sqrt(max(num_detectors / max(rounds, 1), 1.0)))))
    padded = np.zeros((shots, rounds * side * side), dtype=np.int8)
    padded[:, :num_detectors] = detection_events
    return padded.reshape(shots, rounds, side, side)


def generate_surface_code_samples(config: ProjectConfig) -> dict[str, Any]:
    """Generate a real or fallback surface-code sampling bundle."""
    distance = config.qec.distances[0]
    error_rate = config.qec.physical_error_rates[0]
    noise_params = {"after_clifford_depolarization": error_rate}
    circuit = build_surface_code_circuit(
        distance=distance,
        rounds=config.qec.rounds,
        basis=config.qec.basis,
        noise_params=noise_params,
    )
    dem = extract_detector_error_model(circuit)
    sampled = sample_syndromes(circuit, num_shots=config.qec.num_shots, seed=config.seed)
    detection_events = np.asarray(sampled["detection_events"], dtype=np.int8)
    observables = np.asarray(sampled["observables"], dtype=np.int8)
    layout: DetectorLayout | None = build_detector_layout_from_dem(dem)
    if detection_events.ndim == 2:
        syndrome = detection_events
    else:
        syndrome = detection_events
    metadata = {
        "toy": bool(sampled["metadata"].get("toy", False)),
        "num_shots": int(config.qec.num_shots),
        "distance": int(distance),
        "rounds": int(config.qec.rounds),
        **sampled["metadata"],
    }
    if dem is None:
        metadata["placeholder"] = True
    return {
        "circuit": circuit,
        "dem": dem,
        "syndrome": syndrome,
        "observables": observables,
        "layout": layout,
        "metadata": metadata,
    }
