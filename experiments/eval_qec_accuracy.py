"""Real Stim + PyMatching baseline evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

from rt_preqec.config import ProjectConfig
from rt_preqec.data.dem_parser import filter_local_candidates, index_candidates_by_detector, parse_dem_error_candidates
from rt_preqec.data.stim_surface_code import generate_surface_code_samples
from rt_preqec.decoders.pymatching_decoder import PyMatchingDecoder
from rt_preqec.metrics.qec_metrics import summarize_qec_baseline


def run_pymatching_baseline(config: ProjectConfig) -> dict[str, Any]:
    """Run a real Stim sampling plus PyMatching decoding baseline."""
    bundle = generate_surface_code_samples(config)
    decoder = PyMatchingDecoder.from_detector_error_model(bundle["dem"])
    syndrome = np.asarray(bundle["syndrome"], dtype=np.int8)
    observables = np.asarray(bundle["observables"], dtype=np.int8)
    if syndrome.ndim != 2:
        flat = syndrome.reshape(syndrome.shape[0], -1)
    else:
        flat = syndrome
    predictions, decode_metadata = decoder.decode_batch(flat)
    all_candidates = parse_dem_error_candidates(bundle["dem"], layout=bundle["layout"])
    local_candidates = filter_local_candidates(
        all_candidates,
        max_spatial_diameter=4.0,
        max_time_diameter=2.0,
        allow_observable_flip=False,
    )
    metrics = summarize_qec_baseline(
        predictions,
        observables,
        metadata={
            **bundle["metadata"],
            **decode_metadata,
            "decoder": decoder.name,
            "num_detectors": int(flat.shape[1]),
            "num_candidates": len(all_candidates),
            "num_local_candidates": len(local_candidates),
            "candidates_touching_observable": int(sum(len(candidate.observable_ids) > 0 for candidate in all_candidates)),
            "avg_candidate_detector_count": float(np.mean([len(candidate.detector_ids) for candidate in all_candidates])) if all_candidates else 0.0,
        },
    )
    return {
        "metrics": metrics,
        "predicted_observables": predictions,
        "actual_observables": observables,
        "syndrome": flat,
        "layout": bundle["layout"],
        "metadata": {**bundle["metadata"], **decode_metadata},
        "all_candidates": all_candidates,
        "local_candidates": local_candidates,
        "candidates_by_detector": index_candidates_by_detector(local_candidates),
    }
