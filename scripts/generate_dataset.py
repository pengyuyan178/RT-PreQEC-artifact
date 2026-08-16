"""Generate a small patch dataset for RT-PreQEC."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import typer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rt_preqec.config import load_config
from rt_preqec.data.dem_parser import parse_dem_error_candidates
from rt_preqec.data.layout import save_detector_layout
from rt_preqec.data.dataset import save_patch_dataset
from rt_preqec.data.patch_extractor import extract_local_patches, is_candidate_easy_cluster
from rt_preqec.data.schemas import PatchSample
from rt_preqec.data.stim_surface_code import generate_surface_code_samples
from rt_preqec.logging_utils import configure_logging, get_logger
from rt_preqec.utils import ensure_parent

app = typer.Typer(add_completion=False)
logger = get_logger(__name__)


def _patch_to_sample(patch_item: dict[str, object]) -> PatchSample:
    patch = np.asarray(patch_item["patch"], dtype=np.float32)
    location = tuple(int(v) for v in patch_item["location"])
    label = is_candidate_easy_cluster(patch, max_cluster_size=6, min_boundary_distance=1)
    correction_target = patch[-1].reshape(-1).astype(np.float32)
    confidence_target = 0.9 if label else 0.2
    risk_target = 0.1 if label else 0.8
    return PatchSample(
        patch=patch,
        location=location,
        correction_target=correction_target,
        is_correct=label,
        confidence_target=confidence_target,
        risk_target=risk_target,
        metadata={"toy_target": True},
    )


def _reshape_for_patch_pipeline(
    detection_events: np.ndarray,
    rounds: int,
    distance: int,
) -> np.ndarray:
    """Pad flat detector vectors into `[shots, rounds, H, W]` for patch extraction."""
    if detection_events.ndim != 2:
        return detection_events
    shots, num_detectors = detection_events.shape
    side = max(distance, int(np.ceil(np.sqrt(max(num_detectors / max(rounds, 1), 1.0)))))
    padded = np.zeros((shots, rounds * side * side), dtype=np.int8)
    padded[:, :num_detectors] = detection_events
    return padded.reshape(shots, rounds, side, side)


@app.command()
def main(config: str = "configs/data_surface_code.yaml", out: str = "data/processed/predecoder_dataset_legacy.npz") -> None:
    """Generate dataset from Stim if possible, else toy fallback."""
    cfg = load_config(config)
    configure_logging(cfg.log_level)
    qec = cfg.qec
    bundle = generate_surface_code_samples(cfg)
    sampled_metadata = dict(bundle["metadata"])
    syndromes = _reshape_for_patch_pipeline(np.asarray(bundle["syndrome"]), qec.rounds, qec.distances[0])
    samples: list[PatchSample] = []
    for syndrome in syndromes:
        patches = extract_local_patches(
            syndrome,
            patch_size=cfg.predecoder.patch_size,
            temporal_window=cfg.predecoder.temporal_window,
        )
        samples.extend(_patch_to_sample(item) for item in patches)
    save_patch_dataset(samples, Path(out))
    out_path = Path(out)
    layout = bundle["layout"]
    candidates = parse_dem_error_candidates(bundle["dem"], layout=layout)
    if layout is not None:
        save_detector_layout(layout, out_path.with_name(f"{out_path.stem}_detector_layout.csv"))
    if len(candidates) > 0:
        candidate_rows = [
            {
                "candidate_id": candidate.candidate_id,
                "detector_ids": ",".join(str(v) for v in candidate.detector_ids.tolist()),
                "observable_ids": ",".join(str(v) for v in candidate.observable_ids.tolist()),
                "probability": candidate.probability,
                "weight": candidate.weight,
                "coord_span": str(candidate.coord_span),
            }
            for candidate in candidates
        ]
        pd.DataFrame(candidate_rows).to_csv(ensure_parent(out_path.with_name(f"{out_path.stem}_candidates.csv")), index=False)
    sampled_metadata.update(
        {
            "has_layout": layout is not None,
            "has_dem_candidates": len(candidates) > 0,
            "num_detectors": int(np.asarray(bundle["syndrome"]).shape[-1]) if np.asarray(bundle["syndrome"]).ndim >= 2 else int(np.asarray(bundle["syndrome"]).shape[0]),
            "num_candidates": len(candidates),
        }
    )
    logger.info("saved %s samples to %s", len(samples), out)


if __name__ == "__main__":
    app()
