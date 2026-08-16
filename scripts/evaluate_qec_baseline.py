"""Evaluate real Stim + PyMatching QEC baseline."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import typer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.eval_qec_accuracy import run_pymatching_baseline
from rt_preqec.config import load_config
from rt_preqec.data.layout import save_detector_layout
from rt_preqec.logging_utils import configure_logging, get_logger
from rt_preqec.metrics.aggregation import save_metrics_json
from rt_preqec.utils import ensure_parent

app = typer.Typer(add_completion=False)
logger = get_logger(__name__)


def _parse_bool_flag(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


@app.command()
def main(
    config: str = "configs/eval_realtime.yaml",
    out: str = "results/runs/real_baseline",
    export_candidates: str = "true",
) -> None:
    """Run the real or fallback QEC baseline and persist outputs."""
    cfg = load_config(config)
    configure_logging(cfg.log_level)
    result = run_pymatching_baseline(cfg)
    out_dir = Path(out)
    save_metrics_json(result["metrics"], out_dir / "metrics.json")
    predictions_path = ensure_parent(out_dir / "predictions.npz")
    np.savez_compressed(
        predictions_path,
        syndrome=result["syndrome"],
        predicted_observables=result["predicted_observables"],
        actual_observables=result["actual_observables"],
    )
    if result["layout"] is not None:
        save_detector_layout(result["layout"], out_dir / "detector_layout.csv")
    if _parse_bool_flag(export_candidates):
        local_candidates = result.get("local_candidates", [])
        candidate_rows = [
            {
                "candidate_id": candidate.candidate_id,
                "detector_ids": ",".join(str(v) for v in candidate.detector_ids.tolist()),
                "observable_ids": ",".join(str(v) for v in candidate.observable_ids.tolist()),
                "probability": candidate.probability,
                "weight": candidate.weight,
                "spatial_diameter": candidate.coord_span.get("spatial_diameter"),
                "time_diameter": candidate.coord_span.get("time_diameter"),
            }
            for candidate in local_candidates
        ]
        pd.DataFrame(candidate_rows).to_csv(ensure_parent(out_dir / "local_candidates.csv"), index=False)
    logger.info("saved baseline outputs to %s", out)


if __name__ == "__main__":
    app()
