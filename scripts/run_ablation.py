"""Run simple threshold ablations."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import typer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rt_preqec.config import load_config
from rt_preqec.data.dataset import ArrayPredecoderDataset
from rt_preqec.decoders.lookup_decoder import LookupDecoder
from rt_preqec.decoders.pymatching_decoder import PyMatchingDecoder
from rt_preqec.metrics.aggregation import save_metrics_csv, save_metrics_json
from rt_preqec.predecode.selective_predecoder import SelectivePredecoder
from rt_preqec.runtime.pipeline import RTPreQECPipeline
from rt_preqec.runtime.stream_simulator import SyndromeStreamSimulator
from rt_preqec.scheduler.lag_scheduler import LagBoundedScheduler

app = typer.Typer(add_completion=False)


@app.command()
def main(config: str = "configs/ablation_thresholds.yaml", out: str = "results/runs/ablation_thresholds") -> None:
    """Sweep confidence thresholds over the toy runtime."""
    cfg = load_config(config)
    threshold_grid = cfg.to_dict()["predecoder"].get("confidence_threshold_grid", [cfg.predecoder.confidence_threshold])
    samples = ArrayPredecoderDataset("data/processed/predecoder_dataset_v1_300k.npz", split="test")
    simulator = SyndromeStreamSimulator(cfg.runtime.round_period_us, cfg.runtime.decode_deadline_us)
    rows: list[dict[str, float]] = []
    for threshold in threshold_grid:
        cfg.predecoder.confidence_threshold = float(threshold)
        predecoder = SelectivePredecoder(
            model=None,
            confidence_threshold=cfg.predecoder.confidence_threshold,
            risk_threshold=cfg.predecoder.risk_threshold,
            correction_threshold=cfg.predecoder.correction_threshold,
            enable_validation=cfg.predecoder.enable_validation,
            enable_abstention=cfg.predecoder.enable_abstention,
            device=cfg.device,
        )
        pipeline = RTPreQECPipeline(
            cfg,
            predecoder,
            LagBoundedScheduler(cfg),
            {"lookup": LookupDecoder(), "pymatching": PyMatchingDecoder()},
        )
        sample_count = min(len(samples), 32)
        patch_stream = (samples[idx]["patch"].numpy().squeeze(0) for idx in range(sample_count))
        metrics = pipeline.run_stream(simulator.from_syndromes(patch_stream))
        row = {"confidence_threshold": float(threshold), **metrics}
        rows.append(row)
    run_dir = Path(out)
    table = pd.DataFrame(rows)
    table.to_csv(run_dir / "ablation.csv", index=False)
    save_metrics_json({"rows": rows}, run_dir / "ablation.json")
    save_metrics_csv({"num_rows": len(rows)}, run_dir / "summary.csv")


if __name__ == "__main__":
    app()
