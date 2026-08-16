"""CLI for paired real-stream evaluation over shared Stim shots."""

from __future__ import annotations

from pathlib import Path
import sys

import typer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rt_preqec.config import load_config
from rt_preqec.evaluation.real_stream import run_real_stream_eval
from rt_preqec.logging_utils import configure_logging, get_logger

app = typer.Typer(add_completion=False)
logger = get_logger(__name__)


@app.command()
def main(
    config: str = "configs/real_stream_eval.yaml",
    risk_checkpoint: str = "checkpoints/tiny_risk_profiler.pt",
    out: str = "results/runs/real_stream_eval_d3",
    risk_dataset: str | None = typer.Option(None, "--risk-dataset"),
    split: str = typer.Option("test", "--split"),
    calibration: str | None = typer.Option(None, "--calibration"),
    eval_seed: int | None = typer.Option(None, "--eval-seed"),
    train_seed: int | None = typer.Option(None, "--train-seed"),
) -> None:
    """Run paired real-stream evaluation across multiple scheduler modes."""
    cfg = load_config(config)
    configure_logging(cfg.log_level)
    summary = run_real_stream_eval(
        cfg,
        risk_checkpoint=None if risk_checkpoint.lower() == "none" else risk_checkpoint,
        out_dir=out,
        risk_dataset_path=risk_dataset,
        split=split,
        calibration_path=calibration,
        train_seed=train_seed,
        eval_seed=eval_seed,
    )
    logger.info("saved real-stream evaluation to %s", out)
    logger.info("modes evaluated: %s", ",".join([row["mode"] for row in summary.get("summary", [])]))


if __name__ == "__main__":
    app()
