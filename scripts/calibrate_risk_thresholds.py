"""CLI for validation-only risk threshold calibration."""

from __future__ import annotations

from pathlib import Path
import sys

import typer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.calibrate_risk_thresholds import run_calibration
from rt_preqec.config import load_config, load_yaml
from rt_preqec.logging_utils import configure_logging, get_logger

app = typer.Typer(add_completion=False)
logger = get_logger(__name__)


@app.command()
def main(
    config: str = "configs/risk_calibration.yaml",
    data: str = "data/processed/risk_dataset.npz",
    checkpoint: str = "checkpoints/risk_lstm.pt",
    split: str = typer.Option("val", "--split"),
    out: str = "checkpoints/risk_lstm_calibration.json",
) -> None:
    """Calibrate AI risk/confidence thresholds using validation predictions."""
    cfg = load_config(config)
    configure_logging(cfg.log_level)
    objective = load_yaml(config).get("objective", {"type": "maximize_f1"})
    payload = run_calibration(cfg, data, checkpoint, split, out, objective=objective)
    logger.info("saved calibration to %s", out)
    logger.info(
        "selected thresholds: risk=%.3f confidence=%.3f safe_fast=%.3f",
        payload["selected_ai_risk_threshold"],
        payload["selected_ai_confidence_threshold"],
        payload.get("selected_safe_fast_threshold", 0.5),
    )


if __name__ == "__main__":
    app()
