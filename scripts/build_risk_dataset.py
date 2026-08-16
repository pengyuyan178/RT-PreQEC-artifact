"""CLI for building the risk-profiler dataset."""

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

from experiments.build_risk_dataset import run_build_risk_dataset
from rt_preqec.config import load_config
from rt_preqec.logging_utils import configure_logging, get_logger
from rt_preqec.utils import dump_json

app = typer.Typer(add_completion=False)
logger = get_logger(__name__)


def _parse_bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected boolean true/false, got {value!r}")


@app.command()
def main(
    config: str = "configs/risk_profiler.yaml",
    out: str = "data/processed/risk_dataset.npz",
    summary_out: str = "results/runs/risk_dataset_build/summary.json",
    split_policy: str | None = typer.Option(None, "--split-policy"),
    preset: str | None = typer.Option(None, "--preset"),
    shots_per_setting: int | None = typer.Option(None, "--shots-per-setting"),
    seed: int | None = typer.Option(None, "--seed"),
    verbose: str = typer.Option("false", "--verbose"),
) -> None:
    """Build a risk-only profiling dataset from decoding records."""
    cfg = load_config(config)
    if seed is not None:
        cfg.seed = int(seed)
    configure_logging(cfg.log_level)
    if split_policy is not None:
        cfg.risk_dataset.split_policy = str(split_policy)
    if shots_per_setting is not None:
        cfg.risk_dataset.num_shots = int(shots_per_setting)
    summary = run_build_risk_dataset(
        cfg,
        out,
        preset=preset,
        shots_per_setting=shots_per_setting,
        verbose=_parse_bool(verbose),
    )
    dump_json(summary, summary_out)
    logger.info("saved risk dataset to %s", out)
    logger.info("positive risk label rate: %.4f", summary.get("positive_risk_label_rate", 0.0))
    logger.info("feature names: %s", ",".join(summary.get("feature_names", [])))


if __name__ == "__main__":
    app()
