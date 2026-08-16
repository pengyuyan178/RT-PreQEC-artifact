from pathlib import Path

from experiments.build_risk_dataset import run_build_risk_dataset
from experiments.train_risk_profiler import train_risk_profiler
from rt_preqec.config import load_config
from scripts.evaluate_realtime import main as evaluate_realtime_main


def test_risk_smoke_end_to_end(tmp_path: Path) -> None:
    config = load_config("configs/risk_profiler.yaml")
    config.qec.num_shots = 16
    config.risk_dataset.num_shots = 16
    config.risk_training.epochs = 1
    config.risk_training.batch_size = 8
    data_path = tmp_path / "risk_dataset.npz"
    checkpoint_path = tmp_path / "tiny_risk_profiler.pt"
    summary = run_build_risk_dataset(config, data_path)
    assert summary["num_shots"] == 16
    train_summary = train_risk_profiler(config, data_path, checkpoint_path)
    assert Path(train_summary["checkpoint"]).exists()
    out_dir = tmp_path / "eval"
    evaluate_realtime_main(
        config="configs/eval_realtime.yaml",
        data=str(data_path),
        checkpoint="checkpoints/predecoder_v1_300k.pt",
        risk_checkpoint=str(checkpoint_path),
        mode="ai_risk",
        out=str(out_dir),
    )
    assert (out_dir / "metrics.json").exists()
