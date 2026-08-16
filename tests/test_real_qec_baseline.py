import importlib.util

from rt_preqec.config import load_config
from experiments.eval_qec_accuracy import run_pymatching_baseline


def test_real_qec_baseline_small_run() -> None:
    config = load_config("configs/eval_realtime.yaml")
    config.qec.distances = [3]
    config.qec.rounds = 3
    config.qec.num_shots = 100
    result = run_pymatching_baseline(config)
    metrics = result["metrics"]
    assert "logical_error_rate" in metrics

    stim_available = importlib.util.find_spec("stim") is not None
    pymatching_available = importlib.util.find_spec("pymatching") is not None
    if stim_available and pymatching_available and not metrics.get("toy", False):
        assert 0.0 <= metrics["logical_error_rate"] <= 1.0
        assert metrics.get("placeholder", False) is False
    else:
        assert result["metadata"].get("placeholder", False) is True or metrics.get("toy", False) is True
