import torch

from rt_preqec.models.losses import compute_risk_runtime_metrics, decomposed_risk_loss, risk_runtime_loss
from rt_preqec.models.risk_decomposition_model import RiskDecompositionModel


def _batch() -> dict[str, torch.Tensor]:
    fast_wrong = torch.tensor([0.0, 1.0, 0.0, 1.0])
    fast_fail = torch.tensor([0.0, 0.0, 1.0, 1.0])
    return {
        "features": torch.randn(4, 6),
        "risk_label": torch.tensor([0.0, 1.0, 1.0, 1.0]),
        "fast_wrong": fast_wrong,
        "fast_logical_fail": fast_fail,
        "hard_runtime": torch.tensor([0.0, 0.0, 1.0, 0.0]),
        "syndrome_tail": torch.tensor([0.0, 1.0, 0.0, 1.0]),
        "safe_fast": ((fast_wrong <= 0.5) & (fast_fail <= 0.5)).float(),
        "accurate_runtime_us": torch.tensor([1.0, 2.0, 4.0, 8.0]),
        "hard_runtime_label_valid": torch.ones(4),
    }


def test_decomposed_loss_backward_and_runtime_dispatch() -> None:
    model = RiskDecompositionModel(feature_dim=6, hidden_dim=12)
    batch = _batch()
    output = model(batch["features"])
    loss = decomposed_risk_loss(output, batch)
    dispatched = risk_runtime_loss(output, batch)
    loss.backward()
    assert loss.item() >= 0.0
    assert dispatched.item() >= 0.0


def test_decomposed_metrics_include_requested_fields() -> None:
    model = RiskDecompositionModel(feature_dim=6, hidden_dim=12)
    batch = _batch()
    output = model(batch["features"])
    metrics = compute_risk_runtime_metrics(output, batch)
    for key in [
        "combined_risk_accuracy",
        "combined_risk_precision",
        "combined_risk_recall",
        "combined_risk_fnr",
        "combined_risk_fpr",
        "fast_wrong_precision",
        "fast_wrong_recall",
        "fast_wrong_fnr",
        "fast_wrong_fpr",
        "fast_logical_fail_precision",
        "fast_logical_fail_recall",
        "fast_logical_fail_fnr",
        "fast_logical_fail_fpr",
        "safe_fast_precision",
        "safe_fast_recall",
        "hard_runtime_accuracy",
        "runtime_mae",
    ]:
        assert key in metrics
