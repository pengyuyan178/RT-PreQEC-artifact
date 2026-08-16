import torch

from rt_preqec.models.losses import risk_runtime_loss
from rt_preqec.models.risk_runtime_model import RiskRuntimeModel


def _batch() -> dict[str, torch.Tensor]:
    return {
        "features": torch.randn(4, 8),
        "risk_label": torch.tensor([0.0, 1.0, 0.0, 1.0]),
        "hard_runtime": torch.tensor([0.0, 0.0, 1.0, 1.0]),
        "accurate_runtime_us": torch.rand(4) * 10.0,
        "fast_wrong": torch.zeros(4),
        "fast_logical_fail": torch.zeros(4),
    }


def test_risk_runtime_model_mlp_forward() -> None:
    model = RiskRuntimeModel(feature_dim=8, hidden_dim=16, history_encoder_type="none")
    output = model(torch.randn(4, 8))
    payload = output.to_dict()
    assert payload["risk_logit"].shape == (4,)
    assert payload["runtime_logit"].shape == (4,)


def test_risk_runtime_model_lstm_history_forward() -> None:
    model = RiskRuntimeModel(
        feature_dim=8,
        hidden_dim=16,
        history_encoder_type="lstm",
        history_length=5,
        history_hidden_dim=12,
    )
    output = model(torch.randn(4, 8), torch.randn(4, 5, 8))
    assert output.risk_logit.shape == (4, 1)


def test_risk_runtime_loss_backward() -> None:
    model = RiskRuntimeModel(feature_dim=8, hidden_dim=16, history_encoder_type="none")
    batch = _batch()
    output = model(batch["features"])
    loss = risk_runtime_loss(output, batch)
    loss.backward()
    assert loss.item() >= 0.0


def test_risk_runtime_loss_skips_invalid_hard_runtime() -> None:
    model = RiskRuntimeModel(feature_dim=8, hidden_dim=16, history_encoder_type="none")
    batch = _batch()
    batch["hard_runtime_label_valid"] = torch.zeros(4)
    output = model(batch["features"])
    loss = risk_runtime_loss(output, batch, {"risk_weight": 1.0, "hard_runtime_weight": 10.0, "runtime_weight": 0.0, "confidence_weight": 0.0})
    loss.backward()
    assert loss.item() >= 0.0
