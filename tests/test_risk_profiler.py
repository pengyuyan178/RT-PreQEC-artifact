import torch

from rt_preqec.models.risk_losses import risk_profiler_loss
from rt_preqec.models.risk_profiler import TinyRiskProfiler


def test_tiny_risk_profiler_forward_and_backward() -> None:
    model = TinyRiskProfiler(input_dim=8, hidden_dim=16, num_layers=2, dropout=0.0)
    batch = {
        "features": torch.randn(4, 8),
        "risk_label": torch.rand(4),
        "hard_runtime": torch.rand(4),
        "fast_wrong": torch.rand(4),
        "fast_logical_fail": torch.rand(4),
        "accurate_runtime_us": torch.rand(4) * 10.0,
    }
    outputs = model(batch["features"])
    assert {"risk_logit", "runtime_logit", "confidence_logit", "runtime_pred"} <= set(outputs.keys())
    loss = risk_profiler_loss(outputs, batch)
    loss.backward()
    assert loss.item() >= 0.0
