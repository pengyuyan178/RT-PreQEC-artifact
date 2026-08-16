import torch

from rt_preqec.models.losses import predecoder_loss
from rt_preqec.models.predecoder import TinyNeuralPredecoder


def test_predecoder_spatial_head_backward() -> None:
    model = TinyNeuralPredecoder(temporal_window=3, patch_size=5, hidden_channels=8)
    batch = {
        "patch": torch.randn(4, 1, 3, 5, 5),
        "correction_target": torch.randint(0, 2, (4, 25)).float(),
        "confidence_target": torch.rand(4),
        "risk_target": torch.rand(4),
    }
    outputs = model(batch["patch"])
    assert outputs["correction_logits"].shape == (4, 25)
    assert outputs["confidence_logit"].shape == (4,)
    assert outputs["risk_logit"].shape == (4,)
    loss = predecoder_loss(outputs, batch, weights={"correction_pos_weight": 3.0})
    loss.backward()
    assert loss.item() >= 0.0
