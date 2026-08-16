import torch

from rt_preqec.models.candidate_predecoder_model import CandidatePredecoderModel
from rt_preqec.models.losses import candidate_predecoder_loss


def test_candidate_predecoder_forward_mask_and_loss_backward() -> None:
    model = CandidatePredecoderModel(
        detector_feature_dim=4,
        candidate_feature_dim=6,
        hidden_dim=12,
        scorer="bilinear",
    )
    batch = {
        "detector_features": torch.randn(3, 5, 4),
        "detector_mask": torch.tensor(
            [[True, True, False, False, False], [True, True, True, True, True], [True, False, False, False, False]]
        ),
        "candidate_features": torch.randn(3, 4, 6),
        "candidate_mask": torch.tensor([[True, False, True, False], [True, True, True, True], [False, False, False, False]]),
        "candidate_label": torch.tensor([0, 3, 4]),
        "risk_label": torch.tensor([0.0, 1.0, 1.0]),
    }
    output = model(
        batch["detector_features"],
        batch["detector_mask"],
        batch["candidate_features"],
        batch["candidate_mask"],
    )
    assert output.candidate_logits.shape == (3, 4)
    assert output.abstain_logit.shape == (3, 1)
    assert output.candidate_logits[0, 1] < -1e20
    loss = candidate_predecoder_loss(output, batch)
    loss.backward()
    assert loss.item() >= 0.0
