import torch

from rt_preqec.models.encoders import (
    CausalHistoryEncoder,
    DEMCandidateEncoder,
    DetectorPatchEncoder,
    FeatureProjectionEncoder,
    PatchCandidateCompatibility,
)


def test_feature_projection_encoder_forward() -> None:
    encoder = FeatureProjectionEncoder(input_dim=5, hidden_dim=8, num_layers=2)
    output = encoder(torch.randn(4, 5))
    assert output.shape == (4, 8)


def test_causal_history_encoder_modes_forward() -> None:
    history = torch.randn(3, 6, 5)
    for mode in ["none", "gru", "lstm", "tcn"]:
        encoder = CausalHistoryEncoder(input_dim=5, hidden_dim=7, encoder_type=mode, num_layers=1)
        output = encoder(history)
        assert output.shape == (3, 7)


def test_causal_history_encoder_rejects_bidirectional() -> None:
    try:
        CausalHistoryEncoder(input_dim=5, hidden_dim=7, encoder_type="lstm", bidirectional=True)
    except ValueError:
        return
    raise AssertionError("bidirectional history must be rejected")


def test_detector_patch_encoder_masked_pooling() -> None:
    encoder = DetectorPatchEncoder(detector_feature_dim=4, hidden_dim=8, pooling="mean_max")
    features = torch.randn(2, 5, 4)
    mask = torch.tensor([[True, True, False, False, False], [False, False, False, False, False]])
    output = encoder(features, mask)
    assert output.shape == (2, 8)
    assert torch.isfinite(output).all()


def test_dem_candidate_encoder_mask_shape() -> None:
    encoder = DEMCandidateEncoder(candidate_feature_dim=6, hidden_dim=8)
    features = torch.randn(2, 4, 6)
    mask = torch.tensor([[True, False, True, False], [True, True, True, False]])
    output = encoder(features, mask)
    assert output.shape == (2, 4, 8)
    assert torch.all(output[0, 1] == 0)


def test_patch_candidate_compatibility_masks_logits() -> None:
    scorer = PatchCandidateCompatibility(hidden_dim=8, scorer="bilinear")
    logits = scorer(
        torch.randn(2, 8),
        torch.randn(2, 3, 8),
        torch.tensor([[True, False, True], [False, False, False]]),
    )
    assert logits.shape == (2, 3)
    assert logits[0, 1] < -1e20
    assert torch.all(logits[1] < -1e20)
