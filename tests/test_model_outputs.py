import torch

from rt_preqec.models.outputs import CandidatePredecoderOutput, RiskRuntimeOutput


def test_risk_runtime_output_scores_and_to_dict() -> None:
    output = RiskRuntimeOutput(
        risk_logit=torch.tensor([[0.0], [2.0]]),
        hard_runtime_logit=torch.tensor([[1.0], [-1.0]]),
        runtime_pred=torch.tensor([[0.5], [1.5]]),
        confidence_logit=torch.tensor([[0.0], [4.0]]),
    )
    payload = output.to_dict()
    assert payload["risk_score"].shape == (2,)
    assert payload["runtime_logit"].shape == (2,)
    assert payload["runtime_pred"].shape == (2,)
    assert "hard_runtime_score" in payload


def test_candidate_predecoder_output_selection_and_to_dict() -> None:
    output = CandidatePredecoderOutput(
        candidate_logits=torch.tensor([[1.0, 3.0, -1.0], [5.0, 0.0, 1.0]]),
        abstain_logit=torch.tensor([[0.0], [1.0]]),
        confidence_logit=torch.tensor([[2.0], [2.0]]),
        risk_logit=torch.tensor([[-2.0], [1.0]]),
        candidate_mask=torch.tensor([[True, True, False], [False, False, False]]),
    )
    assert output.selected_candidate().tolist() == [1, -1]
    payload = output.to_dict()
    assert payload["candidate_logits"].shape == (2, 3)
    assert payload["abstain_prob"].shape == (2,)
    assert payload["selected_candidate"].tolist() == [1, -1]
