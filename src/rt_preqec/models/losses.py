"""Training losses for RT-PreQEC model components."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from rt_preqec.models.outputs import CandidatePredecoderOutput, DecomposedRiskOutput, RiskRuntimeOutput


def _as_output_dict(
    outputs: RiskRuntimeOutput | DecomposedRiskOutput | CandidatePredecoderOutput | dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if hasattr(outputs, "to_dict"):
        return outputs.to_dict()
    return outputs  # type: ignore[return-value]


def _decomposed_defaults(weights: dict[str, float] | None) -> dict[str, float]:
    merged = {
        "fast_wrong": 1.0,
        "fast_logical_fail": 1.5,
        "hard_runtime": 0.5,
        "syndrome_tail": 0.2,
        "safe_fast": 1.0,
        "runtime": 0.2,
        "confidence": 0.1,
    }
    if weights:
        merged.update(weights)
    for name in [
        "fast_wrong",
        "fast_logical_fail",
        "hard_runtime",
        "syndrome_tail",
        "safe_fast",
        "runtime",
        "confidence",
    ]:
        weighted_name = f"{name}_weight"
        if weighted_name in merged:
            merged[name] = float(merged[weighted_name])
    if "runtime_regression_weight" in merged:
        merged["runtime"] = float(merged["runtime_regression_weight"])
    return merged


def _loss_weight(weights: dict[str, float], name: str, legacy_name: str | None = None) -> float:
    if name in weights:
        return float(weights[name])
    key = f"{name}_weight"
    if key in weights:
        return float(weights[key])
    if legacy_name and legacy_name in weights:
        return float(weights[legacy_name])
    return float(weights.get(f"{legacy_name}_weight", 0.0)) if legacy_name else 0.0


def _bce_with_optional_pos_weight(
    logit: torch.Tensor,
    target: torch.Tensor,
    pos_weight_value: float | None,
) -> torch.Tensor:
    kwargs: dict[str, torch.Tensor] = {}
    if pos_weight_value is not None:
        kwargs["pos_weight"] = torch.as_tensor(pos_weight_value, dtype=torch.float32, device=logit.device)
    return F.binary_cross_entropy_with_logits(logit.float(), target.float(), **kwargs)


def compute_loss_breakdown(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weights: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute per-head losses."""
    merged_weights = {"correction": 1.0, "confidence": 1.0, "risk": 1.0}
    if weights:
        merged_weights.update(weights)
    correction_kwargs: dict[str, torch.Tensor] = {}
    correction_pos_weight = merged_weights.get("correction_pos_weight")
    if correction_pos_weight is not None:
        correction_kwargs["pos_weight"] = torch.as_tensor(
            float(correction_pos_weight),
            dtype=outputs["correction_logits"].dtype,
            device=outputs["correction_logits"].device,
        )
    correction = F.binary_cross_entropy_with_logits(
        outputs["correction_logits"],
        batch["correction_target"],
        **correction_kwargs,
    )
    dice_weight = float(merged_weights.get("correction_dice", 0.0))
    if dice_weight > 0.0:
        probs = torch.sigmoid(outputs["correction_logits"]).float()
        target = batch["correction_target"].float()
        intersection = (probs * target).sum(dim=1)
        denominator = probs.sum(dim=1) + target.sum(dim=1)
        dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0))
        correction = correction + dice_weight * dice_loss.mean()
    confidence = F.binary_cross_entropy_with_logits(outputs["confidence_logit"], batch["confidence_target"])
    risk = F.binary_cross_entropy_with_logits(outputs["risk_logit"], batch["risk_target"])
    return {
        "correction": correction * merged_weights["correction"],
        "confidence": confidence * merged_weights["confidence"],
        "risk": risk * merged_weights["risk"],
    }


def predecoder_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Return total training loss."""
    breakdown = compute_loss_breakdown(outputs, batch, weights)
    return sum(breakdown.values())


def decomposed_risk_loss(
    output: DecomposedRiskOutput | dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Compute multi-head decomposed risk/runtime loss."""
    values = _as_output_dict(output)
    weights = _decomposed_defaults(weights)
    fast_wrong = batch["fast_wrong"].float()
    fast_logical_fail = batch["fast_logical_fail"].float()
    hard_runtime = batch["hard_runtime"].float()
    syndrome_tail = batch.get("syndrome_tail", batch.get("syndrome_weight_tail"))
    if syndrome_tail is None:
        syndrome_tail = torch.zeros_like(hard_runtime)
    syndrome_tail = syndrome_tail.float()
    safe_fast = batch.get("safe_fast")
    if safe_fast is None:
        safe_fast = ((fast_wrong <= 0.5) & (fast_logical_fail <= 0.5)).float()
    safe_fast = safe_fast.float()
    runtime_target = torch.log1p(batch["accurate_runtime_us"].float())

    hard_runtime_valid = batch.get("hard_runtime_label_valid")
    hard_runtime_valid_flag = True
    if hard_runtime_valid is not None:
        hard_runtime_valid_flag = bool((hard_runtime_valid.float() > 0.5).all().item())

    fast_wrong_logit = values["fast_wrong_logit"].float()
    fast_logical_fail_logit = values["fast_logical_fail_logit"].float()
    hard_runtime_logit = values["hard_runtime_logit"].float()
    syndrome_tail_logit = values["syndrome_tail_logit"].float()
    safe_fast_logit = values["safe_fast_logit"].float()
    runtime_pred = values["runtime_pred"].float()
    confidence_logit = values["confidence_logit"].float()

    fast_wrong_loss = _bce_with_optional_pos_weight(
        fast_wrong_logit,
        fast_wrong,
        weights.get("fast_wrong_pos_weight"),
    )
    fast_fail_loss = _bce_with_optional_pos_weight(
        fast_logical_fail_logit,
        fast_logical_fail,
        weights.get("fast_logical_fail_pos_weight"),
    )
    if hard_runtime_valid_flag and _loss_weight(weights, "hard_runtime") > 0.0:
        hard_loss = _bce_with_optional_pos_weight(
            hard_runtime_logit,
            hard_runtime,
            weights.get("hard_runtime_pos_weight"),
        )
    else:
        hard_loss = hard_runtime_logit.sum() * 0.0
    syndrome_tail_loss = _bce_with_optional_pos_weight(
        syndrome_tail_logit,
        syndrome_tail,
        weights.get("syndrome_tail_pos_weight"),
    )
    safe_fast_loss = _bce_with_optional_pos_weight(
        safe_fast_logit,
        safe_fast,
        weights.get("safe_fast_pos_weight"),
    )
    runtime_loss = F.smooth_l1_loss(runtime_pred, runtime_target)

    combined_risk = values.get("combined_scheduler_risk", torch.sigmoid(values["risk_logit"].float()))
    safe_fast_prob = torch.sigmoid(safe_fast_logit)
    risk_label = batch.get("risk_label")
    if risk_label is None:
        risk_label = torch.maximum(torch.maximum(fast_wrong, fast_logical_fail), torch.maximum(hard_runtime, syndrome_tail))
    risk_pred = (combined_risk.detach() >= 0.5).float()
    safe_pred = (safe_fast_prob.detach() >= 0.5).float()
    confidence_target = ((risk_pred == risk_label.float()) & (safe_pred == safe_fast)).float()
    confidence_loss = F.binary_cross_entropy_with_logits(confidence_logit, confidence_target)

    return (
        _loss_weight(weights, "fast_wrong") * fast_wrong_loss
        + _loss_weight(weights, "fast_logical_fail") * fast_fail_loss
        + (_loss_weight(weights, "hard_runtime") if hard_runtime_valid_flag else 0.0) * hard_loss
        + _loss_weight(weights, "syndrome_tail") * syndrome_tail_loss
        + _loss_weight(weights, "safe_fast") * safe_fast_loss
        + _loss_weight(weights, "runtime", "runtime_regression") * runtime_loss
        + _loss_weight(weights, "confidence") * confidence_loss
    )


def risk_runtime_loss(
    output: RiskRuntimeOutput | DecomposedRiskOutput | dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Compute Risk/Runtime Profiler training loss.

    Components:
    - BCE on scheduler risk.
    - BCE on accurate-decoder hard-runtime tail label.
    - SmoothL1 on `log1p(accurate_runtime_us)`.
    - Confidence BCE where target is whether the current risk prediction is
      correct at threshold 0.5.
    """
    values = _as_output_dict(output)
    if "fast_wrong_logit" in values and "safe_fast_logit" in values:
        return decomposed_risk_loss(values, batch, weights)
    weights = weights or {
        "risk_weight": 1.0,
        "hard_runtime_weight": 0.5,
        "runtime_weight": 0.2,
        "confidence_weight": 0.1,
    }
    risk_label = batch["risk_label"].float()
    hard_runtime = batch["hard_runtime"].float()
    hard_runtime_valid = batch.get("hard_runtime_label_valid")
    hard_runtime_valid_flag = True
    if hard_runtime_valid is not None:
        hard_runtime_valid_flag = bool((hard_runtime_valid.float() > 0.5).all().item())
    runtime_target = torch.log1p(batch["accurate_runtime_us"].float())
    risk_logit = values["risk_logit"].float()
    hard_runtime_logit = values.get("hard_runtime_logit", values["runtime_logit"]).float()
    runtime_pred = values["runtime_pred"].float()
    confidence_logit = values["confidence_logit"].float()

    risk_pos_weight = weights.get("risk_pos_weight")
    hard_pos_weight = weights.get("hard_runtime_pos_weight")
    risk_kwargs = {}
    hard_kwargs = {}
    if risk_pos_weight is not None:
        risk_kwargs["pos_weight"] = torch.as_tensor(risk_pos_weight, dtype=torch.float32, device=risk_logit.device)
    if hard_pos_weight is not None:
        hard_kwargs["pos_weight"] = torch.as_tensor(hard_pos_weight, dtype=torch.float32, device=hard_runtime_logit.device)

    risk_loss = F.binary_cross_entropy_with_logits(risk_logit, risk_label, **risk_kwargs)
    if hard_runtime_valid_flag and float(weights.get("hard_runtime_weight", weights.get("hard_runtime", 0.5))) > 0.0:
        hard_runtime_loss = F.binary_cross_entropy_with_logits(hard_runtime_logit, hard_runtime, **hard_kwargs)
    else:
        hard_runtime_loss = hard_runtime_logit.sum() * 0.0
    runtime_loss = F.smooth_l1_loss(runtime_pred, runtime_target)
    risk_pred = (torch.sigmoid(risk_logit).detach() > 0.5).float()
    confidence_target = (risk_pred == risk_label).float()
    confidence_loss = F.binary_cross_entropy_with_logits(confidence_logit, confidence_target)
    return (
        float(weights.get("risk_weight", weights.get("risk", 1.0))) * risk_loss
        + (float(weights.get("hard_runtime_weight", weights.get("hard_runtime", 0.5))) if hard_runtime_valid_flag else 0.0)
        * hard_runtime_loss
        + float(weights.get("runtime_weight", weights.get("runtime_regression", 0.2))) * runtime_loss
        + float(weights.get("confidence_weight", weights.get("confidence", 0.1))) * confidence_loss
    )


def candidate_predecoder_loss(
    output: CandidatePredecoderOutput | dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Compute selective candidate predecoder loss.

    Candidate labels use `0..C-1` for DEM candidates and `C` for abstain. This
    keeps abstention part of the supervised action space rather than a
    post-hoc threshold only.
    """
    values = _as_output_dict(output)
    weights = weights or {
        "candidate_weight": 1.0,
        "risk_weight": 0.5,
        "confidence_weight": 0.1,
    }
    candidate_logits = values["candidate_logits"].float()
    abstain_logit = values["abstain_logit"].float().unsqueeze(-1)
    action_logits = torch.cat([candidate_logits, abstain_logit], dim=-1)
    candidate_loss = F.cross_entropy(action_logits, batch["candidate_label"].long())
    risk_loss = F.binary_cross_entropy_with_logits(values["risk_logit"].float(), batch["risk_label"].float())
    action_pred = torch.argmax(action_logits.detach(), dim=-1)
    confidence_target = (action_pred == batch["candidate_label"].long()).float()
    confidence_loss = F.binary_cross_entropy_with_logits(values["confidence_logit"].float(), confidence_target)
    return (
        float(weights.get("candidate_weight", 1.0)) * candidate_loss
        + float(weights.get("risk_weight", 0.5)) * risk_loss
        + float(weights.get("confidence_weight", 0.1)) * confidence_loss
    )


@torch.no_grad()
def compute_risk_runtime_metrics(
    output: RiskRuntimeOutput | DecomposedRiskOutput | dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute scheduler-facing metrics for risk/runtime outputs."""
    values = _as_output_dict(output)
    if "fast_wrong_logit" in values and "safe_fast_logit" in values:
        return compute_decomposed_risk_metrics(values, batch, threshold=threshold)
    labels = batch["risk_label"].detach().cpu().numpy().astype(int)
    scores = torch.sigmoid(values["risk_logit"]).detach().cpu().numpy().astype(float)
    predictions = (scores >= threshold).astype(int)
    hard_labels = batch["hard_runtime"].detach().cpu().numpy().astype(int)
    hard_scores = torch.sigmoid(values.get("hard_runtime_logit", values["runtime_logit"])).detach().cpu().numpy().astype(float)
    hard_predictions = (hard_scores >= threshold).astype(int)
    hard_valid = batch.get("hard_runtime_label_valid")
    hard_runtime_label_valid = True if hard_valid is None else bool((hard_valid.detach().cpu().float() > 0.5).all().item())
    tp = int(((predictions == 1) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    positives = max(int((labels == 1).sum()), 1)
    negatives = max(int((labels == 0).sum()), 1)
    runtime_target = np.log1p(batch["accurate_runtime_us"].detach().cpu().numpy().astype(float))
    runtime_pred = values["runtime_pred"].detach().cpu().numpy().astype(float)
    confidence = torch.sigmoid(values["confidence_logit"]).detach().cpu().numpy().astype(float)
    return {
        "risk_accuracy": float((tp + tn) / max(len(labels), 1)),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / positives),
        "fpr": float(fp / negatives),
        "fnr": float(fn / positives),
        "hard_runtime_accuracy": float((hard_predictions == hard_labels).mean()) if len(hard_labels) and hard_runtime_label_valid else 0.0,
        "hard_runtime_label_valid": float(hard_runtime_label_valid),
        "runtime_mae": float(np.mean(np.abs(runtime_pred - runtime_target))) if len(runtime_target) else 0.0,
        "confidence_mean": float(np.mean(confidence)) if len(confidence) else 0.0,
    }


def _binary_metric_values(labels: np.ndarray, predictions: np.ndarray, prefix: str | None = None) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int).reshape(-1)
    predictions = np.asarray(predictions, dtype=int).reshape(-1)
    tp = int(((predictions == 1) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    positives = max(int((labels == 1).sum()), 1)
    negatives = max(int((labels == 0).sum()), 1)
    values = {
        "accuracy": float((tp + tn) / max(len(labels), 1)),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / positives),
        "fnr": float(fn / positives),
        "fpr": float(fp / negatives),
    }
    if prefix:
        return {f"{prefix}_{key}": value for key, value in values.items()}
    return values


@torch.no_grad()
def compute_decomposed_risk_metrics(
    output: DecomposedRiskOutput | dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute decomposed risk metrics plus legacy summary fields."""
    values = _as_output_dict(output)
    risk_labels = batch["risk_label"].detach().cpu().numpy().astype(int)
    combined_scores = values.get("combined_scheduler_risk", torch.sigmoid(values["risk_logit"]))
    combined_predictions = (combined_scores.detach().cpu().numpy().astype(float) >= threshold).astype(int)
    metrics = _binary_metric_values(risk_labels, combined_predictions, prefix="combined_risk")
    metrics.update(
        {
            "risk_accuracy": metrics["combined_risk_accuracy"],
            "precision": metrics["combined_risk_precision"],
            "recall": metrics["combined_risk_recall"],
            "fnr": metrics["combined_risk_fnr"],
            "fpr": metrics["combined_risk_fpr"],
        }
    )
    for label_key, logit_key, prefix in [
        ("fast_wrong", "fast_wrong_logit", "fast_wrong"),
        ("fast_logical_fail", "fast_logical_fail_logit", "fast_logical_fail"),
    ]:
        labels = batch[label_key].detach().cpu().numpy().astype(int)
        predictions = (torch.sigmoid(values[logit_key]).detach().cpu().numpy().astype(float) >= threshold).astype(int)
        metrics.update(_binary_metric_values(labels, predictions, prefix=prefix))
    safe_labels_tensor = batch.get("safe_fast")
    if safe_labels_tensor is None:
        safe_labels_tensor = ((batch["fast_wrong"] <= 0.5) & (batch["fast_logical_fail"] <= 0.5)).float()
    safe_labels = safe_labels_tensor.detach().cpu().numpy().astype(int)
    safe_predictions = (torch.sigmoid(values["safe_fast_logit"]).detach().cpu().numpy().astype(float) >= threshold).astype(int)
    safe_metrics = _binary_metric_values(safe_labels, safe_predictions, prefix="safe_fast")
    metrics["safe_fast_precision"] = safe_metrics["safe_fast_precision"]
    metrics["safe_fast_recall"] = safe_metrics["safe_fast_recall"]

    hard_valid = batch.get("hard_runtime_label_valid")
    hard_runtime_label_valid = True if hard_valid is None else bool((hard_valid.detach().cpu().float() > 0.5).all().item())
    hard_labels = batch["hard_runtime"].detach().cpu().numpy().astype(int)
    hard_predictions = (torch.sigmoid(values["hard_runtime_logit"]).detach().cpu().numpy().astype(float) >= threshold).astype(int)
    runtime_target = np.log1p(batch["accurate_runtime_us"].detach().cpu().numpy().astype(float))
    runtime_pred = values["runtime_pred"].detach().cpu().numpy().astype(float)
    confidence = torch.sigmoid(values["confidence_logit"]).detach().cpu().numpy().astype(float)
    metrics.update(
        {
            "hard_runtime_accuracy": float((hard_predictions == hard_labels).mean()) if len(hard_labels) and hard_runtime_label_valid else 0.0,
            "hard_runtime_label_valid": float(hard_runtime_label_valid),
            "runtime_mae": float(np.mean(np.abs(runtime_pred - runtime_target))) if len(runtime_target) else 0.0,
            "confidence_mean": float(np.mean(confidence)) if len(confidence) else 0.0,
        }
    )
    return metrics


@torch.no_grad()
def compute_candidate_metrics(
    output: CandidatePredecoderOutput | dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Compute candidate-predecoder metrics for scaffold training."""
    values = _as_output_dict(output)
    candidate_logits = values["candidate_logits"].float()
    abstain_logit = values["abstain_logit"].float().unsqueeze(-1)
    action_logits = torch.cat([candidate_logits, abstain_logit], dim=-1)
    predictions = torch.argmax(action_logits, dim=-1)
    labels = batch["candidate_label"].long()
    abstain_class = candidate_logits.shape[1]
    confidence = torch.sigmoid(values["confidence_logit"].float())
    accept_predictions = predictions != abstain_class
    return {
        "candidate_accuracy": float((predictions == labels).float().mean().item()),
        "abstain_rate": float((predictions == abstain_class).float().mean().item()),
        "false_accept_rate": float(((accept_predictions) & (labels == abstain_class)).float().mean().item()),
        "confidence_mean": float(confidence.mean().item()),
    }
