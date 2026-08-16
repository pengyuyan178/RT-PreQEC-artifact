"""Confidence calibration helpers."""

from __future__ import annotations

import numpy as np


def temperature_scale_logits(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to logits."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return np.asarray(logits) / temperature


def choose_confidence_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    target_false_accept_rate: float,
) -> float:
    """Choose the lowest threshold satisfying the false-accept target."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    thresholds = np.unique(scores)
    best = 1.0
    for threshold in np.sort(thresholds):
        accepted = scores >= threshold
        false_accepts = accepted & ~labels
        rate = false_accepts.sum() / max(accepted.sum(), 1)
        if rate <= target_false_accept_rate:
            best = float(threshold)
            break
    return best


def evaluate_selective_risk(
    confidence: np.ndarray,
    is_correct: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Evaluate selective acceptance quality at a threshold."""
    confidence = np.asarray(confidence, dtype=float)
    is_correct = np.asarray(is_correct, dtype=bool)
    accepted = confidence >= threshold
    accepted_count = int(accepted.sum())
    correct_count = int((accepted & is_correct).sum())
    return {
        "accept_rate": float(accepted.mean()) if len(accepted) else 0.0,
        "accepted_accuracy": float(correct_count / max(accepted_count, 1)),
        "false_accept_rate": float(((accepted) & (~is_correct)).sum() / max(accepted_count, 1)),
    }


def _binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    tp = int(((predictions == 1) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    positives = max(int((labels == 1).sum()), 1)
    negatives = max(int((labels == 0).sum()), 1)
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / positives)
    return {
        "risk_accuracy": float((tp + tn) / max(len(labels), 1)),
        "precision": precision,
        "recall": recall,
        "false_negative_rate": float(fn / positives),
        "false_positive_rate": float(fp / negatives),
        "f1": float(2.0 * precision * recall / max(precision + recall, 1e-12)),
    }


def _safe_fast_labels(labels: np.ndarray, fast_wrong_labels: np.ndarray | None, fast_fail_labels: np.ndarray | None) -> np.ndarray:
    if fast_wrong_labels is None or fast_fail_labels is None:
        return 1 - np.asarray(labels, dtype=int)
    return ((np.asarray(fast_wrong_labels, dtype=int) == 0) & (np.asarray(fast_fail_labels, dtype=int) == 0)).astype(int)


def sweep_risk_confidence_thresholds(
    risk_scores: np.ndarray,
    confidence_scores: np.ndarray,
    labels: np.ndarray,
    risk_thresholds: np.ndarray | None = None,
    confidence_thresholds: np.ndarray | None = None,
) -> list[dict[str, float]]:
    """Evaluate risk/confidence threshold grid on validation predictions."""
    risk_scores = np.asarray(risk_scores, dtype=float).reshape(-1)
    confidence_scores = np.asarray(confidence_scores, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    risk_thresholds = (
        np.asarray(risk_thresholds, dtype=float)
        if risk_thresholds is not None
        else np.round(np.arange(0.05, 1.0, 0.05), 4)
    )
    confidence_thresholds = (
        np.asarray(confidence_thresholds, dtype=float)
        if confidence_thresholds is not None
        else np.round(np.concatenate([np.arange(0.0, 1.0, 0.1), np.asarray([0.95])]), 4)
    )
    rows: list[dict[str, float]] = []
    for risk_threshold in risk_thresholds.tolist():
        for confidence_threshold in confidence_thresholds.tolist():
            high_risk = risk_scores >= float(risk_threshold)
            low_confidence = confidence_scores < float(confidence_threshold)
            accurate_selected = high_risk | low_confidence
            fast_selected = ~accurate_selected
            predictions = high_risk.astype(int)
            metrics = _binary_metrics(labels, predictions)
            false_negative_fast = fast_selected & (labels == 1)
            row = {
                "risk_threshold": float(risk_threshold),
                "confidence_threshold": float(confidence_threshold),
                **metrics,
                "expected_fast_selection_rate": float(fast_selected.mean()) if len(fast_selected) else 0.0,
                "expected_accurate_selection_rate": float(accurate_selected.mean()) if len(accurate_selected) else 0.0,
                "expected_oracle_gap": float(false_negative_fast.mean()) if len(false_negative_fast) else 0.0,
            }
            rows.append(row)
    return rows


def sweep_decomposed_thresholds(
    combined_fast_risk: np.ndarray,
    safe_fast_prob: np.ndarray,
    confidence_scores: np.ndarray,
    labels: np.ndarray,
    fast_wrong_labels: np.ndarray | None = None,
    fast_logical_fail_labels: np.ndarray | None = None,
    risk_thresholds: np.ndarray | None = None,
    safe_fast_thresholds: np.ndarray | None = None,
    confidence_thresholds: np.ndarray | None = None,
) -> list[dict[str, float]]:
    """Evaluate decomposed fast-risk, safe-fast, and confidence threshold grid."""
    combined_fast_risk = np.asarray(combined_fast_risk, dtype=float).reshape(-1)
    safe_fast_prob = np.asarray(safe_fast_prob, dtype=float).reshape(-1)
    confidence_scores = np.asarray(confidence_scores, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    safe_labels = _safe_fast_labels(labels, fast_wrong_labels, fast_logical_fail_labels)
    risk_thresholds = (
        np.asarray(risk_thresholds, dtype=float)
        if risk_thresholds is not None
        else np.round(np.arange(0.05, 1.0, 0.05), 4)
    )
    safe_fast_thresholds = (
        np.asarray(safe_fast_thresholds, dtype=float)
        if safe_fast_thresholds is not None
        else np.round(np.arange(0.05, 1.0, 0.05), 4)
    )
    confidence_thresholds = (
        np.asarray(confidence_thresholds, dtype=float)
        if confidence_thresholds is not None
        else np.round(np.concatenate([np.arange(0.0, 1.0, 0.1), np.asarray([0.95])]), 4)
    )
    rows: list[dict[str, float]] = []
    for risk_threshold in risk_thresholds.tolist():
        for safe_fast_threshold in safe_fast_thresholds.tolist():
            for confidence_threshold in confidence_thresholds.tolist():
                fast_selected = (
                    (safe_fast_prob >= float(safe_fast_threshold))
                    & (combined_fast_risk <= float(risk_threshold))
                    & (confidence_scores >= float(confidence_threshold))
                )
                accurate_selected = ~fast_selected
                high_risk_prediction = (~fast_selected).astype(int)
                metrics = _binary_metrics(labels, high_risk_prediction)
                safe_predictions = (safe_fast_prob >= float(safe_fast_threshold)).astype(int)
                safe_metrics = _binary_metrics(safe_labels, safe_predictions)
                false_negative_fast = fast_selected & (labels == 1)
                row = {
                    "risk_threshold": float(risk_threshold),
                    "safe_fast_threshold": float(safe_fast_threshold),
                    "confidence_threshold": float(confidence_threshold),
                    **metrics,
                    "safe_fast_precision": safe_metrics["precision"],
                    "safe_fast_recall": safe_metrics["recall"],
                    "expected_fast_selection_rate": float(fast_selected.mean()) if len(fast_selected) else 0.0,
                    "expected_accurate_selection_rate": float(accurate_selected.mean()) if len(accurate_selected) else 0.0,
                    "expected_oracle_gap": float(false_negative_fast.mean()) if len(false_negative_fast) else 0.0,
                }
                rows.append(row)
    return rows


def select_calibration_thresholds(
    sweep_rows: list[dict[str, float]],
    objective: dict,
) -> dict[str, float]:
    """Select a calibration row according to the configured validation objective."""
    if not sweep_rows:
        raise ValueError("sweep_rows is empty")
    objective_type = str(objective.get("type", "maximize_f1"))
    if objective_type == "minimize_fnr_under_fast_rate":
        max_fnr = float(objective.get("max_false_negative_rate", 0.05))
        min_fast = float(objective.get("min_fast_selection_rate", 0.2))
        feasible = [
            row
            for row in sweep_rows
            if row["false_negative_rate"] <= max_fnr and row["expected_fast_selection_rate"] >= min_fast
        ]
        candidates = feasible or sweep_rows
        return min(
            candidates,
            key=lambda row: (
                row["false_negative_rate"],
                -row["expected_fast_selection_rate"],
                -row["f1"],
            ),
        )
    if objective_type == "latency_accuracy_tradeoff":
        lambda_fnr = float(objective.get("lambda_fnr", 10.0))
        lambda_fast = float(objective.get("lambda_fast_rate", -1.0))
        return min(
            sweep_rows,
            key=lambda row: lambda_fnr * row["false_negative_rate"]
            + lambda_fast * row["expected_fast_selection_rate"]
            + row["expected_oracle_gap"],
        )
    if objective_type == "maximize_f1":
        return max(sweep_rows, key=lambda row: (row["f1"], row["expected_fast_selection_rate"], -row["false_negative_rate"]))
    raise ValueError(f"Unsupported calibration objective: {objective_type}")
