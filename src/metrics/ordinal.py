"""Ordinal and asymmetric cost metrics for IPC phase prediction.

Standard classification metrics don't capture the ordinal nature of IPC
phases or the asymmetric costs of different errors. Predicting Phase 1
when the truth is Phase 4 (missing a crisis) is far worse than predicting
Phase 3 when the truth is Phase 2 (false alarm).
"""

import numpy as np
import pandas as pd
from typing import Optional

from src.config import N_STATES, IPC_LABELS, MISCLASS_COST_MATRIX_WEIGHTS


def ordinal_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_states: int = N_STATES,
) -> np.ndarray:
    """Compute confusion matrix for ordinal IPC phases.

    Parameters
    ----------
    y_true : np.ndarray
        True labels (0-indexed).
    y_pred : np.ndarray
        Predicted labels (0-indexed).
    n_states : int
        Number of ordinal categories.

    Returns
    -------
    np.ndarray
        Shape (n_states, n_states). Row = true, Column = predicted.
    """
    cm = np.zeros((n_states, n_states), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def asymmetric_cost_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    crisis_threshold: int = 2,  # 0-indexed: phases 3,4,5 = indices 2,3,4
    fn_weight: float = MISCLASS_COST_MATRIX_WEIGHTS["false_negative_crisis"],
    fp_weight: float = MISCLASS_COST_MATRIX_WEIGHTS["false_positive_crisis"],
) -> float:
    """Compute asymmetric cost score for crisis detection.

    False negatives (missing a crisis, Phase 3+) are weighted more heavily
    than false positives (false alarm for Phase 3+).

    Parameters
    ----------
    y_true : np.ndarray
        True labels (0-indexed).
    y_pred : np.ndarray
        Predicted labels (0-indexed).
    crisis_threshold : int
        Phase index threshold for crisis (0-indexed). Default 2 = Phase 3.
    fn_weight : float
        Cost multiplier for false negatives (missing crisis).
    fp_weight : float
        Cost multiplier for false positives (false alarm).

    Returns
    -------
    float
        Weighted cost. Lower is better.
    """
    true_crisis = y_true >= crisis_threshold
    pred_crisis = y_pred >= crisis_threshold

    false_negatives = np.sum(true_crisis & ~pred_crisis)
    false_positives = np.sum(~true_crisis & pred_crisis)
    true_positives = np.sum(true_crisis & pred_crisis)
    true_negatives = np.sum(~true_crisis & ~pred_crisis)

    total = len(y_true)
    cost = (fn_weight * false_negatives + fp_weight * false_positives) / total
    return float(cost)


def weighted_f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_states: int = N_STATES,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Compute weighted F1 score with humanitarian cost weights.

    Higher-severity phases get higher weight to prioritize
    correct prediction of crises.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        True and predicted labels (0-indexed).
    n_states : int
        Number of classes.
    weights : np.ndarray, optional
        Per-class weights. Default: [1, 1, 2, 3, 4] (crisis phases weighted higher).

    Returns
    -------
    float
        Weighted F1 score in [0, 1]. Higher is better.
    """
    if weights is None:
        weights = np.array([1.0, 1.0, 2.0, 3.0, 4.0])[:n_states]

    f1_per_class = np.zeros(n_states)
    for c in range(n_states):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        if precision + recall > 0:
            f1_per_class[c] = 2 * precision * recall / (precision + recall)

    return float(np.average(f1_per_class, weights=weights))


def quadratic_weighted_kappa(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_states: int = N_STATES,
) -> float:
    """Compute Quadratic Weighted Kappa for ordinal agreement.

    QWK penalizes disagreements proportional to the square of the ordinal
    distance. Off-by-3 errors are penalized 9x more than off-by-1 errors.

    kappa = 1 - (sum W_ij * O_ij) / (sum W_ij * E_ij)

    where W_ij = (i-j)^2 / (K-1)^2, O is observed, E is expected by chance.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Labels (0-indexed).
    n_states : int
        Number of ordinal categories.

    Returns
    -------
    float
        QWK in [-1, 1]. 1 = perfect agreement, 0 = chance, <0 = worse than chance.
    """
    # Observed confusion matrix
    O = ordinal_confusion_matrix(y_true, y_pred, n_states).astype(float)

    # Expected confusion matrix (outer product of marginals)
    hist_true = np.bincount(y_true.astype(int), minlength=n_states).astype(float)
    hist_pred = np.bincount(y_pred.astype(int), minlength=n_states).astype(float)
    E = np.outer(hist_true, hist_pred) / len(y_true)

    # Quadratic weight matrix
    W = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(n_states):
            W[i, j] = (i - j) ** 2 / (n_states - 1) ** 2

    numerator = np.sum(W * O)
    denominator = np.sum(W * E)

    if denominator == 0:
        return 1.0  # Perfect agreement

    return 1.0 - numerator / denominator


def crisis_detection_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    crisis_threshold: int = 2,
) -> dict:
    """Compute Phase 3+ crisis detection metrics.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Labels (0-indexed).
    crisis_threshold : int
        Index threshold for crisis (default 2 = Phase 3).

    Returns
    -------
    dict with keys: precision, recall, f1, false_alarm_rate, hit_rate,
                    critical_success_index
    """
    true_crisis = y_true >= crisis_threshold
    pred_crisis = y_pred >= crisis_threshold

    tp = np.sum(true_crisis & pred_crisis)
    fp = np.sum(~true_crisis & pred_crisis)
    fn = np.sum(true_crisis & ~pred_crisis)
    tn = np.sum(~true_crisis & ~pred_crisis)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Hit rate / POD
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0  # Critical Success Index

    return {
        "precision": float(precision),
        "recall": float(recall),  # Same as hit_rate / POD
        "f1": float(f1),
        "false_alarm_rate": float(false_alarm_rate),
        "hit_rate": float(recall),
        "critical_success_index": float(csi),
    }


def mean_absolute_ordinal_error(
    y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    """Mean absolute error in ordinal distance.

    MAE = (1/N) * sum |true_phase - pred_phase|

    Returns
    -------
    float
        Mean ordinal distance of errors. Lower is better.
    """
    return float(np.mean(np.abs(y_true.astype(float) - y_pred.astype(float))))


def heidke_skill_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_states: int = N_STATES,
) -> float:
    """Heidke Skill Score for categorical forecasts.

    HSS = (PC - PC_chance) / (1 - PC_chance)

    where PC is the proportion correct and PC_chance is the proportion
    correct expected by a forecast that is independent of observations
    (outer product of marginals). Chance-corrected accuracy; ranges
    (-inf, 1], with 0 = no skill and 1 = perfect. Reference:
    Mason & Stephenson 2008, *Forecast Verification*, Wiley.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Labels (0-indexed).
    n_states : int
        Number of categories.

    Returns
    -------
    float
        HSS value.
    """
    cm = ordinal_confusion_matrix(y_true, y_pred, n_states).astype(float)
    n = cm.sum()
    if n == 0:
        return 0.0

    pc = np.trace(cm) / n
    row_marg = cm.sum(axis=1) / n
    col_marg = cm.sum(axis=0) / n
    pc_chance = float(np.sum(row_marg * col_marg))

    denom = 1.0 - pc_chance
    if denom == 0:
        return 0.0
    return float((pc - pc_chance) / denom)


def f1_macro(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_states: int = N_STATES,
) -> float:
    """Unweighted mean F1 across classes.

    Standard F1-macro for ordinal classification. Differs from
    :func:`weighted_f1_score` which applies humanitarian cost weights.
    Use F1-macro when comparing against literature (Andrée et al. 2022
    *Science Advances*, Busker et al. 2024 *Earth's Future*).

    Returns
    -------
    float
        F1-macro in [0, 1]. Higher is better.
    """
    f1_per_class = np.zeros(n_states)
    for c in range(n_states):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        if precision + recall > 0:
            f1_per_class[c] = 2 * precision * recall / (precision + recall)
    return float(np.mean(f1_per_class))
