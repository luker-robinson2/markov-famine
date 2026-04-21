"""Lead-time dependent accuracy analysis.

Evaluates how forecast skill degrades (or improves) with prediction horizon.
This is critical for operational food security early warning systems.

Key insight from literature: 7-month forecasts can be more reliable than
3-month forecasts because longer horizons smooth seasonal volatility.
"""

import numpy as np
import pandas as pd
from typing import Optional

from src.config import N_STATES, LEAD_TIME_HORIZONS
from src.metrics.probabilistic import ranked_probability_skill_score, multi_class_log_loss
from src.metrics.ordinal import (
    crisis_detection_metrics,
    quadratic_weighted_kappa,
    weighted_f1_score,
    mean_absolute_ordinal_error,
)


def leadtime_accuracy_curve(
    predictions: dict[int, np.ndarray],
    y_true: dict[int, np.ndarray],
    horizons: list[int] = LEAD_TIME_HORIZONS,
) -> pd.DataFrame:
    """Compute accuracy metrics at each lead time.

    Parameters
    ----------
    predictions : dict
        Maps lead_time -> (n_samples, n_states) predicted probabilities.
    y_true : dict
        Maps lead_time -> (n_samples,) true labels (0-indexed).
    horizons : list
        Lead times in months to evaluate.

    Returns
    -------
    pd.DataFrame
        Columns: lead_time, accuracy, rpss, qwk, weighted_f1,
                 crisis_recall, crisis_precision, crisis_f1, log_loss, mae
    """
    records = []
    for h in horizons:
        if h not in predictions or h not in y_true:
            continue

        probs = predictions[h]
        true = y_true[h]
        pred_labels = np.argmax(probs, axis=1)

        accuracy = float(np.mean(pred_labels == true))
        rpss = ranked_probability_skill_score(probs, true)
        qwk = quadratic_weighted_kappa(true, pred_labels)
        wf1 = weighted_f1_score(true, pred_labels)
        crisis = crisis_detection_metrics(true, pred_labels)
        logloss = multi_class_log_loss(probs, true)
        mae = mean_absolute_ordinal_error(true, pred_labels)

        records.append({
            "lead_time": h,
            "accuracy": accuracy,
            "rpss": rpss,
            "qwk": qwk,
            "weighted_f1": wf1,
            "crisis_recall": crisis["recall"],
            "crisis_precision": crisis["precision"],
            "crisis_f1": crisis["f1"],
            "false_alarm_rate": crisis["false_alarm_rate"],
            "log_loss": logloss,
            "mae": mae,
        })

    return pd.DataFrame(records)


def crisis_detection_by_leadtime(
    predictions: dict[int, np.ndarray],
    y_true: dict[int, np.ndarray],
    horizons: list[int] = LEAD_TIME_HORIZONS,
    crisis_threshold: int = 2,
) -> pd.DataFrame:
    """Compute Phase 3+ detection rate at each lead time.

    The operationally critical metric: how early can we detect crisis onset?

    Parameters
    ----------
    predictions : dict
        Maps lead_time -> predicted probabilities.
    y_true : dict
        Maps lead_time -> true labels (0-indexed).
    horizons : list
        Lead times to evaluate.
    crisis_threshold : int
        Phase index for crisis (0-indexed, default 2 = Phase 3).

    Returns
    -------
    pd.DataFrame
        Columns: lead_time, detection_rate, false_alarm_rate,
                 crisis_probability_mean, n_crisis_cases, n_total
    """
    records = []
    for h in horizons:
        if h not in predictions or h not in y_true:
            continue

        probs = predictions[h]
        true = y_true[h]

        true_crisis = true >= crisis_threshold
        # Probability of Phase 3+ = sum of probabilities for phases 3,4,5
        p_crisis = probs[:, crisis_threshold:].sum(axis=1)

        # Binary prediction: crisis if P(3+) > 0.5
        pred_crisis = p_crisis > 0.5

        tp = np.sum(true_crisis & pred_crisis)
        fp = np.sum(~true_crisis & pred_crisis)
        fn = np.sum(true_crisis & ~pred_crisis)
        tn = np.sum(~true_crisis & ~pred_crisis)

        detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        records.append({
            "lead_time": h,
            "detection_rate": float(detection_rate),
            "false_alarm_rate": float(far),
            "crisis_probability_mean": float(p_crisis[true_crisis].mean()) if true_crisis.any() else 0.0,
            "n_crisis_cases": int(true_crisis.sum()),
            "n_total": len(true),
        })

    return pd.DataFrame(records)


def onset_detection_analysis(
    phase_sequences: dict[str, np.ndarray],
    predicted_sequences: dict[str, np.ndarray],
    crisis_threshold: int = 3,
) -> pd.DataFrame:
    """Analyze how well the model detects crisis onsets.

    An onset is when a region transitions from Phase <3 to Phase >=3.
    Measures how many months before the onset the model first predicted it.

    Parameters
    ----------
    phase_sequences : dict
        Maps region_code -> array of true IPC phases (1-indexed) over time.
    predicted_sequences : dict
        Maps region_code -> array of predicted IPC phases (1-indexed) over time.

    Returns
    -------
    pd.DataFrame
        Columns: region_code, onset_time_idx, first_prediction_idx,
                 lead_time_months, detected
    """
    records = []

    for region, true_seq in phase_sequences.items():
        pred_seq = predicted_sequences.get(region)
        if pred_seq is None:
            continue

        true_seq = np.asarray(true_seq)
        pred_seq = np.asarray(pred_seq)

        # Find onset events: transition from <3 to >=3
        for t in range(1, len(true_seq)):
            if true_seq[t] >= crisis_threshold and true_seq[t - 1] < crisis_threshold:
                # This is an onset at time t
                # Look back to find earliest prediction of crisis
                first_pred = None
                for tp in range(max(0, t - 12), t + 1):
                    if tp < len(pred_seq) and pred_seq[tp] >= crisis_threshold:
                        first_pred = tp
                        break

                detected = first_pred is not None
                lead = t - first_pred if detected else 0

                records.append({
                    "region_code": region,
                    "onset_time_idx": t,
                    "first_prediction_idx": first_pred,
                    "lead_time_months": lead if detected else None,
                    "detected": detected,
                })

    return pd.DataFrame(records)
