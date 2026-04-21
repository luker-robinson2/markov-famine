"""Probabilistic forecast metrics for food security prediction.

Implements metrics designed for evaluating probabilistic predictions
of ordinal outcomes (IPC phases 1-5).

Key metrics:
- Ranked Probability Score / Skill Score (RPS/RPSS)
- Brier Score / Skill Score (BS/BSS)
- Log-loss (multi-class cross-entropy)
- Calibration curve data
"""

import numpy as np
import pandas as pd
from typing import Optional

from src.config import N_STATES


def ranked_probability_score(
    predicted_probs: np.ndarray,
    true_label: int,
    n_states: int = N_STATES,
) -> float:
    """Compute Ranked Probability Score for a single prediction.

    RPS measures the sum of squared differences between cumulative
    predicted and observed distributions. It penalizes predictions
    that are far from the true category more heavily than those nearby.

    RPS = (1 / (K-1)) * sum_{k=1}^{K} (CDF_pred(k) - CDF_obs(k))^2

    where K is the number of ordered categories.

    Parameters
    ----------
    predicted_probs : np.ndarray
        Shape (n_states,). Predicted probability for each IPC phase.
    true_label : int
        True IPC phase (0-indexed).
    n_states : int
        Number of ordered categories.

    Returns
    -------
    float
        RPS value in [0, 1]. Lower is better.
    """
    # Cumulative predicted distribution
    cdf_pred = np.cumsum(predicted_probs)

    # Cumulative observed distribution (step function at true label)
    cdf_obs = np.zeros(n_states)
    cdf_obs[true_label:] = 1.0

    rps = np.sum((cdf_pred - cdf_obs) ** 2) / (n_states - 1)
    return float(rps)


def ranked_probability_skill_score(
    predicted_probs: np.ndarray,
    y_true: np.ndarray,
    reference_probs: Optional[np.ndarray] = None,
    n_states: int = N_STATES,
) -> float:
    """Compute Ranked Probability Skill Score.

    RPSS = 1 - RPS_forecast / RPS_reference

    RPSS > 0: forecast beats reference (climatology by default)
    RPSS = 0: forecast equals reference
    RPSS < 0: forecast worse than reference

    Parameters
    ----------
    predicted_probs : np.ndarray
        Shape (n_samples, n_states). Model predictions.
    y_true : np.ndarray
        Shape (n_samples,). True labels (0-indexed).
    reference_probs : np.ndarray, optional
        Shape (n_states,) or (n_samples, n_states). Reference forecast.
        If None, uses climatological frequencies from y_true.
    n_states : int
        Number of ordered categories.

    Returns
    -------
    float
        RPSS value. Range (-inf, 1]. Higher is better.
    """
    n_samples = len(y_true)

    # Compute reference if not provided (climatological frequencies)
    if reference_probs is None:
        counts = np.zeros(n_states)
        for y in y_true:
            counts[int(y)] += 1
        reference_probs = counts / counts.sum()

    # Expand reference to per-sample if needed
    if reference_probs.ndim == 1:
        ref = np.tile(reference_probs, (n_samples, 1))
    else:
        ref = reference_probs

    # Compute mean RPS for forecast and reference
    rps_forecast = np.mean([
        ranked_probability_score(predicted_probs[i], int(y_true[i]), n_states)
        for i in range(n_samples)
    ])

    rps_reference = np.mean([
        ranked_probability_score(ref[i], int(y_true[i]), n_states)
        for i in range(n_samples)
    ])

    if rps_reference == 0:
        return 0.0  # Perfect reference, can't beat it

    return 1.0 - rps_forecast / rps_reference


def brier_score(
    predicted_probs: np.ndarray,
    y_true: np.ndarray,
    class_idx: int,
) -> float:
    """Compute Brier Score for a single class (one-vs-rest).

    BS = (1/N) * sum_{i=1}^{N} (p_i - o_i)^2

    where p_i is predicted probability for the class, o_i is 1 if
    true class, 0 otherwise.

    Parameters
    ----------
    predicted_probs : np.ndarray
        Shape (n_samples, n_states). Full probability matrix.
    y_true : np.ndarray
        Shape (n_samples,). True labels (0-indexed).
    class_idx : int
        Which class to evaluate (0-indexed).

    Returns
    -------
    float
        Brier score in [0, 1]. Lower is better.
    """
    p = predicted_probs[:, class_idx]
    o = (y_true == class_idx).astype(float)
    return float(np.mean((p - o) ** 2))


def brier_skill_score(
    predicted_probs: np.ndarray,
    y_true: np.ndarray,
    class_idx: int,
    reference_prob: Optional[float] = None,
) -> float:
    """Compute Brier Skill Score for a single class.

    BSS = 1 - BS_forecast / BS_reference

    Parameters
    ----------
    predicted_probs : np.ndarray
        Shape (n_samples, n_states).
    y_true : np.ndarray
        Shape (n_samples,). 0-indexed.
    class_idx : int
        Which class (0-indexed).
    reference_prob : float, optional
        Climatological frequency. If None, computed from y_true.

    Returns
    -------
    float
        BSS value. Range (-inf, 1]. Higher is better.
    """
    bs_forecast = brier_score(predicted_probs, y_true, class_idx)

    if reference_prob is None:
        reference_prob = float(np.mean(y_true == class_idx))

    # Reference Brier score (constant climatology prediction)
    o = (y_true == class_idx).astype(float)
    bs_reference = float(np.mean((reference_prob - o) ** 2))

    if bs_reference == 0:
        return 0.0

    return 1.0 - bs_forecast / bs_reference


def multi_class_log_loss(
    predicted_probs: np.ndarray,
    y_true: np.ndarray,
    eps: float = 1e-15,
) -> float:
    """Compute multi-class log-loss (cross-entropy).

    L = -(1/N) * sum_{i=1}^{N} log(p_{i, y_i})

    Parameters
    ----------
    predicted_probs : np.ndarray
        Shape (n_samples, n_states).
    y_true : np.ndarray
        Shape (n_samples,). 0-indexed.
    eps : float
        Clipping value to avoid log(0).

    Returns
    -------
    float
        Log-loss. Lower is better.
    """
    clipped = np.clip(predicted_probs, eps, 1 - eps)
    log_probs = np.log(clipped[np.arange(len(y_true)), y_true.astype(int)])
    return float(-np.mean(log_probs))


def compute_all_metrics(
    predicted_probs: np.ndarray,
    y_true: np.ndarray,
    reference_probs: Optional[np.ndarray] = None,
    n_states: int = N_STATES,
) -> dict:
    """Compute all probabilistic metrics at once.

    Returns
    -------
    dict with keys:
        rpss, rps_mean, log_loss,
        brier_score_phase_{1..5}, brier_skill_score_phase_{1..5}
    """
    results = {}

    # RPSS
    results["rpss"] = ranked_probability_skill_score(
        predicted_probs, y_true, reference_probs, n_states
    )

    # Mean RPS
    results["rps_mean"] = np.mean([
        ranked_probability_score(predicted_probs[i], int(y_true[i]), n_states)
        for i in range(len(y_true))
    ])

    # Log-loss
    results["log_loss"] = multi_class_log_loss(predicted_probs, y_true)

    # Per-phase Brier scores
    for phase in range(n_states):
        results[f"brier_score_phase_{phase+1}"] = brier_score(
            predicted_probs, y_true, phase
        )
        results[f"brier_skill_score_phase_{phase+1}"] = brier_skill_score(
            predicted_probs, y_true, phase
        )

    return results
