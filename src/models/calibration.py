"""Probabilistic calibration for ensemble model outputs.

Ensures that predicted probabilities are reliable:
if we predict P(Phase 3) = 0.3, then ~30% of the time it should be Phase 3.

Three calibration methods:
1. Isotonic regression (sklearn CalibratedClassifierCV)
2. Temperature scaling (single parameter, preserves relative ordering)
3. Conformal prediction (CQR) for guaranteed coverage intervals
"""

import numpy as np
import pandas as pd
from typing import Optional
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.optimize import minimize_scalar
from scipy.special import softmax

from src.config import N_STATES, RANDOM_STATE


class TemperatureScaler:
    """Temperature scaling for multi-class probability calibration.

    Finds a single temperature T that minimizes negative log-likelihood:
        p_calibrated = softmax(logits / T)

    Temperature T > 1 makes predictions less confident (flatter),
    T < 1 makes predictions more confident (peakier).
    """

    def __init__(self):
        self.temperature: float = 1.0

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> "TemperatureScaler":
        """Fit temperature on validation set.

        Parameters
        ----------
        logits : np.ndarray
            Shape (n_samples, n_classes). Pre-softmax model outputs or log-probabilities.
        y_true : np.ndarray
            Shape (n_samples,). True class labels (0-indexed).
        """
        def nll(T):
            scaled = logits / T
            log_probs = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))
            # Negative log-likelihood
            return -np.mean(log_probs[np.arange(len(y_true)), y_true.astype(int)])

        result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        self.temperature = result.x
        return self

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to logits.

        Returns
        -------
        np.ndarray
            Shape (n_samples, n_classes) calibrated probabilities.
        """
        return softmax(logits / self.temperature, axis=1)


class IsotonicCalibrator:
    """Isotonic regression calibration wrapper.

    Fits a separate isotonic regression for each class (one-vs-rest).
    """

    def __init__(self, n_states: int = N_STATES):
        self.n_states = n_states
        self.calibrators: list[Optional[CalibratedClassifierCV]] = [None] * n_states

    def fit(
        self,
        model: BaseEstimator,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "IsotonicCalibrator":
        """Fit isotonic calibration on validation data.

        Parameters
        ----------
        model : BaseEstimator
            Trained classifier with predict_proba method.
        X_val : np.ndarray
            Validation features.
        y_val : np.ndarray
            Validation labels (0-indexed).
        """
        calibrated = CalibratedClassifierCV(
            model, method="isotonic", cv="prefit"
        )
        calibrated.fit(X_val, y_val)
        self.calibrated_model = calibrated
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities."""
        return self.calibrated_model.predict_proba(X)


class ConformalPredictor:
    """Conformalized prediction for guaranteed coverage.

    Given a trained model and calibration data, produces prediction sets
    that contain the true label with probability >= 1 - alpha.

    Implements split conformal prediction with adaptive prediction sets.
    """

    def __init__(self, alpha: float = 0.1, n_states: int = N_STATES):
        """
        Parameters
        ----------
        alpha : float
            Miscoverage rate. Default 0.1 gives 90% coverage.
        n_states : int
            Number of IPC phases.
        """
        self.alpha = alpha
        self.n_states = n_states
        self.quantile_threshold: float = 0.0
        self._calibrated = False

    def calibrate(
        self,
        probabilities: np.ndarray,
        y_true: np.ndarray,
    ) -> "ConformalPredictor":
        """Calibrate using held-out data.

        Computes nonconformity scores on calibration data to determine
        the threshold for prediction set construction.

        Parameters
        ----------
        probabilities : np.ndarray
            Shape (n_cal, n_states). Model's predicted probabilities on
            calibration set.
        y_true : np.ndarray
            Shape (n_cal,). True labels (0-indexed).

        The nonconformity score is 1 - p(true_class).
        """
        n_cal = len(y_true)
        scores = 1.0 - probabilities[np.arange(n_cal), y_true.astype(int)]

        # Compute the (1-alpha)(1 + 1/n) quantile of scores
        adjusted_quantile = min(1.0, (1 - self.alpha) * (1 + 1 / n_cal))
        self.quantile_threshold = float(np.quantile(scores, adjusted_quantile))

        self._calibrated = True
        return self

    def predict_sets(self, probabilities: np.ndarray) -> list[list[int]]:
        """Construct prediction sets with guaranteed coverage.

        For each sample, include all classes whose predicted probability
        exceeds 1 - threshold.

        Parameters
        ----------
        probabilities : np.ndarray
            Shape (n_samples, n_states). Model predictions.

        Returns
        -------
        list of lists
            Each inner list contains the IPC phases (1-indexed) in the
            prediction set for that sample.
        """
        assert self._calibrated, "Must call calibrate() first"

        prediction_sets = []
        for i in range(probabilities.shape[0]):
            # Include class j if p(j) >= 1 - threshold
            included = []
            for j in range(self.n_states):
                if probabilities[i, j] >= 1 - self.quantile_threshold:
                    included.append(j + 1)  # 1-indexed

            # Ensure non-empty prediction set
            if not included:
                best = int(np.argmax(probabilities[i])) + 1
                included = [best]

            prediction_sets.append(sorted(included))

        return prediction_sets

    def coverage(
        self, probabilities: np.ndarray, y_true: np.ndarray
    ) -> float:
        """Compute empirical coverage on test data.

        Returns fraction of samples where true label is in prediction set.
        """
        sets = self.predict_sets(probabilities)
        covered = sum(
            1 for i, s in enumerate(sets) if (int(y_true[i]) + 1) in s
        )
        return covered / len(y_true)

    def average_set_size(self, probabilities: np.ndarray) -> float:
        """Compute average prediction set size (smaller is better)."""
        sets = self.predict_sets(probabilities)
        return np.mean([len(s) for s in sets])


def reliability_diagram_data(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10,
    class_idx: int = 0,
) -> pd.DataFrame:
    """Compute data for a reliability diagram.

    Bins predicted probabilities and computes the observed frequency
    in each bin. A perfectly calibrated model has observed = predicted.

    Parameters
    ----------
    probabilities : np.ndarray
        Shape (n_samples, n_states). Predicted probabilities.
    y_true : np.ndarray
        Shape (n_samples,). True labels (0-indexed).
    n_bins : int
        Number of probability bins.
    class_idx : int
        Which class to plot (0-indexed).

    Returns
    -------
    pd.DataFrame
        Columns: bin_center, predicted_prob, observed_freq, count
    """
    p_class = probabilities[:, class_idx]
    y_binary = (y_true == class_idx).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    records = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (p_class >= lo) & (p_class < hi)
        if i == n_bins - 1:
            mask = (p_class >= lo) & (p_class <= hi)

        count = mask.sum()
        if count > 0:
            records.append({
                "bin_center": (lo + hi) / 2,
                "predicted_prob": p_class[mask].mean(),
                "observed_freq": y_binary[mask].mean(),
                "count": int(count),
            })

    return pd.DataFrame(records)
