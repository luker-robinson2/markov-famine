"""Baseline models for food security phase prediction.

Three baselines to compare against the Non-Homogeneous Markov Chain:
1. Homogeneous Markov Chain (time-invariant transition matrix)
2. Climatology (historical mode per region-month)
3. Persistence (current phase continues)
"""

import numpy as np
import pandas as pd
from typing import Optional

from src.config import N_STATES, RANDOM_STATE


class HomogeneousMarkovChain:
    """Standard time-homogeneous Markov chain with MLE-estimated transition matrix.

    The transition matrix P is constant over time:
        P(i,j) = count(i -> j) / count(i -> *)

    This is the maximum likelihood estimator for a stationary Markov chain.
    """

    def __init__(self, n_states: int = N_STATES, smoothing: float = 0.01):
        self.n_states = n_states
        self.smoothing = smoothing
        self.transition_matrix: Optional[np.ndarray] = None
        self.transition_counts: Optional[np.ndarray] = None
        self.stationary_dist: Optional[np.ndarray] = None

    def fit(self, state_sequences: list[np.ndarray]) -> "HomogeneousMarkovChain":
        """Estimate transition matrix from observed state sequences via MLE.

        Parameters
        ----------
        state_sequences : list of arrays
            Each array is a sequence of IPC phases (1-indexed) for one region.
            Values in {1, 2, ..., n_states}.
        """
        counts = np.zeros((self.n_states, self.n_states))

        for seq in state_sequences:
            seq_0indexed = np.asarray(seq) - 1  # Convert to 0-indexed
            for t in range(len(seq_0indexed) - 1):
                i, j = seq_0indexed[t], seq_0indexed[t + 1]
                if 0 <= i < self.n_states and 0 <= j < self.n_states:
                    counts[i, j] += 1

        self.transition_counts = counts

        # MLE with Laplace smoothing to avoid zero probabilities
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        self.transition_matrix = (counts + self.smoothing) / (
            row_sums + self.smoothing * self.n_states
        )

        self._compute_stationary()
        return self

    def _compute_stationary(self) -> None:
        """Compute stationary distribution pi such that pi @ P = pi.

        Uses eigenvalue decomposition: the stationary distribution is the
        left eigenvector corresponding to eigenvalue 1.
        """
        if self.transition_matrix is None:
            return

        # Left eigenvectors of P^T
        eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)

        # Find eigenvector for eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])

        # Normalize to probability distribution
        stationary = np.abs(stationary)
        self.stationary_dist = stationary / stationary.sum()

    def predict_proba(self, current_state: int, horizon: int = 1) -> np.ndarray:
        """Predict probability distribution over states at given horizon.

        Uses the Chapman-Kolmogorov equation: P^(n) = P^n

        Parameters
        ----------
        current_state : int
            Current IPC phase (1-indexed).
        horizon : int
            Number of time steps ahead.

        Returns
        -------
        np.ndarray
            Shape (n_states,) probability vector.
        """
        assert self.transition_matrix is not None, "Must call fit() first"

        state_0idx = current_state - 1
        P_n = np.linalg.matrix_power(self.transition_matrix, horizon)
        return P_n[state_0idx]

    def forecast(self, current_state: int, horizon: int) -> np.ndarray:
        """Forecast state distributions for each step up to horizon.

        Returns
        -------
        np.ndarray
            Shape (horizon, n_states). Row t is P(S_{t+1} | S_0).
        """
        assert self.transition_matrix is not None, "Must call fit() first"

        results = np.zeros((horizon, self.n_states))
        state_vec = np.zeros(self.n_states)
        state_vec[current_state - 1] = 1.0

        for t in range(horizon):
            state_vec = state_vec @ self.transition_matrix
            results[t] = state_vec

        return results

    def verify_chapman_kolmogorov(self, n: int = 3, m: int = 4, tol: float = 1e-10) -> bool:
        """Verify Chapman-Kolmogorov equation: P^(n+m) = P^n @ P^m."""
        assert self.transition_matrix is not None
        P = self.transition_matrix
        P_nm = np.linalg.matrix_power(P, n + m)
        P_n_times_P_m = np.linalg.matrix_power(P, n) @ np.linalg.matrix_power(P, m)
        return np.allclose(P_nm, P_n_times_P_m, atol=tol)

    def verify_stationary(self, tol: float = 1e-8) -> bool:
        """Verify pi @ P = pi."""
        if self.stationary_dist is None or self.transition_matrix is None:
            return False
        result = self.stationary_dist @ self.transition_matrix
        return np.allclose(result, self.stationary_dist, atol=tol)


class ClimatologyBaseline:
    """Predict the historical modal IPC phase per region-month.

    For each (region, month_of_year), predicts the most frequent
    historical IPC phase as a probability distribution.
    """

    def __init__(self, n_states: int = N_STATES):
        self.n_states = n_states
        self.climatology: dict[tuple[str, int], np.ndarray] = {}

    def fit(self, df: pd.DataFrame) -> "ClimatologyBaseline":
        """Fit from panel DataFrame with columns [region_code, date, ipc_phase].

        Parameters
        ----------
        df : pd.DataFrame
            Must have 'region_code', 'date' (datetime), 'ipc_phase' (1-5).
        """
        df = df.copy()
        df["month"] = pd.to_datetime(df["date"]).dt.month

        for (region, month), group in df.groupby(["region_code", "month"]):
            counts = np.zeros(self.n_states)
            for phase in group["ipc_phase"]:
                if 1 <= phase <= self.n_states:
                    counts[int(phase) - 1] += 1
            total = counts.sum()
            if total > 0:
                self.climatology[(region, month)] = counts / total
            else:
                # Uniform if no data
                self.climatology[(region, month)] = np.ones(self.n_states) / self.n_states

        return self

    def predict_proba(self, region_code: str, month: int) -> np.ndarray:
        """Predict climatological probability distribution."""
        key = (region_code, month)
        if key in self.climatology:
            return self.climatology[key]
        return np.ones(self.n_states) / self.n_states

    def predict(self, region_code: str, month: int) -> int:
        """Predict modal phase (1-indexed)."""
        proba = self.predict_proba(region_code, month)
        return int(np.argmax(proba)) + 1


class PersistenceBaseline:
    """Predict that the current IPC phase persists unchanged.

    The simplest baseline: P(S_{t+1} = s | S_t = s) = 1.
    """

    def __init__(self, n_states: int = N_STATES):
        self.n_states = n_states

    def predict_proba(self, current_state: int) -> np.ndarray:
        """Return delta distribution at current state."""
        proba = np.zeros(self.n_states)
        proba[current_state - 1] = 1.0
        return proba

    def predict(self, current_state: int) -> int:
        """Predict current phase persists."""
        return current_state
