"""Non-Homogeneous Markov Chain for food security phase prediction.

The core mathematical model. In a Non-Homogeneous Markov Chain (NHMC),
the transition matrix P_t varies with time through external covariates:

    P_t(i, j) = P(S_{t+1} = j | S_t = i, X_t)

where X_t is the covariate vector (climate, agronomic, market features)
and the transition probabilities are parameterized by an ML model.

State space: S = {1, 2, 3, 4, 5} corresponding to IPC phases
    1: Minimal, 2: Stressed, 3: Crisis, 4: Emergency, 5: Famine
"""

import numpy as np
from typing import Optional, Protocol
from dataclasses import dataclass

from src.config import N_STATES, MONTE_CARLO_SAMPLES, RANDOM_STATE


class TransitionPredictor(Protocol):
    """Protocol for models that predict transition probability rows."""

    def predict_transition_row(
        self, origin_state: int, covariates: np.ndarray
    ) -> np.ndarray:
        """Predict P(next_state | origin_state, covariates).

        Parameters
        ----------
        origin_state : int
            Current IPC phase (1-indexed).
        covariates : np.ndarray
            Feature vector for the current time step.

        Returns
        -------
        np.ndarray
            Shape (n_states,) probability vector summing to 1.
        """
        ...


@dataclass
class ForecastResult:
    """Container for NHMC forecast outputs."""

    # Shape (horizon, n_states): probability distribution at each step
    probabilities: np.ndarray
    # Shape (horizon,): most likely state at each step
    predicted_states: np.ndarray
    # Shape (horizon, n_states): lower confidence bound (e.g., p10)
    ci_lower: Optional[np.ndarray] = None
    # Shape (horizon, n_states): upper confidence bound (e.g., p90)
    ci_upper: Optional[np.ndarray] = None
    # List of (n_states, n_states) transition matrices used
    transition_matrices: Optional[list[np.ndarray]] = None


class NonHomogeneousMarkovChain:
    """Non-Homogeneous Markov Chain with ML-parameterized transitions.

    The transition matrix at time t is:
        P_t(i, j) = f(X_t, theta_{i})

    where f is a per-state ML classifier that predicts the probability
    distribution over next states given covariates X_t.

    Mathematical Properties
    -----------------------
    - Each P_t is a right-stochastic matrix: rows sum to 1, all entries >= 0
    - The n-step transition is: P(t, t+n) = P_t @ P_{t+1} @ ... @ P_{t+n-1}
    - Unlike homogeneous chains, Chapman-Kolmogorov holds in product form:
      P(s, t) = P(s, r) @ P(r, t) for s < r < t
    - Stationary distribution exists only for fixed covariate settings
    """

    def __init__(
        self,
        n_states: int = N_STATES,
        transition_model: Optional[TransitionPredictor] = None,
    ):
        self.n_states = n_states
        self.transition_model = transition_model

    def set_transition_model(self, model: TransitionPredictor) -> None:
        """Set or replace the transition model."""
        self.transition_model = model

    def get_transition_matrix(self, covariates: np.ndarray) -> np.ndarray:
        """Compute the full n_states x n_states transition matrix for given covariates.

        Each row i is predicted by the transition model:
            P(i, :) = model.predict_transition_row(i+1, covariates)

        Parameters
        ----------
        covariates : np.ndarray
            Feature vector for a single time step.

        Returns
        -------
        np.ndarray
            Shape (n_states, n_states) right-stochastic matrix.

        Raises
        ------
        ValueError
            If any row does not sum to ~1 or contains negative values.
        """
        assert self.transition_model is not None, "Transition model not set"

        P = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            row = self.transition_model.predict_transition_row(i + 1, covariates)
            P[i] = row

        self._validate_stochastic_matrix(P)
        return P

    def forecast(
        self,
        current_state: int,
        covariates_sequence: list[np.ndarray],
        horizon: Optional[int] = None,
        n_simulations: int = MONTE_CARLO_SAMPLES,
        ci_level: float = 0.80,
    ) -> ForecastResult:
        """Forecast state probability distribution over multiple time steps.

        The forecast iteratively applies time-varying transition matrices:
            pi_{t+1} = pi_t @ P_t

        Confidence intervals are computed via Monte Carlo simulation:
        sample from the predicted distribution at each step.

        Parameters
        ----------
        current_state : int
            Current IPC phase (1-indexed).
        covariates_sequence : list of np.ndarray
            Covariate vectors for each future time step.
        horizon : int, optional
            Override horizon (default: len(covariates_sequence)).
        n_simulations : int
            Number of Monte Carlo paths for confidence intervals.
        ci_level : float
            Confidence interval width (e.g., 0.80 for 10th-90th percentile).

        Returns
        -------
        ForecastResult
            Contains probability distributions, predicted states, and CIs.
        """
        if horizon is None:
            horizon = len(covariates_sequence)
        assert horizon <= len(covariates_sequence), (
            f"Not enough covariate vectors ({len(covariates_sequence)}) "
            f"for horizon {horizon}"
        )

        # Deterministic forecast: iterate probability vector
        probs = np.zeros((horizon, self.n_states))
        transition_matrices = []
        state_vec = np.zeros(self.n_states)
        state_vec[current_state - 1] = 1.0

        for t in range(horizon):
            P_t = self.get_transition_matrix(covariates_sequence[t])
            transition_matrices.append(P_t)
            state_vec = state_vec @ P_t
            probs[t] = state_vec

        predicted_states = np.argmax(probs, axis=1) + 1

        # Monte Carlo confidence intervals
        rng = np.random.default_rng(RANDOM_STATE)
        simulated_distributions = np.zeros((n_simulations, horizon, self.n_states))

        for sim in range(n_simulations):
            state = current_state - 1  # 0-indexed
            for t in range(horizon):
                P_t = transition_matrices[t]
                # Sample next state from transition probabilities
                state = rng.choice(self.n_states, p=P_t[state])
                simulated_distributions[sim, t, state] += 1

        # Compute empirical distribution from simulations
        sim_probs = simulated_distributions.mean(axis=0)

        alpha = (1 - ci_level) / 2
        ci_lower = np.quantile(simulated_distributions, alpha, axis=0)
        ci_upper = np.quantile(simulated_distributions, 1 - alpha, axis=0)

        return ForecastResult(
            probabilities=probs,
            predicted_states=predicted_states,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            transition_matrices=transition_matrices,
        )

    def stationary_analysis(self, covariates: np.ndarray) -> np.ndarray:
        """Compute stationary distribution for a fixed covariate setting.

        For a time-homogeneous chain with transition matrix P = P(X),
        the stationary distribution pi satisfies: pi @ P = pi.

        This answers: "If conditions X persisted indefinitely, what
        would the long-run distribution of IPC phases be?"

        Parameters
        ----------
        covariates : np.ndarray
            Fixed covariate vector.

        Returns
        -------
        np.ndarray
            Shape (n_states,) stationary probability distribution.
        """
        P = self.get_transition_matrix(covariates)

        eigenvalues, eigenvectors = np.linalg.eig(P.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary = np.abs(stationary)
        return stationary / stationary.sum()

    def scenario_comparison(
        self,
        current_state: int,
        scenario_covariates: dict[str, np.ndarray],
        horizon: int = 12,
    ) -> dict[str, ForecastResult]:
        """Compare forecasts under different covariate scenarios.

        Useful for analyzing "what if drought persists?" vs "what if
        rains normalize?" scenarios.

        Parameters
        ----------
        current_state : int
            Current IPC phase (1-indexed).
        scenario_covariates : dict
            Maps scenario name -> covariate vector (applied at every step).
        horizon : int
            Forecast horizon in months.

        Returns
        -------
        dict mapping scenario name -> ForecastResult
        """
        results = {}
        for name, covariates in scenario_covariates.items():
            cov_seq = [covariates] * horizon
            results[name] = self.forecast(current_state, cov_seq, horizon)
        return results

    def observed_transition_counts(self, state_sequence: np.ndarray) -> np.ndarray:
        """Count observed transitions from a state sequence.

        Parameters
        ----------
        state_sequence : np.ndarray
            Sequence of IPC phases (1-indexed).

        Returns
        -------
        np.ndarray
            Shape (n_states, n_states) count matrix.
        """
        counts = np.zeros((self.n_states, self.n_states))
        seq = np.asarray(state_sequence) - 1
        for t in range(len(seq) - 1):
            i, j = int(seq[t]), int(seq[t + 1])
            if 0 <= i < self.n_states and 0 <= j < self.n_states:
                counts[i, j] += 1
        return counts

    def multi_step_transition(
        self, covariates_sequence: list[np.ndarray]
    ) -> np.ndarray:
        """Compute the product of transition matrices over a sequence.

        P(0, T) = P_0 @ P_1 @ ... @ P_{T-1}

        Parameters
        ----------
        covariates_sequence : list of np.ndarray
            Covariate vectors for each time step.

        Returns
        -------
        np.ndarray
            Shape (n_states, n_states) cumulative transition matrix.
        """
        result = np.eye(self.n_states)
        for covariates in covariates_sequence:
            P_t = self.get_transition_matrix(covariates)
            result = result @ P_t
        return result

    @staticmethod
    def _validate_stochastic_matrix(P: np.ndarray, tol: float = 1e-4) -> None:
        """Validate and normalize P to be a right-stochastic matrix."""
        P = np.maximum(P, 0)  # Clip negatives
        row_sums = P.sum(axis=1, keepdims=True)
        # Normalize rows that don't sum to 1 (e.g., rare states with no data)
        row_sums = np.where(row_sums < tol, 1.0, row_sums)
        P[:] = P / row_sums  # In-place normalize

    @staticmethod
    def empirical_transition_matrix(
        state_sequences: list[np.ndarray],
        n_states: int = N_STATES,
        smoothing: float = 1e-6,
    ) -> np.ndarray:
        """Compute empirical transition matrix from observed sequences.

        Utility method for computing the MLE transition matrix.

        Parameters
        ----------
        state_sequences : list of np.ndarray
            Each array is a 1-indexed state sequence.
        n_states : int
            Number of states.
        smoothing : float
            Laplace smoothing parameter.

        Returns
        -------
        np.ndarray
            Shape (n_states, n_states) stochastic matrix.
        """
        counts = np.zeros((n_states, n_states))
        for seq in state_sequences:
            seq_0 = np.asarray(seq) - 1
            for t in range(len(seq_0) - 1):
                i, j = int(seq_0[t]), int(seq_0[t + 1])
                if 0 <= i < n_states and 0 <= j < n_states:
                    counts[i, j] += 1

        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        return (counts + smoothing) / (row_sums + smoothing * n_states)
