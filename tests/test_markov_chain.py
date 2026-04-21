"""Tests for Markov chain implementations."""

import numpy as np
import pytest

from src.models.baseline import HomogeneousMarkovChain, ClimatologyBaseline, PersistenceBaseline
from src.models.markov_chain import NonHomogeneousMarkovChain
from src.models.transition_model import EmpiricalTransitionModel


class TestHomogeneousMarkovChain:
    """Test the homogeneous MC baseline."""

    def setup_method(self):
        """Create a simple test chain."""
        self.mc = HomogeneousMarkovChain(n_states=3, smoothing=0.0)
        # Simple sequence: mostly stays in same state, some transitions
        sequences = [
            np.array([1, 1, 1, 2, 2, 3, 3, 3, 3, 1]),
            np.array([1, 1, 2, 2, 2, 3, 1, 1, 1, 1]),
        ]
        self.mc.fit(sequences)

    def test_transition_matrix_rows_sum_to_one(self):
        """Every row of the transition matrix must sum to 1."""
        row_sums = self.mc.transition_matrix.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_transition_matrix_non_negative(self):
        """All entries must be non-negative."""
        assert np.all(self.mc.transition_matrix >= 0)

    def test_stationary_distribution_sums_to_one(self):
        """Stationary distribution must be a valid probability distribution."""
        assert self.mc.stationary_dist is not None
        np.testing.assert_allclose(self.mc.stationary_dist.sum(), 1.0, atol=1e-10)

    def test_stationary_distribution_identity(self):
        """pi @ P = pi."""
        assert self.mc.verify_stationary()

    def test_chapman_kolmogorov(self):
        """P^(n+m) = P^n @ P^m."""
        assert self.mc.verify_chapman_kolmogorov(n=2, m=3)
        assert self.mc.verify_chapman_kolmogorov(n=5, m=7)

    def test_predict_proba_sums_to_one(self):
        """Predicted probability must be valid distribution."""
        proba = self.mc.predict_proba(current_state=1, horizon=5)
        np.testing.assert_allclose(proba.sum(), 1.0, atol=1e-10)
        assert np.all(proba >= 0)

    def test_forecast_shape(self):
        """Forecast returns correct shape."""
        result = self.mc.forecast(current_state=1, horizon=10)
        assert result.shape == (10, 3)

    def test_forecast_rows_sum_to_one(self):
        """Each row of forecast must sum to 1."""
        result = self.mc.forecast(current_state=2, horizon=5)
        row_sums = result.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_known_transition_matrix(self):
        """For a deterministic sequence, verify MLE recovery."""
        mc = HomogeneousMarkovChain(n_states=2, smoothing=0.0)
        # Always: 1 -> 2 -> 1 -> 2 -> ...
        mc.fit([np.array([1, 2, 1, 2, 1, 2, 1, 2])])
        # P(1->2) = 1, P(2->1) = 1
        np.testing.assert_allclose(mc.transition_matrix[0, 1], 1.0, atol=1e-10)
        np.testing.assert_allclose(mc.transition_matrix[1, 0], 1.0, atol=1e-10)


class TestNonHomogeneousMarkovChain:
    """Test the NHMC with a mock transition model."""

    def setup_method(self):
        """Set up NHMC with empirical transition model."""
        # Create a known transition matrix
        P = np.array([
            [0.7, 0.2, 0.1, 0.0, 0.0],
            [0.1, 0.6, 0.2, 0.1, 0.0],
            [0.0, 0.1, 0.5, 0.3, 0.1],
            [0.0, 0.0, 0.2, 0.5, 0.3],
            [0.0, 0.0, 0.1, 0.3, 0.6],
        ])
        model = EmpiricalTransitionModel(P)
        self.nhmc = NonHomogeneousMarkovChain(n_states=5, transition_model=model)
        self.dummy_covariates = np.zeros(10)

    def test_transition_matrix_valid(self):
        """Get transition matrix and verify it's stochastic."""
        P = self.nhmc.get_transition_matrix(self.dummy_covariates)
        row_sums = P.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)
        assert np.all(P >= 0)

    def test_forecast_shape(self):
        """Forecast returns correct shape."""
        cov_seq = [self.dummy_covariates] * 6
        result = self.nhmc.forecast(current_state=1, covariates_sequence=cov_seq)
        assert result.probabilities.shape == (6, 5)
        assert result.predicted_states.shape == (6,)

    def test_forecast_probs_sum_to_one(self):
        """Each row of forecast probabilities sums to 1."""
        cov_seq = [self.dummy_covariates] * 3
        result = self.nhmc.forecast(current_state=3, covariates_sequence=cov_seq)
        row_sums = result.probabilities.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_stationary_analysis(self):
        """Stationary distribution is valid and satisfies pi @ P = pi."""
        pi = self.nhmc.stationary_analysis(self.dummy_covariates)
        assert pi.shape == (5,)
        np.testing.assert_allclose(pi.sum(), 1.0, atol=1e-10)
        assert np.all(pi >= 0)

        P = self.nhmc.get_transition_matrix(self.dummy_covariates)
        result = pi @ P
        np.testing.assert_allclose(result, pi, atol=1e-8)

    def test_multi_step_transition(self):
        """Multi-step transition product is valid stochastic matrix."""
        cov_seq = [self.dummy_covariates] * 5
        P_product = self.nhmc.multi_step_transition(cov_seq)
        row_sums = P_product.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_observed_transition_counts(self):
        """Count matrix has correct total."""
        seq = np.array([1, 2, 3, 2, 1, 1, 2])
        counts = self.nhmc.observed_transition_counts(seq)
        assert counts.sum() == len(seq) - 1  # n-1 transitions


class TestBaselines:
    """Test climatology and persistence baselines."""

    def test_persistence_returns_delta(self):
        """Persistence predicts current state with probability 1."""
        baseline = PersistenceBaseline()
        for state in range(1, 6):
            proba = baseline.predict_proba(state)
            assert proba[state - 1] == 1.0
            assert proba.sum() == 1.0
            assert baseline.predict(state) == state

    def test_climatology_fit_and_predict(self):
        """Climatology returns valid distributions."""
        import pandas as pd
        df = pd.DataFrame({
            "region_code": ["KE001"] * 12,
            "date": pd.date_range("2020-01-01", periods=12, freq="MS"),
            "ipc_phase": [1, 1, 2, 2, 2, 3, 3, 2, 2, 1, 1, 1],
        })
        baseline = ClimatologyBaseline()
        baseline.fit(df)

        proba = baseline.predict_proba("KE001", 1)  # January
        assert proba.sum() == pytest.approx(1.0)
        assert np.all(proba >= 0)
