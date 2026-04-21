"""Tests for ensemble model outputs."""

import numpy as np
import pytest

from src.models.ensemble import PerStateClassifier, EnsembleConfig


class TestPerStateClassifier:
    """Test individual per-state classifier."""

    def test_predict_proba_shape(self):
        """Output shape should be (n_samples, n_states)."""
        config = EnsembleConfig(n_states=5, min_samples_per_state=5)
        clf = PerStateClassifier(origin_state=1, model_type="random_forest", config=config)

        rng = np.random.default_rng(42)
        X = rng.normal(size=(50, 10))
        y = rng.choice(5, size=50)
        clf.fit(X, y)

        proba = clf.predict_proba(X[:5])
        assert proba.shape == (5, 5)

    def test_predict_proba_valid_distribution(self):
        """Each row must sum to 1 and be non-negative."""
        config = EnsembleConfig(n_states=5, min_samples_per_state=5)
        clf = PerStateClassifier(origin_state=1, model_type="random_forest", config=config)

        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 10))
        y = rng.choice(5, size=100)
        clf.fit(X, y)

        proba = clf.predict_proba(X[:10])
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        assert np.all(proba >= 0)

    def test_fallback_to_empirical(self):
        """With too few samples, should use empirical distribution."""
        config = EnsembleConfig(n_states=5, min_samples_per_state=100)  # High threshold
        clf = PerStateClassifier(origin_state=1, model_type="random_forest", config=config)

        X = np.random.normal(size=(5, 10))  # Only 5 samples
        y = np.array([0, 0, 1, 1, 2])
        clf.fit(X, y)

        assert clf.model is None  # Should not have trained
        proba = clf.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_empirical_distribution_correct(self):
        """Empirical distribution should match observed frequencies."""
        config = EnsembleConfig(n_states=3, min_samples_per_state=1000)
        clf = PerStateClassifier(origin_state=1, model_type="random_forest", config=config)

        X = np.random.normal(size=(10, 5))
        y = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])  # 4:3:3 ratio
        clf.fit(X, y)

        expected = np.array([0.4, 0.3, 0.3])
        np.testing.assert_allclose(clf.empirical_dist, expected, atol=1e-10)
