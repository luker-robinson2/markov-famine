"""Tests for probabilistic and ordinal metrics."""

import numpy as np
import pytest

from src.metrics.probabilistic import (
    ranked_probability_score,
    ranked_probability_skill_score,
    brier_score,
    brier_skill_score,
    multi_class_log_loss,
)
from src.metrics.ordinal import (
    ordinal_confusion_matrix,
    quadratic_weighted_kappa,
    weighted_f1_score,
    crisis_detection_metrics,
    mean_absolute_ordinal_error,
    asymmetric_cost_score,
)


class TestRankedProbabilityScore:
    """Test RPS and RPSS."""

    def test_perfect_prediction(self):
        """RPS should be 0 for perfect prediction."""
        probs = np.array([0, 0, 1, 0, 0])  # Predict Phase 3 with certainty
        rps = ranked_probability_score(probs, true_label=2)  # 0-indexed
        assert rps == pytest.approx(0.0, abs=1e-10)

    def test_worst_prediction(self):
        """RPS should be high for maximally wrong prediction."""
        probs = np.array([1, 0, 0, 0, 0])  # Predict Phase 1
        rps = ranked_probability_score(probs, true_label=4)  # True is Phase 5
        assert rps > 0.5  # Should be close to 1

    def test_uniform_prediction(self):
        """Uniform prediction should have moderate RPS."""
        probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        rps = ranked_probability_score(probs, true_label=2)
        assert 0 < rps < 1

    def test_rpss_positive_for_good_model(self):
        """RPSS > 0 when model beats climatology."""
        n = 100
        rng = np.random.default_rng(42)
        y_true = rng.choice(5, size=n, p=[0.3, 0.3, 0.2, 0.15, 0.05])

        # Good predictions: high probability on true class
        good_probs = np.zeros((n, 5))
        for i in range(n):
            good_probs[i, y_true[i]] = 0.7
            remaining = 0.3 / 4
            for j in range(5):
                if j != y_true[i]:
                    good_probs[i, j] = remaining

        rpss = ranked_probability_skill_score(good_probs, y_true)
        assert rpss > 0

    def test_rpss_zero_for_climatology(self):
        """RPSS should be ~0 when model predictions match climatology."""
        n = 1000
        rng = np.random.default_rng(42)
        climatology = np.array([0.3, 0.3, 0.2, 0.15, 0.05])
        y_true = rng.choice(5, size=n, p=climatology)
        probs = np.tile(climatology, (n, 1))

        rpss = ranked_probability_skill_score(probs, y_true)
        assert abs(rpss) < 0.05  # Should be near zero


class TestBrierScore:
    """Test Brier score and skill score."""

    def test_perfect_brier(self):
        """Brier score = 0 for perfect predictions."""
        probs = np.eye(5)[[0, 1, 2, 3, 4]]  # Perfect predictions
        y_true = np.array([0, 1, 2, 3, 4])
        for c in range(5):
            bs = brier_score(probs, y_true, class_idx=c)
            assert bs == pytest.approx(0.0, abs=1e-10)

    def test_brier_range(self):
        """Brier score should be in [0, 1]."""
        rng = np.random.default_rng(42)
        probs = rng.dirichlet(np.ones(5), size=100)
        y_true = rng.choice(5, size=100)
        for c in range(5):
            bs = brier_score(probs, y_true, class_idx=c)
            assert 0 <= bs <= 1


class TestOrdinalMetrics:
    """Test ordinal-aware metrics."""

    def test_confusion_matrix_shape(self):
        """Confusion matrix has correct shape and total."""
        y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 3, 4, 1, 2, 2])
        cm = ordinal_confusion_matrix(y_true, y_pred)
        assert cm.shape == (5, 5)
        assert cm.sum() == len(y_true)

    def test_qwk_perfect_agreement(self):
        """QWK should be 1 for perfect predictions."""
        y = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        qwk = quadratic_weighted_kappa(y, y)
        assert qwk == pytest.approx(1.0, abs=1e-10)

    def test_qwk_penalizes_large_errors(self):
        """Off-by-3 should be penalized more than off-by-1."""
        # Need varied true labels for QWK's expected matrix to be non-degenerate
        y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])

        # Small errors (off by 1)
        y_pred_small = np.array([1, 1, 2, 2, 3, 3, 4, 4, 3, 3])
        qwk_small = quadratic_weighted_kappa(y_true, y_pred_small)

        # Large errors (off by 3)
        y_pred_large = np.array([3, 3, 4, 4, 0, 0, 0, 0, 1, 1])
        qwk_large = quadratic_weighted_kappa(y_true, y_pred_large)

        assert qwk_small > qwk_large  # Small errors get higher agreement

    def test_crisis_detection(self):
        """Test Phase 3+ detection metrics."""
        y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3, 4, 0, 1, 1, 3, 4])  # One missed crisis

        metrics = crisis_detection_metrics(y_true, y_pred, crisis_threshold=2)
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert metrics["recall"] < 1.0  # Missed one crisis case

    def test_mae_ordinal(self):
        """Mean absolute ordinal error."""
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3, 4])
        assert mean_absolute_ordinal_error(y_true, y_pred) == 0.0

        y_pred_off = np.array([1, 2, 3, 4, 3])
        mae = mean_absolute_ordinal_error(y_true, y_pred_off)
        assert mae == pytest.approx(1.0)

    def test_asymmetric_cost(self):
        """False negatives should cost more than false positives."""
        y_true = np.array([2, 2, 0, 0])  # 2 crisis, 2 non-crisis
        y_pred_fn = np.array([0, 0, 0, 0])  # Miss all crises
        y_pred_fp = np.array([2, 2, 2, 2])  # False alarm on all

        cost_fn = asymmetric_cost_score(y_true, y_pred_fn, fn_weight=3.0, fp_weight=1.0)
        cost_fp = asymmetric_cost_score(y_true, y_pred_fp, fn_weight=3.0, fp_weight=1.0)
        assert cost_fn > cost_fp  # Missing crises costs more
