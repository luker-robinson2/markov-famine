"""Tests for feature engineering utilities."""

import numpy as np
import pytest

from src.engineering.temporal import encode_cyclical_month


class TestCyclicalEncoding:
    """Test cyclical month encoding."""

    def test_january(self):
        """January encoding values."""
        sin_val, cos_val = encode_cyclical_month(1)
        expected_sin = np.sin(2 * np.pi * 1 / 12)
        expected_cos = np.cos(2 * np.pi * 1 / 12)
        assert sin_val == pytest.approx(expected_sin, abs=1e-10)
        assert cos_val == pytest.approx(expected_cos, abs=1e-10)

    def test_july(self):
        """July encoding values."""
        sin_val, cos_val = encode_cyclical_month(7)
        expected_sin = np.sin(2 * np.pi * 7 / 12)
        expected_cos = np.cos(2 * np.pi * 7 / 12)
        assert sin_val == pytest.approx(expected_sin, abs=1e-10)
        assert cos_val == pytest.approx(expected_cos, abs=1e-10)

    def test_december_wraps_to_january(self):
        """December and January should be close in encoded space."""
        sin_dec, cos_dec = encode_cyclical_month(12)
        sin_jan, cos_jan = encode_cyclical_month(1)
        # Euclidean distance in encoded space should be small
        dist = np.sqrt((sin_dec - sin_jan) ** 2 + (cos_dec - cos_jan) ** 2)
        # Distance between adjacent months
        sin_jan2, cos_jan2 = encode_cyclical_month(2)
        dist_jan_feb = np.sqrt((sin_jan - sin_jan2) ** 2 + (cos_jan - cos_jan2) ** 2)
        # Dec-Jan distance should be similar to Jan-Feb distance
        assert dist == pytest.approx(dist_jan_feb, abs=0.01)

    def test_all_months_unit_circle(self):
        """All encoded points should lie on the unit circle."""
        for month in range(1, 13):
            sin_val, cos_val = encode_cyclical_month(month)
            radius = np.sqrt(sin_val ** 2 + cos_val ** 2)
            assert radius == pytest.approx(1.0, abs=1e-10)
