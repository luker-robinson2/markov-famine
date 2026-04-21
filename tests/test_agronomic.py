"""Tests for agronomic index computation."""

import numpy as np
import pandas as pd
import pytest

from src.engineering.agronomic import (
    compute_gdd,
    compute_vci,
    compute_vhi,
    compute_et0_hargreaves,
    compute_spi,
    compute_spei,
)


class TestGrowingDegreeDays:
    """Test GDD computation."""

    def test_basic_gdd(self):
        """GDD = cumulative max(0, (Tmax+Tmin)/2 - Tbase)."""
        tmax = pd.Series([30.0, 25.0, 15.0, 35.0])
        tmin = pd.Series([20.0, 15.0, 5.0, 25.0])
        gdd = compute_gdd(tmax, tmin, tbase=10.0)

        # Per-step GDD: [15, 10, 0, 20], cumulative: [15, 25, 25, 45]
        expected = pd.Series([15.0, 25.0, 25.0, 45.0])
        pd.testing.assert_series_equal(gdd, expected, check_names=False)

    def test_gdd_non_negative(self):
        """GDD is always >= 0."""
        tmax = pd.Series([5.0, 0.0, -10.0])
        tmin = pd.Series([0.0, -5.0, -20.0])
        gdd = compute_gdd(tmax, tmin, tbase=10.0)
        assert (gdd >= 0).all()

    def test_gdd_with_different_base(self):
        """Different base temperatures give different results."""
        tmax = pd.Series([20.0])
        tmin = pd.Series([10.0])
        gdd_10 = compute_gdd(tmax, tmin, tbase=10.0)
        gdd_5 = compute_gdd(tmax, tmin, tbase=5.0)
        assert gdd_5.iloc[0] > gdd_10.iloc[0]


class TestVegetationIndices:
    """Test VCI and VHI computation."""

    def test_vci_range(self):
        """VCI should be in [0, 100] (WMO convention)."""
        ndvi = pd.Series([0.3, 0.5, 0.7, 0.2, 0.8])
        vci = compute_vci(ndvi, ndvi_min=0.1, ndvi_max=0.9)
        assert (vci >= 0).all()
        assert (vci <= 100).all()

    def test_vci_extremes(self):
        """VCI = 0 at min, VCI = 100 at max (WMO percentage scale)."""
        vci_min = compute_vci(pd.Series([0.1]), ndvi_min=0.1, ndvi_max=0.9)
        vci_max = compute_vci(pd.Series([0.9]), ndvi_min=0.1, ndvi_max=0.9)
        assert vci_min.iloc[0] == pytest.approx(0.0)
        assert vci_max.iloc[0] == pytest.approx(100.0)

    def test_vhi_average(self):
        """VHI = 0.5*VCI + 0.5*TCI."""
        vci = pd.Series([0.8, 0.2])
        tci = pd.Series([0.4, 0.6])
        vhi = compute_vhi(vci, tci)
        expected = pd.Series([0.6, 0.4])
        pd.testing.assert_series_equal(vhi, expected, check_names=False)


class TestHargreaves:
    """Test Hargreaves ET0 computation."""

    def test_et0_positive(self):
        """ET0 should always be positive."""
        temp_mean = pd.Series([25.0, 20.0, 30.0])
        temp_max = pd.Series([30.0, 25.0, 35.0])
        temp_min = pd.Series([20.0, 15.0, 25.0])
        et0 = compute_et0_hargreaves(temp_mean, temp_max, temp_min, latitude=0.0)
        assert (et0 > 0).all()

    def test_et0_increases_with_temperature(self):
        """Higher temperatures should give higher ET0."""
        temp_low = pd.Series([15.0])
        temp_high = pd.Series([30.0])
        et0_low = compute_et0_hargreaves(temp_low, temp_low + 5, temp_low - 5, latitude=0.0)
        et0_high = compute_et0_hargreaves(temp_high, temp_high + 5, temp_high - 5, latitude=0.0)
        assert et0_high.iloc[0] > et0_low.iloc[0]
