"""Tests for statistics primitives"""

import numpy as np
import pytest
from pmtvs.individual import (
    mean, std, variance, min_max, percentiles,
    skewness, kurtosis, rms, peak_to_peak,
    crest_factor, zero_crossings, mean_crossings,
)


class TestBasicStats:
    """Test basic statistical functions"""

    def test_mean_simple(self):
        data = np.array([1, 2, 3, 4, 5])
        assert mean(data) == 3.0

    def test_mean_with_nan(self):
        data = np.array([1, 2, np.nan, 4, 5])
        assert mean(data) == 3.0

    def test_std_population(self):
        data = np.array([2, 4, 4, 4, 5, 5, 7, 9])
        assert np.isclose(std(data, ddof=0), 2.0)

    def test_std_sample(self):
        data = np.array([2, 4, 4, 4, 5, 5, 7, 9])
        result = std(data, ddof=1)
        assert result > std(data, ddof=0)

    def test_variance(self):
        data = np.array([2, 4, 4, 4, 5, 5, 7, 9])
        assert np.isclose(variance(data), 4.0)

    def test_min_max(self):
        data = np.array([3, 1, 4, 1, 5, 9, 2, 6])
        mn, mx = min_max(data)
        assert mn == 1.0
        assert mx == 9.0

    def test_percentiles_default(self):
        data = np.arange(100)
        p = percentiles(data)
        assert len(p) == 3
        assert np.isclose(p[1], 49.5)  # median

    def test_percentiles_custom(self):
        data = np.arange(100)
        p = percentiles(data, [10, 90])
        assert len(p) == 2


class TestHigherMoments:
    """Test skewness and kurtosis"""

    def test_skewness_symmetric(self):
        # Symmetric distribution should have ~0 skewness
        np.random.seed(42)
        data = np.random.randn(10000)
        assert abs(skewness(data)) < 0.1

    def test_skewness_right_skewed(self):
        # Exponential is right-skewed
        np.random.seed(42)
        data = np.random.exponential(1, 10000)
        assert skewness(data) > 1.0

    def test_kurtosis_normal(self):
        # Normal distribution has excess kurtosis ~0
        np.random.seed(42)
        data = np.random.randn(10000)
        assert abs(kurtosis(data, fisher=True)) < 0.2

    def test_kurtosis_not_fisher(self):
        # Non-excess kurtosis of normal is ~3
        np.random.seed(42)
        data = np.random.randn(10000)
        assert abs(kurtosis(data, fisher=False) - 3.0) < 0.2


class TestSignalMetrics:
    """Test signal-specific metrics"""

    def test_rms_sine(self):
        # RMS of sine wave is 1/sqrt(2)
        t = np.linspace(0, 2*np.pi, 1000)
        data = np.sin(t)
        expected = 1.0 / np.sqrt(2)
        assert np.isclose(rms(data), expected, rtol=0.01)

    def test_peak_to_peak(self):
        data = np.array([-2, 0, 3, 1])
        assert peak_to_peak(data) == 5.0

    def test_crest_factor_sine(self):
        # Crest factor of sine is sqrt(2)
        t = np.linspace(0, 2*np.pi, 1000)
        data = np.sin(t)
        assert np.isclose(crest_factor(data), np.sqrt(2), rtol=0.01)

    def test_crest_factor_dc(self):
        # DC signal: peak = rms, crest factor = 1
        data = np.ones(100) * 5
        assert np.isclose(crest_factor(data), 1.0)

    def test_zero_crossings(self):
        data = np.array([1, -1, 1, -1, 1])
        assert zero_crossings(data) == 4

    def test_zero_crossings_no_crossing(self):
        data = np.array([1, 2, 3, 4, 5])
        assert zero_crossings(data) == 0

    def test_mean_crossings(self):
        data = np.array([0, 2, 0, 2, 0])  # mean = 0.8
        # centered: [-0.8, 1.2, -0.8, 1.2, -0.8]
        assert mean_crossings(data) == 4

    def test_mean_crossings_constant(self):
        data = np.ones(100) * 5
        assert mean_crossings(data) == 0
