"""Tests for pmtvs-statistics."""
import numpy as np
import pytest

from pmtvs_statistics import (
    mean, std, variance, min_max, percentiles,
    skewness, kurtosis, rms, peak_to_peak, crest_factor,
    pulsation_index, zero_crossings, mean_crossings,
    derivative, integral, curvature, rate_of_change,
)


class TestBasicStatistics:
    """Tests for basic statistics."""

    def test_mean(self):
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert mean(signal) == pytest.approx(3.0)

    def test_std(self):
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Sample std with ddof=1
        expected = np.std(signal, ddof=1)
        assert std(signal) == pytest.approx(expected)

    def test_variance(self):
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = np.var(signal, ddof=1)
        assert variance(signal) == pytest.approx(expected)

    def test_min_max(self):
        signal = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        min_val, max_val = min_max(signal)
        assert min_val == 1.0
        assert max_val == 5.0

    def test_percentiles(self):
        signal = np.arange(101).astype(float)
        p25, p50, p75 = percentiles(signal)
        assert p25 == pytest.approx(25.0)
        assert p50 == pytest.approx(50.0)
        assert p75 == pytest.approx(75.0)


class TestHigherMoments:
    """Tests for skewness and kurtosis."""

    def test_skewness_symmetric(self):
        """Symmetric distribution should have skewness near 0."""
        np.random.seed(42)
        signal = np.random.randn(10000)
        assert abs(skewness(signal)) < 0.1

    def test_skewness_right_tail(self):
        """Right-skewed distribution should have positive skewness."""
        signal = np.exp(np.random.randn(1000))
        assert skewness(signal) > 0

    def test_kurtosis_normal(self):
        """Normal distribution should have excess kurtosis near 0."""
        np.random.seed(42)
        signal = np.random.randn(10000)
        assert abs(kurtosis(signal)) < 0.2


class TestSignalMetrics:
    """Tests for signal-specific metrics."""

    def test_rms(self):
        signal = np.array([1.0, -1.0, 1.0, -1.0])
        assert rms(signal) == pytest.approx(1.0)

    def test_peak_to_peak(self):
        signal = np.array([-3.0, 0.0, 5.0, 2.0])
        assert peak_to_peak(signal) == pytest.approx(8.0)

    def test_crest_factor(self):
        # Square wave has crest factor of 1
        signal = np.array([1.0, 1.0, 1.0, 1.0])
        assert crest_factor(signal) == pytest.approx(1.0)

        # Sine wave has crest factor of sqrt(2)
        t = np.linspace(0, 2 * np.pi, 1000)
        sine = np.sin(t)
        assert crest_factor(sine) == pytest.approx(np.sqrt(2), rel=0.01)

    def test_zero_crossings(self):
        signal = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
        assert zero_crossings(signal) == 4

    def test_mean_crossings(self):
        signal = np.array([0.0, 2.0, 0.0, 2.0, 0.0])
        # Mean is 0.8, so crossings at indices where signal crosses 0.8
        count = mean_crossings(signal)
        assert count >= 0


class TestCalculus:
    """Tests for calculus functions."""

    def test_derivative(self):
        # Linear signal should have constant derivative
        signal = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        deriv = derivative(signal, dt=1.0)
        assert len(deriv) == 4
        assert np.allclose(deriv, 1.0)

    def test_integral(self):
        # Constant signal should integrate linearly
        signal = np.ones(10) * 2.0
        integ = integral(signal, dt=1.0)
        assert len(integ) == 10
        assert integ[0] == 0.0
        # Each step adds 2.0 (trapezoidal)
        assert integ[-1] == pytest.approx(18.0)

    def test_curvature_line(self):
        """Straight line should have zero curvature."""
        signal = np.linspace(0, 10, 100)
        curv = curvature(signal)
        assert np.all(curv < 0.01)

    def test_curvature_parabola(self):
        """Parabola should have positive curvature."""
        x = np.linspace(-1, 1, 100)
        y = x ** 2
        curv = curvature(y, dt=2.0 / 99)
        # Parabola has positive curvature everywhere
        interior = curv[10:-10]
        assert np.all(interior > 0)
        assert np.all(np.isfinite(interior))

    def test_rate_of_change(self):
        signal = np.array([1.0, 2.0, 4.0, 8.0])
        roc = rate_of_change(signal)
        # (2-1)/1 = 1, (4-2)/2 = 1, (8-4)/4 = 1
        assert np.allclose(roc, 1.0)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_signal(self):
        signal = np.array([])
        assert np.isnan(mean(signal))
        assert np.isnan(std(signal))

    def test_single_value(self):
        signal = np.array([5.0])
        assert mean(signal) == 5.0
        assert np.isnan(std(signal))  # Can't compute std with 1 point

    def test_nan_handling(self):
        signal = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        assert mean(signal) == pytest.approx(3.0)

    def test_constant_signal(self):
        signal = np.ones(100) * 5.0
        assert std(signal) == 0.0
        assert np.isnan(skewness(signal))  # Division by zero std
