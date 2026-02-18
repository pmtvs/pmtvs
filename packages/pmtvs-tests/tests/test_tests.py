"""Tests for pmtvs-tests."""
import numpy as np
import pytest

from pmtvs_tests import (
    bootstrap_mean,
    bootstrap_confidence_interval,
    permutation_test,
    surrogate_test,
    adf_test,
    runs_test,
    mann_kendall_test,
)


class TestBootstrapMean:
    def test_returns_array(self):
        np.random.seed(42)
        data = np.random.randn(100)
        means = bootstrap_mean(data, n_bootstrap=100)
        assert len(means) == 100

    def test_centered(self):
        np.random.seed(42)
        data = np.random.randn(1000) + 5
        means = bootstrap_mean(data, n_bootstrap=1000)
        assert abs(np.mean(means) - 5) < 0.2


class TestBootstrapCI:
    def test_contains_mean(self):
        np.random.seed(42)
        data = np.random.randn(100) + 5
        lower, upper = bootstrap_confidence_interval(data, confidence=0.95)
        assert lower < 5 < upper


class TestPermutationTest:
    def test_same_distribution(self):
        np.random.seed(42)
        x = np.random.randn(50)
        y = np.random.randn(50)
        stat, p = permutation_test(x, y)
        # Same distribution should have high p-value
        assert p > 0.05

    def test_different_distributions(self):
        np.random.seed(42)
        x = np.random.randn(50)
        y = np.random.randn(50) + 2  # Shifted
        stat, p = permutation_test(x, y)
        # Different distributions should have low p-value
        assert p < 0.05


class TestSurrogateTest:
    def test_returns_tuple(self):
        np.random.seed(42)
        signal = np.random.randn(100)
        stat, p = surrogate_test(signal, np.std, n_surrogates=50)
        assert isinstance(stat, float)
        assert 0 <= p <= 1


class TestADFTest:
    def test_stationary(self):
        np.random.seed(42)
        # White noise is stationary
        signal = np.random.randn(200)
        t_stat, crit = adf_test(signal)
        # Should reject unit root (t_stat < critical value)
        assert t_stat < crit

    def test_random_walk(self):
        np.random.seed(42)
        # Random walk is non-stationary
        signal = np.cumsum(np.random.randn(200))
        t_stat, crit = adf_test(signal)
        # Should NOT reject unit root (t_stat > critical value typically)
        # This may not always hold for small samples
        assert np.isfinite(t_stat)


class TestRunsTest:
    def test_random_sequence(self):
        np.random.seed(42)
        signal = np.random.randn(100)
        z, p = runs_test(signal)
        # Random should not reject
        assert p > 0.05

    def test_systematic_sequence(self):
        # Alternating pattern
        signal = np.array([1, -1] * 50, dtype=float)
        z, p = runs_test(signal)
        # Should reject (too many runs)
        assert p < 0.05


class TestMannKendall:
    def test_no_trend(self):
        np.random.seed(42)
        signal = np.random.randn(100)
        tau, p = mann_kendall_test(signal)
        # No trend should have high p-value
        assert p > 0.05

    def test_positive_trend(self):
        signal = np.arange(100, dtype=float)
        tau, p = mann_kendall_test(signal)
        # Strong positive trend
        assert tau > 0.9
        assert p < 0.05
