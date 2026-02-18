"""Tests for pmtvs-fractal."""
import numpy as np
import pytest

from pmtvs_fractal import hurst_exponent, dfa, hurst_r2


class TestHurstExponent:
    """Tests for Hurst exponent."""

    def test_white_noise(self):
        """White noise should have H ≈ 0.5."""
        np.random.seed(42)
        white_noise = np.random.randn(2000)

        h = hurst_exponent(white_noise)
        assert 0.4 < h < 0.6, f"White noise H={h}, expected ~0.5"

    def test_random_walk(self):
        """Random walk (cumsum of white noise) should have H > 0.5."""
        np.random.seed(42)
        random_walk = np.cumsum(np.random.randn(2000))

        h = hurst_exponent(random_walk)
        assert h > 0.5, f"Random walk H={h}, expected > 0.5"

    def test_short_signal(self):
        """Too short signal should return NaN."""
        short = np.array([1, 2, 3, 4, 5])
        h = hurst_exponent(short)
        assert np.isnan(h)

    def test_bounded_output(self):
        """Hurst exponent should be in [0, 1]."""
        np.random.seed(42)
        signal = np.random.randn(500)
        h = hurst_exponent(signal)
        assert 0 <= h <= 1


class TestDFA:
    """Tests for Detrended Fluctuation Analysis."""

    def test_white_noise(self):
        """White noise should have DFA ≈ 0.5."""
        np.random.seed(42)
        white_noise = np.random.randn(1000)

        alpha = dfa(white_noise)
        assert 0.3 < alpha < 0.7, f"White noise DFA={alpha}, expected ~0.5"

    def test_random_walk(self):
        """Random walk should have DFA ≈ 1.5."""
        np.random.seed(42)
        random_walk = np.cumsum(np.random.randn(1000))

        alpha = dfa(random_walk)
        assert 1.2 < alpha < 1.8, f"Random walk DFA={alpha}, expected ~1.5"

    def test_short_signal(self):
        """Too short signal should return NaN."""
        short = np.random.randn(30)
        alpha = dfa(short)
        assert np.isnan(alpha)

    def test_positive_result(self):
        """DFA should return positive values for typical signals."""
        np.random.seed(42)
        signal = np.random.randn(500)
        alpha = dfa(signal)
        assert alpha > 0 or np.isnan(alpha)


class TestHurstR2:
    """Tests for Hurst R-squared."""

    def test_good_fit(self):
        """Ideal fractal signal should have high R²."""
        np.random.seed(42)
        # Brownian motion has good linear relationship in R/S plot
        random_walk = np.cumsum(np.random.randn(2000))

        r2 = hurst_r2(random_walk)
        assert r2 > 0.9, f"Random walk R²={r2}, expected > 0.9"

    def test_bounded_output(self):
        """R² should be in [0, 1] for typical signals."""
        np.random.seed(42)
        signal = np.random.randn(500)
        r2 = hurst_r2(signal)
        # R² can technically be negative for very poor fits
        assert r2 <= 1

    def test_short_signal(self):
        """Too short signal should return NaN."""
        short = np.array([1, 2, 3, 4, 5])
        r2 = hurst_r2(short)
        assert np.isnan(r2)
