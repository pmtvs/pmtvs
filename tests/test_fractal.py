"""Tests for fractal primitives"""

import numpy as np
import pytest
from pmtvs.individual import (
    hurst_exponent,
    dfa,
    hurst_r2,
)


class TestHurstExponent:
    """Test Hurst exponent function"""

    def test_hurst_white_noise(self):
        # White noise should have H ≈ 0.5
        np.random.seed(42)
        data = np.random.randn(1000)
        h = hurst_exponent(data)
        assert 0.3 < h < 0.7, f"Hurst {h} not near 0.5 for white noise"

    def test_hurst_random_walk(self):
        # Random walk (cumsum of white noise) should have H > 0.5
        np.random.seed(42)
        data = np.cumsum(np.random.randn(1000))
        h = hurst_exponent(data)
        assert h > 0.4, f"Hurst {h} too low for random walk"

    def test_hurst_trending(self):
        # Strong trend should have H near 1
        data = np.arange(500).astype(float) + np.random.randn(500) * 10
        h = hurst_exponent(data)
        assert h > 0.6, f"Hurst {h} too low for trending signal"

    def test_hurst_short_signal(self):
        # Too short signal returns NaN
        data = np.array([1, 2, 3, 4, 5])
        h = hurst_exponent(data)
        assert np.isnan(h)

    def test_hurst_bounds(self):
        # Hurst should be clipped to [0, 1]
        np.random.seed(42)
        data = np.random.randn(500)
        h = hurst_exponent(data)
        assert 0 <= h <= 1


class TestDFA:
    """Test Detrended Fluctuation Analysis function"""

    def test_dfa_white_noise(self):
        # White noise should have alpha ≈ 0.5
        np.random.seed(42)
        data = np.random.randn(500)
        alpha = dfa(data)
        assert 0.3 < alpha < 0.8, f"DFA {alpha} not near 0.5 for white noise"

    def test_dfa_brownian(self):
        # Brownian motion (random walk) should have alpha ≈ 1.5
        np.random.seed(42)
        data = np.cumsum(np.random.randn(500))
        alpha = dfa(data)
        assert alpha > 1.0, f"DFA {alpha} too low for Brownian motion"

    def test_dfa_short_signal(self):
        # Too short signal returns NaN
        data = np.random.randn(30)
        alpha = dfa(data)
        assert np.isnan(alpha)

    def test_dfa_via_hurst(self):
        # Can call DFA via hurst_exponent with method='dfa'
        np.random.seed(42)
        data = np.random.randn(500)
        alpha1 = dfa(data)
        alpha2 = hurst_exponent(data, method='dfa')
        assert np.isclose(alpha1, alpha2)


class TestHurstR2:
    """Test Hurst R-squared function"""

    def test_hurst_r2_good_fit(self):
        # Random walk should have reasonably good R²
        np.random.seed(42)
        data = np.cumsum(np.random.randn(1000))
        r2 = hurst_r2(data)
        assert r2 > 0.7, f"R² {r2} too low for random walk"

    def test_hurst_r2_range(self):
        # R² should be in [0, 1] for reasonable fits
        np.random.seed(42)
        data = np.random.randn(500)
        r2 = hurst_r2(data)
        if not np.isnan(r2):
            assert 0 <= r2 <= 1

    def test_hurst_r2_short_signal(self):
        # Too short signal returns NaN
        data = np.array([1, 2, 3, 4, 5])
        r2 = hurst_r2(data)
        assert np.isnan(r2)
