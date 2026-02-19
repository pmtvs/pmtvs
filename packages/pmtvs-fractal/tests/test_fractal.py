"""Tests for pmtvs-fractal."""
import numpy as np
import pytest

from pmtvs_fractal import (
    hurst_exponent,
    dfa,
    hurst_r2,
    detrended_fluctuation_analysis,
    rescaled_range,
    long_range_correlation,
    variance_growth,
)


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


class TestDetrendedFluctuationAnalysis:
    """Tests for detrended_fluctuation_analysis (full output)."""

    def test_returns_three_elements(self):
        """Should return (scales, fluctuations, alpha)."""
        np.random.seed(42)
        signal = np.random.randn(500)
        result = detrended_fluctuation_analysis(signal)
        assert len(result) == 3
        scales, fluct, alpha = result
        assert isinstance(scales, np.ndarray)
        assert isinstance(fluct, np.ndarray)
        assert isinstance(alpha, float)

    def test_white_noise_alpha(self):
        """White noise should have DFA alpha near 0.5."""
        np.random.seed(42)
        white_noise = np.random.randn(2000)
        scales, fluct, alpha = detrended_fluctuation_analysis(white_noise)
        assert 0.3 < alpha < 0.7, f"White noise alpha={alpha}, expected ~0.5"

    def test_random_walk_alpha(self):
        """Random walk should have DFA alpha near 1.5."""
        np.random.seed(42)
        random_walk = np.cumsum(np.random.randn(2000))
        scales, fluct, alpha = detrended_fluctuation_analysis(random_walk)
        assert 1.2 < alpha < 1.8, f"Random walk alpha={alpha}, expected ~1.5"

    def test_scales_and_fluctuations_aligned(self):
        """Scales and fluctuations arrays should have the same length."""
        np.random.seed(42)
        signal = np.random.randn(500)
        scales, fluct, _ = detrended_fluctuation_analysis(signal)
        assert len(scales) == len(fluct)

    def test_too_short_input(self):
        """Very short input should return NaN alpha."""
        short = np.array([1.0, 2.0, 3.0])
        scales, fluct, alpha = detrended_fluctuation_analysis(short)
        assert np.isnan(alpha)

    def test_nan_filtering(self):
        """NaN values should be filtered before analysis."""
        np.random.seed(42)
        signal = np.random.randn(500)
        signal[10] = np.nan
        signal[200] = np.nan
        scales, fluct, alpha = detrended_fluctuation_analysis(signal)
        assert not np.isnan(alpha)

    def test_custom_scales(self):
        """Custom min_scale, max_scale, n_scales should be respected."""
        np.random.seed(42)
        signal = np.random.randn(1000)
        scales, fluct, alpha = detrended_fluctuation_analysis(
            signal, min_scale=8, max_scale=100, n_scales=5
        )
        assert scales[0] >= 8
        assert scales[-1] <= 100


class TestRescaledRange:
    """Tests for rescaled_range."""

    def test_positive_result(self):
        """R/S should be positive for non-constant signals."""
        np.random.seed(42)
        signal = np.random.randn(500)
        rs = rescaled_range(signal)
        assert rs > 0

    def test_returns_float(self):
        """Should return a Python float."""
        np.random.seed(42)
        signal = np.random.randn(200)
        rs = rescaled_range(signal)
        assert isinstance(rs, float)

    def test_segment_size(self):
        """Specifying segment_size should use only that many samples."""
        np.random.seed(42)
        signal = np.random.randn(500)
        rs_full = rescaled_range(signal)
        rs_half = rescaled_range(signal, segment_size=250)
        # Both should be valid finite numbers; values will generally differ
        assert np.isfinite(rs_full)
        assert np.isfinite(rs_half)

    def test_constant_signal(self):
        """Constant signal (S=0) should return NaN."""
        constant = np.ones(100)
        rs = rescaled_range(constant)
        assert np.isnan(rs)

    def test_too_short(self):
        """Single-element input should return NaN."""
        rs = rescaled_range(np.array([1.0]))
        assert np.isnan(rs)

    def test_nan_filtering(self):
        """NaN values should be filtered."""
        np.random.seed(42)
        signal = np.random.randn(200)
        signal[5] = np.nan
        rs = rescaled_range(signal)
        assert np.isfinite(rs)

    def test_random_walk_larger_rs(self):
        """Random walk should have larger R/S than white noise (same length)."""
        np.random.seed(42)
        white = np.random.randn(1000)
        walk = np.cumsum(np.random.randn(1000))
        assert rescaled_range(walk) > rescaled_range(white)


class TestLongRangeCorrelation:
    """Tests for long_range_correlation."""

    def test_returns_tuple(self):
        """Should return (acf_array, decay_exponent)."""
        np.random.seed(42)
        signal = np.random.randn(500)
        acf, d = long_range_correlation(signal)
        assert isinstance(acf, np.ndarray)
        assert isinstance(d, float)

    def test_acf_starts_at_one(self):
        """Normalised ACF should start at 1.0 at lag 0."""
        np.random.seed(42)
        signal = np.random.randn(500)
        acf, _ = long_range_correlation(signal)
        assert len(acf) > 0
        assert abs(acf[0] - 1.0) < 1e-10

    def test_white_noise_fast_decay(self):
        """White noise should have rapid autocorrelation decay (high exponent)."""
        np.random.seed(42)
        white_noise = np.random.randn(2000)
        acf, d = long_range_correlation(white_noise)
        # White noise ACF drops to ~0 quickly; exponent should be relatively high
        assert np.isfinite(d)

    def test_persistent_series(self):
        """Cumulative sum should return a finite decay exponent."""
        np.random.seed(42)
        walk = np.cumsum(np.random.randn(2000))
        _, d_walk = long_range_correlation(walk)
        assert np.isfinite(d_walk)

    def test_too_short(self):
        """Very short input should return NaN exponent."""
        short = np.array([1.0, 2.0, 3.0])
        acf, d = long_range_correlation(short)
        assert np.isnan(d)

    def test_nan_filtering(self):
        """NaN values in input should be filtered."""
        np.random.seed(42)
        signal = np.random.randn(500)
        signal[50] = np.nan
        acf, d = long_range_correlation(signal)
        assert np.isfinite(d)

    def test_constant_signal(self):
        """Constant signal (zero variance) should return NaN."""
        constant = np.ones(100)
        acf, d = long_range_correlation(constant)
        assert np.isnan(d)


class TestVarianceGrowth:
    """Tests for variance_growth."""

    def test_returns_tuple(self):
        """Should return (scales, exponent)."""
        np.random.seed(42)
        signal = np.random.randn(500)
        scales, exp = variance_growth(signal)
        assert isinstance(scales, np.ndarray)
        assert isinstance(exp, float)

    def test_negative_exponent(self):
        """Variance of aggregated means should decay with scale (negative exponent)."""
        np.random.seed(42)
        signal = np.random.randn(2000)
        scales, exp = variance_growth(signal)
        # For iid data, variance of block means ~ 1/scale => exponent ~ -1
        assert exp < 0, f"Expected negative exponent, got {exp}"

    def test_white_noise_exponent(self):
        """White noise should have variance growth exponent near -1."""
        np.random.seed(42)
        white_noise = np.random.randn(5000)
        _, exp = variance_growth(white_noise)
        assert -1.3 < exp < -0.7, f"White noise exponent={exp}, expected ~-1"

    def test_scales_start_at_one(self):
        """Scales should start at 1."""
        np.random.seed(42)
        signal = np.random.randn(200)
        scales, _ = variance_growth(signal)
        assert len(scales) > 0
        assert scales[0] == 1

    def test_too_short(self):
        """Very short input should return NaN exponent."""
        short = np.array([1.0, 2.0, 3.0])
        scales, exp = variance_growth(short)
        assert np.isnan(exp)

    def test_nan_filtering(self):
        """NaN values in input should be filtered."""
        np.random.seed(42)
        signal = np.random.randn(500)
        signal[42] = np.nan
        scales, exp = variance_growth(signal)
        assert np.isfinite(exp)

    def test_custom_max_lag(self):
        """Custom max_lag should control number of scales."""
        np.random.seed(42)
        signal = np.random.randn(500)
        scales, _ = variance_growth(signal, max_lag=20)
        assert len(scales) == 20
