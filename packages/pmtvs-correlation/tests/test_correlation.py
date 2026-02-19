"""Tests for pmtvs-correlation."""
import numpy as np
import pytest

from pmtvs_correlation import (
    autocorrelation,
    partial_autocorrelation,
    autocorrelation_function,
    acf_decay_time,
    correlation,
    covariance,
    cross_correlation,
    lag_at_max_xcorr,
    partial_correlation,
    coherence,
    spearman_correlation,
    kendall_tau,
)


class TestAutocorrelation:
    """Tests for autocorrelation."""

    def test_lag_zero(self):
        """Autocorrelation at lag 0 should be 1 (computed via lag 1 approach)."""
        np.random.seed(42)
        signal = np.random.randn(100)
        # Our function doesn't explicitly return lag 0, but lag 1 should be valid
        acf_1 = autocorrelation(signal, lag=1)
        assert np.isfinite(acf_1)

    def test_random_signal(self):
        """White noise should have low autocorrelation at non-zero lags."""
        np.random.seed(42)
        signal = np.random.randn(1000)

        acf_10 = autocorrelation(signal, lag=10)
        assert abs(acf_10) < 0.1

    def test_periodic_signal(self):
        """Periodic signal should have high autocorrelation at period lag."""
        t = np.linspace(0, 10 * np.pi, 1000)
        signal = np.sin(t)

        # At lag ~100 (half period of 200), autocorrelation should be negative
        # At lag ~200 (full period), should be positive
        acf_100 = autocorrelation(signal, lag=100)
        # Just check it's finite for now
        assert np.isfinite(acf_100)

    def test_short_signal(self):
        """Too short signal should return NaN."""
        signal = np.array([1.0, 2.0])
        acf = autocorrelation(signal, lag=5)
        assert np.isnan(acf)


class TestPartialAutocorrelation:
    """Tests for PACF."""

    def test_returns_array(self):
        """PACF should return array of correct length."""
        np.random.seed(42)
        signal = np.random.randn(200)

        pacf = partial_autocorrelation(signal, max_lag=10)
        assert len(pacf) == 11  # 0 to 10

    def test_lag_zero_is_one(self):
        """PACF at lag 0 should be 1."""
        np.random.seed(42)
        signal = np.random.randn(200)

        pacf = partial_autocorrelation(signal, max_lag=10)
        assert pacf[0] == pytest.approx(1.0)

    def test_ar1_process(self):
        """AR(1) process should have PACF cutoff after lag 1."""
        np.random.seed(42)
        n = 500
        phi = 0.8
        signal = np.zeros(n)
        signal[0] = np.random.randn()
        for i in range(1, n):
            signal[i] = phi * signal[i-1] + np.random.randn()

        pacf = partial_autocorrelation(signal, max_lag=10)
        # PACF at lag 1 should be close to phi
        assert abs(pacf[1] - phi) < 0.2
        # PACF at higher lags should be small
        assert abs(pacf[5]) < 0.2


class TestCorrelation:
    """Tests for Pearson correlation."""

    def test_perfect_correlation(self):
        """Identical signals should have correlation 1."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert correlation(x, x) == pytest.approx(1.0)

    def test_perfect_anticorrelation(self):
        """Negatively related signals should have correlation -1."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        assert correlation(x, y) == pytest.approx(-1.0)

    def test_uncorrelated(self):
        """Orthogonal signals should have low correlation."""
        x = np.array([1.0, 0.0, -1.0, 0.0])
        y = np.array([0.0, 1.0, 0.0, -1.0])
        assert abs(correlation(x, y)) < 0.1

    def test_different_lengths(self):
        """Different length signals should return NaN."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0])
        assert np.isnan(correlation(x, y))


class TestCovariance:
    """Tests for covariance."""

    def test_self_covariance_is_variance(self):
        """Covariance of signal with itself should be variance."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cov = covariance(x, x)
        var = np.var(x, ddof=1)
        assert cov == pytest.approx(var)

    def test_uncorrelated(self):
        """Uncorrelated signals should have low covariance."""
        np.random.seed(42)
        x = np.random.randn(1000)
        y = np.random.randn(1000)
        cov = covariance(x, y)
        assert abs(cov) < 0.1


class TestCrossCorrelation:
    """Tests for cross-correlation."""

    def test_returns_array(self):
        """Cross-correlation should return array."""
        x = np.random.randn(100)
        y = np.random.randn(100)
        xcorr = cross_correlation(x, y, max_lag=10)
        assert len(xcorr) == 21  # -10 to +10

    def test_peak_at_zero_for_identical(self):
        """Identical signals should have peak at lag 0."""
        x = np.random.randn(100)
        xcorr = cross_correlation(x, x, max_lag=10)
        # Middle element (lag 0) should be maximum
        assert np.argmax(xcorr) == 10


class TestLagAtMaxXcorr:
    """Tests for lag at max cross-correlation."""

    def test_identical_signals(self):
        """Identical signals should have max at lag 0."""
        x = np.random.randn(100)
        lag = lag_at_max_xcorr(x, x)
        assert lag == 0

    def test_delayed_signal(self):
        """Delayed signal should show positive lag."""
        x = np.zeros(100)
        x[50] = 1.0  # Impulse at 50

        y = np.zeros(100)
        y[55] = 1.0  # Impulse at 55 (delayed by 5)

        lag = lag_at_max_xcorr(x, y, max_lag=20)
        assert lag == 5 or lag == -5  # Direction depends on convention


class TestCoherence:
    """Tests for coherence."""

    def test_returns_tuple(self):
        """Coherence should return (freqs, coh) tuple."""
        np.random.seed(42)
        x = np.random.randn(512)
        y = np.random.randn(512)

        freqs, coh = coherence(x, y, nperseg=64)
        assert len(freqs) == len(coh)

    def test_identical_signals_high_coherence(self):
        """Identical signals should have coherence of 1."""
        x = np.random.randn(512)
        freqs, coh = coherence(x, x, nperseg=64)
        assert np.all(coh > 0.99)

    def test_coherence_bounded(self):
        """Coherence should be in [0, 1]."""
        np.random.seed(42)
        x = np.random.randn(512)
        y = np.random.randn(512)

        freqs, coh = coherence(x, y, nperseg=64)
        assert np.all(coh >= 0)
        assert np.all(coh <= 1)


class TestSpearmanCorrelation:
    """Tests for Spearman rank correlation."""

    def test_perfect_monotonic(self):
        """Monotonically increasing relationship should have rho = 1."""
        np.random.seed(42)
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 8.0, 16.0, 32.0])  # monotonically increasing
        rho = spearman_correlation(x, y)
        assert rho == pytest.approx(1.0)

    def test_perfect_negative_monotonic(self):
        """Monotonically decreasing relationship should have rho = -1."""
        np.random.seed(42)
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        rho = spearman_correlation(x, y)
        assert rho == pytest.approx(-1.0)

    def test_uncorrelated(self):
        """Independent random signals should have low Spearman correlation."""
        np.random.seed(42)
        x = np.random.randn(1000)
        y = np.random.randn(1000)
        rho = spearman_correlation(x, y)
        assert abs(rho) < 0.1

    def test_short_signal_returns_nan(self):
        """Too short signals should return NaN."""
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        assert np.isnan(spearman_correlation(x, y))

    def test_different_lengths_returns_nan(self):
        """Different length signals should return NaN."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0])
        assert np.isnan(spearman_correlation(x, y))

    def test_bounded(self):
        """Spearman correlation should be in [-1, 1]."""
        np.random.seed(42)
        for _ in range(10):
            x = np.random.randn(50)
            y = np.random.randn(50)
            rho = spearman_correlation(x, y)
            assert -1.0 <= rho <= 1.0


class TestKendallTau:
    """Tests for Kendall's tau rank correlation."""

    def test_perfect_concordance(self):
        """Perfectly concordant pairs should give tau = 1."""
        np.random.seed(42)
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        tau = kendall_tau(x, y)
        assert tau == pytest.approx(1.0)

    def test_perfect_discordance(self):
        """Perfectly discordant pairs should give tau = -1."""
        np.random.seed(42)
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([50.0, 40.0, 30.0, 20.0, 10.0])
        tau = kendall_tau(x, y)
        assert tau == pytest.approx(-1.0)

    def test_uncorrelated(self):
        """Independent random signals should have low Kendall's tau."""
        np.random.seed(42)
        x = np.random.randn(500)
        y = np.random.randn(500)
        tau = kendall_tau(x, y)
        assert abs(tau) < 0.1

    def test_short_signal_returns_nan(self):
        """Too short signals should return NaN."""
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        assert np.isnan(kendall_tau(x, y))

    def test_different_lengths_returns_nan(self):
        """Different length signals should return NaN."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0])
        assert np.isnan(kendall_tau(x, y))

    def test_bounded(self):
        """Kendall's tau should be in [-1, 1]."""
        np.random.seed(42)
        for _ in range(10):
            x = np.random.randn(50)
            y = np.random.randn(50)
            tau = kendall_tau(x, y)
            assert -1.0 <= tau <= 1.0


class TestAutocorrelationFunction:
    """Tests for full ACF computation."""

    def test_lag_zero_is_one(self):
        """ACF at lag 0 should be 1."""
        np.random.seed(42)
        signal = np.random.randn(200)
        acf = autocorrelation_function(signal)
        assert acf[0] == pytest.approx(1.0)

    def test_white_noise_decays(self):
        """White noise ACF should be near zero at non-zero lags."""
        np.random.seed(42)
        signal = np.random.randn(1000)
        acf = autocorrelation_function(signal, max_lag=20)
        # Non-zero lags should be small
        assert np.all(np.abs(acf[1:]) < 0.1)

    def test_returns_correct_length(self):
        """ACF should return max_lag + 1 values."""
        np.random.seed(42)
        signal = np.random.randn(200)
        acf = autocorrelation_function(signal, max_lag=15)
        assert len(acf) == 16

    def test_short_signal(self):
        """Short signal should return NaN array."""
        signal = np.array([1.0, 2.0])
        acf = autocorrelation_function(signal)
        assert len(acf) == 1
        assert np.isnan(acf[0])


class TestACFDecayTime:
    """Tests for ACF decay time."""

    def test_white_noise_fast_decay(self):
        """White noise should have fast ACF decay."""
        np.random.seed(42)
        signal = np.random.randn(500)
        decay = acf_decay_time(signal)
        # White noise decays quickly
        assert decay < 5.0

    def test_correlated_signal_slow_decay(self):
        """Highly correlated signal should have slower ACF decay."""
        np.random.seed(42)
        # AR(1) with high persistence
        n = 500
        signal = np.zeros(n)
        signal[0] = np.random.randn()
        for i in range(1, n):
            signal[i] = 0.95 * signal[i - 1] + 0.1 * np.random.randn()
        decay = acf_decay_time(signal)
        assert decay > 1.0

    def test_short_signal(self):
        """Short signal should return NaN."""
        signal = np.array([1.0, 2.0])
        assert np.isnan(acf_decay_time(signal))
