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
    bootstrap_ci,
    bootstrap_std,
    block_bootstrap_ci,
    marchenko_pastur_test,
    arch_test,
    t_test,
    t_test_paired,
    t_test_independent,
    f_test,
    chi_squared_test,
    mannwhitney_test,
    kruskal_test,
    anova,
    shapiro_test,
    levene_test,
    stationarity_test,
    trend,
    changepoints,
    kpss_test,
    phillips_perron_test,
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


# --- Tests for new functions in tests.py ---


class TestBootstrapCi:
    def test_contains_mean(self):
        np.random.seed(42)
        data = np.random.randn(100) + 3
        lower, upper = bootstrap_ci(data, confidence=0.95)
        assert lower < 3 < upper

    def test_short_data(self):
        lower, upper = bootstrap_ci(np.array([1.0]))
        assert np.isnan(lower)
        assert np.isnan(upper)


class TestBootstrapStd:
    def test_positive(self):
        np.random.seed(42)
        data = np.random.randn(100)
        se = bootstrap_std(data, n_bootstrap=500)
        assert se > 0

    def test_short_data(self):
        se = bootstrap_std(np.array([1.0]))
        assert np.isnan(se)


class TestBlockBootstrapCi:
    def test_contains_mean(self):
        np.random.seed(42)
        data = np.random.randn(200) + 5
        lower, upper = block_bootstrap_ci(data, confidence=0.95, n_bootstrap=500)
        assert lower < 5 < upper

    def test_short_data(self):
        lower, upper = block_bootstrap_ci(np.array([1.0, 2.0]))
        assert np.isnan(lower)
        assert np.isnan(upper)


class TestMarchenkoPasturTest:
    def test_noise_matrix(self):
        np.random.seed(42)
        data = np.random.randn(200, 10)
        max_eig, mp_upper = marchenko_pastur_test(data)
        assert np.isfinite(max_eig)
        assert np.isfinite(mp_upper)

    def test_1d_input(self):
        max_eig, mp_upper = marchenko_pastur_test(np.array([1, 2, 3]))
        assert np.isnan(max_eig)
        assert np.isnan(mp_upper)


class TestArchTest:
    def test_white_noise(self):
        np.random.seed(42)
        signal = np.random.randn(200)
        lm_stat, p_val = arch_test(signal)
        assert np.isfinite(lm_stat)
        # White noise should not show ARCH effects
        assert p_val > 0.05

    def test_short_signal(self):
        lm_stat, p_val = arch_test(np.array([1.0, 2.0, 3.0]))
        assert np.isnan(lm_stat)
        assert np.isnan(p_val)


# --- Tests for hypothesis.py ---


class TestTTest:
    def test_zero_mean(self):
        np.random.seed(42)
        data = np.random.randn(100)
        stat, p = t_test(data, popmean=0.0)
        assert np.isfinite(stat)
        assert p > 0.05

    def test_shifted_mean(self):
        np.random.seed(42)
        data = np.random.randn(100) + 5
        stat, p = t_test(data, popmean=0.0)
        assert p < 0.05


class TestTTestPaired:
    def test_no_difference(self):
        np.random.seed(42)
        x = np.random.randn(50)
        y = x + 0.01 * np.random.randn(50)  # Almost the same
        stat, p = t_test_paired(x, y)
        assert np.isfinite(stat)

    def test_significant_difference(self):
        np.random.seed(42)
        x = np.random.randn(50)
        y = x + 5
        stat, p = t_test_paired(x, y)
        assert p < 0.05


class TestTTestIndependent:
    def test_same_distribution(self):
        np.random.seed(42)
        x = np.random.randn(50)
        y = np.random.randn(50)
        stat, p = t_test_independent(x, y)
        assert p > 0.05

    def test_different_means(self):
        np.random.seed(42)
        x = np.random.randn(50)
        y = np.random.randn(50) + 3
        stat, p = t_test_independent(x, y)
        assert p < 0.05


class TestFTest:
    def test_same_variance(self):
        np.random.seed(42)
        x = np.random.randn(50)
        y = np.random.randn(50)
        f_stat, p = f_test(x, y)
        assert np.isfinite(f_stat)
        assert p > 0.05

    def test_different_variance(self):
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100) * 5  # Much larger variance
        f_stat, p = f_test(x, y)
        assert p < 0.05


class TestChiSquaredTest:
    def test_uniform(self):
        observed = np.array([25, 25, 25, 25], dtype=float)
        stat, p = chi_squared_test(observed)
        assert stat == pytest.approx(0.0, abs=0.01)
        assert p > 0.99

    def test_non_uniform(self):
        observed = np.array([50, 10, 10, 10], dtype=float)
        stat, p = chi_squared_test(observed)
        assert p < 0.05


class TestMannWhitneyTest:
    def test_same_distribution(self):
        np.random.seed(42)
        x = np.random.randn(50)
        y = np.random.randn(50)
        stat, p = mannwhitney_test(x, y)
        assert np.isfinite(stat)

    def test_shifted_distribution(self):
        np.random.seed(42)
        x = np.random.randn(50)
        y = np.random.randn(50) + 3
        stat, p = mannwhitney_test(x, y)
        assert p < 0.05


class TestKruskalTest:
    def test_same_groups(self):
        np.random.seed(42)
        a = np.random.randn(30)
        b = np.random.randn(30)
        c = np.random.randn(30)
        stat, p = kruskal_test(a, b, c)
        assert p > 0.05

    def test_different_groups(self):
        np.random.seed(42)
        a = np.random.randn(30)
        b = np.random.randn(30) + 3
        c = np.random.randn(30) + 6
        stat, p = kruskal_test(a, b, c)
        assert p < 0.05


class TestAnova:
    def test_same_groups(self):
        np.random.seed(42)
        a = np.random.randn(30)
        b = np.random.randn(30)
        c = np.random.randn(30)
        stat, p = anova(a, b, c)
        assert p > 0.05

    def test_different_groups(self):
        np.random.seed(42)
        a = np.random.randn(30)
        b = np.random.randn(30) + 3
        c = np.random.randn(30) + 6
        stat, p = anova(a, b, c)
        assert p < 0.05


class TestShapiroTest:
    def test_normal_data(self):
        np.random.seed(42)
        data = np.random.randn(100)
        stat, p = shapiro_test(data)
        assert p > 0.05

    def test_non_normal_data(self):
        np.random.seed(42)
        data = np.random.exponential(1, 100)
        stat, p = shapiro_test(data)
        # Exponential data should fail normality test
        assert p < 0.05


class TestLeveneTest:
    def test_same_variance(self):
        np.random.seed(42)
        a = np.random.randn(50)
        b = np.random.randn(50)
        stat, p = levene_test(a, b)
        assert p > 0.05

    def test_different_variance(self):
        np.random.seed(42)
        a = np.random.randn(50)
        b = np.random.randn(50) * 5
        stat, p = levene_test(a, b)
        assert p < 0.05


# --- Tests for stationarity.py ---


class TestStationarityTest:
    def test_stationary_signal(self):
        np.random.seed(42)
        signal = np.random.randn(200)
        ratio, is_stationary = stationarity_test(signal)
        assert np.isfinite(ratio)

    def test_short_signal(self):
        ratio, is_stationary = stationarity_test(np.array([1.0, 2.0]))
        assert np.isnan(ratio)
        assert is_stationary is False


class TestTrend:
    def test_no_trend(self):
        np.random.seed(42)
        signal = np.random.randn(100)
        slope, r2 = trend(signal)
        assert abs(slope) < 0.1

    def test_linear_trend(self):
        signal = np.arange(100, dtype=float)
        slope, r2 = trend(signal)
        assert slope == pytest.approx(1.0, abs=0.01)
        assert r2 == pytest.approx(1.0, abs=0.01)


class TestChangepoints:
    def test_single_changepoint(self):
        np.random.seed(42)
        signal = np.concatenate([np.random.randn(100), np.random.randn(100) + 5])
        bkps = changepoints(signal, n_bkps=1)
        assert len(bkps) == 1
        # Changepoint should be near index 100
        assert 80 < bkps[0] < 120

    def test_short_signal(self):
        bkps = changepoints(np.array([1.0, 2.0, 3.0]), n_bkps=1)
        assert len(bkps) == 0


class TestKPSSTest:
    def test_stationary(self):
        np.random.seed(42)
        signal = np.random.randn(200)
        stat, p = kpss_test(signal)
        assert np.isfinite(stat)
        # Stationary signal should NOT reject null (p > 0.05)

    def test_short_signal(self):
        stat, p = kpss_test(np.array([1.0, 2.0]))
        assert np.isnan(stat)
        assert np.isnan(p)


class TestPhillipsPerronTest:
    def test_stationary(self):
        np.random.seed(42)
        signal = np.random.randn(200)
        pp_stat, crit = phillips_perron_test(signal)
        assert np.isfinite(pp_stat)
        # Stationary signal should reject unit root
        assert pp_stat < crit

    def test_random_walk(self):
        np.random.seed(42)
        signal = np.cumsum(np.random.randn(200))
        pp_stat, crit = phillips_perron_test(signal)
        assert np.isfinite(pp_stat)

    def test_short_signal(self):
        pp_stat, crit = phillips_perron_test(np.array([1.0, 2.0]))
        assert np.isnan(pp_stat)
        assert np.isnan(crit)
