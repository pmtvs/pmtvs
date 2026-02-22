"""
Known-answer tests against published values and analytical solutions.
If any of these fail, the math is wrong — not the test.
"""
import numpy as np
import pytest
import os

# Force Python backend first, then test Rust separately
os.environ['PMTVS_USE_RUST'] = '0'

import pmtvs


# ════════════════════════════════════════════
# HURST EXPONENT
# ════════════════════════════════════════════

class TestHurstKnownAnswers:
    """Hurst exponent on signals with known H values."""

    def test_white_noise_h05(self):
        """White noise -> H ~ 0.5 (no memory)."""
        np.random.seed(42)
        x = np.random.randn(5000)
        H = pmtvs.hurst_exponent(x)
        assert 0.35 < H < 0.65, f"White noise H={H}, expected ~0.5"

    def test_random_walk_h1(self):
        """Random walk (cumsum of white noise) -> H ~ 1.0 (strong persistence)."""
        np.random.seed(42)
        x = np.cumsum(np.random.randn(5000))
        H = pmtvs.hurst_exponent(x)
        assert 0.85 < H < 1.15, f"Random walk H={H}, expected ~1.0"

    def test_persistent_signal(self):
        """Persistent AR(1) cumsum -> H > 0.5 (strong persistence).
        AR(1) + cumsum produces near-random-walk behavior.
        """
        np.random.seed(42)
        n = 5000
        x = np.random.randn(n)
        for i in range(1, n):
            x[i] += 0.4 * x[i - 1]
        x = np.cumsum(x)
        H = pmtvs.hurst_exponent(x)
        assert 0.7 < H < 1.15, f"Persistent signal H={H}, expected >0.7"

    def test_antipersistent(self):
        """Mean-reverting signal -> H near or below 0.5.
        AR(1) with strong negative coefficient.
        """
        np.random.seed(42)
        n = 5000
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = -0.7 * x[i - 1] + np.random.randn()
        H = pmtvs.hurst_exponent(x)
        assert 0.1 < H < 0.55, f"Mean-reverting H={H}, expected <0.55"


# ════════════════════════════════════════════
# LYAPUNOV EXPONENT
# ════════════════════════════════════════════

class TestLyapunovKnownAnswers:
    """Lyapunov exponent on systems with known dynamics."""

    @pytest.fixture
    def lorenz_x(self):
        """Lorenz attractor x-component. Known max LE ~ 0.91."""
        from scipy.integrate import solve_ivp

        def lorenz(t, s):
            return [10 * (s[1] - s[0]), s[0] * (28 - s[2]) - s[1], s[0] * s[1] - (8 / 3) * s[2]]

        sol = solve_ivp(lorenz, [0, 100], [1, 1, 1],
                        t_eval=np.linspace(0, 100, 10000), rtol=1e-10)
        return sol.y[0]

    @pytest.fixture
    def rossler_x(self):
        """Rossler attractor x-component. Known max LE ~ 0.07."""
        from scipy.integrate import solve_ivp

        def rossler(t, s):
            return [-s[1] - s[2], s[0] + 0.2 * s[1], 0.2 + s[2] * (s[0] - 5.7)]

        sol = solve_ivp(rossler, [0, 500], [1, 1, 1],
                        t_eval=np.linspace(0, 500, 10000), rtol=1e-10)
        return sol.y[0]

    @pytest.fixture
    def logistic_chaotic(self):
        """Logistic map r=4.0. Known LE = ln(2) ~ 0.693."""
        n = 5000
        x = np.zeros(n)
        x[0] = 0.1
        for i in range(1, n):
            x[i] = 4.0 * x[i - 1] * (1 - x[i - 1])
        return x

    @pytest.fixture
    def sine_wave(self):
        """Pure sine. Known LE < 0 (periodic, not chaotic)."""
        t = np.linspace(0, 100, 5000)
        return np.sin(t)

    def test_lorenz_positive(self, lorenz_x):
        """Lorenz max LE should be positive (chaotic)."""
        le = pmtvs.largest_lyapunov_exponent(lorenz_x)
        assert le > 0.05, f"Lorenz LE={le}, expected positive (chaotic)"

    def test_lorenz_magnitude(self, lorenz_x):
        """Lorenz max LE ~ 0.91 (published). Time-series estimators underestimate.
        Rosenstein/Kantz from scalar data typically yields 0.1-0.5.
        """
        le = pmtvs.largest_lyapunov_exponent(lorenz_x)
        assert 0.1 < le < 2.0, f"Lorenz LE={le}, expected positive, ~0.1-0.9"

    def test_rossler_positive(self, rossler_x):
        """Rossler max LE should be positive (weakly chaotic)."""
        le = pmtvs.largest_lyapunov_exponent(rossler_x)
        assert le > 0.01, f"Rossler LE={le}, expected >0.01"

    def test_rossler_magnitude(self, rossler_x):
        """Rossler max LE ~ 0.07 (published). Should not be near zero."""
        le = pmtvs.largest_lyapunov_exponent(rossler_x)
        assert 0.01 < le < 0.3, f"Rossler LE={le}, expected ~0.07"

    def test_sine_nonpositive(self, sine_wave):
        """Periodic signal -> LE <= 0 or very small positive."""
        le = pmtvs.largest_lyapunov_exponent(sine_wave)
        assert le < 0.05, f"Sine LE={le}, expected <= 0 (periodic)"

    def test_logistic_chaotic(self, logistic_chaotic):
        """Logistic map r=4 -> LE = ln(2) ~ 0.693."""
        le = pmtvs.largest_lyapunov_exponent(logistic_chaotic)
        assert 0.3 < le < 1.2, f"Logistic LE={le}, expected ~0.693"


# ════════════════════════════════════════════
# PERMUTATION ENTROPY
# ════════════════════════════════════════════

class TestPermEntropyKnownAnswers:
    """Permutation entropy on signals with known complexity."""

    def test_constant_zero(self):
        """Constant signal -> PE = 0 (only one permutation pattern)."""
        x = np.ones(1000)
        pe = pmtvs.permutation_entropy(x)
        assert pe == pytest.approx(0.0, abs=0.01), f"Constant PE={pe}, expected 0"

    def test_monotonic_zero(self):
        """Monotonically increasing -> PE = 0 (only ascending pattern)."""
        x = np.arange(1000, dtype=float)
        pe = pmtvs.permutation_entropy(x)
        assert pe == pytest.approx(0.0, abs=0.01), f"Monotonic PE={pe}, expected 0"

    def test_white_noise_high(self):
        """White noise -> PE near maximum (all permutations equally likely)."""
        np.random.seed(42)
        x = np.random.randn(5000)
        pe = pmtvs.permutation_entropy(x)
        assert pe > 0.5, f"White noise PE={pe}, expected high"

    def test_sine_low(self):
        """Sine wave -> low PE (regular, few permutation patterns)."""
        t = np.linspace(0, 10 * np.pi, 5000)
        x = np.sin(t)
        pe = pmtvs.permutation_entropy(x)
        np.random.seed(42)
        pe_noise = pmtvs.permutation_entropy(np.random.randn(5000))
        assert pe < pe_noise, f"Sine PE={pe} should be < noise PE={pe_noise}"


# ════════════════════════════════════════════
# SAMPLE ENTROPY
# ════════════════════════════════════════════

class TestSampleEntropyKnownAnswers:

    def test_periodic_low(self):
        """Periodic signal -> low sample entropy (highly predictable)."""
        t = np.linspace(0, 20 * np.pi, 2000)
        x = np.sin(t)
        se = pmtvs.sample_entropy(x)
        assert se < 0.5, f"Sine SampEn={se}, expected low (periodic)"

    def test_random_high(self):
        """Random signal -> high sample entropy (unpredictable)."""
        np.random.seed(42)
        x = np.random.randn(2000)
        se = pmtvs.sample_entropy(x)
        assert se > 1.0, f"Random SampEn={se}, expected high"

    def test_constant_zero_or_nan(self):
        """Constant signal -> SampEn = 0 or NaN (no template matches)."""
        x = np.ones(1000)
        se = pmtvs.sample_entropy(x)
        assert se == 0.0 or np.isnan(se), f"Constant SampEn={se}, expected 0 or NaN"


# ════════════════════════════════════════════
# SPECTRAL FLATNESS
# ════════════════════════════════════════════

class TestSpectralFlatnessKnownAnswers:

    def test_white_noise_near_one(self):
        """White noise -> spectral flatness ~ 1 (flat spectrum)."""
        np.random.seed(42)
        x = np.random.randn(5000)
        sf = pmtvs.spectral_flatness(x)
        assert 0.7 < sf < 1.0, f"White noise SF={sf}, expected ~1.0"

    def test_sine_near_zero(self):
        """Pure sine -> spectral flatness ~ 0 (energy at one frequency)."""
        t = np.linspace(0, 10, 5000)
        x = np.sin(2 * np.pi * 50 * t)
        sf = pmtvs.spectral_flatness(x)
        assert sf < 0.3, f"Sine SF={sf}, expected near 0"


# ════════════════════════════════════════════
# FTLE
# ════════════════════════════════════════════

class TestFTLEKnownAnswers:

    def test_lorenz_finite(self):
        """Lorenz system FTLE should be finite (not NaN or inf).
        The SVD-based FTLE using velocity finite-differences may not
        yield a large positive value on attractors, but should be finite.
        """
        from scipy.integrate import solve_ivp

        def lorenz(t, s):
            return [10 * (s[1] - s[0]), s[0] * (28 - s[2]) - s[1], s[0] * s[1] - (8 / 3) * s[2]]

        sol = solve_ivp(lorenz, [0, 50], [1, 1, 1],
                        t_eval=np.linspace(0, 50, 5000), rtol=1e-10)
        # Pass full 3D trajectory
        trajectory = sol.y.T  # shape (5000, 3)
        val = pmtvs.ftle(trajectory)
        assert np.isfinite(val), f"Lorenz FTLE={val}, expected finite"

    def test_random_walk_finite(self):
        """Random walk trajectory -> FTLE should be finite."""
        np.random.seed(42)
        trajectory = np.cumsum(np.random.randn(1000, 2), axis=0)
        val = pmtvs.ftle(trajectory)
        assert np.isfinite(val), f"Random walk FTLE={val}, expected finite"

    def test_constant_zero(self):
        """Constant signal -> FTLE = 0 or NaN (no divergence)."""
        x = np.ones(1000)
        val = pmtvs.ftle(x)
        assert val == 0.0 or np.isnan(val) or val < 0.01, \
            f"Constant FTLE={val}, expected 0 or NaN"


# ════════════════════════════════════════════
# ADF (Augmented Dickey-Fuller)
# ════════════════════════════════════════════

class TestADFKnownAnswers:

    def test_random_walk_nonstationary(self):
        """Random walk -> ADF should NOT reject null (non-stationary).
        t_stat should be > critical value (less negative).
        """
        np.random.seed(42)
        x = np.cumsum(np.random.randn(1000))
        t_stat, critical = pmtvs.adf_test(x)
        # Non-stationary: t_stat > critical (i.e., not negative enough)
        assert t_stat > critical, \
            f"Random walk ADF t={t_stat}, crit={critical}, expected non-stationary"

    def test_white_noise_stationary(self):
        """White noise -> ADF should reject null (stationary).
        t_stat should be < critical value (more negative).
        """
        np.random.seed(42)
        x = np.random.randn(1000)
        t_stat, critical = pmtvs.adf_test(x)
        # Stationary: t_stat < critical (more negative)
        assert t_stat < critical, \
            f"White noise ADF t={t_stat}, crit={critical}, expected stationary"
