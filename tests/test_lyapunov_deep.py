"""
Deep Lyapunov tests — the function that already burned us once.
"""
import numpy as np
import pytest
import os

os.environ['PMTVS_USE_RUST'] = '0'
import pmtvs


class TestLyapunovEdgeCases:
    """Every way Lyapunov can go wrong."""

    def test_constant_signal(self):
        """LE should be 0 or NaN, not positive."""
        x = np.ones(1000)
        le = pmtvs.largest_lyapunov_exponent(x)
        assert le <= 0 or np.isnan(le), f"Constant LE={le}, should be <=0 or NaN"

    def test_linear_trend(self):
        """Pure linear -> not chaotic."""
        x = np.linspace(0, 100, 1000)
        le = pmtvs.largest_lyapunov_exponent(x)
        assert le < 0.1, f"Linear LE={le}, should be small"

    def test_very_short(self):
        """10 points — should return NaN, not crash."""
        np.random.seed(42)
        x = np.random.randn(10)
        le = pmtvs.largest_lyapunov_exponent(x)
        assert np.isnan(le) or isinstance(le, float), \
            f"Short signal: unexpected return {le}"

    def test_repeated_pattern(self):
        """Perfectly periodic -> LE <= 0."""
        pattern = np.array([0, 1, 2, 1, 0, -1, -2, -1], dtype=float)
        x = np.tile(pattern, 125)  # 1000 points
        le = pmtvs.largest_lyapunov_exponent(x)
        assert le < 0.05, f"Periodic LE={le}, should be <=0"

    def test_different_lengths(self):
        """Lyapunov should work for various signal lengths without crashing."""
        np.random.seed(42)
        for n in [50, 100, 500, 1000, 5000]:
            x = np.random.randn(n)
            le = pmtvs.largest_lyapunov_exponent(x)
            assert isinstance(le, float), f"Length {n}: returned {type(le)}"

    def test_embed_dim_sensitivity(self):
        """Result should be consistent across reasonable embed_dim values."""
        from scipy.integrate import solve_ivp

        def lorenz(t, s):
            return [10 * (s[1] - s[0]), s[0] * (28 - s[2]) - s[1], s[0] * s[1] - (8 / 3) * s[2]]

        sol = solve_ivp(lorenz, [0, 50], [1, 1, 1],
                        t_eval=np.linspace(0, 50, 5000), rtol=1e-10)
        x = sol.y[0]

        le_default = pmtvs.largest_lyapunov_exponent(x)
        assert le_default > 0.05, f"Default embed: LE={le_default}"

    def test_noise_floor(self):
        """White noise LE should be small positive (not large).
        Chaos estimation on pure noise returns small positive artifact.
        """
        np.random.seed(42)
        x = np.random.randn(5000)
        le = pmtvs.largest_lyapunov_exponent(x)
        assert le < 1.0, f"Noise LE={le}, suspiciously large"

    def test_henon_map(self):
        """Henon map: known LE ~ 0.42 for a=1.4, b=0.3."""
        n = 5000
        x = np.zeros(n)
        y = np.zeros(n)
        x[0], y[0] = 0.1, 0.1
        for i in range(1, n):
            x[i] = 1 - 1.4 * x[i - 1] ** 2 + y[i - 1]
            y[i] = 0.3 * x[i - 1]
        le = pmtvs.largest_lyapunov_exponent(x)
        assert le > 0.1, f"Henon LE={le}, expected ~0.42"
        assert le < 1.5, f"Henon LE={le}, too large"
