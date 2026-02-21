"""
Test Lyapunov exponent accuracy against known analytical values.

The logistic map x_{n+1} = r * x_n * (1 - x_n) has analytically known
Lyapunov exponents for specific r values. These serve as ground truth
for validating the Rosenstein implementation.

Reference:
    Strogatz, "Nonlinear Dynamics and Chaos", Chapter 10
    lambda = lim_{N->inf} (1/N) sum ln|f'(x_n)| = lim_{N->inf} (1/N) sum ln|r(1-2x_n)|
"""

import numpy as np
import pytest

from pmtvs_dynamics._lyapunov_utils import fit_linear_region


# -- Test data generators --

def logistic_map(r: float, n: int, x0: float = 0.4, transient: int = 500) -> np.ndarray:
    """Generate logistic map time series, discarding transient."""
    x = np.zeros(n + transient)
    x[0] = x0
    for i in range(1, n + transient):
        x[i] = r * x[i - 1] * (1 - x[i - 1])
    return x[transient:]


def analytical_lyapunov(r: float, n: int = 100000, transient: int = 1000) -> float:
    """Compute analytical Lyapunov exponent for logistic map."""
    x = logistic_map(r, n, transient=transient)
    return float(np.mean(np.log(np.abs(r * (1 - 2 * x)) + 1e-15)))


# -- fit_linear_region unit tests --

class TestFitLinearRegion:
    """Tests for the linear region detection algorithm."""

    def test_perfectly_linear(self):
        """Perfect line should return exact slope."""
        x = np.arange(50, dtype=float)
        y = 0.5 * x + 3.0
        slope, start, end, r2 = fit_linear_region(y, x)
        assert abs(slope - 0.5) < 1e-10
        assert r2 > 0.999

    def test_linear_then_flat(self):
        """Should detect the linear portion and ignore the plateau."""
        x = np.arange(100, dtype=float)
        y = np.minimum(0.7 * x, 10.0)  # linear to 10, then flat
        slope, start, end, r2 = fit_linear_region(y, x)
        assert abs(slope - 0.7) < 0.07
        assert end < 20  # should stop well before the plateau

    def test_short_signal(self):
        """Should handle signals shorter than min_pts."""
        y = np.array([1.0, 2.0, 3.0])
        x = np.arange(3, dtype=float)
        slope, start, end, r2 = fit_linear_region(y, x, min_pts=4)
        assert abs(slope - 1.0) < 0.1

    def test_all_nan(self):
        """Should handle all-NaN input gracefully."""
        y = np.array([np.nan, np.nan, np.nan])
        slope, start, end, r2 = fit_linear_region(y)
        assert np.isnan(slope)

    def test_constant_signal(self):
        """Constant divergence curve should give ~zero slope."""
        y = np.full(50, 5.0)
        slope, start, end, r2 = fit_linear_region(y)
        assert abs(slope) < 0.01

    def test_r2_threshold_sensitivity(self):
        """Higher R2 threshold should give steeper slope on curved data."""
        x = np.arange(50, dtype=float)
        y = np.log(x + 1)  # concave -- initial slope is steepest
        slope_strict, _, end_strict, _ = fit_linear_region(y, x, r2_threshold=0.99)
        slope_loose, _, end_loose, _ = fit_linear_region(y, x, r2_threshold=0.90)
        # Strict threshold should find shorter region with steeper slope
        assert end_strict <= end_loose
        assert slope_strict >= slope_loose


# -- Accuracy tests against known Lyapunov exponents --

class TestLyapunovAccuracy:
    """
    Validate Lyapunov estimates against analytically known values.

    These tests use the Rosenstein divergence curve from pmtvs_dynamics,
    then apply fit_linear_region to extract the exponent.
    """

    @pytest.fixture
    def rosenstein_fn(self):
        """Get the Rosenstein function."""
        from pmtvs_dynamics import lyapunov_rosenstein
        return lyapunov_rosenstein

    @pytest.mark.parametrize("r,expected,tolerance", [
        (3.57, 0.014, 0.02),    # Edge of chaos -- near-zero exponent
        (3.70, 0.352, 0.06),    # Mild chaos
        (3.80, 0.432, 0.06),    # Moderate chaos
        (3.90, 0.501, 0.08),    # Strong chaos
        (4.00, 0.693, 0.10),    # Full chaos (ln 2)
    ])
    def test_logistic_map(self, rosenstein_fn, r, expected, tolerance):
        """Lyapunov estimate should be within tolerance of analytical value."""
        x = logistic_map(r, 2000)
        _, div_curve, iters = rosenstein_fn(x)
        div_arr = np.asarray(div_curve)
        iter_arr = np.asarray(iters)

        slope, _, _, r2 = fit_linear_region(div_arr, iter_arr)
        assert abs(slope - expected) < tolerance, (
            f"r={r}: got {slope:.4f}, expected {expected:.3f} +/- {tolerance}"
        )

    def test_monotonic_ordering(self, rosenstein_fn):
        """Higher r should give higher Lyapunov exponent."""
        r_values = [3.60, 3.70, 3.80, 3.90, 4.00]
        exponents = []
        for r in r_values:
            x = logistic_map(r, 2000)
            _, div_curve, iters = rosenstein_fn(x)
            slope, _, _, _ = fit_linear_region(
                np.asarray(div_curve), np.asarray(iters)
            )
            exponents.append(slope)

        for i in range(len(exponents) - 1):
            assert exponents[i] < exponents[i + 1], (
                f"Non-monotonic: r={r_values[i]}->{exponents[i]:.4f}, "
                f"r={r_values[i+1]}->{exponents[i+1]:.4f}"
            )

    @pytest.mark.parametrize("n", [200, 500, 1000, 2000, 5000])
    def test_signal_length_robustness(self, rosenstein_fn, n):
        """Estimate should be stable across signal lengths (r=4.0, theory=0.693)."""
        x = logistic_map(4.0, n)
        _, div_curve, iters = rosenstein_fn(x)
        slope, _, _, _ = fit_linear_region(
            np.asarray(div_curve), np.asarray(iters)
        )
        # Should be in [0.45, 0.75] regardless of length
        assert 0.45 < slope < 0.75, (
            f"n={n}: got {slope:.4f}, expected ~0.6 (range 0.45-0.75)"
        )

    def test_periodic_signal(self, rosenstein_fn):
        """Periodic signal (r=3.2) should give near-zero or negative exponent."""
        x = logistic_map(3.2, 2000)
        _, div_curve, iters = rosenstein_fn(x)
        div_arr = np.asarray(div_curve)
        iter_arr = np.asarray(iters)

        if len(div_arr) >= 4:
            slope, _, _, _ = fit_linear_region(div_arr, iter_arr)
        else:
            slope = 0.0
        # Should be near zero or slightly negative
        assert slope < 0.05, f"Periodic signal gave lambda={slope:.4f}, expected <=0"


# -- Integration test: full wrapper --

class TestLyapunovWrapper:
    """Test the patched top-level lyapunov_rosenstein function."""

    def test_wrapper_returns_tuple(self):
        """Should still return (lambda, curve, iterations) tuple."""
        from pmtvs_dynamics import lyapunov_rosenstein
        x = logistic_map(4.0, 1000)
        result = lyapunov_rosenstein(x)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_wrapper_accuracy(self):
        """Wrapper should give improved accuracy on r=4.0."""
        from pmtvs_dynamics import lyapunov_rosenstein
        x = logistic_map(4.0, 2000)
        lam, _, _ = lyapunov_rosenstein(x)
        # With the fix, should be > 0.5 (was ~0.12 before)
        assert lam > 0.5, f"Wrapper returned {lam:.4f}, expected > 0.5"

    def test_wrapper_short_signal(self):
        """Should handle short signals gracefully."""
        from pmtvs_dynamics import lyapunov_rosenstein
        x = logistic_map(4.0, 100)
        lam, div_curve, iters = lyapunov_rosenstein(x)
        assert not np.isnan(lam)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
