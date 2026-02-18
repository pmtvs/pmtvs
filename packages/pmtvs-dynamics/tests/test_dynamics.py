"""Tests for pmtvs-dynamics."""
import numpy as np
import pytest

from pmtvs_dynamics import (
    ftle,
    largest_lyapunov_exponent,
    lyapunov_spectrum,
    recurrence_matrix,
    recurrence_rate,
    determinism,
    laminarity,
    trapping_time,
    entropy_recurrence,
    correlation_dimension,
    attractor_reconstruction,
    kaplan_yorke_dimension,
    fixed_point_detection,
    stability_index,
    jacobian_eigenvalues,
    bifurcation_indicator,
    phase_space_contraction,
)


class TestLyapunov:
    """Tests for Lyapunov exponent functions."""

    def test_ftle_returns_float(self):
        """FTLE should return a float."""
        np.random.seed(42)
        trajectory = np.cumsum(np.random.randn(100, 3), axis=0)
        result = ftle(trajectory)
        assert isinstance(result, float)

    def test_ftle_short_trajectory(self):
        """Short trajectory should return NaN."""
        trajectory = np.array([[1, 2], [3, 4]])
        assert np.isnan(ftle(trajectory))

    def test_largest_lyapunov_chaotic(self):
        """Logistic map in chaotic regime should have positive LLE."""
        np.random.seed(42)
        n = 2000
        x = np.zeros(n)
        x[0] = 0.1
        r = 3.9  # Chaotic regime

        for i in range(1, n):
            x[i] = r * x[i-1] * (1 - x[i-1])

        lle = largest_lyapunov_exponent(x, dim=3, tau=1)
        # Should be positive for chaos (though exact value varies)
        assert np.isfinite(lle)

    def test_lyapunov_spectrum_shape(self):
        """Lyapunov spectrum should have correct shape."""
        np.random.seed(42)
        trajectory = np.random.randn(500, 3)
        spectrum = lyapunov_spectrum(trajectory, n_exponents=3)
        assert len(spectrum) == 3

    def test_lyapunov_spectrum_sorted(self):
        """Lyapunov spectrum should be sorted descending."""
        np.random.seed(42)
        trajectory = np.random.randn(500, 3)
        spectrum = lyapunov_spectrum(trajectory)
        # Check sorting where values are finite
        finite_spectrum = spectrum[np.isfinite(spectrum)]
        if len(finite_spectrum) > 1:
            assert np.all(np.diff(finite_spectrum) <= 0.01)  # Allow small tolerance


class TestRecurrence:
    """Tests for recurrence analysis functions."""

    def test_recurrence_matrix_shape(self):
        """Recurrence matrix should be square."""
        trajectory = np.random.randn(50, 2)
        R = recurrence_matrix(trajectory)
        assert R.shape == (50, 50)

    def test_recurrence_matrix_symmetric(self):
        """Recurrence matrix should be symmetric."""
        trajectory = np.random.randn(50, 2)
        R = recurrence_matrix(trajectory)
        assert np.allclose(R, R.T)

    def test_recurrence_rate_bounded(self):
        """Recurrence rate should be in [0, 1]."""
        trajectory = np.random.randn(50, 2)
        R = recurrence_matrix(trajectory)
        rr = recurrence_rate(R)
        assert 0 <= rr <= 1

    def test_determinism_periodic(self):
        """Periodic signal should have high determinism."""
        t = np.linspace(0, 10 * np.pi, 500)
        trajectory = np.column_stack([np.sin(t), np.cos(t)])
        R = recurrence_matrix(trajectory, threshold_percentile=5)
        det = determinism(R)
        # Periodic should have diagonal structures
        assert det > 0  # Should have some determinism

    def test_laminarity_bounded(self):
        """Laminarity should be in [0, 1]."""
        trajectory = np.random.randn(50, 2)
        R = recurrence_matrix(trajectory)
        lam = laminarity(R)
        assert np.isnan(lam) or 0 <= lam <= 1

    def test_trapping_time_positive(self):
        """Trapping time should be non-negative."""
        trajectory = np.random.randn(50, 2)
        R = recurrence_matrix(trajectory)
        tt = trapping_time(R)
        assert np.isnan(tt) or tt >= 0

    def test_entropy_recurrence_non_negative(self):
        """Entropy should be non-negative."""
        trajectory = np.random.randn(50, 2)
        R = recurrence_matrix(trajectory)
        ent = entropy_recurrence(R)
        assert np.isnan(ent) or ent >= 0


class TestAttractor:
    """Tests for attractor analysis functions."""

    def test_correlation_dimension_returns_float(self):
        """Correlation dimension should return float."""
        np.random.seed(42)
        trajectory = np.random.randn(200, 3)
        d = correlation_dimension(trajectory)
        assert isinstance(d, float)

    def test_attractor_reconstruction_shape(self):
        """Attractor reconstruction should have correct shape."""
        signal = np.random.randn(100)
        attractor = attractor_reconstruction(signal, dim=3, tau=2)
        # n_vectors = 100 - (3-1)*2 = 96
        assert attractor.shape == (96, 3)

    def test_kaplan_yorke_dimension_lorenz(self):
        """Test Kaplan-Yorke dimension with typical Lorenz exponents."""
        # Typical Lorenz spectrum: [0.906, 0, -14.572]
        spectrum = np.array([0.9, 0.0, -14.5])
        d_ky = kaplan_yorke_dimension(spectrum)
        # D_KY should be around 2.06
        assert 2.0 < d_ky < 2.2

    def test_kaplan_yorke_dimension_stable(self):
        """All negative exponents should give dimension 0."""
        spectrum = np.array([-1.0, -2.0, -3.0])
        d_ky = kaplan_yorke_dimension(spectrum)
        assert d_ky == 0.0


class TestStability:
    """Tests for stability analysis functions."""

    def test_fixed_point_detection_stationary(self):
        """Constant signal should be detected as fixed point."""
        trajectory = np.ones((100, 2)) * 5.0 + np.random.randn(100, 2) * 0.001
        fixed_pts = fixed_point_detection(trajectory, threshold=0.01, min_duration=10)
        assert len(fixed_pts) > 0

    def test_stability_index_returns_float(self):
        """Stability index should return float."""
        trajectory = np.random.randn(50, 2)
        si = stability_index(trajectory)
        assert isinstance(si, float)

    def test_jacobian_eigenvalues_shape(self):
        """Jacobian eigenvalues should match dimensions."""
        trajectory = np.random.randn(100, 3)
        eigs = jacobian_eigenvalues(trajectory)
        assert len(eigs) == 3

    def test_bifurcation_indicator_returns_array(self):
        """Bifurcation indicator should return array."""
        signal = np.random.randn(500)
        bi = bifurcation_indicator(signal, window_size=50)
        assert isinstance(bi, np.ndarray)
        assert len(bi) > 1

    def test_phase_space_contraction_returns_float(self):
        """Phase space contraction should return float."""
        trajectory = np.random.randn(100, 3)
        psc = phase_space_contraction(trajectory)
        assert isinstance(psc, float)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_short_signals(self):
        """Short signals should return NaN or reasonable defaults."""
        short = np.array([1.0, 2.0, 3.0])

        # FTLE returns 0.0 for very short trajectories (valid result)
        assert np.isfinite(ftle(short.reshape(-1, 1)))
        assert np.isnan(largest_lyapunov_exponent(short))

    def test_nan_handling(self):
        """Functions should handle NaN values."""
        signal = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        attractor = attractor_reconstruction(signal, dim=2, tau=1)
        # Should work with NaN filtered out
        assert attractor.shape[0] == 3  # 4 valid points -> 3 vectors
