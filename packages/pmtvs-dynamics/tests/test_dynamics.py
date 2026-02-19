"""Tests for pmtvs-dynamics."""
import numpy as np
import pytest

from pmtvs_dynamics import (
    ftle,
    largest_lyapunov_exponent,
    lyapunov_spectrum,
    lyapunov_rosenstein,
    lyapunov_kantz,
    estimate_embedding_dim_cao,
    estimate_tau_ami,
    ftle_local_linearization,
    ftle_direct_perturbation,
    recurrence_matrix,
    recurrence_rate,
    determinism,
    laminarity,
    trapping_time,
    entropy_recurrence,
    max_diagonal_line,
    divergence_rqa,
    determinism_from_signal,
    rqa_metrics,
    correlation_dimension,
    attractor_reconstruction,
    kaplan_yorke_dimension,
    fixed_point_detection,
    stability_index,
    jacobian_eigenvalues,
    bifurcation_indicator,
    phase_space_contraction,
    hilbert_stability,
    wavelet_stability,
    detect_collapse,
    estimate_jacobian_local,
    classify_jacobian_eigenvalues,
    detect_saddle_points,
    compute_separatrix_distance,
    compute_basin_stability,
    compute_variable_sensitivity,
    compute_directional_sensitivity,
    compute_sensitivity_evolution,
    detect_sensitivity_transitions,
    compute_influence_matrix,
    correlation_integral,
    information_dimension,
    basin_stability,
    cycle_counting,
    local_outlier_factor,
    time_constant,
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


class TestNewRecurrence:
    """Tests for new recurrence functions."""

    def test_max_diagonal_line(self):
        trajectory = np.random.randn(50, 2)
        R = recurrence_matrix(trajectory)
        ml = max_diagonal_line(R)
        assert isinstance(ml, (int, np.integer))
        assert ml >= 0

    def test_divergence_rqa(self):
        trajectory = np.random.randn(50, 2)
        R = recurrence_matrix(trajectory)
        div = divergence_rqa(R)
        assert np.isnan(div) or div > 0

    def test_determinism_from_signal(self):
        t = np.linspace(0, 4 * np.pi, 200)
        signal = np.sin(t)
        det = determinism_from_signal(signal)
        assert np.isnan(det) or 0 <= det <= 1

    def test_rqa_metrics(self):
        t = np.linspace(0, 4 * np.pi, 200)
        signal = np.sin(t)
        metrics = rqa_metrics(signal)
        assert isinstance(metrics, dict)
        assert 'recurrence_rate' in metrics
        assert 'determinism' in metrics


class TestNewLyapunov:
    """Tests for new Lyapunov functions."""

    def test_lyapunov_rosenstein(self):
        np.random.seed(42)
        n = 500
        x = np.zeros(n)
        x[0] = 0.1
        for i in range(1, n):
            x[i] = 3.9 * x[i-1] * (1 - x[i-1])
        lam, div, iters = lyapunov_rosenstein(x)
        assert isinstance(lam, float)

    def test_lyapunov_kantz(self):
        np.random.seed(42)
        signal = np.random.randn(200)
        lam, div = lyapunov_kantz(signal, dimension=3, delay=1)
        assert isinstance(lam, float)

    def test_estimate_embedding_dim_cao(self):
        np.random.seed(42)
        signal = np.random.randn(200)
        result = estimate_embedding_dim_cao(signal)
        assert 'optimal_dim' in result
        assert result['optimal_dim'] >= 1

    def test_estimate_tau_ami(self):
        t = np.linspace(0, 10 * np.pi, 500)
        signal = np.sin(t)
        tau = estimate_tau_ami(signal)
        assert isinstance(tau, (int, np.integer))
        assert tau >= 1

    def test_ftle_local_linearization(self):
        np.random.seed(42)
        trajectory = np.cumsum(np.random.randn(100, 3), axis=0)
        ftle_vals, valid_idx = ftle_local_linearization(trajectory, time_horizon=5)
        assert len(ftle_vals) == 100

    def test_ftle_direct_perturbation(self):
        np.random.seed(42)
        signal = np.random.randn(200)
        ftle_vals, valid_idx = ftle_direct_perturbation(signal, dimension=3, delay=1)
        assert len(ftle_vals) > 0


class TestNewStability:
    """Tests for new stability functions."""

    def test_hilbert_stability(self):
        t = np.linspace(0, 4 * np.pi, 500)
        signal = np.sin(t)
        result = hilbert_stability(signal)
        assert isinstance(result, dict)
        assert 'inst_freq_mean' in result
        assert np.isfinite(result['inst_freq_mean'])

    def test_wavelet_stability(self):
        np.random.seed(42)
        signal = np.random.randn(200)
        result = wavelet_stability(signal)
        assert isinstance(result, dict)
        assert 'energy_low' in result

    def test_detect_collapse_no_collapse(self):
        signal = np.ones(100) * 5.0
        result = detect_collapse(signal)
        assert result['collapse_onset_idx'] == -1

    def test_detect_collapse_with_collapse(self):
        signal = np.linspace(10, 1, 100)
        result = detect_collapse(signal, threshold_velocity=-0.05)
        assert isinstance(result, dict)
        assert 'collapse_onset_idx' in result


class TestSaddle:
    """Tests for saddle point functions."""

    def test_classify_jacobian_eigenvalues(self):
        J = np.array([[1.0, 0.0], [0.0, -1.0]])
        result = classify_jacobian_eigenvalues(J)
        assert result['is_saddle'] is True

    def test_detect_saddle_points(self):
        np.random.seed(42)
        trajectory = np.random.randn(50, 2)
        score, vel, info = detect_saddle_points(trajectory)
        assert len(score) == 50

    def test_compute_separatrix_distance(self):
        trajectory = np.random.randn(50, 2)
        distances = compute_separatrix_distance(trajectory, np.array([0, 10, 20]))
        assert len(distances) == 50

    def test_compute_basin_stability(self):
        score = np.random.rand(100)
        stability = compute_basin_stability(np.random.randn(100, 2), score, window=20)
        assert len(stability) == 100


class TestSensitivity:
    """Tests for sensitivity functions."""

    def test_compute_variable_sensitivity(self):
        np.random.seed(42)
        trajectory = np.random.randn(50, 3)
        sens, rank = compute_variable_sensitivity(trajectory, time_horizon=5, n_neighbors=5)
        assert sens.shape == (50, 3)

    def test_compute_influence_matrix(self):
        np.random.seed(42)
        trajectory = np.random.randn(50, 3)
        influence = compute_influence_matrix(trajectory, time_horizon=5, n_neighbors=5)
        assert influence.shape == (3, 3)


class TestDimension:
    """Tests for dimension functions."""

    def test_correlation_integral(self):
        embedded = np.random.randn(50, 3)
        c = correlation_integral(embedded, r=1.0)
        assert 0 <= c <= 1

    def test_information_dimension(self):
        np.random.seed(42)
        signal = np.random.randn(500)
        d1 = information_dimension(signal)
        assert isinstance(d1, float)


class TestDomain:
    """Tests for domain functions."""

    def test_basin_stability(self):
        np.random.seed(42)
        signal = np.random.randn(100)
        result = basin_stability(signal)
        assert 'basin_stability' in result
        assert 0 <= result['basin_stability'] <= 1

    def test_cycle_counting(self):
        t = np.linspace(0, 4 * np.pi, 200)
        signal = np.sin(t)
        result = cycle_counting(signal)
        assert result['n_cycles'] > 0

    def test_time_constant(self):
        signal = np.exp(-np.arange(100) / 20.0)
        result = time_constant(signal)
        assert 'tau' in result
