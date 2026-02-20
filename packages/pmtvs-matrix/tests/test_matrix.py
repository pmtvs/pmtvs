"""Tests for pmtvs-matrix."""
import numpy as np
import pytest

from pmtvs_matrix import (
    covariance_matrix,
    correlation_matrix,
    eigendecomposition,
    svd_decomposition,
    matrix_rank,
    condition_number,
    effective_rank,
    graph_laplacian,
    # geometry
    effective_dimension,
    participation_ratio,
    alignment_metric,
    eigenvalue_spread,
    matrix_entropy,
    geometric_mean_eigenvalue,
    explained_variance_ratio,
    cumulative_variance_ratio,
    # dmd
    dynamic_mode_decomposition,
    dmd_frequencies,
    dmd_growth_rates,
    # information
    mutual_information_matrix,
    transfer_entropy_matrix,
    granger_matrix,
)


class TestCovarianceMatrix:
    def test_shape(self):
        data = np.random.randn(100, 5)
        cov = covariance_matrix(data)
        assert cov.shape == (5, 5)

    def test_symmetric(self):
        data = np.random.randn(100, 5)
        cov = covariance_matrix(data)
        assert np.allclose(cov, cov.T)


class TestCorrelationMatrix:
    def test_diagonal_ones(self):
        data = np.random.randn(100, 5)
        corr = correlation_matrix(data)
        assert np.allclose(np.diag(corr), 1.0)

    def test_bounded(self):
        data = np.random.randn(100, 5)
        corr = correlation_matrix(data)
        assert np.all(corr >= -1) and np.all(corr <= 1)


class TestEigendecomposition:
    def test_reconstruction(self):
        A = np.random.randn(5, 5)
        A = A @ A.T  # Make symmetric
        eigenvalues, eigenvectors = eigendecomposition(A)
        # Reconstruction: A = V @ diag(lambda) @ V^T
        reconstructed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        assert np.allclose(A, reconstructed)


class TestSVD:
    def test_shapes(self):
        A = np.random.randn(10, 5)
        U, s, Vh = svd_decomposition(A)
        assert len(s) == 5
        assert U.shape == (10, 5)
        assert Vh.shape == (5, 5)


class TestMatrixRank:
    def test_full_rank(self):
        A = np.eye(5)
        assert matrix_rank(A) == 5

    def test_rank_deficient(self):
        A = np.array([[1, 2], [2, 4]])  # Rank 1
        assert matrix_rank(A) == 1


class TestConditionNumber:
    def test_identity(self):
        A = np.eye(5)
        assert condition_number(A) == pytest.approx(1.0)

    def test_ill_conditioned(self):
        A = np.array([[1, 1], [1, 1.0001]])
        assert condition_number(A) > 1000

    def test_near_singular_clamped_not_inf(self):
        """Near-singular matrix: clamped finite, not inf. Must exceed true ratio."""
        np.random.seed(42)
        data = np.random.randn(100, 5)
        data[:, 1:] *= 1e-10  # ratio ~ 1e10
        cn = condition_number(data)
        assert np.isfinite(cn), "Condition number must be finite after clamping"
        assert cn > 1e8, f"Expected > 1e8 for 1e-10 scaling, got {cn:.2e}"

    def test_known_diagonal_condition_number(self):
        """Diagonal matrix: condition number = max(|diag|) / min(|diag|)."""
        A = np.diag([10.0, 5.0, 1.0])
        cn = condition_number(A)
        assert cn == pytest.approx(10.0, rel=0.01), f"diag([10,5,1]) should have CN=10, got {cn}"

    def test_condition_number_scales_with_perturbation(self):
        """Progressively singular matrices should have increasing condition numbers."""
        cn_values = []
        for scale in [1.0, 0.01, 1e-6]:
            A = np.diag([1.0, scale])
            cn_values.append(condition_number(A))
        # Each should be larger than the previous
        assert cn_values[1] > cn_values[0], "CN should grow as matrix becomes more singular"
        assert cn_values[2] > cn_values[1], "CN should grow as matrix becomes more singular"
        # Known values: 1/1=1, 1/0.01=100, 1/1e-6=1e6
        assert cn_values[0] == pytest.approx(1.0)
        assert cn_values[1] == pytest.approx(100.0, rel=0.01)
        assert cn_values[2] == pytest.approx(1e6, rel=0.01)


class TestEffectiveRank:
    def test_identity(self):
        A = np.eye(5)
        # Effective rank of identity should be close to actual rank
        assert effective_rank(A) == pytest.approx(5.0, rel=0.1)


class TestGraphLaplacian:
    def test_row_sum_zero(self):
        adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        L = graph_laplacian(adj)
        # Rows should sum to zero
        assert np.allclose(np.sum(L, axis=1), 0)


# ---------------------------------------------------------------------------
# geometry.py tests
# ---------------------------------------------------------------------------


class TestEffectiveDimension:
    def test_uniform_eigenvalues(self):
        """Uniform eigenvalues should give maximum effective dimension."""
        np.random.seed(42)
        eigenvals = np.ones(5)
        result = effective_dimension(eigenvals)
        assert result == pytest.approx(5.0)

    def test_single_dominant(self):
        """One dominant eigenvalue should give effective dimension near 1."""
        eigenvals = np.array([100.0, 0.01, 0.01, 0.01])
        result = effective_dimension(eigenvals)
        assert result == pytest.approx(1.0, abs=0.1)

    def test_empty(self):
        """Empty eigenvalues should return 0."""
        result = effective_dimension(np.array([]))
        assert result == 0.0

    def test_all_zero(self):
        """All-zero eigenvalues should return 0."""
        result = effective_dimension(np.zeros(5))
        assert result == 0.0

    def test_normalized_entropy_method(self):
        """Normalized entropy method should work."""
        eigenvals = np.ones(4)
        result = effective_dimension(eigenvals, method='normalized_entropy')
        assert result == pytest.approx(4.0, abs=0.1)

    def test_inverse_participation_method(self):
        """Inverse participation method should work."""
        eigenvals = np.ones(4)
        result = effective_dimension(eigenvals, method='inverse_participation')
        assert result == pytest.approx(4.0, abs=0.1)

    def test_unknown_method_raises(self):
        """Unknown method should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            effective_dimension(np.ones(3), method='bogus')


class TestParticipationRatio:
    def test_uniform(self):
        """Uniform eigenvalues: PR = N."""
        eigenvals = np.ones(10)
        assert participation_ratio(eigenvals) == pytest.approx(10.0)

    def test_single(self):
        """Single nonzero eigenvalue: PR = 1."""
        eigenvals = np.array([5.0, 0.0, 0.0])
        assert participation_ratio(eigenvals) == pytest.approx(1.0)

    def test_returns_float(self):
        result = participation_ratio(np.array([3.0, 2.0, 1.0]))
        assert isinstance(result, float)


class TestAlignmentMetric:
    def test_uniform_is_one(self):
        """Uniform eigenvalues should give alignment of 1.0."""
        eigenvals = np.ones(5)
        result = alignment_metric(eigenvals, method='cosine')
        assert result == pytest.approx(1.0)

    def test_single_eigenvalue(self):
        """Single eigenvalue should return 1.0."""
        result = alignment_metric(np.array([5.0]))
        assert result == 1.0

    def test_kl_divergence_uniform(self):
        """Uniform eigenvalues with KL divergence should be near 1.0."""
        eigenvals = np.ones(5)
        result = alignment_metric(eigenvals, method='kl_divergence')
        assert result == pytest.approx(1.0, abs=0.05)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            alignment_metric(np.ones(3), method='bogus')

    def test_returns_float(self):
        result = alignment_metric(np.array([3.0, 1.0]))
        assert isinstance(result, float)


class TestEigenvalueSpread:
    def test_uniform_is_zero(self):
        """Uniform eigenvalues should have zero spread."""
        eigenvals = np.ones(5)
        assert eigenvalue_spread(eigenvals) == pytest.approx(0.0)

    def test_single_is_zero(self):
        """Single eigenvalue should have zero spread."""
        assert eigenvalue_spread(np.array([5.0])) == 0.0

    def test_positive_for_varied(self):
        """Varied eigenvalues should have positive spread."""
        eigenvals = np.array([10.0, 1.0, 0.1])
        assert eigenvalue_spread(eigenvals) > 0.0

    def test_returns_float(self):
        result = eigenvalue_spread(np.array([3.0, 1.0]))
        assert isinstance(result, float)


class TestMatrixEntropy:
    def test_identity_normalized(self):
        """Identity matrix (uniform eigenvalues) should have entropy near 1."""
        A = np.eye(5)
        result = matrix_entropy(A, normalize=True)
        assert result == pytest.approx(1.0, abs=0.05)

    def test_rank_one_low_entropy(self):
        """Rank-1 matrix should have low normalized entropy."""
        v = np.array([1.0, 0, 0, 0, 0])
        A = np.outer(v, v)
        result = matrix_entropy(A, normalize=True)
        assert result < 0.1

    def test_unnormalized(self):
        """Unnormalized entropy should be non-negative."""
        np.random.seed(42)
        A = np.random.randn(4, 4)
        A = A @ A.T
        result = matrix_entropy(A, normalize=False)
        assert result >= 0.0

    def test_zero_matrix(self):
        """Zero matrix should have zero entropy."""
        result = matrix_entropy(np.zeros((3, 3)))
        assert result == 0.0

    def test_returns_float(self):
        A = np.eye(3)
        assert isinstance(matrix_entropy(A), float)


class TestGeometricMeanEigenvalue:
    def test_uniform(self):
        """Geometric mean of equal values is that value."""
        eigenvals = np.array([5.0, 5.0, 5.0])
        result = geometric_mean_eigenvalue(eigenvals)
        assert result == pytest.approx(5.0, rel=0.01)

    def test_empty(self):
        """Empty should return 0."""
        assert geometric_mean_eigenvalue(np.array([])) == 0.0

    def test_positive(self):
        """Result should be positive for positive eigenvalues."""
        np.random.seed(42)
        eigenvals = np.abs(np.random.randn(5)) + 0.1
        result = geometric_mean_eigenvalue(eigenvals)
        assert result > 0

    def test_returns_float(self):
        result = geometric_mean_eigenvalue(np.array([2.0, 8.0]))
        assert isinstance(result, float)


class TestExplainedVarianceRatio:
    def test_sums_to_one(self):
        """Ratios should sum to 1."""
        eigenvals = np.array([5.0, 3.0, 2.0])
        ratios = explained_variance_ratio(eigenvals)
        assert np.sum(ratios) == pytest.approx(1.0)

    def test_proportional(self):
        """Ratios should be proportional to eigenvalues."""
        eigenvals = np.array([6.0, 3.0, 1.0])
        ratios = explained_variance_ratio(eigenvals)
        assert ratios[0] == pytest.approx(0.6)
        assert ratios[1] == pytest.approx(0.3)
        assert ratios[2] == pytest.approx(0.1)

    def test_zero_eigenvalues(self):
        """Zero eigenvalues should give zero ratios."""
        ratios = explained_variance_ratio(np.zeros(3))
        assert np.allclose(ratios, 0.0)

    def test_returns_array(self):
        ratios = explained_variance_ratio(np.array([3.0, 1.0]))
        assert isinstance(ratios, np.ndarray)


class TestCumulativeVarianceRatio:
    def test_last_is_one(self):
        """Last cumulative ratio should be 1.0."""
        eigenvals = np.array([5.0, 3.0, 2.0])
        cum = cumulative_variance_ratio(eigenvals)
        assert cum[-1] == pytest.approx(1.0)

    def test_monotonically_increasing(self):
        """Cumulative ratios should be non-decreasing."""
        eigenvals = np.array([5.0, 3.0, 2.0, 1.0])
        cum = cumulative_variance_ratio(eigenvals)
        assert np.all(np.diff(cum) >= 0)

    def test_first_equals_explained(self):
        """First cumulative equals first explained ratio."""
        eigenvals = np.array([6.0, 3.0, 1.0])
        cum = cumulative_variance_ratio(eigenvals)
        ratios = explained_variance_ratio(eigenvals)
        assert cum[0] == pytest.approx(ratios[0])

    def test_returns_array(self):
        cum = cumulative_variance_ratio(np.array([3.0, 1.0]))
        assert isinstance(cum, np.ndarray)


# ---------------------------------------------------------------------------
# dmd.py tests
# ---------------------------------------------------------------------------


class TestDynamicModeDecomposition:
    def test_basic_shapes(self):
        """DMD output shapes should be consistent."""
        np.random.seed(42)
        n_samples, n_signals = 50, 3
        signals = np.random.randn(n_samples, n_signals)
        modes, eigenvalues, dynamics, amplitudes = dynamic_mode_decomposition(signals)
        r = len(eigenvalues)
        assert modes.shape[0] == n_signals
        assert modes.shape[1] == r
        assert dynamics.shape == (r, n_samples)
        assert len(amplitudes) == r

    def test_rank_truncation(self):
        """Rank parameter should control decomposition rank."""
        np.random.seed(42)
        signals = np.random.randn(50, 5)
        modes, eigenvalues, dynamics, amplitudes = dynamic_mode_decomposition(
            signals, rank=2
        )
        assert len(eigenvalues) == 2
        assert modes.shape[1] == 2

    def test_too_few_samples(self):
        """Fewer than 3 samples should return NaN."""
        signals = np.array([[1.0, 2.0], [3.0, 4.0]])
        modes, eigenvalues, dynamics, amplitudes = dynamic_mode_decomposition(signals)
        assert np.all(np.isnan(np.real(modes)))

    def test_1d_input(self):
        """1D input should be reshaped automatically."""
        np.random.seed(42)
        signal = np.random.randn(50)
        modes, eigenvalues, dynamics, amplitudes = dynamic_mode_decomposition(signal)
        assert modes.shape[0] == 1

    def test_projected_dmd(self):
        """Projected DMD (exact=False) should work."""
        np.random.seed(42)
        signals = np.random.randn(50, 3)
        modes, eigenvalues, dynamics, amplitudes = dynamic_mode_decomposition(
            signals, exact=False
        )
        assert modes.shape[0] == 3

    def test_oscillatory_signal(self):
        """DMD should detect oscillation in a sinusoidal signal."""
        np.random.seed(42)
        t = np.linspace(0, 4 * np.pi, 100)
        signals = np.column_stack([np.sin(t), np.cos(t)])
        modes, eigenvalues, dynamics, amplitudes = dynamic_mode_decomposition(
            signals, dt=t[1] - t[0]
        )
        # Eigenvalues should have magnitude near 1 (oscillatory, not growing)
        magnitudes = np.abs(eigenvalues)
        assert np.any(magnitudes > 0.9)


class TestDmdFrequencies:
    def test_unit_eigenvalues(self):
        """Eigenvalues on unit circle should give finite frequencies."""
        eigenvalues = np.array([np.exp(1j * 0.5), np.exp(1j * 1.0)])
        freqs = dmd_frequencies(eigenvalues, dt=1.0)
        assert freqs.shape == (2,)
        assert np.all(np.isfinite(freqs))

    def test_real_eigenvalues(self):
        """Real positive eigenvalues should have near-zero frequency."""
        eigenvalues = np.array([0.9, 1.1])
        freqs = dmd_frequencies(eigenvalues, dt=1.0)
        assert np.all(freqs < 0.01)

    def test_returns_array(self):
        freqs = dmd_frequencies(np.array([1.0 + 0.1j]))
        assert isinstance(freqs, np.ndarray)


class TestDmdGrowthRates:
    def test_growing_mode(self):
        """Eigenvalue > 1 should give positive growth rate."""
        eigenvalues = np.array([1.5])
        rates = dmd_growth_rates(eigenvalues, dt=1.0)
        assert rates[0] > 0

    def test_decaying_mode(self):
        """Eigenvalue < 1 should give negative growth rate."""
        eigenvalues = np.array([0.5])
        rates = dmd_growth_rates(eigenvalues, dt=1.0)
        assert rates[0] < 0

    def test_neutral_mode(self):
        """Eigenvalue ~ 1 should give growth rate near zero."""
        eigenvalues = np.array([1.0])
        rates = dmd_growth_rates(eigenvalues, dt=1.0)
        assert rates[0] == pytest.approx(0.0, abs=0.01)

    def test_returns_array(self):
        rates = dmd_growth_rates(np.array([0.8, 1.2]))
        assert isinstance(rates, np.ndarray)
        assert len(rates) == 2


# ---------------------------------------------------------------------------
# information.py tests
# ---------------------------------------------------------------------------


class TestMutualInformationMatrix:
    def test_shape(self):
        """MI matrix should be (n_signals x n_signals)."""
        np.random.seed(42)
        signals = np.random.randn(200, 4)
        mi = mutual_information_matrix(signals)
        assert mi.shape == (4, 4)

    def test_symmetric(self):
        """MI matrix should be symmetric."""
        np.random.seed(42)
        signals = np.random.randn(200, 3)
        mi = mutual_information_matrix(signals)
        assert np.allclose(mi, mi.T)

    def test_non_negative(self):
        """MI values should be non-negative."""
        np.random.seed(42)
        signals = np.random.randn(200, 3)
        mi = mutual_information_matrix(signals)
        assert np.all(mi >= 0)

    def test_diagonal_is_entropy(self):
        """Diagonal should be the entropy of each signal."""
        np.random.seed(42)
        signals = np.random.randn(200, 3)
        mi = mutual_information_matrix(signals)
        # Diagonal (entropy) should be >= off-diagonal (MI)
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert mi[i, i] >= mi[i, j] - 1e-10

    def test_correlated_signals(self):
        """Correlated signals should have higher MI than independent ones."""
        np.random.seed(42)
        x = np.random.randn(500)
        y = x + 0.1 * np.random.randn(500)  # highly correlated
        z = np.random.randn(500)             # independent
        signals = np.column_stack([x, y, z])
        mi = mutual_information_matrix(signals)
        assert mi[0, 1] > mi[0, 2]

    def test_1d_input(self):
        """1D input should work."""
        np.random.seed(42)
        signal = np.random.randn(100)
        mi = mutual_information_matrix(signal)
        assert mi.shape == (1, 1)


class TestTransferEntropyMatrix:
    def test_shape(self):
        """TE matrix should be (n_signals x n_signals)."""
        np.random.seed(42)
        signals = np.random.randn(200, 3)
        te = transfer_entropy_matrix(signals)
        assert te.shape == (3, 3)

    def test_diagonal_zero(self):
        """Diagonal should be zero (no self-transfer)."""
        np.random.seed(42)
        signals = np.random.randn(200, 3)
        te = transfer_entropy_matrix(signals)
        assert np.allclose(np.diag(te), 0.0)

    def test_non_negative(self):
        """TE values should be non-negative."""
        np.random.seed(42)
        signals = np.random.randn(200, 3)
        te = transfer_entropy_matrix(signals)
        assert np.all(te >= 0)

    def test_causal_signal(self):
        """Signal with causal relationship should show nonzero TE from driver."""
        np.random.seed(42)
        n = 2000
        x = np.random.randn(n)
        y = np.zeros(n)
        for i in range(1, n):
            y[i] = 0.95 * x[i - 1] + 0.01 * np.random.randn()
        signals = np.column_stack([x, y])
        te = transfer_entropy_matrix(signals, lag=1, n_bins=6)
        # TE from x -> y should be positive (x drives y)
        assert te[0, 1] > 0.0

    def test_short_signal(self):
        """Very short signal should return zeros."""
        signals = np.random.randn(5, 3)
        te = transfer_entropy_matrix(signals)
        assert np.allclose(te, 0.0)

    def test_1d_input(self):
        """1D input should return 1x1 zero matrix."""
        np.random.seed(42)
        signal = np.random.randn(100)
        te = transfer_entropy_matrix(signal)
        assert te.shape == (1, 1)
        assert te[0, 0] == 0.0


class TestGrangerMatrix:
    def test_shape(self):
        """Granger matrices should be (n_signals x n_signals)."""
        np.random.seed(42)
        signals = np.random.randn(200, 3)
        f_mat, p_mat = granger_matrix(signals)
        assert f_mat.shape == (3, 3)
        assert p_mat.shape == (3, 3)

    def test_diagonal_zero(self):
        """Diagonal F-statistics should be zero and p-values should be 1."""
        np.random.seed(42)
        signals = np.random.randn(200, 3)
        f_mat, p_mat = granger_matrix(signals)
        assert np.allclose(np.diag(f_mat), 0.0)
        assert np.allclose(np.diag(p_mat), 1.0)

    def test_f_non_negative(self):
        """F-statistics should be non-negative."""
        np.random.seed(42)
        signals = np.random.randn(200, 3)
        f_mat, _ = granger_matrix(signals)
        assert np.all(f_mat >= 0)

    def test_p_bounded(self):
        """P-values should be in [0, 1]."""
        np.random.seed(42)
        signals = np.random.randn(200, 3)
        _, p_mat = granger_matrix(signals)
        assert np.all(p_mat >= 0) and np.all(p_mat <= 1)

    def test_causal_signal(self):
        """Known causal relationship should yield low p-value."""
        np.random.seed(42)
        n = 500
        x = np.random.randn(n)
        y = np.zeros(n)
        for i in range(1, n):
            y[i] = 0.9 * x[i - 1] + 0.05 * np.random.randn()
        signals = np.column_stack([x, y])
        f_mat, p_mat = granger_matrix(signals, max_lag=3)
        # x -> y should be significant
        assert p_mat[0, 1] < 0.05

    def test_short_signal(self):
        """Very short signal should return default matrices."""
        signals = np.random.randn(5, 3)
        f_mat, p_mat = granger_matrix(signals)
        assert np.allclose(f_mat, 0.0)
        assert np.allclose(p_mat, 1.0)

    def test_1d_input(self):
        """1D input should return 1x1 matrices."""
        np.random.seed(42)
        signal = np.random.randn(100)
        f_mat, p_mat = granger_matrix(signal)
        assert f_mat.shape == (1, 1)
        assert p_mat.shape == (1, 1)
