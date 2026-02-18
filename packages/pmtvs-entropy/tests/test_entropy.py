"""Tests for pmtvs-entropy."""
import numpy as np
import pytest

from pmtvs_entropy import sample_entropy, permutation_entropy, approximate_entropy


class TestSampleEntropy:
    """Tests for sample entropy."""

    def test_random_signal(self):
        """Random signal should have higher entropy than periodic."""
        np.random.seed(42)
        random_sig = np.random.randn(500)
        periodic_sig = np.sin(np.linspace(0, 10 * np.pi, 500))

        se_random = sample_entropy(random_sig)
        se_periodic = sample_entropy(periodic_sig)

        assert se_random > se_periodic

    def test_constant_signal(self):
        """Constant signal should have low entropy (or NaN due to zero std)."""
        constant = np.ones(100)
        se = sample_entropy(constant)
        # With r=0 (from 0.2*std=0), should return NaN
        assert np.isnan(se)

    def test_short_signal(self):
        """Too short signal should return NaN."""
        short = np.array([1, 2, 3])
        se = sample_entropy(short)
        assert np.isnan(se)

    def test_embedding_dimension(self):
        """Different embedding dimensions should produce valid results."""
        np.random.seed(42)
        signal = np.random.randn(200)

        se_m2 = sample_entropy(signal, m=2)
        se_m3 = sample_entropy(signal, m=3)

        assert np.isfinite(se_m2)
        assert np.isfinite(se_m3)


class TestPermutationEntropy:
    """Tests for permutation entropy."""

    def test_normalized_range(self):
        """Normalized permutation entropy should be in [0, 1]."""
        np.random.seed(42)
        signal = np.random.randn(200)

        pe = permutation_entropy(signal, normalize=True)
        assert 0 <= pe <= 1

    def test_random_high_entropy(self):
        """Random signal should have high normalized entropy."""
        np.random.seed(42)
        random_sig = np.random.randn(1000)

        pe = permutation_entropy(random_sig, normalize=True)
        assert pe > 0.9  # Should be close to 1 for random

    def test_monotonic_low_entropy(self):
        """Monotonic signal should have zero entropy."""
        monotonic = np.arange(100).astype(float)
        pe = permutation_entropy(monotonic, normalize=True)
        assert pe < 0.1  # Should be close to 0

    def test_order_parameter(self):
        """Different order parameters should work."""
        np.random.seed(42)
        signal = np.random.randn(200)

        pe_3 = permutation_entropy(signal, order=3)
        pe_4 = permutation_entropy(signal, order=4)

        assert np.isfinite(pe_3)
        assert np.isfinite(pe_4)


class TestApproximateEntropy:
    """Tests for approximate entropy."""

    def test_random_vs_periodic(self):
        """Random signal should have higher ApEn than periodic."""
        np.random.seed(42)
        random_sig = np.random.randn(300)
        periodic_sig = np.sin(np.linspace(0, 10 * np.pi, 300))

        ae_random = approximate_entropy(random_sig)
        ae_periodic = approximate_entropy(periodic_sig)

        assert ae_random > ae_periodic

    def test_short_signal(self):
        """Too short signal should return NaN."""
        short = np.array([1, 2, 3])
        ae = approximate_entropy(short)
        assert np.isnan(ae)

    def test_positive_value(self):
        """Approximate entropy should be non-negative for typical signals."""
        np.random.seed(42)
        signal = np.random.randn(200)
        ae = approximate_entropy(signal)
        assert ae >= 0 or np.isnan(ae)
