"""Tests for entropy primitives"""

import numpy as np
import pytest
from pmtvs.individual import (
    sample_entropy,
    permutation_entropy,
    approximate_entropy,
)


class TestSampleEntropy:
    """Test sample entropy function"""

    def test_sample_entropy_constant(self):
        # Constant signal should have low/nan entropy
        data = np.ones(100)
        se = sample_entropy(data, m=2, r=0.2)
        assert np.isnan(se) or se < 0.1

    def test_sample_entropy_random(self):
        # Random signal should have higher entropy
        np.random.seed(42)
        data = np.random.randn(500)
        se = sample_entropy(data, m=2, r=0.2 * np.std(data))
        assert se > 0.5

    def test_sample_entropy_periodic(self):
        # Periodic signal should have low entropy
        t = np.linspace(0, 10 * np.pi, 500)
        data = np.sin(t)
        se = sample_entropy(data, m=2, r=0.2 * np.std(data))
        # Periodic signals typically have low sample entropy
        assert se < 1.5

    def test_sample_entropy_short_signal(self):
        # Too short signal returns NaN
        data = np.array([1, 2, 3])
        se = sample_entropy(data, m=2, r=0.2)
        assert np.isnan(se)


class TestPermutationEntropy:
    """Test permutation entropy function"""

    def test_permutation_entropy_constant(self):
        # Constant signal should have zero entropy (only one pattern)
        data = np.ones(100)
        pe = permutation_entropy(data, order=3, delay=1)
        assert pe == 0.0 or np.isnan(pe)

    def test_permutation_entropy_monotonic(self):
        # Monotonic signal should have low entropy
        data = np.arange(100).astype(float)
        pe = permutation_entropy(data, order=3, delay=1)
        assert pe < 0.2

    def test_permutation_entropy_random(self):
        # Random signal should have high entropy (near 1 when normalized)
        np.random.seed(42)
        data = np.random.randn(1000)
        pe = permutation_entropy(data, order=3, delay=1, normalize=True)
        assert pe > 0.9

    def test_permutation_entropy_normalized_range(self):
        # Normalized entropy should be in [0, 1]
        np.random.seed(42)
        data = np.random.randn(500)
        pe = permutation_entropy(data, order=4, delay=1, normalize=True)
        assert 0 <= pe <= 1

    def test_permutation_entropy_not_normalized(self):
        # Unnormalized entropy can be > 1
        np.random.seed(42)
        data = np.random.randn(500)
        pe_norm = permutation_entropy(data, order=4, delay=1, normalize=True)
        pe_unnorm = permutation_entropy(data, order=4, delay=1, normalize=False)
        assert pe_unnorm >= pe_norm


class TestApproximateEntropy:
    """Test approximate entropy function"""

    def test_approximate_entropy_constant(self):
        # Constant signal should have low entropy
        data = np.ones(100)
        ae = approximate_entropy(data, m=2, r=0.2)
        assert np.isnan(ae) or abs(ae) < 0.1

    def test_approximate_entropy_random(self):
        # Random signal should have higher entropy
        np.random.seed(42)
        data = np.random.randn(300)
        ae = approximate_entropy(data, m=2, r=0.2 * np.std(data))
        assert ae > 0.3

    def test_approximate_entropy_vs_sample_entropy(self):
        # ApEn and SampEn should be correlated but not identical
        np.random.seed(42)
        data = np.random.randn(300)
        r = 0.2 * np.std(data)
        ae = approximate_entropy(data, m=2, r=r)
        se = sample_entropy(data, m=2, r=r)
        # They measure similar things, so should both be positive for random data
        assert ae > 0
        if not np.isnan(se):
            assert se > 0

    def test_approximate_entropy_short_signal(self):
        # Too short signal returns NaN
        data = np.array([1, 2, 3])
        ae = approximate_entropy(data, m=2, r=0.2)
        assert np.isnan(ae)
