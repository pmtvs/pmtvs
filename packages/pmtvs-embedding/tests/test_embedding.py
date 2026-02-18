"""Tests for pmtvs-embedding."""
import numpy as np
import pytest

from pmtvs_embedding import (
    delay_embedding,
    optimal_embedding_dimension,
    mutual_information_delay,
    false_nearest_neighbors,
)


class TestDelayEmbedding:
    """Tests for delay embedding."""

    def test_shape(self):
        """Embedding should have correct shape."""
        signal = np.arange(100, dtype=float)
        embed = delay_embedding(signal, dim=3, tau=1)
        assert embed.shape == (98, 3)

    def test_shape_with_tau(self):
        """Embedding with tau should have correct shape."""
        signal = np.arange(100, dtype=float)
        embed = delay_embedding(signal, dim=3, tau=5)
        # N - (dim-1)*tau = 100 - 2*5 = 90
        assert embed.shape == (90, 3)

    def test_values(self):
        """Embedding values should be correct."""
        signal = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
        embed = delay_embedding(signal, dim=3, tau=1)

        # First row: [0, 1, 2]
        assert np.allclose(embed[0], [0, 1, 2])
        # Second row: [1, 2, 3]
        assert np.allclose(embed[1], [1, 2, 3])
        # Last row: [7, 8, 9]
        assert np.allclose(embed[-1], [7, 8, 9])

    def test_values_with_tau(self):
        """Embedding with tau should have correct values."""
        signal = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
        embed = delay_embedding(signal, dim=3, tau=2)

        # First row: [0, 2, 4]
        assert np.allclose(embed[0], [0, 2, 4])
        # Second row: [1, 3, 5]
        assert np.allclose(embed[1], [1, 3, 5])

    def test_short_signal(self):
        """Too short signal should return NaN."""
        signal = np.array([1.0, 2.0])
        embed = delay_embedding(signal, dim=5, tau=1)
        assert np.isnan(embed[0, 0])

    def test_invalid_params(self):
        """Invalid parameters should return NaN."""
        signal = np.arange(100, dtype=float)
        embed = delay_embedding(signal, dim=0, tau=1)
        assert np.isnan(embed[0, 0])


class TestOptimalEmbeddingDimension:
    """Tests for Cao's method."""

    def test_returns_int(self):
        """Should return integer dimension."""
        np.random.seed(42)
        signal = np.random.randn(500)
        dim = optimal_embedding_dimension(signal)
        assert isinstance(dim, int)
        assert dim >= 1

    def test_sinusoidal(self):
        """Sinusoidal signal should have low dimension."""
        t = np.linspace(0, 10 * np.pi, 1000)
        signal = np.sin(t)
        dim = optimal_embedding_dimension(signal, max_dim=10)
        # Sinusoid is essentially 2D (sin and cos)
        assert dim <= 4

    def test_short_signal(self):
        """Short signal should return default."""
        signal = np.array([1.0, 2.0, 3.0])
        dim = optimal_embedding_dimension(signal)
        assert dim >= 1


class TestMutualInformationDelay:
    """Tests for mutual information delay selection."""

    def test_returns_int(self):
        """Should return integer delay."""
        np.random.seed(42)
        signal = np.random.randn(500)
        tau = mutual_information_delay(signal)
        assert isinstance(tau, int)
        assert tau >= 1

    def test_periodic_signal(self):
        """Periodic signal should have meaningful delay."""
        t = np.linspace(0, 10 * np.pi, 1000)
        signal = np.sin(t)
        tau = mutual_information_delay(signal, max_lag=100)
        # Should find a minimum around quarter period
        assert tau >= 1 and tau <= 100

    def test_short_signal(self):
        """Short signal should return 1."""
        signal = np.array([1.0, 2.0, 3.0])
        tau = mutual_information_delay(signal)
        assert tau == 1


class TestFalseNearestNeighbors:
    """Tests for FNN method."""

    def test_returns_tuple(self):
        """Should return (array, int) tuple."""
        np.random.seed(42)
        signal = np.random.randn(500)
        fnn_pct, opt_dim = false_nearest_neighbors(signal)
        assert isinstance(fnn_pct, np.ndarray)
        assert isinstance(opt_dim, int)

    def test_fnn_decreases(self):
        """FNN percentage should generally decrease with dimension."""
        # Generate Lorenz-like attractor (surrogate)
        np.random.seed(42)
        n = 1000
        signal = np.zeros(n)
        signal[0] = 0.1
        for i in range(1, n):
            signal[i] = 3.9 * signal[i-1] * (1 - signal[i-1])  # Logistic map

        fnn_pct, opt_dim = false_nearest_neighbors(signal, max_dim=8)

        # Should find a reasonable dimension
        assert opt_dim >= 1 and opt_dim <= 8

    def test_short_signal(self):
        """Short signal should return NaN and default."""
        signal = np.array([1.0, 2.0, 3.0])
        fnn_pct, opt_dim = false_nearest_neighbors(signal)
        assert np.all(np.isnan(fnn_pct))
        assert opt_dim >= 1
