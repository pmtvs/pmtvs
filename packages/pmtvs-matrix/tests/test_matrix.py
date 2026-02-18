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
