"""
Matrix Analysis Functions
"""

import numpy as np
from typing import Tuple, Optional


def covariance_matrix(data: np.ndarray, rowvar: bool = False) -> np.ndarray:
    """
    Compute covariance matrix.

    Parameters
    ----------
    data : np.ndarray
        Data matrix (observations x variables if rowvar=False)
    rowvar : bool
        If True, rows are variables

    Returns
    -------
    np.ndarray
        Covariance matrix
    """
    data = np.asarray(data)
    return np.cov(data, rowvar=rowvar)


def correlation_matrix(data: np.ndarray, rowvar: bool = False) -> np.ndarray:
    """
    Compute correlation matrix.

    Parameters
    ----------
    data : np.ndarray
        Data matrix
    rowvar : bool
        If True, rows are variables

    Returns
    -------
    np.ndarray
        Correlation matrix
    """
    data = np.asarray(data)
    return np.corrcoef(data, rowvar=rowvar)


def eigendecomposition(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute eigendecomposition.

    Parameters
    ----------
    matrix : np.ndarray
        Square matrix

    Returns
    -------
    tuple
        (eigenvalues, eigenvectors) sorted by eigenvalue magnitude
    """
    matrix = np.asarray(matrix)
    if matrix.ndim < 2:
        return np.array([]), np.empty((0, 0))
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Sort by magnitude
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    return eigenvalues[idx], eigenvectors[:, idx]


def svd_decomposition(
    matrix: np.ndarray,
    full_matrices: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute SVD decomposition.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix
    full_matrices : bool
        Return full U and V matrices

    Returns
    -------
    tuple
        (U, singular_values, Vh)
    """
    matrix = np.asarray(matrix)
    if matrix.ndim < 2:
        return np.empty((0, 0)), np.array([]), np.empty((0, 0))
    return np.linalg.svd(matrix, full_matrices=full_matrices)


def matrix_rank(matrix: np.ndarray, tol: Optional[float] = None) -> int:
    """
    Compute matrix rank.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix
    tol : float, optional
        Tolerance for singular values

    Returns
    -------
    int
        Matrix rank
    """
    matrix = np.asarray(matrix)
    return int(np.linalg.matrix_rank(matrix, tol=tol))


def condition_number(matrix: np.ndarray) -> float:
    """
    Compute condition number.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix

    Returns
    -------
    float
        Condition number (ratio of largest to smallest singular value).
        Clamped to a finite maximum to prevent inf/NaN propagation
        when the matrix is numerically singular.
    """
    matrix = np.asarray(matrix)
    if matrix.ndim < 2:
        return np.nan
    _, s, _ = np.linalg.svd(matrix, full_matrices=False)
    if len(s) == 0 or s[0] == 0:
        return 1.0
    # Clamp smallest singular value to eps * largest to prevent inf
    eps = np.finfo(float).eps
    s_min = max(s[-1], eps * s[0])
    return float(s[0] / s_min)


def effective_rank(matrix: np.ndarray) -> float:
    """
    Compute effective rank using Shannon entropy of singular values.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix

    Returns
    -------
    float
        Effective rank
    """
    matrix = np.asarray(matrix)
    if matrix.ndim < 2:
        return 0.0
    _, s, _ = np.linalg.svd(matrix, full_matrices=False)

    s = s[s > 0]
    if len(s) == 0:
        return 0.0

    # Normalize to probabilities
    p = s / np.sum(s)

    # Shannon entropy
    entropy = -np.sum(p * np.log(p))

    # Effective rank = exp(entropy)
    return float(np.exp(entropy))


def graph_laplacian(
    adjacency: np.ndarray,
    normalized: bool = False
) -> np.ndarray:
    """
    Compute graph Laplacian matrix.

    Parameters
    ----------
    adjacency : np.ndarray
        Adjacency matrix
    normalized : bool
        Return normalized Laplacian

    Returns
    -------
    np.ndarray
        Laplacian matrix
    """
    adjacency = np.asarray(adjacency)
    if adjacency.ndim < 2:
        return np.array([])
    degree = np.sum(adjacency, axis=1)
    D = np.diag(degree)
    L = D - adjacency

    if normalized:
        # Normalized Laplacian: L_norm = D^(-1/2) L D^(-1/2)
        d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degree, 1e-10)))
        L = d_inv_sqrt @ L @ d_inv_sqrt

    return L
