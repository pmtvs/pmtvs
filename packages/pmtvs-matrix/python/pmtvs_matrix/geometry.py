"""
Geometry and Eigenstructure Analysis Functions

Functions for effective dimension, participation ratio, alignment metrics,
and eigenvalue distribution analysis.
"""

import numpy as np
from scipy.linalg import eigvals
from typing import Optional


def effective_dimension(
    eigenvalues: np.ndarray,
    method: str = 'participation_ratio'
) -> float:
    """
    Compute effective dimension from eigenvalues.

    The effective dimension quantifies how many dimensions actively
    contribute to the system's behavior.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Array of eigenvalues (should be non-negative).
    method : str
        Method to compute effective dimension:
        - 'participation_ratio' : Standard participation ratio.
        - 'normalized_entropy' : Information-theoretic measure.
        - 'inverse_participation' : Physics-based measure.

    Returns
    -------
    float
        Effective dimension (0 to len(eigenvalues)).
    """
    eigenvals = np.abs(np.asarray(eigenvalues, dtype=np.float64))
    eigenvals = eigenvals[eigenvals > 1e-12]

    if len(eigenvals) == 0:
        return 0.0

    if method == 'participation_ratio':
        sum_evals = np.sum(eigenvals)
        sum_squared = np.sum(eigenvals ** 2)

        if sum_squared == 0:
            return 0.0

        return float(sum_evals ** 2 / sum_squared)

    elif method == 'normalized_entropy':
        probs = eigenvals / np.sum(eigenvals)
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        return float(2 ** entropy)

    elif method == 'inverse_participation':
        probs = eigenvals / np.sum(eigenvals)
        ipr = 1.0 / np.sum(probs ** 2)
        return float(ipr)

    else:
        raise ValueError(f"Unknown method: {method}")


def participation_ratio(eigenvalues: np.ndarray) -> float:
    """
    Compute participation ratio from eigenvalues.

    PR = (sum(lambda_i))^2 / sum(lambda_i^2)

    Parameters
    ----------
    eigenvalues : np.ndarray
        Array of eigenvalues.

    Returns
    -------
    float
        Participation ratio.
    """
    return effective_dimension(eigenvalues, method='participation_ratio')


def alignment_metric(
    eigenvalues: np.ndarray,
    method: str = 'cosine'
) -> float:
    """
    Compute alignment of eigenvalue distribution.

    Measures how aligned the eigenvalue distribution is with
    a uniform distribution (high alignment = more uniform).

    Parameters
    ----------
    eigenvalues : np.ndarray
        Array of eigenvalues.
    method : str
        'cosine' or 'kl_divergence'.

    Returns
    -------
    float
        Alignment measure.
    """
    eigenvals = np.abs(np.asarray(eigenvalues, dtype=np.float64))
    eigenvals = eigenvals[eigenvals > 1e-12]

    if len(eigenvals) == 0:
        return np.nan
    if len(eigenvals) == 1:
        return 1.0

    probs = eigenvals / np.sum(eigenvals)
    uniform = np.ones_like(probs) / len(probs)

    if method == 'cosine':
        dot_product = np.dot(probs, uniform)
        norm_probs = np.linalg.norm(probs)
        norm_uniform = np.linalg.norm(uniform)

        if norm_probs == 0 or norm_uniform == 0:
            return 0.0

        return float(dot_product / (norm_probs * norm_uniform))

    elif method == 'kl_divergence':
        kl_div = np.sum(probs * np.log(probs / (uniform + 1e-12) + 1e-12))
        max_kl = np.log(len(probs))
        alignment = 1.0 - (kl_div / max_kl)
        return float(max(0.0, alignment))

    else:
        raise ValueError(f"Unknown method: {method}")


def eigenvalue_spread(eigenvalues: np.ndarray) -> float:
    """
    Compute spread of eigenvalues (coefficient of variation).

    Parameters
    ----------
    eigenvalues : np.ndarray
        Array of eigenvalues.

    Returns
    -------
    float
        Eigenvalue spread (std / mean of absolute eigenvalues).
    """
    eigenvals = np.abs(np.asarray(eigenvalues, dtype=np.float64))
    eigenvals = eigenvals[eigenvals > 1e-12]

    if len(eigenvals) <= 1:
        return 0.0

    mean_eval = np.mean(eigenvals)
    std_eval = np.std(eigenvals)

    if mean_eval == 0:
        return 0.0

    return float(std_eval / mean_eval)


def matrix_entropy(
    matrix: np.ndarray,
    normalize: bool = True
) -> float:
    """
    Compute entropy of a matrix using its eigenvalues.

    Parameters
    ----------
    matrix : np.ndarray
        Input square matrix.
    normalize : bool
        If True, normalize entropy to [0, 1].

    Returns
    -------
    float
        Matrix entropy.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim < 2 or matrix.shape[0] != matrix.shape[1]:
        return np.nan
    eigenvals = np.abs(eigvals(matrix))
    eigenvals = eigenvals[eigenvals > 1e-12]

    if len(eigenvals) == 0:
        return 0.0

    probs = eigenvals / np.sum(eigenvals)
    entropy = -np.sum(probs * np.log2(probs + 1e-12))

    if normalize:
        max_entropy = np.log2(len(probs))
        if max_entropy > 0:
            entropy = entropy / max_entropy

    return float(entropy)


def geometric_mean_eigenvalue(eigenvalues: np.ndarray) -> float:
    """
    Compute geometric mean of eigenvalues.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Array of eigenvalues.

    Returns
    -------
    float
        Geometric mean of absolute eigenvalues.
    """
    eigenvals = np.abs(np.asarray(eigenvalues, dtype=np.float64))
    eigenvals = eigenvals[eigenvals > 1e-12]

    if len(eigenvals) == 0:
        return 0.0

    log_mean = np.mean(np.log(eigenvals + 1e-12))
    return float(np.exp(log_mean))


def explained_variance_ratio(eigenvalues: np.ndarray) -> np.ndarray:
    """
    Compute explained variance ratio for each eigenvalue.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Array of eigenvalues.

    Returns
    -------
    np.ndarray
        Array of variance ratios (sum to 1).
    """
    eigenvals = np.abs(np.asarray(eigenvalues, dtype=np.float64))
    total = np.sum(eigenvals)

    if total == 0:
        return np.zeros_like(eigenvals)

    return eigenvals / total


def cumulative_variance_ratio(eigenvalues: np.ndarray) -> np.ndarray:
    """
    Compute cumulative explained variance ratio.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Array of eigenvalues (should be sorted descending).

    Returns
    -------
    np.ndarray
        Cumulative variance ratios.
    """
    ratios = explained_variance_ratio(eigenvalues)
    return np.cumsum(ratios)
