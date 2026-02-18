"""
Distance Metrics

Distance measures for signal comparison: Euclidean, cosine, Manhattan, DTW.
"""

import numpy as np
from typing import Optional

from pmtvs_distance._dispatch import use_rust


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Euclidean (L2) distance between two signals.

    Parameters
    ----------
    x : np.ndarray
        First signal
    y : np.ndarray
        Second signal

    Returns
    -------
    float
        Euclidean distance (L2 norm of difference)
    """
    if use_rust('euclidean_distance'):
        from pmtvs_distance import _get_rust
        rust_fn = _get_rust('euclidean_distance')
        if rust_fn is not None:
            return rust_fn(x, y)

    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    if len(x) != len(y):
        return np.nan

    # Remove NaN pairs
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        return np.nan

    return float(np.sqrt(np.sum((x - y) ** 2)))


def cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute cosine distance between two signals.

    Cosine distance = 1 - cosine similarity

    Parameters
    ----------
    x : np.ndarray
        First signal
    y : np.ndarray
        Second signal

    Returns
    -------
    float
        Cosine distance in [0, 2]
    """
    if use_rust('cosine_distance'):
        from pmtvs_distance import _get_rust
        rust_fn = _get_rust('cosine_distance')
        if rust_fn is not None:
            return rust_fn(x, y)

    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    if len(x) != len(y):
        return np.nan

    # Remove NaN pairs
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        return np.nan

    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)

    if norm_x == 0 or norm_y == 0:
        return np.nan

    cosine_similarity = np.dot(x, y) / (norm_x * norm_y)
    # Clip to handle floating point errors
    cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)

    return float(1.0 - cosine_similarity)


def manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Manhattan (L1) distance between two signals.

    Parameters
    ----------
    x : np.ndarray
        First signal
    y : np.ndarray
        Second signal

    Returns
    -------
    float
        Manhattan distance (L1 norm of difference)
    """
    if use_rust('manhattan_distance'):
        from pmtvs_distance import _get_rust
        rust_fn = _get_rust('manhattan_distance')
        if rust_fn is not None:
            return rust_fn(x, y)

    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    if len(x) != len(y):
        return np.nan

    # Remove NaN pairs
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        return np.nan

    return float(np.sum(np.abs(x - y)))


def dtw_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[int] = None
) -> float:
    """
    Compute Dynamic Time Warping distance between two signals.

    DTW finds the optimal alignment between two time series,
    allowing for stretching and compression.

    Parameters
    ----------
    x : np.ndarray
        First signal
    y : np.ndarray
        Second signal
    window : int, optional
        Sakoe-Chiba band width for constraint (default: no constraint)

    Returns
    -------
    float
        DTW distance
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    # Remove NaN values
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    n, m = len(x), len(y)

    if n == 0 or m == 0:
        return np.nan

    # Initialize cost matrix with infinity
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    # Fill the matrix
    for i in range(1, n + 1):
        j_start = 1
        j_end = m + 1

        if window is not None:
            j_start = max(1, i - window)
            j_end = min(m + 1, i + window + 1)

        for j in range(j_start, j_end):
            cost = abs(x[i - 1] - y[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],      # insertion
                dtw_matrix[i, j - 1],      # deletion
                dtw_matrix[i - 1, j - 1]   # match
            )

    return float(dtw_matrix[n, m])
