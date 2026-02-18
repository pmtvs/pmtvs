"""
Recurrence Quantification Analysis (RQA)

Functions for computing recurrence plots and RQA measures
that characterize dynamical system structure.
"""

import numpy as np
from typing import Optional


def recurrence_matrix(
    trajectory: np.ndarray,
    threshold: Optional[float] = None,
    threshold_percentile: float = 10.0
) -> np.ndarray:
    """
    Compute recurrence matrix.

    R[i,j] = 1 if distance(x_i, x_j) < threshold, else 0

    Parameters
    ----------
    trajectory : np.ndarray
        State-space trajectory of shape (n_points, n_dims) or 1D signal
    threshold : float, optional
        Distance threshold (if None, uses percentile of distances)
    threshold_percentile : float
        Percentile of distances to use as threshold (default: 10%)

    Returns
    -------
    np.ndarray
        Binary recurrence matrix
    """
    trajectory = np.asarray(trajectory)

    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(-1, 1)

    n = len(trajectory)

    if n < 2:
        return np.array([[1]])

    # Compute distance matrix
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(trajectory[i] - trajectory[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    # Set threshold
    if threshold is None:
        # Use percentile of non-zero distances
        non_diag = dist_matrix[np.triu_indices(n, k=1)]
        threshold = np.percentile(non_diag, threshold_percentile)

    # Create recurrence matrix
    R = (dist_matrix <= threshold).astype(int)

    return R


def recurrence_rate(R: np.ndarray) -> float:
    """
    Compute recurrence rate (RR).

    RR = fraction of recurrence points in the recurrence matrix.

    Parameters
    ----------
    R : np.ndarray
        Recurrence matrix

    Returns
    -------
    float
        Recurrence rate in [0, 1]
    """
    R = np.asarray(R)
    n = len(R)

    if n < 2:
        return 1.0

    # Exclude main diagonal
    mask = ~np.eye(n, dtype=bool)
    return float(np.mean(R[mask]))


def determinism(R: np.ndarray, min_line_length: int = 2) -> float:
    """
    Compute determinism (DET).

    DET = fraction of recurrence points that form diagonal lines.
    High DET indicates deterministic dynamics.

    Parameters
    ----------
    R : np.ndarray
        Recurrence matrix
    min_line_length : int
        Minimum diagonal line length to count

    Returns
    -------
    float
        Determinism in [0, 1]
    """
    R = np.asarray(R)
    n = len(R)

    if n < min_line_length:
        return np.nan

    # Count diagonal lines
    total_points = 0
    diagonal_points = 0

    # Check all diagonals (excluding main diagonal)
    for k in range(1, n):
        # Upper diagonal
        diag = np.diag(R, k)
        total_points += np.sum(diag)
        diagonal_points += _count_line_points(diag, min_line_length)

        # Lower diagonal
        diag = np.diag(R, -k)
        total_points += np.sum(diag)
        diagonal_points += _count_line_points(diag, min_line_length)

    if total_points == 0:
        return 0.0

    return float(diagonal_points / total_points)


def laminarity(R: np.ndarray, min_line_length: int = 2) -> float:
    """
    Compute laminarity (LAM).

    LAM = fraction of recurrence points that form vertical lines.
    High LAM indicates intermittent or laminar states.

    Parameters
    ----------
    R : np.ndarray
        Recurrence matrix
    min_line_length : int
        Minimum vertical line length to count

    Returns
    -------
    float
        Laminarity in [0, 1]
    """
    R = np.asarray(R)
    n = len(R)

    if n < min_line_length:
        return np.nan

    total_points = 0
    vertical_points = 0

    # Check all columns
    for j in range(n):
        col = R[:, j]
        total_points += np.sum(col)
        vertical_points += _count_line_points(col, min_line_length)

    if total_points == 0:
        return 0.0

    return float(vertical_points / total_points)


def trapping_time(R: np.ndarray, min_line_length: int = 2) -> float:
    """
    Compute trapping time (TT).

    TT = average length of vertical lines.
    Measures average time the system stays in a state.

    Parameters
    ----------
    R : np.ndarray
        Recurrence matrix
    min_line_length : int
        Minimum vertical line length to count

    Returns
    -------
    float
        Average trapping time
    """
    R = np.asarray(R)
    n = len(R)

    if n < min_line_length:
        return np.nan

    line_lengths = []

    # Check all columns
    for j in range(n):
        col = R[:, j]
        lengths = _get_line_lengths(col, min_line_length)
        line_lengths.extend(lengths)

    if len(line_lengths) == 0:
        return 0.0

    return float(np.mean(line_lengths))


def entropy_recurrence(R: np.ndarray, min_line_length: int = 2) -> float:
    """
    Compute entropy of diagonal line length distribution (ENTR).

    Higher entropy indicates more complex dynamics.

    Parameters
    ----------
    R : np.ndarray
        Recurrence matrix
    min_line_length : int
        Minimum diagonal line length to count

    Returns
    -------
    float
        Shannon entropy of line length distribution
    """
    R = np.asarray(R)
    n = len(R)

    if n < min_line_length:
        return np.nan

    line_lengths = []

    # Check all diagonals (excluding main diagonal)
    for k in range(1, n):
        diag = np.diag(R, k)
        lengths = _get_line_lengths(diag, min_line_length)
        line_lengths.extend(lengths)

        diag = np.diag(R, -k)
        lengths = _get_line_lengths(diag, min_line_length)
        line_lengths.extend(lengths)

    if len(line_lengths) == 0:
        return 0.0

    # Compute histogram
    unique_lengths, counts = np.unique(line_lengths, return_counts=True)
    probs = counts / np.sum(counts)

    # Shannon entropy
    entropy = -np.sum(probs * np.log(probs + 1e-12))

    return float(entropy)


def _count_line_points(line: np.ndarray, min_length: int) -> int:
    """Count points in lines of at least min_length."""
    count = 0
    current_length = 0

    for point in line:
        if point == 1:
            current_length += 1
        else:
            if current_length >= min_length:
                count += current_length
            current_length = 0

    if current_length >= min_length:
        count += current_length

    return count


def _get_line_lengths(line: np.ndarray, min_length: int) -> list:
    """Get lengths of all lines of at least min_length."""
    lengths = []
    current_length = 0

    for point in line:
        if point == 1:
            current_length += 1
        else:
            if current_length >= min_length:
                lengths.append(current_length)
            current_length = 0

    if current_length >= min_length:
        lengths.append(current_length)

    return lengths
