"""
Attractor Analysis

Functions for characterizing strange attractors and
computing attractor dimensions.
"""

import numpy as np
from typing import Optional, Tuple


def correlation_dimension(
    trajectory: np.ndarray,
    r_values: Optional[np.ndarray] = None,
    n_reference: int = 500
) -> float:
    """
    Compute correlation dimension using Grassberger-Procaccia algorithm.

    The correlation dimension measures the fractal dimension of an attractor
    by analyzing how the correlation sum scales with distance.

    Parameters
    ----------
    trajectory : np.ndarray
        State-space trajectory of shape (n_points, n_dims)
    r_values : np.ndarray, optional
        Radius values to use (default: log-spaced)
    n_reference : int
        Number of reference points to use

    Returns
    -------
    float
        Correlation dimension estimate
    """
    trajectory = np.asarray(trajectory)

    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(-1, 1)

    n_points, n_dims = trajectory.shape

    if n_points < 20:
        return np.nan

    # Subsample reference points
    n_ref = min(n_reference, n_points)
    ref_indices = np.random.choice(n_points, n_ref, replace=False)

    # Compute all distances from reference points
    distances = []
    for i in ref_indices:
        for j in range(n_points):
            if i != j:
                d = np.linalg.norm(trajectory[i] - trajectory[j])
                if d > 0:
                    distances.append(d)

    if len(distances) < 10:
        return np.nan

    distances = np.array(distances)

    # Set radius values
    if r_values is None:
        r_min = np.percentile(distances, 1)
        r_max = np.percentile(distances, 50)
        if r_min <= 0 or r_max <= r_min:
            return np.nan
        r_values = np.logspace(np.log10(r_min), np.log10(r_max), 20)

    # Compute correlation sum C(r)
    C_values = []
    valid_r = []

    for r in r_values:
        C = np.mean(distances <= r)
        if C > 0:
            C_values.append(C)
            valid_r.append(r)

    if len(C_values) < 5:
        return np.nan

    # Linear regression on log-log plot
    log_r = np.log(valid_r)
    log_C = np.log(C_values)

    # Use middle portion for linear fit (scaling region)
    n_fit = len(log_r)
    start = n_fit // 4
    end = 3 * n_fit // 4

    log_r_fit = log_r[start:end]
    log_C_fit = log_C[start:end]

    if len(log_r_fit) < 3:
        return np.nan

    # Simple linear regression
    x_mean = np.mean(log_r_fit)
    y_mean = np.mean(log_C_fit)

    num = np.sum((log_r_fit - x_mean) * (log_C_fit - y_mean))
    den = np.sum((log_r_fit - x_mean) ** 2)

    if den == 0:
        return np.nan

    return float(num / den)


def attractor_reconstruction(
    signal: np.ndarray,
    dim: int = 3,
    tau: int = 1
) -> np.ndarray:
    """
    Reconstruct attractor using time-delay embedding.

    Wrapper for delay embedding specific to attractor analysis.

    Parameters
    ----------
    signal : np.ndarray
        Input time series
    dim : int
        Embedding dimension
    tau : int
        Time delay

    Returns
    -------
    np.ndarray
        Reconstructed attractor trajectory
    """
    signal = np.asarray(signal).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if n < (dim - 1) * tau + 1:
        return np.array([[np.nan]])

    n_vectors = n - (dim - 1) * tau

    embedding = np.zeros((n_vectors, dim))
    for i in range(dim):
        embedding[:, i] = signal[i * tau:i * tau + n_vectors]

    return embedding


def kaplan_yorke_dimension(lyapunov_spectrum: np.ndarray) -> float:
    """
    Compute Kaplan-Yorke (Lyapunov) dimension.

    D_KY = j + (sum of first j exponents) / |lambda_{j+1}|

    where j is the largest index for which the sum is non-negative.

    Parameters
    ----------
    lyapunov_spectrum : np.ndarray
        Lyapunov exponents (sorted descending)

    Returns
    -------
    float
        Kaplan-Yorke dimension
    """
    spectrum = np.asarray(lyapunov_spectrum).flatten()
    spectrum = spectrum[~np.isnan(spectrum)]

    if len(spectrum) == 0:
        return np.nan

    # Sort descending (should already be)
    spectrum = np.sort(spectrum)[::-1]

    # Find j where cumsum goes negative
    cumsum = np.cumsum(spectrum)

    j = 0
    for i, s in enumerate(cumsum):
        if s >= 0:
            j = i + 1
        else:
            break

    if j == 0:
        return 0.0

    if j >= len(spectrum):
        # All exponents sum to non-negative
        return float(len(spectrum))

    # D_KY = j + cumsum[j-1] / |spectrum[j]|
    if abs(spectrum[j]) < 1e-10:
        return float(j)

    return float(j + cumsum[j - 1] / abs(spectrum[j]))
