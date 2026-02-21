"""
Time-Delay Embedding

Functions for constructing and optimizing time-delay embeddings for
dynamical systems analysis (Takens' theorem).
"""

import numpy as np
from typing import Optional, Tuple

from pmtvs_embedding._dispatch import use_rust


def delay_embedding(
    signal: np.ndarray,
    dim: int,
    tau: int = 1
) -> np.ndarray:
    """
    Construct time-delay embedding matrix.

    Creates an m-dimensional embedding from a 1D time series using
    Takens' delay embedding theorem.

    Parameters
    ----------
    signal : np.ndarray
        Input time series (1D)
    dim : int
        Embedding dimension (m)
    tau : int
        Time delay (default: 1)

    Returns
    -------
    np.ndarray
        Embedding matrix of shape (N - (dim-1)*tau, dim)
        where N is the length of the signal
    """
    if use_rust('delay_embedding'):
        from pmtvs_embedding import _get_rust
        rust_fn = _get_rust('delay_embedding')
        if rust_fn is not None:
            return rust_fn(signal, dim, tau)

    signal = np.asarray(signal).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if dim < 1 or tau < 1:
        return np.array([[np.nan]])

    n_vectors = n - (dim - 1) * tau
    if n_vectors < 1:
        return np.array([[np.nan]])

    # Build embedding matrix
    embedding = np.zeros((n_vectors, dim))
    for i in range(dim):
        embedding[:, i] = signal[i * tau:i * tau + n_vectors]

    return embedding


def optimal_embedding_dimension(
    signal: np.ndarray,
    tau: int = 1,
    max_dim: int = 10,
    threshold: float = 0.95
) -> int:
    """
    Find optimal embedding dimension using Cao's method.

    Cao's method computes E1(d) and E2(d) statistics to determine
    the minimum embedding dimension that unfolds the attractor.

    Parameters
    ----------
    signal : np.ndarray
        Input time series
    tau : int
        Time delay
    max_dim : int
        Maximum dimension to search
    threshold : float
        E1 saturation threshold (default: 0.95)

    Returns
    -------
    int
        Optimal embedding dimension
    """
    signal = np.asarray(signal).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if n < (max_dim + 1) * tau + 10:
        return 2  # Default minimum

    e1_values = []

    for d in range(1, max_dim + 1):
        embed_d = delay_embedding(signal, d, tau)
        embed_d1 = delay_embedding(signal, d + 1, tau)

        if len(embed_d) < 2 or len(embed_d1) < 2:
            break

        # Compute a(i, d) for each point
        n_points = min(len(embed_d), len(embed_d1))
        a_values = []

        for i in range(n_points):
            # Find nearest neighbor in d-dimensional space
            distances = np.linalg.norm(embed_d[:n_points] - embed_d[i], axis=1)
            distances[i] = np.inf  # Exclude self

            nn_idx = np.argmin(distances)
            nn_dist = distances[nn_idx]

            if nn_dist > 0:
                # Distance in (d+1)-dimensional space
                dist_d1 = np.linalg.norm(embed_d1[i] - embed_d1[nn_idx])
                a_values.append(dist_d1 / nn_dist)

        if a_values:
            e1_values.append(np.mean(a_values))

    if len(e1_values) < 2:
        return 2

    # Find where E1 saturates
    e1 = np.array(e1_values)
    for d in range(1, len(e1)):
        if d > 0 and e1[d] / e1[d - 1] > threshold:
            return d + 1

    return len(e1) + 1


def mutual_information_delay(
    signal: np.ndarray,
    max_lag: int = 50,
    n_bins: int = 16
) -> int:
    """
    Find optimal time delay using mutual information.

    The optimal delay is typically the first minimum of the
    mutual information function.

    Parameters
    ----------
    signal : np.ndarray
        Input time series
    max_lag : int
        Maximum lag to search
    n_bins : int
        Number of bins for histogram

    Returns
    -------
    int
        Optimal time delay
    """
    signal = np.asarray(signal).flatten()
    signal = signal[np.isfinite(signal)]
    n = len(signal)

    if n < max_lag + 10:
        return 1

    max_lag = min(max_lag, n - 10)

    mi_values = []

    for lag in range(1, max_lag + 1):
        x = signal[:-lag]
        y = signal[lag:]

        # Compute mutual information via histograms
        hist_xy, _, _ = np.histogram2d(x, y, bins=n_bins)
        hist_x, _ = np.histogram(x, bins=n_bins)
        hist_y, _ = np.histogram(y, bins=n_bins)

        # Normalize to probabilities
        p_xy = hist_xy / np.sum(hist_xy)
        p_x = hist_x / np.sum(hist_x)
        p_y = hist_y / np.sum(hist_y)

        # Mutual information
        mi = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                    mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))

        mi_values.append(mi)

    # Find first local minimum
    mi_arr = np.array(mi_values)
    for i in range(1, len(mi_arr) - 1):
        if mi_arr[i] < mi_arr[i - 1] and mi_arr[i] < mi_arr[i + 1]:
            return i + 1

    # If no minimum found, use first quarter decay
    return max(1, max_lag // 4)


def false_nearest_neighbors(
    signal: np.ndarray,
    tau: int = 1,
    max_dim: int = 10,
    r_threshold: float = 15.0,
    a_threshold: float = 2.0
) -> Tuple[np.ndarray, int]:
    """
    Find optimal embedding dimension using false nearest neighbors.

    A false nearest neighbor occurs when two points that appear close
    in a lower-dimensional embedding are actually far apart in the
    true higher-dimensional space.

    Parameters
    ----------
    signal : np.ndarray
        Input time series
    tau : int
        Time delay
    max_dim : int
        Maximum dimension to search
    r_threshold : float
        Distance ratio threshold
    a_threshold : float
        Attractor size threshold

    Returns
    -------
    tuple
        (fnn_percentages, optimal_dimension)
        - fnn_percentages: array of FNN percentages for each dimension
        - optimal_dimension: dimension where FNN drops below 1%
    """
    signal = np.asarray(signal).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if n < (max_dim + 1) * tau + 10:
        return np.full(max_dim, np.nan), 2

    # Standard deviation for normalization
    std = np.std(signal)
    if std == 0:
        return np.full(max_dim, np.nan), 2

    fnn_percentages = []

    for d in range(1, max_dim + 1):
        embed_d = delay_embedding(signal, d, tau)
        embed_d1 = delay_embedding(signal, d + 1, tau)

        n_points = min(len(embed_d), len(embed_d1))
        if n_points < 2:
            fnn_percentages.append(np.nan)
            continue

        n_fnn = 0
        n_valid = 0

        for i in range(n_points):
            # Find nearest neighbor in d-dimensional space
            distances = np.linalg.norm(embed_d[:n_points] - embed_d[i], axis=1)
            distances[i] = np.inf

            nn_idx = np.argmin(distances)
            r_d = distances[nn_idx]

            if r_d > 0:
                # Check FNN criteria
                r_d1 = np.linalg.norm(embed_d1[i] - embed_d1[nn_idx])

                # Criterion 1: distance ratio
                criterion1 = abs(r_d1 - r_d) / r_d > r_threshold

                # Criterion 2: distance relative to attractor size
                criterion2 = r_d1 / std > a_threshold

                if criterion1 or criterion2:
                    n_fnn += 1
                n_valid += 1

        if n_valid > 0:
            fnn_percentages.append(100.0 * n_fnn / n_valid)
        else:
            fnn_percentages.append(np.nan)

    fnn_arr = np.array(fnn_percentages)

    # Find dimension where FNN drops below 1%
    optimal_dim = max_dim
    for d, fnn in enumerate(fnn_arr):
        if not np.isnan(fnn) and fnn < 1.0:
            optimal_dim = d + 1
            break

    return fnn_arr, optimal_dim


def multivariate_embedding(
    signals: np.ndarray,
    dim: int = 3,
    tau: int = 1
) -> np.ndarray:
    """
    Construct multivariate time-delay embedding from multiple signals.

    Each signal contributes delay coordinates to the joint embedding space.

    Parameters
    ----------
    signals : np.ndarray
        Input signals, shape (n_signals, n_samples) or (n_samples, n_signals)
        If 2D and n_cols > n_rows, assumes (n_signals, n_samples)
    dim : int
        Embedding dimension per signal
    tau : int
        Time delay

    Returns
    -------
    np.ndarray
        Joint embedding matrix of shape (n_vectors, n_signals * dim)
    """
    signals = np.asarray(signals, dtype=np.float64)

    if signals.ndim == 1:
        return delay_embedding(signals, dim, tau)

    # Ensure shape is (n_signals, n_samples)
    if signals.shape[0] > signals.shape[1]:
        signals = signals.T

    n_signals, n_samples = signals.shape
    n_vectors = n_samples - (dim - 1) * tau

    if n_vectors < 1:
        return np.array([[np.nan]])

    embeddings = []
    for sig in signals:
        emb = delay_embedding(sig, dim, tau)
        if emb.shape[0] < n_vectors:
            n_vectors = emb.shape[0]
        embeddings.append(emb)

    # Truncate all to same length and concatenate
    joint = np.hstack([emb[:n_vectors] for emb in embeddings])
    return joint
