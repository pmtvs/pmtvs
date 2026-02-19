"""Dynamical dimension primitives — correlation integral and information dimension."""

import numpy as np
from typing import Optional


def correlation_integral(
    embedded: np.ndarray,
    r: float
) -> float:
    """
    Compute correlation integral for embedded trajectory.

    C(r) = (2 / N(N-1)) sum_{i<j} Theta(r - ||x_i - x_j||)

    Parameters
    ----------
    embedded : np.ndarray
        Embedded trajectory (n_points x dimension).
    r : float
        Radius.

    Returns
    -------
    float
        Correlation sum C(r).
    """
    embedded = np.asarray(embedded)
    if embedded.ndim == 1:
        embedded = embedded.reshape(-1, 1)
    n = len(embedded)

    if n < 2:
        return 0.0

    count = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(embedded[i] - embedded[j])
            if dist < r:
                count += 1
            total += 1

    return float(count / total) if total > 0 else 0.0


def information_dimension(
    signal: np.ndarray,
    dimension: Optional[int] = None,
    delay: Optional[int] = None,
    n_boxes: int = 20
) -> float:
    """
    Estimate information dimension using box-counting.

    D1 = lim_{eps->0} sum(p_i log(p_i)) / log(eps)

    Parameters
    ----------
    signal : np.ndarray
        1D time series.
    dimension : int, optional
        Embedding dimension.
    delay : int, optional
        Time delay.
    n_boxes : int
        Number of box sizes to test.

    Returns
    -------
    float
        Information dimension D1.
    """
    signal = np.asarray(signal, dtype=np.float64).flatten()
    signal = signal[~np.isnan(signal)]

    if delay is None:
        delay = _auto_delay(signal)
    if dimension is None:
        dimension = min(5, max(2, len(signal) // 50))

    embedded = _embed(signal, dimension, delay)
    n_points = len(embedded)

    if n_points < 50:
        return np.nan

    mins = np.min(embedded, axis=0)
    maxs = np.max(embedded, axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    normalized = (embedded - mins) / ranges

    log_eps = np.linspace(np.log(0.01), np.log(0.5), n_boxes)
    epsilons = np.exp(log_eps)
    log_info = np.zeros(n_boxes)

    for k, eps in enumerate(epsilons):
        n_boxes_per_dim = int(np.ceil(1.0 / eps))
        box_indices = np.floor(normalized / eps).astype(int)
        box_indices = np.clip(box_indices, 0, n_boxes_per_dim - 1)

        multipliers = n_boxes_per_dim ** np.arange(dimension)
        flat_indices = np.sum(box_indices * multipliers, axis=1)

        unique, counts = np.unique(flat_indices, return_counts=True)
        p = counts / n_points
        info = -np.sum(p * np.log(p + 1e-10))
        log_info[k] = info

    slope, _ = np.polyfit(log_eps, log_info, 1)
    return float(-slope)


def _embed(signal: np.ndarray, dimension: int, delay: int) -> np.ndarray:
    """Time delay embedding."""
    n = len(signal)
    n_points = n - (dimension - 1) * delay
    if n_points < 1:
        return np.array([]).reshape(0, dimension)
    embedded = np.zeros((n_points, dimension))
    for d in range(dimension):
        embedded[:, d] = signal[d * delay:d * delay + n_points]
    return embedded


def _auto_delay(signal: np.ndarray) -> int:
    """Auto-detect delay using autocorrelation 1/e decay."""
    n = len(signal)
    centered = signal - np.mean(signal)
    var = np.var(centered)
    if var == 0:
        return 1
    for lag in range(1, n // 4):
        acf = np.mean(centered[:-lag] * centered[lag:]) / var
        if acf < 1 / np.e:
            return lag
    return max(1, n // 10)
