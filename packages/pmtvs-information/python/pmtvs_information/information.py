"""
Information Theory Functions
"""

import numpy as np
from typing import Optional


def mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute mutual information between two signals.

    I(X;Y) = H(X) + H(Y) - H(X,Y)

    Parameters
    ----------
    x : np.ndarray
        First signal
    y : np.ndarray
        Second signal
    n_bins : int
        Number of histogram bins

    Returns
    -------
    float
        Mutual information in bits
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    if len(x) != len(y) or len(x) < 2:
        return np.nan

    # Joint histogram
    hist_xy, _, _ = np.histogram2d(x, y, bins=n_bins)
    hist_x, _ = np.histogram(x, bins=n_bins)
    hist_y, _ = np.histogram(y, bins=n_bins)

    # Convert to probabilities
    p_xy = hist_xy / np.sum(hist_xy)
    p_x = hist_x / np.sum(hist_x)
    p_y = hist_y / np.sum(hist_y)

    # Mutual information
    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))

    return float(mi)


def transfer_entropy(
    source: np.ndarray,
    target: np.ndarray,
    lag: int = 1,
    n_bins: int = 10
) -> float:
    """
    Compute transfer entropy from source to target.

    Measures information flow from source to target beyond
    target's own past.

    Parameters
    ----------
    source : np.ndarray
        Source signal
    target : np.ndarray
        Target signal
    lag : int
        Time lag
    n_bins : int
        Number of histogram bins

    Returns
    -------
    float
        Transfer entropy in bits
    """
    source = np.asarray(source).flatten()
    target = np.asarray(target).flatten()

    n = min(len(source), len(target))
    if n < lag + 2:
        return np.nan

    # Construct lagged variables
    target_future = target[lag:]
    target_past = target[:-lag]
    source_past = source[:-lag]

    n_samples = len(target_future)

    # 3D histogram (target_future, target_past, source_past)
    hist_3d, _ = np.histogramdd(
        [target_future, target_past[:n_samples], source_past[:n_samples]],
        bins=n_bins
    )

    # 2D histograms
    hist_tt, _, _ = np.histogram2d(target_future, target_past[:n_samples], bins=n_bins)
    hist_ts, _, _ = np.histogram2d(target_past[:n_samples], source_past[:n_samples], bins=n_bins)

    # 1D histograms
    hist_t, _ = np.histogram(target_past[:n_samples], bins=n_bins)

    # Normalize
    p_3d = hist_3d / np.sum(hist_3d)
    p_tt = hist_tt / np.sum(hist_tt)
    p_ts = hist_ts / np.sum(hist_ts)
    p_t = hist_t / np.sum(hist_t)

    # Transfer entropy
    te = 0.0
    eps = 1e-12

    for i in range(n_bins):
        for j in range(n_bins):
            for k in range(n_bins):
                if p_3d[i, j, k] > eps and p_tt[i, j] > eps and p_ts[j, k] > eps and p_t[j] > eps:
                    te += p_3d[i, j, k] * np.log2(
                        (p_3d[i, j, k] * p_t[j]) / (p_tt[i, j] * p_ts[j, k])
                    )

    return float(max(0, te))


def conditional_entropy(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute conditional entropy H(X|Y).

    Parameters
    ----------
    x : np.ndarray
        Signal X
    y : np.ndarray
        Conditioning signal Y
    n_bins : int
        Number of histogram bins

    Returns
    -------
    float
        Conditional entropy in bits
    """
    return joint_entropy(x, y, n_bins) - _entropy(y, n_bins)


def joint_entropy(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute joint entropy H(X,Y).

    Parameters
    ----------
    x : np.ndarray
        First signal
    y : np.ndarray
        Second signal
    n_bins : int
        Number of histogram bins

    Returns
    -------
    float
        Joint entropy in bits
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    if len(x) != len(y) or len(x) < 2:
        return np.nan

    hist_xy, _, _ = np.histogram2d(x, y, bins=n_bins)
    p_xy = hist_xy / np.sum(hist_xy)
    p_xy = p_xy[p_xy > 0]

    return float(-np.sum(p_xy * np.log2(p_xy)))


def _entropy(x: np.ndarray, n_bins: int = 10) -> float:
    """Compute Shannon entropy."""
    x = np.asarray(x).flatten()
    hist, _ = np.histogram(x, bins=n_bins)
    p = hist / np.sum(hist)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Kullback-Leibler divergence D_KL(P||Q).

    Parameters
    ----------
    p : np.ndarray
        First distribution (samples)
    q : np.ndarray
        Second distribution (samples)
    n_bins : int
        Number of histogram bins

    Returns
    -------
    float
        KL divergence (non-negative)
    """
    p = np.asarray(p).flatten()
    q = np.asarray(q).flatten()

    # Estimate distributions from samples
    all_data = np.concatenate([p, q])
    bins = np.histogram_bin_edges(all_data, bins=n_bins)

    hist_p, _ = np.histogram(p, bins=bins)
    hist_q, _ = np.histogram(q, bins=bins)

    # Add small constant for numerical stability
    eps = 1e-10
    p_dist = (hist_p + eps) / np.sum(hist_p + eps)
    q_dist = (hist_q + eps) / np.sum(hist_q + eps)

    return float(np.sum(p_dist * np.log(p_dist / q_dist)))


def js_divergence(
    p: np.ndarray,
    q: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Jensen-Shannon divergence.

    JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M) where M = 0.5*(P+Q)

    Parameters
    ----------
    p : np.ndarray
        First distribution (samples)
    q : np.ndarray
        Second distribution (samples)
    n_bins : int
        Number of histogram bins

    Returns
    -------
    float
        JS divergence in [0, 1]
    """
    p = np.asarray(p).flatten()
    q = np.asarray(q).flatten()

    all_data = np.concatenate([p, q])
    bins = np.histogram_bin_edges(all_data, bins=n_bins)

    hist_p, _ = np.histogram(p, bins=bins)
    hist_q, _ = np.histogram(q, bins=bins)

    eps = 1e-10
    p_dist = (hist_p + eps) / np.sum(hist_p + eps)
    q_dist = (hist_q + eps) / np.sum(hist_q + eps)

    m = 0.5 * (p_dist + q_dist)

    kl_pm = np.sum(p_dist * np.log2(p_dist / m))
    kl_qm = np.sum(q_dist * np.log2(q_dist / m))

    return float(0.5 * kl_pm + 0.5 * kl_qm)


def information_gain(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute information gain (reduction in entropy).

    IG(X;Y) = H(X) - H(X|Y)

    Parameters
    ----------
    x : np.ndarray
        Target signal
    y : np.ndarray
        Conditioning signal
    n_bins : int
        Number of histogram bins

    Returns
    -------
    float
        Information gain in bits
    """
    return _entropy(x, n_bins) - conditional_entropy(x, y, n_bins)
