"""Domain-specific dynamical feature primitives."""

import numpy as np


def basin_stability(y: np.ndarray, n_bins: int = 20) -> dict:
    """
    Estimate basin stability from the distribution of signal values.

    Parameters
    ----------
    y : np.ndarray
        Input signal (1D).
    n_bins : int
        Number of histogram bins.

    Returns
    -------
    dict
        basin_stability, transition_prob, n_attractors.
    """
    nan_result = {
        'basin_stability': np.nan,
        'transition_prob': np.nan,
        'n_attractors': np.nan,
    }
    y = np.asarray(y, dtype=np.float64).ravel()
    y = y[np.isfinite(y)]

    if len(y) < 20:
        return nan_result

    counts, bin_edges = np.histogram(y, bins=n_bins)
    total = np.sum(counts)
    if total == 0:
        return nan_result

    bs = float(np.max(counts) / total)
    bin_indices = np.digitize(y, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    transitions = np.sum(np.diff(bin_indices) != 0)
    tp = float(transitions / max(1, len(y) - 1))

    n_attractors = 0
    for i in range(1, len(counts) - 1):
        if counts[i] > counts[i - 1] and counts[i] > counts[i + 1]:
            n_attractors += 1
    if n_attractors == 0:
        n_attractors = 1

    return {
        'basin_stability': bs,
        'transition_prob': tp,
        'n_attractors': int(n_attractors),
    }


def cycle_counting(y: np.ndarray) -> dict:
    """
    Count cycles via zero crossings of the demeaned signal.

    Parameters
    ----------
    y : np.ndarray
        Input signal (1D).

    Returns
    -------
    dict
        n_cycles, mean_amplitude, max_amplitude, mean_period.
    """
    nan_result = {
        'n_cycles': np.nan,
        'mean_amplitude': np.nan,
        'max_amplitude': np.nan,
        'mean_period': np.nan,
    }
    y = np.asarray(y, dtype=np.float64).ravel()
    y = y[np.isfinite(y)]

    if len(y) < 4:
        return nan_result

    y_centered = y - np.mean(y)
    signs = np.sign(y_centered)
    for i in range(len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i - 1] if i > 0 else 1.0
    crossings = np.where(np.diff(signs) != 0)[0]

    n_crossings = len(crossings)
    n_cycles = n_crossings / 2.0

    if n_crossings < 2:
        return {
            'n_cycles': float(n_cycles),
            'mean_amplitude': np.nan,
            'max_amplitude': np.nan,
            'mean_period': float(len(y)) if n_cycles > 0 else np.nan,
        }

    amplitudes = []
    for i in range(len(crossings) - 1):
        segment = y_centered[crossings[i]:crossings[i + 1] + 1]
        if len(segment) > 0:
            amplitudes.append(float(np.max(segment) - np.min(segment)))

    mean_amp = float(np.mean(amplitudes)) if amplitudes else np.nan
    max_amp = float(np.max(amplitudes)) if amplitudes else np.nan
    mean_period = float(len(y) / max(1.0, n_cycles))

    return {
        'n_cycles': float(n_cycles),
        'mean_amplitude': mean_amp,
        'max_amplitude': max_amp,
        'mean_period': mean_period,
    }


def local_outlier_factor(y: np.ndarray, n_neighbors: int = 20) -> dict:
    """
    Compute Local Outlier Factor (LOF) scores for a 1D signal.

    Parameters
    ----------
    y : np.ndarray
        Input signal (1D).
    n_neighbors : int
        Number of neighbors for LOF computation.

    Returns
    -------
    dict
        mean_lof, max_lof, outlier_fraction.
    """
    nan_result = {
        'mean_lof': np.nan,
        'max_lof': np.nan,
        'outlier_fraction': np.nan,
    }
    y = np.asarray(y, dtype=np.float64).ravel()
    y = y[np.isfinite(y)]

    if len(y) < n_neighbors + 1:
        return nan_result

    n = len(y)
    lof_scores = np.empty(n, dtype=np.float64)

    for i in range(n):
        all_dists = np.abs(y[i] - y)
        neighbor_indices = np.argsort(all_dists)[1:n_neighbors + 1]

        reach_dists = np.empty(len(neighbor_indices), dtype=np.float64)
        for j_idx, j in enumerate(neighbor_indices):
            j_dists = np.abs(y[j] - y)
            j_dists_sorted = np.sort(j_dists)
            k_dist_j = j_dists_sorted[min(n_neighbors, n - 1)]
            reach_dists[j_idx] = max(k_dist_j, abs(y[i] - y[j]))

        mean_reach = np.mean(reach_dists)
        if mean_reach < 1e-15:
            lof_scores[i] = 1.0
            continue

        lrd_i = 1.0 / mean_reach

        neighbor_lrds = np.empty(len(neighbor_indices), dtype=np.float64)
        for j_idx, j in enumerate(neighbor_indices):
            j_all_dists = np.abs(y[j] - y)
            j_neighbor_indices = np.argsort(j_all_dists)[1:n_neighbors + 1]
            j_reach_dists = np.empty(len(j_neighbor_indices), dtype=np.float64)
            for k_idx, k in enumerate(j_neighbor_indices):
                k_dists = np.sort(np.abs(y[k] - y))
                k_dist_k = k_dists[min(n_neighbors, n - 1)]
                j_reach_dists[k_idx] = max(k_dist_k, abs(y[j] - y[k]))
            mean_reach_j = np.mean(j_reach_dists)
            neighbor_lrds[j_idx] = 1.0 / max(mean_reach_j, 1e-15)

        lof_scores[i] = float(np.mean(neighbor_lrds) / lrd_i) if lrd_i > 0 else 1.0

    return {
        'mean_lof': float(np.mean(lof_scores)),
        'max_lof': float(np.max(lof_scores)),
        'outlier_fraction': float(np.sum(lof_scores > 1.5) / len(lof_scores)),
    }


def time_constant(y: np.ndarray) -> dict:
    """
    Estimate exponential decay time constant via log-linear regression.

    Parameters
    ----------
    y : np.ndarray
        Input signal (1D).

    Returns
    -------
    dict
        tau (time constant), r_squared (fit quality).
    """
    nan_result = {'tau': np.nan, 'r_squared': np.nan}
    y = np.asarray(y, dtype=np.float64).ravel()
    y = y[np.isfinite(y)]

    if len(y) < 5:
        return nan_result

    diffs = np.diff(y)
    nonzero_diffs = diffs[diffs != 0]
    if len(nonzero_diffs) < 2:
        return nan_result

    n_positive = np.sum(nonzero_diffs > 0)
    n_negative = np.sum(nonzero_diffs < 0)
    dominant_fraction = max(n_positive, n_negative) / len(nonzero_diffs)
    if dominant_fraction < 0.7:
        return nan_result

    y_min = np.min(y)
    epsilon = 1e-10
    y_shifted = y - y_min + epsilon
    t = np.arange(len(y), dtype=np.float64)
    log_y = np.log(y_shifted)

    valid = np.isfinite(log_y)
    if np.sum(valid) < 3:
        return nan_result

    t_valid = t[valid]
    log_y_valid = log_y[valid]
    coeffs = np.polyfit(t_valid, log_y_valid, 1)
    slope = coeffs[0]

    if abs(slope) < 1e-15:
        return nan_result

    tau = abs(-1.0 / slope)
    fitted = np.polyval(coeffs, t_valid)
    ss_res = np.sum((log_y_valid - fitted) ** 2)
    ss_tot = np.sum((log_y_valid - np.mean(log_y_valid)) ** 2)
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-15 else 0.0

    return {'tau': float(tau), 'r_squared': r_squared}
