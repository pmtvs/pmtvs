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
    Count cycles via peak-valley detection.

    Parameters
    ----------
    y : np.ndarray
        Input signal (1D).

    Returns
    -------
    dict
        n_cycles, n_full, n_half, mean_amplitude, max_amplitude,
        mean_period, max_range, mean_range, accumulation.
    """
    nan_result = {
        'n_cycles': np.nan, 'n_full': np.nan, 'n_half': np.nan,
        'mean_amplitude': np.nan, 'max_amplitude': np.nan,
        'mean_period': np.nan, 'max_range': np.nan,
        'mean_range': np.nan, 'accumulation': np.nan,
    }
    y = np.asarray(y, dtype=np.float64).ravel()
    y = y[np.isfinite(y)]

    if len(y) < 4:
        return nan_result

    d = np.diff(y)
    sign_changes = np.where(d[:-1] * d[1:] < 0)[0] + 1

    if len(sign_changes) < 2:
        return {**nan_result, 'n_cycles': 0.0, 'n_full': 0.0, 'n_half': 0.0}

    extrema = y[sign_changes]
    ranges = np.abs(np.diff(extrema))
    n_half = len(ranges)
    n_full = n_half // 2
    n_cycles = float(n_full + 0.5 * (n_half % 2))

    # Zero-crossing based amplitude/period (backward compat)
    y_centered = y - np.mean(y)
    signs = np.sign(y_centered)
    for i in range(len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i - 1] if i > 0 else 1.0
    crossings = np.where(np.diff(signs) != 0)[0]

    amplitudes = []
    for i in range(len(crossings) - 1):
        segment = y_centered[crossings[i]:crossings[i + 1] + 1]
        if len(segment) > 0:
            amplitudes.append(float(np.max(segment) - np.min(segment)))

    mean_amp = float(np.mean(amplitudes)) if amplitudes else np.nan
    max_amp = float(np.max(amplitudes)) if amplitudes else np.nan
    mean_period = float(len(y) / max(1.0, n_cycles))

    return {
        'n_cycles': n_cycles,
        'n_full': float(n_full),
        'n_half': float(n_half),
        'mean_amplitude': mean_amp,
        'max_amplitude': max_amp,
        'mean_period': mean_period,
        'max_range': float(np.max(ranges)) if len(ranges) > 0 else 0.0,
        'mean_range': float(np.mean(ranges)) if len(ranges) > 0 else 0.0,
        'accumulation': float(np.sum(ranges)),
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
        'scores': lof_scores,
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
        tau, r_squared, equilibrium, fit_r2 (alias of r_squared), is_decay.
    """
    nan_result = {
        'tau': np.nan, 'r_squared': np.nan,
        'equilibrium': np.nan, 'fit_r2': np.nan, 'is_decay': np.nan,
    }
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

    equilibrium = float(y[-1])
    y_shifted = y - equilibrium
    mask = np.abs(y_shifted) > 1e-15
    if np.sum(mask) < 5:
        return nan_result

    t = np.arange(len(y), dtype=np.float64)
    log_y = np.log(np.abs(y_shifted[mask]))
    coeffs = np.polyfit(t[mask], log_y, 1)
    slope = coeffs[0]

    if abs(slope) < 1e-15:
        return nan_result

    tau = float(abs(-1.0 / slope))
    fitted = np.polyval(coeffs, t[mask])
    ss_res = np.sum((log_y - fitted) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-15 else 0.0
    is_decay = float(slope < 0)

    return {
        'tau': tau, 'r_squared': r_squared,
        'equilibrium': equilibrium, 'fit_r2': r_squared, 'is_decay': is_decay,
    }


def transition_analysis(y: np.ndarray, n_states: int = 0) -> dict:
    """
    State quantization and transition matrix analysis.

    Parameters
    ----------
    y : np.ndarray
        Input signal (1D).
    n_states : int
        Number of quantization states. 0 = auto from sqrt(n)/5.

    Returns
    -------
    dict
        entropy, self_loop, max_self_loop, asymmetry, n_active, sparsity.
    """
    nan_result = {
        'entropy': np.nan, 'self_loop': np.nan, 'max_self_loop': np.nan,
        'asymmetry': np.nan, 'n_active': np.nan, 'sparsity': np.nan,
    }
    y = np.asarray(y, dtype=np.float64).ravel()
    y = y[np.isfinite(y)]
    n = len(y)

    if n < 20:
        return nan_result

    if n_states <= 0:
        n_states = min(5, max(2, int(np.sqrt(n) / 5)))

    states = np.digitize(y, np.linspace(np.min(y), np.max(y), n_states + 1)) - 1
    states = np.clip(states, 0, n_states - 1)

    tm = np.zeros((n_states, n_states))
    for i in range(n - 1):
        tm[states[i], states[i + 1]] += 1

    row_sums = tm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    tm_norm = tm / row_sums

    diag = np.diag(tm_norm)
    self_loop = float(np.mean(diag))
    max_self = float(np.max(diag))
    asym = float(np.sum(np.abs(tm_norm - tm_norm.T)) / 2)
    nonzero = np.sum(tm > 0)
    total = n_states * n_states
    sparsity = float(1.0 - nonzero / total)

    flat = tm_norm.ravel()
    flat = flat[flat > 1e-15]
    entropy = float(-np.sum(flat * np.log(flat)))

    return {
        'entropy': entropy, 'self_loop': self_loop,
        'max_self_loop': max_self, 'asymmetry': asym,
        'n_active': float(nonzero), 'sparsity': sparsity,
    }


def dwell_analysis(y: np.ndarray, n_bins: int = 0) -> dict:
    """
    State quantization and dwell time statistics.

    Parameters
    ----------
    y : np.ndarray
        Input signal (1D).
    n_bins : int
        Number of quantization bins. 0 = auto from sqrt(n).

    Returns
    -------
    dict
        mean, std, max, min, cv, n.
    """
    nan_result = {
        'mean': np.nan, 'std': np.nan, 'max': np.nan,
        'min': np.nan, 'cv': np.nan, 'n': np.nan,
    }
    y = np.asarray(y, dtype=np.float64).ravel()
    y = y[np.isfinite(y)]
    n = len(y)

    if n < 10:
        return nan_result

    if n_bins <= 0:
        n_bins = min(10, int(np.sqrt(n)))

    bins = np.digitize(y, np.linspace(np.min(y), np.max(y), n_bins + 1))

    dwells = []
    current_len = 1
    for i in range(1, len(bins)):
        if bins[i] == bins[i - 1]:
            current_len += 1
        else:
            dwells.append(current_len)
            current_len = 1
    dwells.append(current_len)

    d = np.array(dwells, dtype=np.float64)
    m = float(np.mean(d))
    s = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0

    return {
        'mean': m, 'std': s,
        'max': float(np.max(d)), 'min': float(np.min(d)),
        'cv': s / m if m > 1e-15 else 0.0,
        'n': float(len(d)),
    }


def garch_fit(y: np.ndarray) -> dict:
    """
    Method-of-moments GARCH(1,1) estimator.

    Parameters
    ----------
    y : np.ndarray
        Input signal (1D).

    Returns
    -------
    dict
        omega, alpha, beta.
    """
    nan_result = {'omega': np.nan, 'alpha': np.nan, 'beta': np.nan}

    y = np.asarray(y, dtype=np.float64).ravel()
    y = y[np.isfinite(y)]

    if len(y) < 20:
        return nan_result

    returns = np.diff(y)
    if len(returns) < 10:
        return nan_result

    returns = returns - np.mean(returns)
    r2 = returns ** 2
    var = np.var(returns)

    if var < 1e-30:
        return nan_result

    # Autocorrelation of squared returns at lag 1
    if len(r2) < 3:
        return nan_result
    r2_centered = r2 - np.mean(r2)
    ac1 = np.sum(r2_centered[:-1] * r2_centered[1:]) / np.sum(r2_centered ** 2)
    ac1 = np.clip(ac1, 0.0, 0.99)

    # Method of moments: alpha + beta ~ ac1, alpha ~ kurt_excess * (1 - ac1)
    kurt = np.mean(r2_centered ** 2) / (np.mean(r2) ** 2) - 1.0
    kurt = max(kurt, 0.01)

    alpha = float(np.clip(ac1 * (1.0 - ac1) / max(kurt, 0.01), 0.001, 0.5))
    beta = float(np.clip(ac1 - alpha, 0.0, 0.999 - alpha))
    omega = float(var * (1.0 - alpha - beta))
    omega = max(omega, 1e-10)

    return {'omega': omega, 'alpha': alpha, 'beta': beta}


def hmm_fit(y: np.ndarray, n_states: int = 3) -> dict:
    """
    Gaussian HMM via k-means + transition counting.

    Parameters
    ----------
    y : np.ndarray
        Input signal (1D).
    n_states : int
        Number of hidden states.

    Returns
    -------
    dict
        n_states, current_state, current_state_prob, state_entropy,
        transition_rate, bic, log_likelihood.
    """
    nan_result = {
        'n_states': np.nan, 'current_state': np.nan,
        'current_state_prob': np.nan, 'state_entropy': np.nan,
        'transition_rate': np.nan, 'bic': np.nan, 'log_likelihood': np.nan,
    }

    y = np.asarray(y, dtype=np.float64).ravel()
    y = y[np.isfinite(y)]
    n = len(y)

    if n < 20:
        return nan_result

    # K-means clustering for state assignment
    # Initialize centroids from quantiles
    centroids = np.quantile(y, np.linspace(0, 1, n_states + 2)[1:-1])

    for _ in range(20):
        dists = np.abs(y[:, None] - centroids[None, :])
        labels = np.argmin(dists, axis=1)
        new_centroids = np.array([
            np.mean(y[labels == k]) if np.sum(labels == k) > 0 else centroids[k]
            for k in range(n_states)
        ])
        if np.allclose(centroids, new_centroids, atol=1e-10):
            break
        centroids = new_centroids

    # State probabilities
    state_counts = np.array([np.sum(labels == k) for k in range(n_states)])
    state_probs = state_counts / n

    # Transition rate
    transitions = np.sum(np.diff(labels) != 0)
    transition_rate = float(transitions / max(1, n - 1))

    # State entropy
    nonzero_probs = state_probs[state_probs > 0]
    state_entropy = float(-np.sum(nonzero_probs * np.log(nonzero_probs)))

    # Current state
    current_state = int(labels[-1])
    current_prob = float(state_probs[current_state])

    # Log-likelihood (Gaussian per state)
    ll = 0.0
    for k in range(n_states):
        mask = labels == k
        if np.sum(mask) > 1:
            mu = np.mean(y[mask])
            sigma = np.std(y[mask])
            if sigma < 1e-15:
                sigma = 1e-15
            ll += np.sum(-0.5 * np.log(2 * np.pi * sigma ** 2) - 0.5 * ((y[mask] - mu) / sigma) ** 2)

    # BIC
    n_params = 3 * n_states - 1  # means + vars + transition probs
    bic = float(-2 * ll + n_params * np.log(n))

    return {
        'n_states': float(n_states),
        'current_state': float(current_state),
        'current_state_prob': current_prob,
        'state_entropy': state_entropy,
        'transition_rate': transition_rate,
        'bic': bic,
        'log_likelihood': float(ll),
    }
