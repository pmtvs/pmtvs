"""
Lyapunov Exponent Analysis

Functions for computing Lyapunov exponents that characterize
chaos and predictability in dynamical systems.
"""

import numpy as np
from typing import Optional, Tuple


def ftle(
    trajectory: np.ndarray,
    dt: float = 1.0,
    method: str = "svd"
) -> float:
    """
    Compute Finite-Time Lyapunov Exponent (FTLE).

    FTLE measures the rate of separation of infinitesimally close
    trajectories over a finite time interval.

    Parameters
    ----------
    trajectory : np.ndarray
        State-space trajectory of shape (n_points, n_dims)
    dt : float
        Time step between observations
    method : str
        Computation method: "svd" or "qr"

    Returns
    -------
    float
        FTLE value (positive indicates chaos)
    """
    trajectory = np.asarray(trajectory)

    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(-1, 1)

    n_points, n_dims = trajectory.shape

    if n_points < 3:
        return np.nan

    # Compute local flow map gradients
    # Using finite differences to approximate Jacobian
    jacobians = []

    for i in range(n_points - 1):
        if i == 0:
            dx = trajectory[1] - trajectory[0]
        else:
            dx = trajectory[i + 1] - trajectory[i - 1]

        if np.linalg.norm(dx) > 0:
            jacobians.append(dx / (2 * dt) if i > 0 else dx / dt)

    if len(jacobians) < 2:
        return np.nan

    jacobians = np.array(jacobians)

    # Compute stretching using SVD
    try:
        if method == "svd":
            # Compute Cauchy-Green tensor approximation
            cg_tensor = jacobians.T @ jacobians / len(jacobians)
            eigenvalues = np.linalg.eigvalsh(cg_tensor)
            max_eigenvalue = np.max(eigenvalues)

            if max_eigenvalue > 0:
                T = n_points * dt
                return float(np.log(np.sqrt(max_eigenvalue)) / T)
            return 0.0
        else:
            # QR method for Lyapunov calculation
            lyap_sum = 0.0
            for j in jacobians:
                if np.linalg.norm(j) > 0:
                    lyap_sum += np.log(np.linalg.norm(j))

            T = len(jacobians) * dt
            return float(lyap_sum / T) if T > 0 else np.nan

    except (np.linalg.LinAlgError, ValueError):
        return np.nan


def largest_lyapunov_exponent(
    signal: np.ndarray,
    dim: int = 3,
    tau: int = 1,
    min_neighbors: int = 5,
    max_iter: int = 500
) -> float:
    """
    Compute largest Lyapunov exponent from a time series.

    Uses the method of Rosenstein et al. (1993) which tracks
    divergence of nearby trajectories in reconstructed phase space.

    Parameters
    ----------
    signal : np.ndarray
        Input time series
    dim : int
        Embedding dimension
    tau : int
        Time delay for embedding
    min_neighbors : int
        Minimum number of neighbors to track
    max_iter : int
        Maximum iterations for divergence tracking

    Returns
    -------
    float
        Largest Lyapunov exponent
    """
    signal = np.asarray(signal).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if n < (dim - 1) * tau + max_iter + 10:
        return np.nan

    # Construct delay embedding
    n_vectors = n - (dim - 1) * tau
    embedding = np.zeros((n_vectors, dim))

    for i in range(dim):
        embedding[:, i] = signal[i * tau:i * tau + n_vectors]

    # For each point, find nearest neighbor (excluding temporal neighbors)
    divergences = []
    min_time_sep = dim * tau  # Minimum temporal separation

    for i in range(n_vectors - max_iter):
        # Find nearest neighbor
        min_dist = np.inf
        nn_idx = -1

        for j in range(n_vectors - max_iter):
            if abs(i - j) < min_time_sep:
                continue

            dist = np.linalg.norm(embedding[i] - embedding[j])
            if dist < min_dist and dist > 0:
                min_dist = dist
                nn_idx = j

        if nn_idx >= 0 and min_dist < np.inf:
            # Track divergence over time
            for k in range(1, min(max_iter, n_vectors - max(i, nn_idx))):
                new_dist = np.linalg.norm(embedding[i + k] - embedding[nn_idx + k])
                if new_dist > 0 and min_dist > 0:
                    if len(divergences) <= k:
                        divergences.append([])
                    divergences[k - 1].append(np.log(new_dist / min_dist))

    if not divergences:
        return np.nan

    # Average divergence at each time step
    avg_divergence = []
    for k, divs in enumerate(divergences):
        if divs:
            avg_divergence.append((k + 1, np.mean(divs)))

    if len(avg_divergence) < 5:
        return np.nan

    # Linear regression to find slope (= largest Lyapunov exponent)
    times = np.array([t for t, _ in avg_divergence])
    log_divs = np.array([d for _, d in avg_divergence])

    # Use only linear region (first quarter typically)
    n_fit = max(5, len(times) // 4)
    times_fit = times[:n_fit]
    log_divs_fit = log_divs[:n_fit]

    # Simple linear regression
    x_mean = np.mean(times_fit)
    y_mean = np.mean(log_divs_fit)

    num = np.sum((times_fit - x_mean) * (log_divs_fit - y_mean))
    den = np.sum((times_fit - x_mean) ** 2)

    if den == 0:
        return np.nan

    return float(num / den)


def lyapunov_spectrum(
    trajectory: np.ndarray,
    dt: float = 1.0,
    n_exponents: Optional[int] = None
) -> np.ndarray:
    """
    Compute Lyapunov spectrum from a trajectory.

    The Lyapunov spectrum consists of all Lyapunov exponents,
    ordered from largest to smallest.

    Parameters
    ----------
    trajectory : np.ndarray
        State-space trajectory of shape (n_points, n_dims)
    dt : float
        Time step
    n_exponents : int, optional
        Number of exponents to compute (default: all)

    Returns
    -------
    np.ndarray
        Array of Lyapunov exponents (sorted descending)
    """
    trajectory = np.asarray(trajectory)

    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(-1, 1)

    n_points, n_dims = trajectory.shape

    if n_exponents is None:
        n_exponents = n_dims

    n_exponents = min(n_exponents, n_dims)

    if n_points < 10:
        return np.full(n_exponents, np.nan)

    # Approximate local Jacobians using finite differences
    lyap_sums = np.zeros(n_exponents)
    count = 0

    # Initialize Q matrix for QR decomposition
    Q = np.eye(n_dims)[:, :n_exponents]

    for i in range(1, n_points - 1):
        # Estimate local Jacobian
        dx_before = trajectory[i] - trajectory[i - 1]
        dx_after = trajectory[i + 1] - trajectory[i]

        # Simple Jacobian approximation
        if np.linalg.norm(dx_before) > 1e-10:
            J_approx = np.outer(dx_after, dx_before) / np.dot(dx_before, dx_before)
        else:
            continue

        # Propagate tangent vectors
        Q_new = J_approx @ Q

        # QR decomposition for re-orthogonalization
        try:
            Q, R = np.linalg.qr(Q_new)

            # Extract diagonal (stretching factors)
            for j in range(n_exponents):
                if j < R.shape[0] and j < R.shape[1]:
                    r_jj = abs(R[j, j])
                    if r_jj > 0:
                        lyap_sums[j] += np.log(r_jj)

            count += 1

        except np.linalg.LinAlgError:
            continue

    if count == 0:
        return np.full(n_exponents, np.nan)

    T = count * dt
    spectrum = lyap_sums / T

    # Sort descending
    return np.sort(spectrum)[::-1]


def lyapunov_rosenstein(
    signal: np.ndarray,
    dimension: int = None,
    delay: int = None,
    min_tsep: int = None,
    max_iter: int = None
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate largest Lyapunov exponent using Rosenstein's algorithm.

    Parameters
    ----------
    signal : np.ndarray
        1D time series.
    dimension : int, optional
        Embedding dimension (auto-detected if None).
    delay : int, optional
        Time delay (auto-detected if None).
    min_tsep : int, optional
        Minimum temporal separation for neighbors.
    max_iter : int, optional
        Maximum number of iterations.

    Returns
    -------
    tuple
        (lambda_max, divergence, iterations)
    """
    signal = np.asarray(signal, dtype=np.float64).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if n < 50:
        return np.nan, np.array([]), np.array([])

    if delay is None:
        delay = _auto_delay_acf(signal)
    if dimension is None:
        dimension = _auto_dim_simple(signal, delay)
    if min_tsep is None:
        min_tsep = delay * dimension
    if max_iter is None:
        max_iter = min(n // 10, 500)

    n_points = n - (dimension - 1) * delay
    if n_points < min_tsep + max_iter + 10:
        max_iter = max(10, n_points - min_tsep - 10)
        if max_iter < 10:
            return np.nan, np.array([]), np.array([])

    embedded = _embed_signal(signal, dimension, delay)
    n_pts = len(embedded)

    nn_indices = np.full(n_pts, -1, dtype=int)
    nn_dists = np.full(n_pts, np.inf)

    for i in range(n_pts):
        for j in range(n_pts):
            if abs(i - j) >= min_tsep:
                dist = np.linalg.norm(embedded[i] - embedded[j])
                if 0 < dist < nn_dists[i]:
                    nn_dists[i] = dist
                    nn_indices[i] = j

    divergence = np.zeros(max_iter)
    counts = np.zeros(max_iter)

    for i in range(n_pts - max_iter):
        j = nn_indices[i]
        if j < 0 or j >= n_pts - max_iter:
            continue
        for k in range(max_iter):
            dist = np.linalg.norm(embedded[i + k] - embedded[j + k])
            if dist > 0:
                divergence[k] += np.log(dist)
                counts[k] += 1

    valid = counts > 0
    divergence[valid] = divergence[valid] / counts[valid]
    divergence[~valid] = np.nan

    iterations = np.arange(max_iter)

    fit_end = max(10, max_iter // 5)
    fit_mask = np.isfinite(divergence[:fit_end])
    if np.sum(fit_mask) < 3:
        return np.nan, divergence, iterations

    x = iterations[:fit_end][fit_mask]
    y = divergence[:fit_end][fit_mask]
    slope, _ = np.polyfit(x, y, 1)

    return float(slope / delay), divergence, iterations


def lyapunov_kantz(
    signal: np.ndarray,
    dimension: int = None,
    delay: int = None,
    min_tsep: int = None,
    epsilon: float = None,
    max_iter: int = None
) -> Tuple[float, np.ndarray]:
    """
    Estimate largest Lyapunov exponent using Kantz's algorithm.

    Parameters
    ----------
    signal : np.ndarray
        1D time series.
    dimension : int, optional
        Embedding dimension.
    delay : int, optional
        Time delay.
    min_tsep : int, optional
        Minimum temporal separation.
    epsilon : float, optional
        Neighborhood radius.
    max_iter : int, optional
        Maximum iterations.

    Returns
    -------
    tuple
        (lambda_max, divergence)
    """
    signal = np.asarray(signal, dtype=np.float64).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if n < 50:
        return np.nan, np.array([])

    if delay is None:
        delay = _auto_delay_acf(signal)
    if dimension is None:
        dimension = _auto_dim_simple(signal, delay)
    if min_tsep is None:
        min_tsep = delay * dimension
    if max_iter is None:
        max_iter = min(n // 10, 500)

    embedded = _embed_signal(signal, dimension, delay)
    n_pts = len(embedded)

    if n_pts < min_tsep + max_iter + 10:
        max_iter = max(10, n_pts - min_tsep - 10)
        if max_iter < 10:
            return np.nan, np.array([])

    if epsilon is None:
        sample_idx = np.random.choice(n_pts, min(100, n_pts), replace=False)
        dists = []
        for i in sample_idx:
            for j in sample_idx:
                if i != j:
                    dists.append(np.linalg.norm(embedded[i] - embedded[j]))
        epsilon = np.percentile(dists, 10) if dists else 1.0

    divergence = np.zeros(max_iter)
    counts = np.zeros(max_iter)

    for i in range(n_pts - max_iter):
        neighbors = []
        for j in range(n_pts):
            if abs(i - j) >= min_tsep:
                dist = np.linalg.norm(embedded[i] - embedded[j])
                if 0 < dist < epsilon:
                    neighbors.append(j)

        if not neighbors:
            continue

        for k in range(max_iter):
            if i + k >= n_pts:
                break
            neighbor_dists = []
            for j in neighbors:
                if j + k < n_pts:
                    dist = np.linalg.norm(embedded[i + k] - embedded[j + k])
                    if dist > 0:
                        neighbor_dists.append(np.log(dist))
            if neighbor_dists:
                divergence[k] += np.mean(neighbor_dists)
                counts[k] += 1

    valid = counts > 0
    divergence[valid] = divergence[valid] / counts[valid]
    divergence[~valid] = np.nan

    fit_end = max(10, max_iter // 5)
    iterations = np.arange(max_iter)
    fit_mask = np.isfinite(divergence[:fit_end])
    if np.sum(fit_mask) < 3:
        return np.nan, divergence

    x = iterations[:fit_end][fit_mask]
    y = divergence[:fit_end][fit_mask]
    slope, _ = np.polyfit(x, y, 1)

    return float(slope / delay), divergence


def estimate_embedding_dim_cao(
    signal: np.ndarray,
    max_dim: int = 10,
    tau: int = None
) -> dict:
    """
    Cao's method for embedding dimension estimation.

    Parameters
    ----------
    signal : np.ndarray
        1D time series.
    max_dim : int
        Maximum embedding dimension to test.
    tau : int, optional
        Time delay (auto-detected if None).

    Returns
    -------
    dict
        optimal_dim, is_deterministic, E1_values, E2_values, E1_ratio, confidence.
    """
    signal = np.asarray(signal, dtype=np.float64).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if n < 50:
        return {'optimal_dim': 3, 'is_deterministic': None,
                'E1_values': None, 'E2_values': None,
                'E1_ratio': None, 'confidence': 0.0}

    if tau is None:
        tau = _auto_delay_acf(signal)

    E1 = np.zeros(max_dim)
    E2 = np.zeros(max_dim)

    for d in range(1, max_dim + 1):
        n_points_d = n - (d - 1) * tau
        n_points_d1 = n - d * tau

        if n_points_d < 10 or n_points_d1 < 10:
            E1[d - 1] = np.nan
            E2[d - 1] = np.nan
            continue

        embedded_d = _embed_signal(signal, d, tau)
        embedded_d1 = _embed_signal(signal, d + 1, tau)
        N = min(len(embedded_d), len(embedded_d1))

        if N < 10:
            E1[d - 1] = np.nan
            E2[d - 1] = np.nan
            continue

        a_values = []
        a_star_values = []
        sample_size = min(300, N)
        sample_idx = np.random.choice(N, sample_size, replace=False)

        for i in sample_idx:
            min_dist = np.inf
            nn_idx = -1
            for j in sample_idx:
                if i == j:
                    continue
                dist = np.max(np.abs(embedded_d[i] - embedded_d[j]))
                if 0 < dist < min_dist:
                    min_dist = dist
                    nn_idx = j

            if nn_idx < 0 or min_dist == 0:
                continue

            dist_d1 = np.max(np.abs(embedded_d1[i] - embedded_d1[nn_idx]))
            a_values.append(dist_d1 / (min_dist + 1e-12))

            idx_d1 = min(i + d * tau, n - 1)
            idx_d1_nn = min(nn_idx + d * tau, n - 1)
            a_star_values.append(abs(signal[idx_d1] - signal[idx_d1_nn]))

        E1[d - 1] = np.mean(a_values) if a_values else np.nan
        E2[d - 1] = np.mean(a_star_values) if a_star_values else np.nan

    E1_ratio = np.zeros(max_dim - 1)
    for d in range(1, max_dim):
        if E1[d - 1] > 1e-10:
            E1_ratio[d - 1] = E1[d] / E1[d - 1]
        else:
            E1_ratio[d - 1] = np.nan

    threshold = 0.95
    optimal_dim = max_dim
    for d in range(len(E1_ratio)):
        if np.isfinite(E1_ratio[d]) and E1_ratio[d] > threshold:
            optimal_dim = d + 1
            break

    E2_valid = E2[np.isfinite(E2)]
    if len(E2_valid) >= 2:
        E2_std = np.std(E2_valid)
        E2_mean = np.mean(E2_valid)
        is_deterministic = E2_std / (E2_mean + 1e-12) > 0.1
    else:
        is_deterministic = None

    valid_ratios = E1_ratio[np.isfinite(E1_ratio)]
    if len(valid_ratios) >= 2:
        confidence = min(1.0, 1.0 / (np.std(valid_ratios) + 0.1))
    else:
        confidence = 0.5

    return {
        'optimal_dim': optimal_dim,
        'is_deterministic': is_deterministic,
        'E1_values': E1,
        'E2_values': E2,
        'E1_ratio': E1_ratio,
        'confidence': float(confidence),
    }


def estimate_tau_ami(
    signal: np.ndarray,
    max_tau: int = 50,
    n_bins: int = 64
) -> int:
    """
    Estimate embedding delay using Average Mutual Information.

    Parameters
    ----------
    signal : np.ndarray
        1D time series.
    max_tau : int
        Maximum delay to test.
    n_bins : int
        Number of bins for histogram.

    Returns
    -------
    int
        Optimal embedding delay.
    """
    signal = np.asarray(signal, dtype=np.float64).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if n < max_tau + 10:
        return 1

    ami_values = []
    for tau in range(1, min(max_tau + 1, n // 2)):
        x = signal[:-tau]
        y = signal[tau:]
        try:
            hist_xy, _, _ = np.histogram2d(x, y, bins=n_bins)
            hist_x, _ = np.histogram(x, bins=n_bins)
            hist_y, _ = np.histogram(y, bins=n_bins)
        except Exception:
            ami_values.append(np.nan)
            continue

        p_xy = hist_xy / hist_xy.sum()
        p_x = hist_x / hist_x.sum()
        p_y = hist_y / hist_y.sum()

        mi = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                    mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
        ami_values.append(mi)

    if not ami_values:
        return 1

    ami_arr = np.array(ami_values)
    for i in range(1, len(ami_arr) - 1):
        if np.isfinite(ami_arr[i]):
            if ami_arr[i] < ami_arr[i - 1] and ami_arr[i] <= ami_arr[i + 1]:
                return i + 1

    if np.isfinite(ami_arr[0]):
        threshold = ami_arr[0] / np.e
        below = np.where(ami_arr < threshold)[0]
        if len(below) > 0:
            return int(below[0]) + 1

    return max_tau // 4


def ftle_local_linearization(
    trajectory: np.ndarray,
    time_horizon: int = 10,
    n_neighbors: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute FTLE via local linearization at each point.

    Parameters
    ----------
    trajectory : np.ndarray
        State-space trajectory (n_points, n_dims).
    time_horizon : int
        Time steps for measuring stretching.
    n_neighbors : int, optional
        Neighbors for local estimation.

    Returns
    -------
    tuple
        (ftle_values, valid_indices)
    """
    trajectory = np.asarray(trajectory)
    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(-1, 1)
    n_points, dim = trajectory.shape

    if n_neighbors is None:
        n_neighbors = 2 * dim + 1

    ftle_values = np.full(n_points, np.nan)

    for i in range(n_points - time_horizon):
        dists = np.linalg.norm(trajectory - trajectory[i], axis=1)
        indices = np.argsort(dists)[1:n_neighbors + 1]
        valid = [j for j in indices if j + time_horizon < n_points]

        if len(valid) < dim + 1:
            continue

        delta_x0 = np.array([trajectory[j] - trajectory[i] for j in valid])
        delta_xT = np.array([trajectory[j + time_horizon] - trajectory[i + time_horizon]
                            for j in valid])

        try:
            Phi_T, _, _, _ = np.linalg.lstsq(delta_x0, delta_xT, rcond=None)
            Phi = Phi_T.T
            C = Phi.T @ Phi
            eigenvalues = np.linalg.eigvalsh(C)
            max_eig = np.max(eigenvalues)
            if max_eig > 0:
                ftle_values[i] = np.log(np.sqrt(max_eig)) / time_horizon
        except (np.linalg.LinAlgError, ValueError):
            continue

    valid_idx = np.where(np.isfinite(ftle_values))[0]
    return ftle_values, valid_idx


def ftle_direct_perturbation(
    signal: np.ndarray,
    dimension: int = 3,
    delay: int = 1,
    time_horizon: int = 10,
    n_neighbors: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute FTLE from scalar time series via embedding + perturbation.

    Parameters
    ----------
    signal : np.ndarray
        1D time series.
    dimension : int
        Embedding dimension.
    delay : int
        Time delay.
    time_horizon : int
        Time horizon for stretching.
    n_neighbors : int, optional
        Neighbors for estimation.

    Returns
    -------
    tuple
        (ftle_values, valid_indices)
    """
    signal = np.asarray(signal, dtype=np.float64).flatten()
    signal = signal[~np.isnan(signal)]

    embedded = _embed_signal(signal, dimension, delay)
    return ftle_local_linearization(embedded, time_horizon, n_neighbors)


# --- Helper functions for new Lyapunov methods ---

def _embed_signal(signal: np.ndarray, dimension: int, delay: int) -> np.ndarray:
    """Time delay embedding."""
    n = len(signal)
    n_points = n - (dimension - 1) * delay
    if n_points < 1:
        return np.zeros((0, dimension))
    embedded = np.zeros((n_points, dimension))
    for d in range(dimension):
        embedded[:, d] = signal[d * delay:d * delay + n_points]
    return embedded


def _auto_delay_acf(signal: np.ndarray) -> int:
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


def _auto_dim_simple(signal: np.ndarray, delay: int) -> int:
    """Simple auto-detect of embedding dimension."""
    n = len(signal)
    for dim in range(2, min(11, n // (3 * delay))):
        n_pts = n - (dim - 1) * delay
        if n_pts < 50:
            return max(2, dim - 1)
    return 3
