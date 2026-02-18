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
