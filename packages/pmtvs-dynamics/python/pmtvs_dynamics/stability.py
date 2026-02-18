"""
Stability Analysis

Functions for analyzing stability properties of dynamical systems:
fixed points, stability indices, bifurcations.
"""

import numpy as np
from typing import Optional, Tuple, List


def fixed_point_detection(
    trajectory: np.ndarray,
    threshold: float = 0.01,
    min_duration: int = 10
) -> List[Tuple[int, int, np.ndarray]]:
    """
    Detect fixed points (stationary regions) in a trajectory.

    Parameters
    ----------
    trajectory : np.ndarray
        State-space trajectory of shape (n_points, n_dims)
    threshold : float
        Maximum velocity to consider as "fixed"
    min_duration : int
        Minimum number of points in a fixed region

    Returns
    -------
    list
        List of (start_idx, end_idx, center_point) tuples
    """
    trajectory = np.asarray(trajectory)

    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(-1, 1)

    n_points = len(trajectory)

    if n_points < min_duration:
        return []

    # Compute velocities
    velocities = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)

    # Find regions where velocity is below threshold
    fixed_regions = []
    in_fixed_region = False
    region_start = 0

    for i, v in enumerate(velocities):
        if v < threshold:
            if not in_fixed_region:
                region_start = i
                in_fixed_region = True
        else:
            if in_fixed_region:
                if i - region_start >= min_duration:
                    center = np.mean(trajectory[region_start:i], axis=0)
                    fixed_regions.append((region_start, i, center))
                in_fixed_region = False

    # Check final region
    if in_fixed_region and n_points - region_start >= min_duration:
        center = np.mean(trajectory[region_start:], axis=0)
        fixed_regions.append((region_start, n_points, center))

    return fixed_regions


def stability_index(
    trajectory: np.ndarray,
    dt: float = 1.0
) -> float:
    """
    Compute local stability index.

    Measures how quickly perturbations decay or grow.
    Negative values indicate stable (convergent) dynamics.
    Positive values indicate unstable (divergent) dynamics.

    Parameters
    ----------
    trajectory : np.ndarray
        State-space trajectory
    dt : float
        Time step

    Returns
    -------
    float
        Stability index
    """
    trajectory = np.asarray(trajectory)

    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(-1, 1)

    n_points = len(trajectory)

    if n_points < 10:
        return np.nan

    # Compute distance from centroid over time
    centroid = np.mean(trajectory, axis=0)
    distances = np.linalg.norm(trajectory - centroid, axis=1)

    if np.all(distances < 1e-10):
        return -np.inf  # Fixed point

    # Fit exponential decay/growth
    # distance(t) ~ exp(lambda * t)
    # log(distance) ~ lambda * t

    valid_mask = distances > 1e-10
    if np.sum(valid_mask) < 5:
        return np.nan

    log_dist = np.log(distances[valid_mask])
    times = np.arange(len(distances))[valid_mask] * dt

    # Linear regression
    x_mean = np.mean(times)
    y_mean = np.mean(log_dist)

    num = np.sum((times - x_mean) * (log_dist - y_mean))
    den = np.sum((times - x_mean) ** 2)

    if den == 0:
        return 0.0

    return float(num / den)


def jacobian_eigenvalues(
    trajectory: np.ndarray,
    at_index: int = -1,
    neighborhood_size: int = 10
) -> np.ndarray:
    """
    Estimate Jacobian eigenvalues at a point in the trajectory.

    Uses local linear approximation to estimate the Jacobian
    of the flow map.

    Parameters
    ----------
    trajectory : np.ndarray
        State-space trajectory
    at_index : int
        Index at which to estimate (default: last point)
    neighborhood_size : int
        Number of points to use for local approximation

    Returns
    -------
    np.ndarray
        Eigenvalues of estimated Jacobian (complex)
    """
    trajectory = np.asarray(trajectory)

    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(-1, 1)

    n_points, n_dims = trajectory.shape

    if at_index < 0:
        at_index = n_points + at_index

    if n_points < neighborhood_size + 2:
        return np.full(n_dims, np.nan + 0j)

    # Get neighborhood
    start = max(0, at_index - neighborhood_size // 2)
    end = min(n_points - 1, start + neighborhood_size)

    if end - start < n_dims + 1:
        return np.full(n_dims, np.nan + 0j)

    # Current and next states
    X = trajectory[start:end]
    Y = trajectory[start + 1:end + 1]

    # Center data
    X_mean = np.mean(X, axis=0)
    Y_mean = np.mean(Y, axis=0)

    X_centered = X - X_mean
    Y_centered = Y - Y_mean

    # Solve for Jacobian: Y = J @ X (in least squares sense)
    try:
        J, residuals, rank, s = np.linalg.lstsq(X_centered, Y_centered, rcond=None)
        eigenvalues = np.linalg.eigvals(J.T)
        return eigenvalues
    except np.linalg.LinAlgError:
        return np.full(n_dims, np.nan + 0j)


def bifurcation_indicator(
    signal: np.ndarray,
    window_size: int = 100,
    overlap: float = 0.5
) -> np.ndarray:
    """
    Compute bifurcation indicator over time.

    Detects changes in dynamical behavior that may indicate
    bifurcations (qualitative changes in dynamics).

    Parameters
    ----------
    signal : np.ndarray
        Input time series
    window_size : int
        Size of analysis windows
    overlap : float
        Fraction of overlap between windows

    Returns
    -------
    np.ndarray
        Bifurcation indicator values for each window
    """
    signal = np.asarray(signal).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if n < window_size * 2:
        return np.array([np.nan])

    stride = int(window_size * (1 - overlap))
    n_windows = (n - window_size) // stride + 1

    indicators = []

    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        window = signal[start:end]

        # Compute multiple indicators
        # 1. Variance change
        var = np.var(window)

        # 2. Autocorrelation at lag 1
        if len(window) > 2:
            autocorr = np.corrcoef(window[:-1], window[1:])[0, 1]
        else:
            autocorr = 0

        # 3. Number of local extrema
        diff = np.diff(window)
        extrema = np.sum(diff[:-1] * diff[1:] < 0)
        extrema_rate = extrema / len(window)

        # Combine into single indicator
        # High indicator = potential bifurcation
        indicator = var * (1 + abs(autocorr)) * (1 + extrema_rate)
        indicators.append(indicator)

    return np.array(indicators)


def phase_space_contraction(
    trajectory: np.ndarray,
    dt: float = 1.0
) -> float:
    """
    Compute phase space contraction rate.

    For dissipative systems, this is related to the sum of
    Lyapunov exponents and measures information loss.

    Parameters
    ----------
    trajectory : np.ndarray
        State-space trajectory
    dt : float
        Time step

    Returns
    -------
    float
        Contraction rate (negative = contraction, positive = expansion)
    """
    trajectory = np.asarray(trajectory)

    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(-1, 1)

    n_points, n_dims = trajectory.shape

    if n_points < 10:
        return np.nan

    # Estimate local divergence of flow
    # div(v) = trace(Jacobian)

    divergences = []

    for i in range(1, n_points - 1):
        # Estimate velocity at point i
        v_before = trajectory[i] - trajectory[i - 1]
        v_after = trajectory[i + 1] - trajectory[i]

        # Estimate Jacobian trace (divergence)
        # Using central difference
        if n_dims == 1:
            if abs(v_before[0]) > 1e-10:
                div = (v_after[0] - v_before[0]) / (2 * dt * v_before[0])
                divergences.append(div)
        else:
            # For multi-dimensional, compute trace of velocity gradient
            dv = v_after - v_before
            dx = trajectory[i] - trajectory[i - 1]

            if np.linalg.norm(dx) > 1e-10:
                # Approximate trace
                div = np.dot(dv, dx) / (np.dot(dx, dx) * 2 * dt)
                divergences.append(div)

    if len(divergences) == 0:
        return np.nan

    return float(np.mean(divergences))
