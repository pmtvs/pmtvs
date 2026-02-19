"""Saddle point detection and basin stability analysis."""

import numpy as np
from typing import Dict, Any, List, Tuple


def estimate_jacobian_local(
    trajectory: np.ndarray,
    point_idx: int,
    n_neighbors: int = None,
) -> np.ndarray:
    """
    Estimate local Jacobian from trajectory using linear regression.

    Parameters
    ----------
    trajectory : np.ndarray
        Embedded trajectory (n_points, dimension).
    point_idx : int
        Index of point to estimate Jacobian at.
    n_neighbors : int, optional
        Number of neighbors for estimation.

    Returns
    -------
    np.ndarray
        Estimated Jacobian matrix (dim x dim).
    """
    trajectory = np.asarray(trajectory)
    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(-1, 1)
    n_points, dim = trajectory.shape

    if n_neighbors is None:
        n_neighbors = 2 * dim + 1

    if point_idx >= n_points - 1:
        return np.full((dim, dim), np.nan)

    # Find neighbors by distance
    dists = np.linalg.norm(trajectory - trajectory[point_idx], axis=1)
    indices = np.argsort(dists)[1:n_neighbors + 1]

    valid = [j for j in indices if j < n_points - 1]
    if len(valid) < dim + 1:
        return np.full((dim, dim), np.nan)

    X = np.array([trajectory[j] - trajectory[point_idx] for j in valid])
    Y = np.array([trajectory[j + 1] - trajectory[point_idx + 1] for j in valid])

    try:
        J_T, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        return J_T.T
    except np.linalg.LinAlgError:
        return np.full((dim, dim), np.nan)


def classify_jacobian_eigenvalues(
    jacobian: np.ndarray,
) -> Dict[str, Any]:
    """
    Classify fixed point type from Jacobian eigenvalues.

    Parameters
    ----------
    jacobian : np.ndarray
        Jacobian matrix.

    Returns
    -------
    dict
        Classification with eigenvalues, is_saddle, is_stable, stability_type.
    """
    jacobian = np.asarray(jacobian)
    empty = {
        'eigenvalues': np.array([]), 'eigenvalues_real': np.array([]),
        'eigenvalues_imag': np.array([]), 'n_positive': 0, 'n_negative': 0,
        'n_zero': 0, 'is_saddle': False, 'is_stable': False,
        'is_unstable': False, 'stability_type': 'unknown',
    }

    if np.any(np.isnan(jacobian)):
        return empty

    try:
        eigenvalues = np.linalg.eigvals(jacobian)
    except np.linalg.LinAlgError:
        return empty

    real_parts = np.real(eigenvalues)
    imag_parts = np.imag(eigenvalues)
    tol = 1e-6

    n_positive = int(np.sum(real_parts > tol))
    n_negative = int(np.sum(real_parts < -tol))
    n_zero = int(np.sum(np.abs(real_parts) <= tol))

    magnitudes = np.abs(eigenvalues)
    n_outside_unit = np.sum(magnitudes > 1 + tol)
    n_inside_unit = np.sum(magnitudes < 1 - tol)

    has_complex = np.any(np.abs(imag_parts) > tol)
    is_saddle = (n_positive > 0 and n_negative > 0) or (n_outside_unit > 0 and n_inside_unit > 0)
    is_stable = n_positive == 0 and n_outside_unit == 0
    is_unstable = n_positive > 0 or n_outside_unit > 0

    if is_saddle:
        stability_type = 'saddle'
    elif is_stable:
        stability_type = 'stable_focus' if has_complex else 'stable_node'
    elif n_positive > 0 and n_negative == 0:
        stability_type = 'unstable_focus' if has_complex else 'unstable_node'
    elif n_zero > 0:
        stability_type = 'center_manifold'
    else:
        stability_type = 'unknown'

    return {
        'eigenvalues': eigenvalues, 'eigenvalues_real': real_parts,
        'eigenvalues_imag': imag_parts, 'n_positive': n_positive,
        'n_negative': n_negative, 'n_zero': n_zero, 'is_saddle': is_saddle,
        'is_stable': is_stable, 'is_unstable': is_unstable,
        'stability_type': stability_type,
    }


def detect_saddle_points(
    trajectory: np.ndarray,
    velocity_threshold: float = 0.1,
    n_neighbors: int = None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Detect saddle point regions in a trajectory.

    Parameters
    ----------
    trajectory : np.ndarray
        Embedded trajectory (n_points, dimension).
    velocity_threshold : float
        Normalized velocity threshold for "near equilibrium".
    n_neighbors : int, optional
        Neighbors for Jacobian estimation.

    Returns
    -------
    tuple
        (saddle_score, velocity, saddle_info)
    """
    trajectory = np.asarray(trajectory)
    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(-1, 1)
    n_points, dim = trajectory.shape

    if n_neighbors is None:
        n_neighbors = 2 * dim + 1

    vel = np.full(n_points, np.nan)
    for i in range(n_points - 1):
        vel[i] = np.linalg.norm(trajectory[i + 1] - trajectory[i])

    max_vel = np.nanmax(vel)
    vel_norm = vel / max_vel if max_vel > 0 else np.zeros_like(vel)

    saddle_score = np.zeros(n_points)
    saddle_info = []

    for i in range(n_points - 1):
        info = {'point_idx': i}
        J = estimate_jacobian_local(trajectory, i, n_neighbors)
        eig_info = classify_jacobian_eigenvalues(J)
        info.update(eig_info)

        velocity_factor = 1.0 - vel_norm[i] if not np.isnan(vel_norm[i]) else 0
        if eig_info['n_positive'] > 0 and eig_info['n_negative'] > 0:
            eigenvalue_factor = 1.0
        elif eig_info['is_unstable'] and not eig_info['is_saddle']:
            eigenvalue_factor = 0.5
        else:
            eigenvalue_factor = 1.0 if eig_info['is_saddle'] else 0.0

        saddle_score[i] = velocity_factor * eigenvalue_factor
        saddle_info.append(info)

    saddle_info.append({'point_idx': n_points - 1, 'stability_type': 'unknown'})
    return saddle_score, vel, saddle_info


def compute_separatrix_distance(
    trajectory: np.ndarray,
    saddle_indices: np.ndarray,
    stable_direction: np.ndarray = None,
) -> np.ndarray:
    """
    Estimate distance to separatrix (stable manifold of saddle).

    Parameters
    ----------
    trajectory : np.ndarray
        Embedded trajectory.
    saddle_indices : np.ndarray
        Indices of saddle points.
    stable_direction : np.ndarray, optional
        Estimated stable eigenvector of saddle.

    Returns
    -------
    np.ndarray
        Estimated distance to separatrix at each point.
    """
    trajectory = np.asarray(trajectory)
    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(-1, 1)
    n_points = len(trajectory)
    saddle_indices = np.asarray(saddle_indices).flatten()

    if len(saddle_indices) == 0:
        return np.full(n_points, np.nan)

    saddle_points = trajectory[saddle_indices]
    distances = np.full(n_points, np.inf)
    for i in range(n_points):
        dists = np.linalg.norm(saddle_points - trajectory[i], axis=1)
        distances[i] = np.min(dists)

    return distances


def compute_basin_stability(
    trajectory: np.ndarray,
    saddle_score: np.ndarray,
    window: int = 50,
) -> np.ndarray:
    """
    Estimate basin stability from saddle proximity over time.

    Parameters
    ----------
    trajectory : np.ndarray
        Embedded trajectory.
    saddle_score : np.ndarray
        Saddle proximity score (from detect_saddle_points).
    window : int
        Rolling window for stability estimation.

    Returns
    -------
    np.ndarray
        Basin stability score (0-1, higher = more stable).
    """
    n_points = len(saddle_score)
    stability = np.full(n_points, np.nan)

    for i in range(window, n_points):
        window_scores = saddle_score[i - window:i]
        valid = ~np.isnan(window_scores)
        if np.sum(valid) > 0:
            stability[i] = 1.0 - np.mean(window_scores[valid])

    return stability
