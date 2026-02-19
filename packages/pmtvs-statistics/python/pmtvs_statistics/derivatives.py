"""Numerical derivative primitives for signal analysis."""

import numpy as np
from typing import Optional


def first_derivative(
    values: np.ndarray,
    dt: float = 1.0,
    method: str = 'central'
) -> np.ndarray:
    """
    Compute first derivative (rate of change).

    Parameters
    ----------
    values : np.ndarray
        Input time series.
    dt : float
        Time step.
    method : str
        'forward', 'backward', or 'central' difference.

    Returns
    -------
    np.ndarray
        First derivative array (same length as input).
    """
    values = np.asarray(values, dtype=np.float64).flatten()

    if len(values) < 2:
        return np.array([np.nan])

    if method == 'forward':
        deriv = np.diff(values) / dt
        deriv = np.append(deriv, deriv[-1])
    elif method == 'backward':
        deriv = np.diff(values) / dt
        deriv = np.insert(deriv, 0, deriv[0])
    elif method == 'central':
        deriv = np.gradient(values, dt)
    else:
        raise ValueError(f"Unknown method: {method}")

    return deriv


def second_derivative(
    values: np.ndarray,
    dt: float = 1.0,
    method: str = 'central'
) -> np.ndarray:
    """
    Compute second derivative (acceleration/curvature).

    Parameters
    ----------
    values : np.ndarray
        Input time series.
    dt : float
        Time step.
    method : str
        'central' or 'finite_difference'.

    Returns
    -------
    np.ndarray
        Second derivative array (same length as input).
    """
    values = np.asarray(values, dtype=np.float64).flatten()

    if len(values) < 3:
        return np.full(len(values), np.nan)

    if method == 'central':
        deriv = np.gradient(np.gradient(values, dt), dt)
    elif method == 'finite_difference':
        n = len(values)
        deriv = np.zeros(n)
        for i in range(1, n - 1):
            deriv[i] = (values[i + 1] - 2 * values[i] + values[i - 1]) / (dt ** 2)
        deriv[0] = deriv[1]
        deriv[-1] = deriv[-2]
    else:
        raise ValueError(f"Unknown method: {method}")

    return deriv


def gradient(
    values: np.ndarray,
    dt: float = 1.0
) -> np.ndarray:
    """
    Compute gradient using numpy's gradient (handles edges appropriately).

    Parameters
    ----------
    values : np.ndarray
        Input array.
    dt : float
        Spacing.

    Returns
    -------
    np.ndarray
        Gradient array.
    """
    values = np.asarray(values, dtype=np.float64).flatten()
    if len(values) < 2:
        return np.array([np.nan])
    return np.gradient(values, dt)


def laplacian(
    values: np.ndarray,
    dt: float = 1.0
) -> np.ndarray:
    """
    Compute Laplacian (second spatial derivative).

    Parameters
    ----------
    values : np.ndarray
        Input array (1D or 2D).
    dt : float
        Spacing.

    Returns
    -------
    np.ndarray
        Laplacian array.
    """
    values = np.asarray(values, dtype=np.float64)

    if values.ndim == 1:
        return second_derivative(values.flatten(), dt)

    if values.ndim == 2:
        lap_x = np.zeros_like(values)
        lap_y = np.zeros_like(values)
        lap_x[:, 1:-1] = (values[:, 2:] - 2 * values[:, 1:-1] + values[:, :-2]) / (dt ** 2)
        lap_x[:, 0] = lap_x[:, 1]
        lap_x[:, -1] = lap_x[:, -2]
        lap_y[1:-1, :] = (values[2:, :] - 2 * values[1:-1, :] + values[:-2, :]) / (dt ** 2)
        lap_y[0, :] = lap_y[1, :]
        lap_y[-1, :] = lap_y[-2, :]
        return lap_x + lap_y

    raise ValueError(f"Laplacian not implemented for {values.ndim}D arrays")


def finite_difference(
    values: np.ndarray,
    order: int = 1,
    dt: float = 1.0
) -> np.ndarray:
    """
    Compute finite difference of specified order.

    Parameters
    ----------
    values : np.ndarray
        Input array.
    order : int
        Derivative order (1, 2, 3, ...).
    dt : float
        Time step.

    Returns
    -------
    np.ndarray
        Finite difference array (same length as input).
    """
    values = np.asarray(values, dtype=np.float64).flatten()
    result = values.copy()

    for _ in range(order):
        if len(result) < 2:
            return np.array([np.nan])
        result = np.diff(result) / dt
        result = np.append(result, result[-1])

    return result


def velocity(
    values: np.ndarray,
    dt: float = 1.0
) -> np.ndarray:
    """
    Compute velocity (first time derivative via central differences).

    Parameters
    ----------
    values : np.ndarray
        Position time series.
    dt : float
        Time step.

    Returns
    -------
    np.ndarray
        Velocity array.
    """
    return first_derivative(values, dt, method='central')


def acceleration(
    values: np.ndarray,
    dt: float = 1.0
) -> np.ndarray:
    """
    Compute acceleration (second time derivative via central differences).

    Parameters
    ----------
    values : np.ndarray
        Position time series.
    dt : float
        Time step.

    Returns
    -------
    np.ndarray
        Acceleration array.
    """
    return second_derivative(values, dt, method='central')


def jerk(
    values: np.ndarray,
    dt: float = 1.0
) -> np.ndarray:
    """
    Compute jerk (third time derivative).

    Parameters
    ----------
    values : np.ndarray
        Position time series.
    dt : float
        Time step.

    Returns
    -------
    np.ndarray
        Jerk array.
    """
    values = np.asarray(values, dtype=np.float64).flatten()
    if len(values) < 4:
        return np.full(len(values), np.nan)
    return np.gradient(np.gradient(np.gradient(values, dt), dt), dt)


def smoothed_derivative(
    values: np.ndarray,
    dt: float = 1.0,
    window: int = 5,
    order: int = 1
) -> np.ndarray:
    """
    Compute smoothed derivative using Savitzky-Golay filter.

    Parameters
    ----------
    values : np.ndarray
        Input time series.
    dt : float
        Time step.
    window : int
        Window length for smoothing (must be odd).
    order : int
        Derivative order.

    Returns
    -------
    np.ndarray
        Smoothed derivative.
    """
    from scipy.signal import savgol_filter

    values = np.asarray(values, dtype=np.float64).flatten()

    if window % 2 == 0:
        window += 1

    if len(values) < window:
        return np.full(len(values), np.nan)

    polyorder = min(3, window - 1)
    return savgol_filter(values, window, polyorder, deriv=order, delta=dt)
