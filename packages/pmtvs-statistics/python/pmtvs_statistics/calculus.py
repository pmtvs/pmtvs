"""
Calculus Primitives

Derivatives, integrals, and curvature for signal analysis.
"""

import numpy as np

from pmtvs_statistics._dispatch import use_rust


def derivative(signal: np.ndarray, order: int = 1, dt: float = 1.0) -> np.ndarray:
    """
    Compute numerical derivative of signal.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    order : int
        Derivative order (1 = first derivative, 2 = second, etc.)
    dt : float
        Time step between samples

    Returns
    -------
    np.ndarray
        Derivative (length = len(signal) - order)
    """
    if use_rust('derivative') and order == 1:
        from pmtvs_statistics import _get_rust
        rust_fn = _get_rust('derivative')
        if rust_fn is not None:
            return rust_fn(signal, dt)

    signal = np.asarray(signal).flatten()
    result = signal.copy()

    for _ in range(order):
        if len(result) < 2:
            return np.array([np.nan])
        result = np.diff(result) / dt

    return result


def integral(signal: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """
    Compute cumulative integral of signal (trapezoidal rule).

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    dt : float
        Time step between samples

    Returns
    -------
    np.ndarray
        Cumulative integral
    """
    if use_rust('integral'):
        from pmtvs_statistics import _get_rust
        rust_fn = _get_rust('integral')
        if rust_fn is not None:
            return rust_fn(signal, dt)

    signal = np.asarray(signal).flatten()
    if len(signal) < 2:
        return np.array([0.0])

    # Trapezoidal integration
    integral_vals = np.zeros(len(signal))
    for i in range(1, len(signal)):
        integral_vals[i] = integral_vals[i-1] + 0.5 * (signal[i] + signal[i-1]) * dt

    return integral_vals


def curvature(signal: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """
    Compute curvature of signal.

    Curvature = |y''| / (1 + y'^2)^(3/2)

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    dt : float
        Time step between samples

    Returns
    -------
    np.ndarray
        Curvature values
    """
    if use_rust('curvature'):
        from pmtvs_statistics import _get_rust
        rust_fn = _get_rust('curvature')
        if rust_fn is not None:
            return rust_fn(signal, dt)

    signal = np.asarray(signal).flatten()
    if len(signal) < 3:
        return np.array([np.nan])

    # First derivative (central difference)
    dy = np.zeros(len(signal))
    dy[1:-1] = (signal[2:] - signal[:-2]) / (2 * dt)
    dy[0] = (signal[1] - signal[0]) / dt
    dy[-1] = (signal[-1] - signal[-2]) / dt

    # Second derivative
    d2y = np.zeros(len(signal))
    d2y[1:-1] = (signal[2:] - 2 * signal[1:-1] + signal[:-2]) / (dt ** 2)
    d2y[0] = d2y[1]
    d2y[-1] = d2y[-2]

    # Curvature formula
    curvature = np.abs(d2y) / (1 + dy ** 2) ** 1.5

    return curvature


def rate_of_change(signal: np.ndarray, window: int = 1) -> np.ndarray:
    """
    Compute rate of change over a sliding window.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    window : int
        Window size for computing rate of change

    Returns
    -------
    np.ndarray
        Rate of change values
    """
    if use_rust('rate_of_change') and window == 1:
        from pmtvs_statistics import _get_rust
        rust_fn = _get_rust('rate_of_change')
        if rust_fn is not None:
            return rust_fn(signal)

    signal = np.asarray(signal).flatten()
    if len(signal) <= window:
        return np.array([np.nan])

    roc = np.zeros(len(signal) - window)
    for i in range(len(signal) - window):
        if signal[i] != 0:
            roc[i] = (signal[i + window] - signal[i]) / abs(signal[i])
        else:
            roc[i] = np.nan

    return roc
