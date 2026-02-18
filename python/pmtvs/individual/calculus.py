"""
Calculus Primitives

Derivative and integral operations.
"""

import numpy as np


def derivative(
    signal: np.ndarray,
    dt: float = 1.0,
    order: int = 1
) -> np.ndarray:
    """
    Compute numerical derivative.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    dt : float
        Time step (sampling interval)
    order : int
        Order of derivative (1, 2, or 3)

    Returns
    -------
    np.ndarray
        Derivative signal (uses central differences)
    """
    signal = np.asarray(signal)
    result = signal.copy()

    for _ in range(order):
        result = np.gradient(result, dt)

    return result


def integral(
    signal: np.ndarray,
    dt: float = 1.0,
    initial: float = 0.0
) -> np.ndarray:
    """
    Compute numerical integral (cumulative).

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    dt : float
        Time step (sampling interval)
    initial : float
        Initial value

    Returns
    -------
    np.ndarray
        Integrated signal (trapezoidal rule)
    """
    signal = np.asarray(signal)

    # Trapezoidal integration
    result = np.zeros_like(signal)
    result[0] = initial

    for i in range(1, len(signal)):
        result[i] = result[i-1] + (signal[i-1] + signal[i]) * dt / 2

    return result


def curvature(
    signal: np.ndarray,
    dt: float = 1.0
) -> np.ndarray:
    """
    Compute curvature (second derivative normalized).

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    dt : float
        Time step

    Returns
    -------
    np.ndarray
        Curvature: |d²y/dt²| / (1 + (dy/dt)²)^(3/2)
    """
    dy = derivative(signal, dt, order=1)
    d2y = derivative(signal, dt, order=2)

    denominator = (1 + dy**2) ** 1.5
    denominator[denominator == 0] = np.finfo(float).eps

    return np.abs(d2y) / denominator
