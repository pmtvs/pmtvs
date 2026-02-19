"""
Pairwise Regression Primitives

Linear regression, ratio, product, difference, sum.
Pure numpy — no scipy dependency.
"""

import numpy as np
from typing import Tuple


def linear_regression(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    Compute linear regression: sig_b = slope * sig_a + intercept.

    Parameters
    ----------
    sig_a : np.ndarray
        Independent variable (x).
    sig_b : np.ndarray
        Dependent variable (y).

    Returns
    -------
    tuple of float
        (slope, intercept, r_squared, std_error)

    Notes
    -----
    Uses ordinary least squares implemented with pure numpy.
    std_error is the standard error of the slope estimate.
    Returns (nan, nan, nan, nan) when fewer than 3 valid points.
    """
    sig_a = np.asarray(sig_a, dtype=np.float64).flatten()
    sig_b = np.asarray(sig_b, dtype=np.float64).flatten()

    n = min(len(sig_a), len(sig_b))
    sig_a, sig_b = sig_a[:n], sig_b[:n]

    # Filter NaN values
    mask = ~(np.isnan(sig_a) | np.isnan(sig_b))
    sig_a, sig_b = sig_a[mask], sig_b[mask]

    n = len(sig_a)
    if n < 3:
        return (float("nan"), float("nan"), float("nan"), float("nan"))

    # OLS via normal equations
    mean_a = np.mean(sig_a)
    mean_b = np.mean(sig_b)

    ss_xx = np.sum((sig_a - mean_a) ** 2)
    ss_xy = np.sum((sig_a - mean_a) * (sig_b - mean_b))
    ss_yy = np.sum((sig_b - mean_b) ** 2)

    if ss_xx == 0.0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))

    slope = ss_xy / ss_xx
    intercept = mean_b - slope * mean_a

    # R-squared
    if ss_yy == 0.0:
        r_squared = 1.0 if ss_xy == 0.0 else 0.0
    else:
        r_squared = (ss_xy ** 2) / (ss_xx * ss_yy)

    # Standard error of slope
    residuals = sig_b - (slope * sig_a + intercept)
    ss_res = np.sum(residuals ** 2)
    mse = ss_res / (n - 2)
    std_error = float(np.sqrt(mse / ss_xx))

    return (float(slope), float(intercept), float(r_squared), float(std_error))


def ratio(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """
    Compute element-wise ratio sig_a / sig_b.

    Parameters
    ----------
    sig_a, sig_b : np.ndarray
        Input signals.
    epsilon : float
        Small value to prevent division by zero.

    Returns
    -------
    np.ndarray
        Ratio signal.

    Notes
    -----
    Where |sig_b| < epsilon, sig_b is replaced by epsilon * sign(sig_b)
    to avoid division by zero while preserving sign.
    """
    sig_a = np.asarray(sig_a, dtype=np.float64).flatten()
    sig_b = np.asarray(sig_b, dtype=np.float64).flatten()

    n = min(len(sig_a), len(sig_b))
    sig_a, sig_b = sig_a[:n], sig_b[:n]

    # Avoid division by zero
    sig_b_safe = np.where(
        np.abs(sig_b) < epsilon,
        epsilon * np.sign(sig_b + epsilon),
        sig_b,
    )

    return sig_a / sig_b_safe


def product(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
) -> np.ndarray:
    """
    Compute element-wise product sig_a * sig_b.

    Parameters
    ----------
    sig_a, sig_b : np.ndarray
        Input signals.

    Returns
    -------
    np.ndarray
        Product signal.

    Notes
    -----
    Useful for power = force * velocity, etc.
    """
    sig_a = np.asarray(sig_a, dtype=np.float64).flatten()
    sig_b = np.asarray(sig_b, dtype=np.float64).flatten()

    n = min(len(sig_a), len(sig_b))
    return sig_a[:n] * sig_b[:n]


def difference(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
) -> np.ndarray:
    """
    Compute element-wise difference sig_a - sig_b.

    Parameters
    ----------
    sig_a, sig_b : np.ndarray
        Input signals.

    Returns
    -------
    np.ndarray
        Difference signal.

    Notes
    -----
    Useful for error signals, residuals, imbalances.
    """
    sig_a = np.asarray(sig_a, dtype=np.float64).flatten()
    sig_b = np.asarray(sig_b, dtype=np.float64).flatten()

    n = min(len(sig_a), len(sig_b))
    return sig_a[:n] - sig_b[:n]


def sum_signals(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
) -> np.ndarray:
    """
    Compute element-wise sum sig_a + sig_b.

    Parameters
    ----------
    sig_a, sig_b : np.ndarray
        Input signals.

    Returns
    -------
    np.ndarray
        Sum signal.
    """
    sig_a = np.asarray(sig_a, dtype=np.float64).flatten()
    sig_b = np.asarray(sig_b, dtype=np.float64).flatten()

    n = min(len(sig_a), len(sig_b))
    return sig_a[:n] + sig_b[:n]
