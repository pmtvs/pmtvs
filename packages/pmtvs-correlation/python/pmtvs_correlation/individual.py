"""
Individual Correlation Primitives

Single-signal correlation measures: autocorrelation, PACF.
"""

import numpy as np
from typing import Optional

from pmtvs_correlation._dispatch import use_rust


def autocorrelation(signal: np.ndarray, lag: int = 1) -> float:
    """
    Compute autocorrelation at specified lag.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    lag : int
        Time lag

    Returns
    -------
    float
        Autocorrelation at lag
    """
    if use_rust('autocorrelation'):
        from pmtvs_correlation import _get_rust
        rust_fn = _get_rust('autocorrelation')
        if rust_fn is not None:
            return rust_fn(signal, lag)

    signal = np.asarray(signal).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if n < lag + 2:
        return np.nan

    # Pearson correlation between signal[:-lag] and signal[lag:]
    x = signal[:-lag]
    y = signal[lag:]

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2))

    if den == 0:
        return np.nan

    return float(num / den)


def partial_autocorrelation(signal: np.ndarray, max_lag: int = 10) -> np.ndarray:
    """
    Compute partial autocorrelation function (PACF).

    Uses Durbin-Levinson recursion.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    max_lag : int
        Maximum lag

    Returns
    -------
    np.ndarray
        PACF values for lags 0 to max_lag
    """
    if use_rust('partial_autocorrelation'):
        from pmtvs_correlation import _get_rust
        rust_fn = _get_rust('partial_autocorrelation')
        if rust_fn is not None:
            return rust_fn(signal, max_lag)

    signal = np.asarray(signal).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if n < max_lag + 2:
        return np.full(max_lag + 1, np.nan)

    # Compute autocorrelations
    acf = np.zeros(max_lag + 1)
    acf[0] = 1.0
    for k in range(1, max_lag + 1):
        acf[k] = autocorrelation(signal, k)

    # Durbin-Levinson recursion
    pacf = np.zeros(max_lag + 1)
    pacf[0] = 1.0

    phi = np.zeros((max_lag + 1, max_lag + 1))

    for k in range(1, max_lag + 1):
        if k == 1:
            phi[1, 1] = acf[1]
        else:
            num = acf[k] - np.sum(phi[k-1, 1:k] * acf[k-1:0:-1])
            den = 1.0 - np.sum(phi[k-1, 1:k] * acf[1:k])

            if abs(den) < 1e-10:
                phi[k, k] = 0.0
            else:
                phi[k, k] = num / den

            for j in range(1, k):
                phi[k, j] = phi[k-1, j] - phi[k, k] * phi[k-1, k-j]

        pacf[k] = phi[k, k]

    return pacf


def autocorrelation_function(signal: np.ndarray, max_lag: Optional[int] = None) -> np.ndarray:
    """
    Compute full autocorrelation function.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    max_lag : int, optional
        Maximum lag (default: len(signal) // 2)

    Returns
    -------
    np.ndarray
        ACF values for lags 0 to max_lag
    """
    signal = np.asarray(signal).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if n < 3:
        return np.array([np.nan])

    if max_lag is None:
        max_lag = n // 2

    max_lag = min(max_lag, n - 2)

    acf = np.zeros(max_lag + 1)
    acf[0] = 1.0

    for lag in range(1, max_lag + 1):
        acf[lag] = autocorrelation(signal, lag)

    return acf


def acf_decay_time(signal: np.ndarray, threshold: float = 1.0 / np.e) -> float:
    """
    Compute ACF decay time (lag where ACF drops below threshold).

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    threshold : float
        Threshold value (default: 1/e)

    Returns
    -------
    float
        Decay time (in samples)
    """
    acf = autocorrelation_function(signal)

    if len(acf) == 1 and np.isnan(acf[0]):
        return np.nan

    below = np.where(acf < threshold)[0]
    if len(below) == 0:
        return float(len(acf) - 1)

    return float(below[0])
