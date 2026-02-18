"""
Statistics Primitives

Basic statistical measures for signal analysis.
"""

import numpy as np
from typing import Tuple

from pmtvs_statistics._dispatch import use_rust


def mean(signal: np.ndarray) -> float:
    """
    Compute mean of signal.

    Parameters
    ----------
    signal : np.ndarray
        Input signal

    Returns
    -------
    float
        Mean value
    """
    if use_rust('mean'):
        from pmtvs_statistics import _get_rust
        rust_fn = _get_rust('mean')
        if rust_fn is not None:
            return rust_fn(signal)

    signal = np.asarray(signal).flatten()
    signal = signal[~np.isnan(signal)]
    if len(signal) == 0:
        return np.nan
    return float(np.mean(signal))


def std(signal: np.ndarray, ddof: int = 1) -> float:
    """
    Compute standard deviation of signal.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    ddof : int
        Delta degrees of freedom (default: 1 for sample std)

    Returns
    -------
    float
        Standard deviation
    """
    if use_rust('std'):
        from pmtvs_statistics import _get_rust
        rust_fn = _get_rust('std')
        if rust_fn is not None:
            return rust_fn(signal, ddof)

    signal = np.asarray(signal).flatten()
    signal = signal[~np.isnan(signal)]
    if len(signal) < ddof + 1:
        return np.nan
    return float(np.std(signal, ddof=ddof))


def variance(signal: np.ndarray, ddof: int = 1) -> float:
    """
    Compute variance of signal.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    ddof : int
        Delta degrees of freedom (default: 1 for sample variance)

    Returns
    -------
    float
        Variance
    """
    if use_rust('variance'):
        from pmtvs_statistics import _get_rust
        rust_fn = _get_rust('variance')
        if rust_fn is not None:
            return rust_fn(signal, ddof)

    signal = np.asarray(signal).flatten()
    signal = signal[~np.isnan(signal)]
    if len(signal) < ddof + 1:
        return np.nan
    return float(np.var(signal, ddof=ddof))


def min_max(signal: np.ndarray) -> Tuple[float, float]:
    """
    Compute minimum and maximum of signal.

    Parameters
    ----------
    signal : np.ndarray
        Input signal

    Returns
    -------
    tuple
        (min, max)
    """
    if use_rust('min_max'):
        from pmtvs_statistics import _get_rust
        rust_fn = _get_rust('min_max')
        if rust_fn is not None:
            return rust_fn(signal)

    signal = np.asarray(signal).flatten()
    signal = signal[~np.isnan(signal)]
    if len(signal) == 0:
        return (np.nan, np.nan)
    return (float(np.min(signal)), float(np.max(signal)))


def percentiles(signal: np.ndarray, q: Tuple[float, ...] = (25, 50, 75)) -> Tuple[float, ...]:
    """
    Compute percentiles of signal.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    q : tuple
        Percentiles to compute (default: 25, 50, 75)

    Returns
    -------
    tuple
        Percentile values
    """
    signal = np.asarray(signal).flatten()
    signal = signal[~np.isnan(signal)]
    if len(signal) == 0:
        return tuple(np.nan for _ in q)
    return tuple(float(np.percentile(signal, p)) for p in q)


def skewness(signal: np.ndarray) -> float:
    """
    Compute skewness of signal.

    Parameters
    ----------
    signal : np.ndarray
        Input signal

    Returns
    -------
    float
        Skewness (0 = symmetric, >0 = right tail, <0 = left tail)
    """
    if use_rust('skewness'):
        from pmtvs_statistics import _get_rust
        rust_fn = _get_rust('skewness')
        if rust_fn is not None:
            return rust_fn(signal)

    signal = np.asarray(signal).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)
    if n < 3:
        return np.nan

    m = np.mean(signal)
    s = np.std(signal, ddof=0)
    if s == 0:
        return np.nan

    return float(np.mean(((signal - m) / s) ** 3))


def kurtosis(signal: np.ndarray, fisher: bool = True) -> float:
    """
    Compute kurtosis of signal.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    fisher : bool
        If True, return excess kurtosis (normal = 0)

    Returns
    -------
    float
        Kurtosis
    """
    if use_rust('kurtosis'):
        from pmtvs_statistics import _get_rust
        rust_fn = _get_rust('kurtosis')
        if rust_fn is not None:
            return rust_fn(signal, fisher)

    signal = np.asarray(signal).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)
    if n < 4:
        return np.nan

    m = np.mean(signal)
    s = np.std(signal, ddof=0)
    if s == 0:
        return np.nan

    k = float(np.mean(((signal - m) / s) ** 4))
    if fisher:
        k -= 3
    return k


def rms(signal: np.ndarray) -> float:
    """
    Compute root mean square of signal.

    Parameters
    ----------
    signal : np.ndarray
        Input signal

    Returns
    -------
    float
        RMS value
    """
    if use_rust('rms'):
        from pmtvs_statistics import _get_rust
        rust_fn = _get_rust('rms')
        if rust_fn is not None:
            return rust_fn(signal)

    signal = np.asarray(signal).flatten()
    signal = signal[~np.isnan(signal)]
    if len(signal) == 0:
        return np.nan
    return float(np.sqrt(np.mean(signal ** 2)))


def peak_to_peak(signal: np.ndarray) -> float:
    """
    Compute peak-to-peak amplitude.

    Parameters
    ----------
    signal : np.ndarray
        Input signal

    Returns
    -------
    float
        Peak-to-peak value (max - min)
    """
    if use_rust('peak_to_peak'):
        from pmtvs_statistics import _get_rust
        rust_fn = _get_rust('peak_to_peak')
        if rust_fn is not None:
            return rust_fn(signal)

    signal = np.asarray(signal).flatten()
    signal = signal[~np.isnan(signal)]
    if len(signal) == 0:
        return np.nan
    return float(np.max(signal) - np.min(signal))


def crest_factor(signal: np.ndarray) -> float:
    """
    Compute crest factor (peak / RMS).

    Parameters
    ----------
    signal : np.ndarray
        Input signal

    Returns
    -------
    float
        Crest factor
    """
    if use_rust('crest_factor'):
        from pmtvs_statistics import _get_rust
        rust_fn = _get_rust('crest_factor')
        if rust_fn is not None:
            return rust_fn(signal)

    signal = np.asarray(signal).flatten()
    signal = signal[~np.isnan(signal)]
    if len(signal) == 0:
        return np.nan

    rms_val = rms(signal)
    if rms_val == 0:
        return np.nan

    peak = np.max(np.abs(signal))
    return float(peak / rms_val)


def pulsation_index(signal: np.ndarray) -> float:
    """
    Compute pulsation index (peak-to-peak / mean).

    Parameters
    ----------
    signal : np.ndarray
        Input signal

    Returns
    -------
    float
        Pulsation index
    """
    if use_rust('pulsation_index'):
        from pmtvs_statistics import _get_rust
        rust_fn = _get_rust('pulsation_index')
        if rust_fn is not None:
            return rust_fn(signal)

    signal = np.asarray(signal).flatten()
    signal = signal[~np.isnan(signal)]
    if len(signal) == 0:
        return np.nan

    m = np.mean(signal)
    if m == 0:
        return np.nan

    return float((np.max(signal) - np.min(signal)) / abs(m))


def zero_crossings(signal: np.ndarray) -> int:
    """
    Count zero crossings in signal.

    Parameters
    ----------
    signal : np.ndarray
        Input signal

    Returns
    -------
    int
        Number of zero crossings
    """
    signal = np.asarray(signal).flatten()
    signal = signal[~np.isnan(signal)]
    if len(signal) < 2:
        return 0

    signs = np.sign(signal)
    signs[signs == 0] = 1  # Treat zeros as positive
    crossings = np.where(np.diff(signs) != 0)[0]
    return len(crossings)


def mean_crossings(signal: np.ndarray) -> int:
    """
    Count mean crossings in signal.

    Parameters
    ----------
    signal : np.ndarray
        Input signal

    Returns
    -------
    int
        Number of mean crossings
    """
    signal = np.asarray(signal).flatten()
    signal = signal[~np.isnan(signal)]
    if len(signal) < 2:
        return 0

    centered = signal - np.mean(signal)
    return zero_crossings(centered)
