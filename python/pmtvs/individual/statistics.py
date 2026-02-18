"""
Statistical Primitives

Basic statistical measures for individual signals.
"""

import numpy as np
from scipy import stats as scipy_stats
from typing import Tuple, List, Optional


def mean(signal: np.ndarray) -> float:
    """
    Compute arithmetic mean.

    Parameters
    ----------
    signal : np.ndarray
        Input signal

    Returns
    -------
    float
        Arithmetic mean
    """
    return float(np.nanmean(signal))


def std(signal: np.ndarray, ddof: int = 0) -> float:
    """
    Compute standard deviation.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    ddof : int
        Delta degrees of freedom (0 for population, 1 for sample)

    Returns
    -------
    float
        Standard deviation
    """
    return float(np.nanstd(signal, ddof=ddof))


def variance(signal: np.ndarray, ddof: int = 0) -> float:
    """
    Compute variance.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    ddof : int
        Delta degrees of freedom

    Returns
    -------
    float
        Variance
    """
    return float(np.nanvar(signal, ddof=ddof))


def min_max(signal: np.ndarray) -> Tuple[float, float]:
    """
    Compute minimum and maximum values.

    Parameters
    ----------
    signal : np.ndarray
        Input signal

    Returns
    -------
    tuple
        (min_value, max_value)
    """
    return float(np.nanmin(signal)), float(np.nanmax(signal))


def percentiles(signal: np.ndarray, qs: Optional[List[float]] = None) -> np.ndarray:
    """
    Compute percentiles.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    qs : list of float, optional
        Percentiles to compute (default: [25, 50, 75])

    Returns
    -------
    np.ndarray
        Percentile values
    """
    if qs is None:
        qs = [25, 50, 75]
    return np.nanpercentile(signal, qs)


def skewness(signal: np.ndarray) -> float:
    """
    Compute skewness (third standardized moment).

    Parameters
    ----------
    signal : np.ndarray
        Input signal

    Returns
    -------
    float
        Skewness (positive = right-skewed, negative = left-skewed)
    """
    return float(scipy_stats.skew(signal, nan_policy='omit'))


def kurtosis(signal: np.ndarray, fisher: bool = True) -> float:
    """
    Compute kurtosis (fourth standardized moment).

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    fisher : bool
        If True, return excess kurtosis (kurtosis - 3)

    Returns
    -------
    float
        Kurtosis (positive = heavy tails, negative = light tails)
    """
    return float(scipy_stats.kurtosis(signal, fisher=fisher, nan_policy='omit'))


def rms(signal: np.ndarray) -> float:
    """
    Compute root mean square.

    Parameters
    ----------
    signal : np.ndarray
        Input signal

    Returns
    -------
    float
        RMS value = sqrt(mean(x^2))
    """
    signal = np.asarray(signal)
    return float(np.sqrt(np.nanmean(signal ** 2)))


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
    return float(np.nanmax(signal) - np.nanmin(signal))


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
        Crest factor (indicates how "peaky" the signal is)
    """
    signal = np.asarray(signal)
    rms_val = rms(signal)
    peak_val = np.nanmax(np.abs(signal))
    if rms_val == 0:
        return np.nan
    return float(peak_val / rms_val)


def zero_crossings(signal: np.ndarray) -> int:
    """
    Count zero crossings.

    Parameters
    ----------
    signal : np.ndarray
        Input signal

    Returns
    -------
    int
        Number of zero crossings
    """
    signal = np.asarray(signal)
    signal = signal[~np.isnan(signal)]
    return int(np.sum(np.diff(np.signbit(signal))))


def mean_crossings(signal: np.ndarray) -> int:
    """
    Count mean crossings.

    Parameters
    ----------
    signal : np.ndarray
        Input signal

    Returns
    -------
    int
        Number of mean crossings (zero crossings of centered signal)
    """
    signal = np.asarray(signal)
    signal = signal[~np.isnan(signal)]
    centered = signal - np.mean(signal)
    return int(np.sum(np.diff(np.signbit(centered))))
