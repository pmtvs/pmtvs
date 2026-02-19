"""
Fractal Primitives

Hurst exponent and Detrended Fluctuation Analysis.
These measure long-range dependence in time series.
"""

import numpy as np
from typing import Optional, Tuple

from pmtvs_fractal._dispatch import use_rust


# Default configuration constants
DEFAULT_RS_MIN_K = 10
DEFAULT_RS_MAX_K_RATIO = 0.25
DEFAULT_RS_MAX_K_CAP = 500
DEFAULT_DFA_MIN_SCALE = 4
DEFAULT_DFA_MAX_SCALE_RATIO = 0.25
DEFAULT_DFA_MAX_SCALE_CAP = 256
DEFAULT_DFA_N_SCALES = 20
DEFAULT_MIN_SAMPLES_DFA = 64


def hurst_exponent(
    signal: np.ndarray,
    method: str = 'rs'
) -> float:
    """
    Compute Hurst exponent.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    method : str
        'rs' (rescaled range) or 'dfa' (detrended fluctuation)

    Returns
    -------
    float
        Hurst exponent H (0.5 = random walk, >0.5 = persistent, <0.5 = anti-persistent)
    """
    if use_rust('hurst_exponent'):
        from pmtvs_fractal import _get_rust
        rust_fn = _get_rust('hurst_exponent')
        if rust_fn is not None:
            return rust_fn(signal, method)

    signal = np.asarray(signal).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if n < DEFAULT_RS_MIN_K:
        return np.nan

    if method == 'dfa':
        return dfa(signal)

    # Rescaled Range (R/S) method
    max_k = min(int(n * DEFAULT_RS_MAX_K_RATIO), DEFAULT_RS_MAX_K_CAP)
    k_values = []
    rs_values = []

    for k in range(DEFAULT_RS_MIN_K, max_k):
        n_subseries = n // k
        rs_sum = 0

        for i in range(n_subseries):
            subseries = signal[i*k:(i+1)*k]
            mean = np.mean(subseries)

            # Cumulative deviation from mean
            Y = np.cumsum(subseries - mean)

            # Range
            R = np.max(Y) - np.min(Y)

            # Standard deviation
            S = np.std(subseries, ddof=1)

            if S > 0:
                rs_sum += R / S

        if n_subseries > 0:
            rs_avg = rs_sum / n_subseries
            if rs_avg > 0:
                k_values.append(np.log(k))
                rs_values.append(np.log(rs_avg))

    if len(k_values) < 3:
        return np.nan

    # Linear fit: log(R/S) = H * log(k) + c
    H, _ = np.polyfit(k_values, rs_values, 1)

    return float(np.clip(H, 0, 1))


def dfa(
    signal: np.ndarray,
    scale_range: Optional[Tuple[int, int]] = None,
    order: int = 1
) -> float:
    """
    Compute Detrended Fluctuation Analysis.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    scale_range : tuple, optional
        (min_scale, max_scale)
    order : int
        Polynomial order for detrending

    Returns
    -------
    float
        DFA exponent (similar interpretation to Hurst)
    """
    if use_rust('dfa'):
        from pmtvs_fractal import _get_rust
        rust_fn = _get_rust('dfa')
        if rust_fn is not None:
            min_s = scale_range[0] if scale_range else DEFAULT_DFA_MIN_SCALE
            max_s = scale_range[1] if scale_range else -1  # -1 means auto
            return rust_fn(signal, min_s, max_s, order)

    signal = np.asarray(signal).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if n < DEFAULT_MIN_SAMPLES_DFA:
        return np.nan

    # Integrate signal
    Y = np.cumsum(signal - np.mean(signal))

    # Define scales
    if scale_range is None:
        min_scale = DEFAULT_DFA_MIN_SCALE
        max_scale = min(int(n * DEFAULT_DFA_MAX_SCALE_RATIO), DEFAULT_DFA_MAX_SCALE_CAP)
    else:
        min_scale, max_scale = scale_range

    scales = np.unique(np.logspace(
        np.log10(min_scale),
        np.log10(max_scale),
        DEFAULT_DFA_N_SCALES
    ).astype(int))

    fluctuations = []

    for scale in scales:
        n_segments = n // scale
        if n_segments < 2:
            continue

        F_sq = []

        for i in range(n_segments):
            segment = Y[i*scale:(i+1)*scale]
            x = np.arange(scale)

            # Polynomial fit (local trend)
            coeffs = np.polyfit(x, segment, order)
            trend = np.polyval(coeffs, x)

            # Fluctuation
            F_sq.append(np.mean((segment - trend) ** 2))

        if F_sq:
            fluctuations.append(np.sqrt(np.mean(F_sq)))

    if len(fluctuations) < 3:
        return np.nan

    # Linear fit in log-log space
    log_scales = np.log(scales[:len(fluctuations)])
    log_fluct = np.log(fluctuations)

    alpha, _ = np.polyfit(log_scales, log_fluct, 1)

    return float(alpha)


def hurst_r2(signal: np.ndarray) -> float:
    """
    Compute R-squared of Hurst exponent fit.

    Parameters
    ----------
    signal : np.ndarray
        Input signal

    Returns
    -------
    float
        R-squared value (goodness of fit)
    """
    if use_rust('hurst_r2'):
        from pmtvs_fractal import _get_rust
        rust_fn = _get_rust('hurst_r2')
        if rust_fn is not None:
            return rust_fn(signal)

    signal = np.asarray(signal).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if n < DEFAULT_RS_MIN_K:
        return np.nan

    max_k = min(int(n * DEFAULT_RS_MAX_K_RATIO), DEFAULT_RS_MAX_K_CAP)
    log_k = []
    log_rs = []

    for k in range(DEFAULT_RS_MIN_K, max_k):
        n_subseries = n // k
        rs_sum = 0

        for i in range(n_subseries):
            subseries = signal[i*k:(i+1)*k]
            Y = np.cumsum(subseries - np.mean(subseries))
            R = np.max(Y) - np.min(Y)
            S = np.std(subseries, ddof=1)

            if S > 0:
                rs_sum += R / S

        if n_subseries > 0:
            rs_avg = rs_sum / n_subseries
            if rs_avg > 0:
                log_k.append(np.log(k))
                log_rs.append(np.log(rs_avg))

    if len(log_k) < 3:
        return np.nan

    # Compute R²
    slope, intercept = np.polyfit(log_k, log_rs, 1)
    predicted = slope * np.array(log_k) + intercept
    ss_res = np.sum((np.array(log_rs) - predicted) ** 2)
    ss_tot = np.sum((np.array(log_rs) - np.mean(log_rs)) ** 2)

    if ss_tot == 0:
        return np.nan

    return float(1 - ss_res / ss_tot)


# ---------------------------------------------------------------------------
# Memory analysis functions
# ---------------------------------------------------------------------------

# Default configuration constants for memory analysis
DEFAULT_MEMORY_MIN_SAMPLES = 20
DEFAULT_MAX_LAG_RATIO = 0.25


def detrended_fluctuation_analysis(
    values: np.ndarray,
    min_scale: int = 4,
    max_scale: Optional[int] = None,
    n_scales: int = 10,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Perform detrended fluctuation analysis (full output version).

    Unlike :func:`dfa` which returns only the scaling exponent, this function
    returns the intermediate scales and fluctuation values as well.

    Parameters
    ----------
    values : np.ndarray
        Input time series.
    min_scale : int
        Minimum window scale (default 4).
    max_scale : int, optional
        Maximum window scale (default ``len(values) // 4``).
    n_scales : int
        Number of logarithmically-spaced scale points (default 10).

    Returns
    -------
    tuple of (scales, fluctuations, alpha)
        * **scales** — array of window sizes used.
        * **fluctuations** — RMS fluctuation at each scale.
        * **alpha** — DFA scaling exponent (related to Hurst: H ≈ alpha).
          Returns 0.5 when the fit cannot be computed.
    """
    values = np.asarray(values).flatten()
    values = values[~np.isnan(values)]
    n = len(values)

    if n < min_scale * 2:
        return np.array([]), np.array([]), float(np.nan)

    if max_scale is None:
        max_scale = n // 4

    if max_scale < min_scale:
        return np.array([]), np.array([]), float(np.nan)

    # Integration: cumulative sum of deviations from mean
    y = np.cumsum(values - np.mean(values))

    # Generate logarithmically-spaced scales
    scales = np.unique(np.round(np.logspace(
        np.log10(min_scale),
        np.log10(max_scale),
        n_scales,
    ))).astype(int)
    scales = scales[(scales >= min_scale) & (scales <= max_scale)]

    fluctuations = []

    for scale in scales:
        n_segments = n // scale
        if n_segments < 1:
            fluctuations.append(np.nan)
            continue

        rms_list = []

        for i in range(n_segments):
            segment = y[i * scale:(i + 1) * scale]

            # Linear detrend
            x = np.arange(len(segment))
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)

            rms = np.sqrt(np.mean((segment - trend) ** 2))
            rms_list.append(rms)

        fluctuations.append(np.mean(rms_list))

    fluctuations = np.array(fluctuations)

    # Remove invalid values for fitting
    valid = ~np.isnan(fluctuations) & (fluctuations > 0)
    if np.sum(valid) < 2:
        return scales, fluctuations, float(np.nan)

    # Log-log fit to obtain alpha
    log_scales = np.log(scales[valid])
    log_fluct = np.log(fluctuations[valid])

    alpha, _ = np.polyfit(log_scales, log_fluct, 1)

    return scales, fluctuations, float(alpha)


def rescaled_range(
    values: np.ndarray,
    segment_size: Optional[int] = None,
) -> float:
    """
    Compute the rescaled range (R/S) statistic — Hurst exponent via R/S.

    The R/S statistic is the ratio of the range of cumulative deviations from
    the mean to the standard deviation of the series.

    Parameters
    ----------
    values : np.ndarray
        Input time series.
    segment_size : int, optional
        Number of samples to use (default: full length).

    Returns
    -------
    float
        R/S statistic.  Returns ``nan`` for degenerate inputs.
    """
    values = np.asarray(values).flatten()
    values = values[~np.isnan(values)]
    n = len(values)

    if n < 2:
        return float(np.nan)

    if segment_size is None:
        segment_size = n

    segment_size = min(segment_size, n)

    segment = values[:segment_size]

    # Mean-adjusted cumulative sum
    mean = np.mean(segment)
    cumsum = np.cumsum(segment - mean)

    # Range
    R = np.max(cumsum) - np.min(cumsum)

    # Standard deviation
    S = np.std(segment, ddof=1)

    if S == 0:
        return float(np.nan)

    return float(R / S)


def long_range_correlation(
    values: np.ndarray,
    max_lag: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    """
    Analyse long-range correlation via autocorrelation power-law decay.

    Fits ``ACF(tau) ~ tau^(-d)`` in log-log space. A decay exponent ``d < 1``
    indicates the presence of long-range correlations.

    Parameters
    ----------
    values : np.ndarray
        Input time series.
    max_lag : int, optional
        Maximum lag to analyse (default: ``int(len(values) * 0.25)``).

    Returns
    -------
    tuple of (acf, decay_exponent)
        * **acf** — normalised autocorrelation from lag 0 to *max_lag*.
        * **decay_exponent** — positive exponent ``d`` of power-law decay.
          Returns ``nan`` when the fit cannot be computed.
    """
    values = np.asarray(values).flatten()
    values = values[~np.isnan(values)]
    n = len(values)

    if n < DEFAULT_MEMORY_MIN_SAMPLES:
        return np.array([]), float(np.nan)

    if max_lag is None:
        max_lag = int(n * DEFAULT_MAX_LAG_RATIO)

    max_lag = max(max_lag, 1)

    # Compute autocorrelation via full cross-correlation
    values_centered = values - np.mean(values)
    autocorr = np.correlate(values_centered, values_centered, mode='full')
    autocorr = autocorr[n - 1:]  # keep non-negative lags

    if autocorr[0] == 0:
        return np.zeros(min(max_lag + 1, n)), float(np.nan)

    autocorr = autocorr / autocorr[0]  # normalise

    acf = autocorr[:max_lag + 1]

    # Fit power-law decay: ACF(tau) ~ tau^(-d)
    lags = np.arange(1, len(acf))
    acf_positive = acf[1:]

    # Only use positive ACF values for log-log fit
    valid = acf_positive > 0
    if np.sum(valid) < 2:
        return acf, float(np.nan)

    log_lags = np.log(lags[valid])
    log_acf = np.log(acf_positive[valid])

    decay_exp, _ = np.polyfit(log_lags, log_acf, 1)

    return acf, float(-decay_exp)


def variance_growth(
    values: np.ndarray,
    max_lag: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    """
    Analyse variance growth with aggregation scale.

    For a random walk the variance of aggregated blocks decays linearly with
    scale (exponent = -1).  Persistent series decay slower (exponent > -1)
    and anti-persistent series decay faster (exponent < -1).

    Parameters
    ----------
    values : np.ndarray
        Input time series.
    max_lag : int, optional
        Maximum aggregation scale (default: ``int(len(values) * 0.25)``).

    Returns
    -------
    tuple of (scales, scaling_exponent)
        * **scales** — aggregation scales from 1 to *max_lag*.
        * **scaling_exponent** — power-law exponent of variance vs. scale.
          Returns ``nan`` when the fit cannot be computed.
    """
    values = np.asarray(values).flatten()
    values = values[~np.isnan(values)]
    n = len(values)

    if n < DEFAULT_MEMORY_MIN_SAMPLES:
        return np.array([]), float(np.nan)

    if max_lag is None:
        max_lag = int(n * DEFAULT_MAX_LAG_RATIO)

    max_lag = max(max_lag, 1)

    scales = np.arange(1, max_lag + 1)
    variances = []

    for scale in scales:
        n_agg = n // scale
        if n_agg < 2:
            variances.append(np.nan)
            continue

        aggregated = [
            np.mean(values[i * scale:(i + 1) * scale])
            for i in range(n_agg)
        ]

        variances.append(np.var(aggregated))

    variances = np.array(variances)

    # Fit power law in log-log space
    valid = ~np.isnan(variances) & (variances > 0)
    if np.sum(valid) < 2:
        return scales, float(np.nan)

    log_scales = np.log(scales[valid])
    log_var = np.log(variances[valid])

    exponent, _ = np.polyfit(log_scales, log_var, 1)

    return scales, float(exponent)
