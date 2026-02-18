"""
Pairwise Correlation Primitives

Two-signal correlation measures: correlation, covariance, cross-correlation, coherence.
"""

import numpy as np
from typing import Optional, Tuple

from pmtvs_correlation._dispatch import use_rust


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Pearson correlation coefficient.

    Parameters
    ----------
    x : np.ndarray
        First signal
    y : np.ndarray
        Second signal

    Returns
    -------
    float
        Correlation coefficient in [-1, 1]
    """
    if use_rust('correlation'):
        from pmtvs_correlation import _get_rust
        rust_fn = _get_rust('correlation')
        if rust_fn is not None:
            return rust_fn(x, y)

    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    if len(x) != len(y) or len(x) < 2:
        return np.nan

    # Remove NaN pairs
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2))

    if den == 0:
        return np.nan

    return float(num / den)


def covariance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute covariance between two signals.

    Parameters
    ----------
    x : np.ndarray
        First signal
    y : np.ndarray
        Second signal

    Returns
    -------
    float
        Covariance
    """
    if use_rust('covariance'):
        from pmtvs_correlation import _get_rust
        rust_fn = _get_rust('covariance')
        if rust_fn is not None:
            return rust_fn(x, y)

    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    if len(x) != len(y) or len(x) < 2:
        return np.nan

    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan

    return float(np.cov(x, y, ddof=1)[0, 1])


def cross_correlation(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: Optional[int] = None,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute cross-correlation between two signals.

    Parameters
    ----------
    x : np.ndarray
        First signal
    y : np.ndarray
        Second signal
    max_lag : int, optional
        Maximum lag (default: len(x) - 1)
    normalize : bool
        If True, normalize to [-1, 1]

    Returns
    -------
    np.ndarray
        Cross-correlation values for lags [-max_lag, max_lag]
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    if len(x) != len(y) or len(x) < 2:
        return np.array([np.nan])

    n = len(x)
    if max_lag is None:
        max_lag = n - 1

    max_lag = min(max_lag, n - 1)

    # Remove means
    x = x - np.mean(x)
    y = y - np.mean(y)

    # Full cross-correlation using FFT
    xcorr = np.correlate(x, y, mode='full')

    # Extract relevant lags
    mid = len(xcorr) // 2
    xcorr = xcorr[mid - max_lag:mid + max_lag + 1]

    if normalize:
        norm = np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))
        if norm > 0:
            xcorr = xcorr / norm

    return xcorr


def lag_at_max_xcorr(x: np.ndarray, y: np.ndarray, max_lag: Optional[int] = None) -> int:
    """
    Find lag at maximum cross-correlation.

    Parameters
    ----------
    x : np.ndarray
        First signal
    y : np.ndarray
        Second signal
    max_lag : int, optional
        Maximum lag to search

    Returns
    -------
    int
        Lag at maximum cross-correlation
    """
    xcorr = cross_correlation(x, y, max_lag=max_lag, normalize=True)

    if len(xcorr) == 1 and np.isnan(xcorr[0]):
        return 0

    max_lag_used = (len(xcorr) - 1) // 2
    return int(np.argmax(xcorr) - max_lag_used)


def partial_correlation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """
    Compute partial correlation between x and y, controlling for z.

    Parameters
    ----------
    x : np.ndarray
        First signal
    y : np.ndarray
        Second signal
    z : np.ndarray
        Control signal

    Returns
    -------
    float
        Partial correlation coefficient
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    z = np.asarray(z).flatten()

    if len(x) != len(y) or len(x) != len(z) or len(x) < 3:
        return np.nan

    # Remove NaN triples
    mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x = x[mask]
    y = y[mask]
    z = z[mask]

    if len(x) < 3:
        return np.nan

    rxy = correlation(x, y)
    rxz = correlation(x, z)
    ryz = correlation(y, z)

    if np.isnan(rxy) or np.isnan(rxz) or np.isnan(ryz):
        return np.nan

    den = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    if den == 0:
        return np.nan

    return float((rxy - rxz * ryz) / den)


def coherence(
    x: np.ndarray,
    y: np.ndarray,
    nperseg: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute magnitude-squared coherence.

    Parameters
    ----------
    x : np.ndarray
        First signal
    y : np.ndarray
        Second signal
    nperseg : int
        Segment length for FFT

    Returns
    -------
    tuple
        (frequencies, coherence values)
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    if len(x) != len(y) or len(x) < nperseg:
        return np.array([np.nan]), np.array([np.nan])

    n = len(x)
    n_segments = n // nperseg

    if n_segments < 1:
        nperseg = n
        n_segments = 1

    # Compute cross-spectral and auto-spectral densities
    pxx = np.zeros(nperseg // 2 + 1)
    pyy = np.zeros(nperseg // 2 + 1)
    pxy = np.zeros(nperseg // 2 + 1, dtype=complex)

    for i in range(n_segments):
        seg_x = x[i * nperseg:(i + 1) * nperseg]
        seg_y = y[i * nperseg:(i + 1) * nperseg]

        # Apply Hann window
        window = np.hanning(nperseg)
        seg_x = seg_x * window
        seg_y = seg_y * window

        fx = np.fft.rfft(seg_x)
        fy = np.fft.rfft(seg_y)

        pxx += np.abs(fx) ** 2
        pyy += np.abs(fy) ** 2
        pxy += fx * np.conj(fy)

    pxx /= n_segments
    pyy /= n_segments
    pxy /= n_segments

    # Coherence
    denom = pxx * pyy
    denom[denom == 0] = np.finfo(float).eps

    coh = np.abs(pxy) ** 2 / denom

    freqs = np.fft.rfftfreq(nperseg)

    return freqs, coh


def cross_spectral_density(
    x: np.ndarray,
    y: np.ndarray,
    nperseg: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cross-spectral density.

    Parameters
    ----------
    x : np.ndarray
        First signal
    y : np.ndarray
        Second signal
    nperseg : int
        Segment length for FFT

    Returns
    -------
    tuple
        (frequencies, cross-spectral density)
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    if len(x) != len(y) or len(x) < nperseg:
        return np.array([np.nan]), np.array([np.nan])

    n = len(x)
    n_segments = n // nperseg

    if n_segments < 1:
        nperseg = n
        n_segments = 1

    pxy = np.zeros(nperseg // 2 + 1, dtype=complex)

    for i in range(n_segments):
        seg_x = x[i * nperseg:(i + 1) * nperseg]
        seg_y = y[i * nperseg:(i + 1) * nperseg]

        window = np.hanning(nperseg)
        seg_x = seg_x * window
        seg_y = seg_y * window

        fx = np.fft.rfft(seg_x)
        fy = np.fft.rfft(seg_y)

        pxy += fx * np.conj(fy)

    pxy /= n_segments

    freqs = np.fft.rfftfreq(nperseg)

    return freqs, pxy


def phase_spectrum(
    x: np.ndarray,
    y: np.ndarray,
    nperseg: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute phase spectrum between two signals.

    Parameters
    ----------
    x : np.ndarray
        First signal
    y : np.ndarray
        Second signal
    nperseg : int
        Segment length for FFT

    Returns
    -------
    tuple
        (frequencies, phase angles in radians)
    """
    freqs, csd = cross_spectral_density(x, y, nperseg)

    if len(freqs) == 1 and np.isnan(freqs[0]):
        return np.array([np.nan]), np.array([np.nan])

    phase = np.angle(csd)

    return freqs, phase


def wavelet_coherence(
    x: np.ndarray,
    y: np.ndarray,
    scales: Optional[np.ndarray] = None,
    wavelet: str = 'morlet'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute wavelet coherence (simplified implementation).

    Parameters
    ----------
    x : np.ndarray
        First signal
    y : np.ndarray
        Second signal
    scales : np.ndarray, optional
        Scales for wavelet transform (default: log-spaced)
    wavelet : str
        Wavelet type (currently only 'morlet' supported)

    Returns
    -------
    tuple
        (scales, coherence values averaged over time)
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    if len(x) != len(y) or len(x) < 10:
        return np.array([np.nan]), np.array([np.nan])

    n = len(x)

    if scales is None:
        scales = np.logspace(0, np.log10(n // 4), 20).astype(int)
        scales = np.unique(scales[scales > 1])

    if len(scales) == 0:
        return np.array([np.nan]), np.array([np.nan])

    # Simplified wavelet coherence using windowed cross-correlation
    coherences = []

    for scale in scales:
        if scale >= n:
            coherences.append(np.nan)
            continue

        # Windowed cross-correlation at this scale
        n_windows = n // scale
        if n_windows < 2:
            coherences.append(np.nan)
            continue

        window_corrs = []
        for i in range(n_windows):
            start = i * scale
            end = start + scale
            window_x = x[start:end]
            window_y = y[start:end]
            r = correlation(window_x, window_y)
            if not np.isnan(r):
                window_corrs.append(r ** 2)

        if window_corrs:
            coherences.append(np.mean(window_corrs))
        else:
            coherences.append(np.nan)

    return scales, np.array(coherences)
