"""
Spectral Analysis Functions

Power spectrum, frequency analysis, and spectral features.
"""

import numpy as np
from typing import Optional, Tuple


def power_spectral_density(
    signal: np.ndarray,
    fs: float = 1.0,
    nperseg: int = 256,
    noverlap: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectral density using Welch's method.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    fs : float
        Sampling frequency
    nperseg : int
        Segment length
    noverlap : int, optional
        Overlap (default: nperseg // 2)

    Returns
    -------
    tuple
        (frequencies, psd)
    """
    signal = np.asarray(signal).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if n < nperseg:
        nperseg = n

    if noverlap is None:
        noverlap = nperseg // 2

    step = nperseg - noverlap
    n_segments = max(1, (n - noverlap) // step)

    psd = np.zeros(nperseg // 2 + 1)

    for i in range(n_segments):
        start = i * step
        end = start + nperseg
        if end > n:
            break

        segment = signal[start:end]
        window = np.hanning(len(segment))
        segment = segment * window

        fft = np.fft.rfft(segment)
        psd += np.abs(fft) ** 2

    psd /= n_segments
    psd *= 2 / (fs * np.sum(np.hanning(nperseg) ** 2))

    freqs = np.fft.rfftfreq(nperseg, 1 / fs)

    return freqs, psd


def dominant_frequency(
    signal: np.ndarray,
    fs: float = 1.0
) -> float:
    """
    Find dominant frequency (frequency with highest power).

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    fs : float
        Sampling frequency

    Returns
    -------
    float
        Dominant frequency
    """
    freqs, psd = power_spectral_density(signal, fs=fs)

    if len(psd) == 0 or np.all(psd == 0):
        return np.nan

    return float(freqs[np.argmax(psd)])


def spectral_entropy(
    signal: np.ndarray,
    fs: float = 1.0,
    normalize: bool = True
) -> float:
    """
    Compute spectral entropy.

    Measures the flatness/uniformity of the power spectrum.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    fs : float
        Sampling frequency
    normalize : bool
        Normalize to [0, 1]

    Returns
    -------
    float
        Spectral entropy
    """
    freqs, psd = power_spectral_density(signal, fs=fs)

    if len(psd) == 0 or np.sum(psd) == 0:
        return np.nan

    # Normalize to probability distribution
    psd_norm = psd / np.sum(psd)
    psd_norm = psd_norm[psd_norm > 0]

    # Shannon entropy
    entropy = -np.sum(psd_norm * np.log2(psd_norm))

    if normalize:
        max_entropy = np.log2(len(psd))
        if max_entropy > 0:
            entropy /= max_entropy

    return float(entropy)


def spectral_centroid(
    signal: np.ndarray,
    fs: float = 1.0
) -> float:
    """
    Compute spectral centroid (center of mass of spectrum).

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    fs : float
        Sampling frequency

    Returns
    -------
    float
        Spectral centroid frequency
    """
    freqs, psd = power_spectral_density(signal, fs=fs)

    if np.sum(psd) == 0:
        return np.nan

    return float(np.sum(freqs * psd) / np.sum(psd))


def spectral_bandwidth(
    signal: np.ndarray,
    fs: float = 1.0,
    p: int = 2
) -> float:
    """
    Compute spectral bandwidth (spread around centroid).

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    fs : float
        Sampling frequency
    p : int
        Order of bandwidth (default: 2, standard deviation)

    Returns
    -------
    float
        Spectral bandwidth
    """
    freqs, psd = power_spectral_density(signal, fs=fs)

    if np.sum(psd) == 0:
        return np.nan

    centroid = np.sum(freqs * psd) / np.sum(psd)
    bandwidth = (np.sum(((freqs - centroid) ** p) * psd) / np.sum(psd)) ** (1 / p)

    return float(bandwidth)


def spectral_rolloff(
    signal: np.ndarray,
    fs: float = 1.0,
    threshold: float = 0.85
) -> float:
    """
    Compute spectral rolloff frequency.

    Frequency below which threshold% of total spectral energy is contained.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    fs : float
        Sampling frequency
    threshold : float
        Energy threshold (default: 85%)

    Returns
    -------
    float
        Rolloff frequency
    """
    freqs, psd = power_spectral_density(signal, fs=fs)

    if np.sum(psd) == 0:
        return np.nan

    cumulative_energy = np.cumsum(psd)
    total_energy = cumulative_energy[-1]

    rolloff_idx = np.searchsorted(cumulative_energy, threshold * total_energy)
    rolloff_idx = min(rolloff_idx, len(freqs) - 1)

    return float(freqs[rolloff_idx])


def spectral_flatness(
    signal: np.ndarray,
    fs: float = 1.0
) -> float:
    """
    Compute spectral flatness (Wiener entropy).

    Ratio of geometric mean to arithmetic mean of spectrum.
    Values close to 1 indicate noise-like signals.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    fs : float
        Sampling frequency

    Returns
    -------
    float
        Spectral flatness in [0, 1]
    """
    freqs, psd = power_spectral_density(signal, fs=fs)

    psd = psd[psd > 0]
    if len(psd) == 0:
        return np.nan

    geometric_mean = np.exp(np.mean(np.log(psd)))
    arithmetic_mean = np.mean(psd)

    if arithmetic_mean == 0:
        return np.nan

    return float(geometric_mean / arithmetic_mean)


def harmonic_ratio(
    signal: np.ndarray,
    fs: float = 1.0,
    n_harmonics: int = 5
) -> float:
    """
    Compute harmonic-to-noise ratio.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    fs : float
        Sampling frequency
    n_harmonics : int
        Number of harmonics to consider

    Returns
    -------
    float
        Harmonic ratio
    """
    freqs, psd = power_spectral_density(signal, fs=fs)

    if len(psd) < 3:
        return np.nan

    # Find fundamental frequency
    f0_idx = np.argmax(psd[1:]) + 1  # Skip DC
    f0 = freqs[f0_idx]

    if f0 == 0:
        return np.nan

    # Sum harmonic power
    harmonic_power = 0
    freq_resolution = freqs[1] - freqs[0] if len(freqs) > 1 else 1

    for h in range(1, n_harmonics + 1):
        harmonic_freq = h * f0
        if harmonic_freq > freqs[-1]:
            break

        # Find nearest frequency bin
        h_idx = int(round(harmonic_freq / freq_resolution))
        if h_idx < len(psd):
            harmonic_power += psd[h_idx]

    total_power = np.sum(psd)
    noise_power = total_power - harmonic_power

    if noise_power <= 0:
        return np.inf

    return float(harmonic_power / noise_power)


def total_harmonic_distortion(
    signal: np.ndarray,
    fs: float = 1.0,
    n_harmonics: int = 10
) -> float:
    """
    Compute total harmonic distortion (THD).

    THD = sqrt(sum of harmonic powers) / fundamental power

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    fs : float
        Sampling frequency
    n_harmonics : int
        Number of harmonics to include

    Returns
    -------
    float
        THD (0 = perfect sine, higher = more distortion)
    """
    freqs, psd = power_spectral_density(signal, fs=fs)

    if len(psd) < 3:
        return np.nan

    # Find fundamental
    f0_idx = np.argmax(psd[1:]) + 1
    f0 = freqs[f0_idx]
    fundamental_power = psd[f0_idx]

    if f0 == 0 or fundamental_power == 0:
        return np.nan

    # Sum harmonic powers (excluding fundamental)
    harmonic_power_sum = 0
    freq_resolution = freqs[1] - freqs[0] if len(freqs) > 1 else 1

    for h in range(2, n_harmonics + 1):  # Start from 2nd harmonic
        harmonic_freq = h * f0
        if harmonic_freq > freqs[-1]:
            break

        h_idx = int(round(harmonic_freq / freq_resolution))
        if h_idx < len(psd):
            harmonic_power_sum += psd[h_idx]

    return float(np.sqrt(harmonic_power_sum) / np.sqrt(fundamental_power))
