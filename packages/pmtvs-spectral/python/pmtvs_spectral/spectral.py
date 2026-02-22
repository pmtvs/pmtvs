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

    if n < 2:
        return np.array([]), np.array([])

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


def fft_magnitude(
    signal: np.ndarray,
    fs: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute FFT magnitude spectrum.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    fs : float
        Sampling frequency

    Returns
    -------
    tuple
        (frequencies, magnitudes)
    """
    signal = np.asarray(signal, dtype=np.float64).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if n == 0:
        return np.array([]), np.array([])

    fft_vals = np.fft.rfft(signal)
    magnitudes = np.abs(fft_vals) * 2.0 / n
    freqs = np.fft.rfftfreq(n, 1.0 / fs)

    return freqs, magnitudes


def hilbert_transform(
    signal: np.ndarray
) -> np.ndarray:
    """
    Compute the analytic signal using the Hilbert transform.

    Parameters
    ----------
    signal : np.ndarray
        Input real-valued signal

    Returns
    -------
    np.ndarray
        Complex analytic signal
    """
    signal = np.asarray(signal, dtype=np.float64).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if n == 0:
        return np.array([], dtype=np.complex128)

    # FFT-based Hilbert transform
    fft_vals = np.fft.fft(signal)
    h = np.zeros(n)
    if n > 0:
        h[0] = 1
        if n % 2 == 0:
            h[n // 2] = 1
            h[1:n // 2] = 2
        else:
            h[1:(n + 1) // 2] = 2

    analytic = np.fft.ifft(fft_vals * h)
    return analytic


def envelope(
    signal: np.ndarray
) -> np.ndarray:
    """
    Compute signal envelope (amplitude modulation) via Hilbert transform.

    Parameters
    ----------
    signal : np.ndarray
        Input signal

    Returns
    -------
    np.ndarray
        Envelope (instantaneous amplitude)
    """
    analytic = hilbert_transform(signal)
    return np.abs(analytic)


def instantaneous_frequency(
    signal: np.ndarray,
    fs: float = 1.0
) -> np.ndarray:
    """
    Compute instantaneous frequency via Hilbert transform.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    fs : float
        Sampling frequency

    Returns
    -------
    np.ndarray
        Instantaneous frequency at each sample
    """
    analytic = hilbert_transform(signal)
    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(phase) / (2.0 * np.pi) * fs
    return inst_freq


def instantaneous_amplitude(
    signal: np.ndarray
) -> np.ndarray:
    """
    Compute instantaneous amplitude via Hilbert transform.

    Parameters
    ----------
    signal : np.ndarray
        Input signal

    Returns
    -------
    np.ndarray
        Instantaneous amplitude
    """
    return envelope(signal)


def instantaneous_phase(
    signal: np.ndarray
) -> np.ndarray:
    """
    Compute instantaneous phase via Hilbert transform.

    Parameters
    ----------
    signal : np.ndarray
        Input signal

    Returns
    -------
    np.ndarray
        Instantaneous phase (unwrapped, in radians)
    """
    analytic = hilbert_transform(signal)
    return np.unwrap(np.angle(analytic))


def spectral_slope(
    signal: np.ndarray,
    fs: float = 1.0
) -> float:
    """
    Compute spectral slope via log-log linear regression of PSD vs frequency.

    Parameters
    ----------
    signal : np.ndarray
        Input signal (1D).
    fs : float
        Sampling frequency.

    Returns
    -------
    float
        Spectral slope (negative = 1/f-like decay).
    """
    signal = np.asarray(signal, dtype=np.float64).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if n < 4:
        return np.nan

    xc = signal - np.mean(signal)
    fft_vals = np.fft.rfft(xc)
    psd = np.abs(fft_vals[1:]) ** 2
    if len(psd) == 0 or np.sum(psd) < 1e-30:
        return np.nan

    freqs = np.arange(1, len(psd) + 1) * (fs / n)
    log_f = np.log(freqs + 1e-30)
    log_p = np.log(psd + 1e-30)
    slope = float(np.polyfit(log_f, log_p, 1)[0])
    return slope


def signal_to_noise(
    signal: np.ndarray,
    kernel_fraction: int = 20
) -> dict:
    """
    Estimate signal-to-noise ratio via moving-average separation.

    Parameters
    ----------
    signal : np.ndarray
        Input signal (1D).
    kernel_fraction : int
        Denominator for kernel size (kernel = n // kernel_fraction).

    Returns
    -------
    dict
        db, linear, signal_power, noise_power.
    """
    nan_result = {
        'db': np.nan, 'linear': np.nan,
        'signal_power': np.nan, 'noise_power': np.nan,
    }

    signal = np.asarray(signal, dtype=np.float64).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if n < 10:
        return nan_result

    k = max(n // kernel_fraction, 3)
    kernel = np.ones(k) / k
    signal_est = np.convolve(signal, kernel, mode='same')
    noise = signal - signal_est
    sig_power = float(np.mean(signal_est ** 2))
    noise_power = float(np.mean(noise ** 2))

    if noise_power < 1e-30:
        return {
            'db': 100.0, 'linear': 1e10,
            'signal_power': sig_power, 'noise_power': noise_power,
        }

    linear = sig_power / noise_power
    db = float(10 * np.log10(linear))

    return {
        'db': db, 'linear': linear,
        'signal_power': sig_power, 'noise_power': noise_power,
    }
