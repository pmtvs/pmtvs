"""
pmtvs-spectral — Spectral analysis primitives.

numpy in, array/number out.
"""

__version__ = "0.3.3"
BACKEND = "python"

from pmtvs_spectral.spectral import (
    power_spectral_density,
    dominant_frequency,
    spectral_entropy,
    spectral_centroid,
    spectral_bandwidth,
    spectral_rolloff,
    spectral_flatness,
    harmonic_ratio,
    total_harmonic_distortion,
    fft_magnitude,
    hilbert_transform,
    envelope,
    instantaneous_frequency,
    instantaneous_amplitude,
    instantaneous_phase,
    spectral_slope,
    signal_to_noise,
)

__all__ = [
    "__version__",
    "BACKEND",
    "power_spectral_density",
    "dominant_frequency",
    "spectral_entropy",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "spectral_flatness",
    "harmonic_ratio",
    "total_harmonic_distortion",
    "fft_magnitude",
    "hilbert_transform",
    "envelope",
    "instantaneous_frequency",
    "instantaneous_amplitude",
    "instantaneous_phase",
    "spectral_slope",
    "signal_to_noise",
]
