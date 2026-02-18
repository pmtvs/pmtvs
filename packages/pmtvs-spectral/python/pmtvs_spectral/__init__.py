"""
pmtvs-spectral — Spectral analysis primitives.

numpy in, array/number out.
"""

__version__ = "0.1.0"
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
]
