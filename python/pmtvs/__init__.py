"""
pmtvs - High-Performance Primitives for Time Series and Dynamical Systems

A standalone library of ~200 numerical primitives for time series analysis,
dynamical systems characterization, and signal processing. Optionally
accelerated by Rust for computationally intensive functions.

Usage:
    import pmtvs
    print(pmtvs.BACKEND)  # 'rust' or 'python'

    # All functions work regardless of backend
    from pmtvs import sample_entropy, hurst_exponent, fft
    result = sample_entropy(data, m=2, r=0.2)

Environment:
    PMTVS_USE_RUST=0  # Force Python-only mode

Modules:
    individual/     - Single-signal computations (stats, spectral, entropy)
    pairwise/       - Two-signal computations (correlation, causality)
    matrix/         - Multi-signal computations (covariance, decomposition)
    embedding/      - Phase space reconstruction
    dynamical/      - Chaos and recurrence analysis
    topology/       - Persistent homology
    network/        - Graph metrics
    information/    - Information theory
    statistical_tests/  - Hypothesis tests, normalization, bootstrap
"""

from pmtvs._dispatch import get_backend, is_rust_available

__version__ = "0.1.0"
BACKEND = get_backend()
RUST_AVAILABLE = is_rust_available()

# Individual primitives
from pmtvs.individual import (
    # Statistics
    mean,
    std,
    variance,
    min_max,
    percentiles,
    skewness,
    kurtosis,
    rms,
    peak_to_peak,
    crest_factor,
    zero_crossings,
    mean_crossings,
    # Calculus
    derivative,
    integral,
    curvature,
    # Entropy
    sample_entropy,
    permutation_entropy,
    approximate_entropy,
    # Fractal
    hurst_exponent,
    dfa,
    hurst_r2,
)

__all__ = [
    "__version__",
    "BACKEND",
    "RUST_AVAILABLE",
    "get_backend",
    "is_rust_available",
    # Statistics
    'mean',
    'std',
    'variance',
    'min_max',
    'percentiles',
    'skewness',
    'kurtosis',
    'rms',
    'peak_to_peak',
    'crest_factor',
    'zero_crossings',
    'mean_crossings',
    # Calculus
    'derivative',
    'integral',
    'curvature',
    # Entropy
    'sample_entropy',
    'permutation_entropy',
    'approximate_entropy',
    # Fractal
    'hurst_exponent',
    'dfa',
    'hurst_r2',
]
