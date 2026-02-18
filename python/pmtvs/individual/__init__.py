"""
Individual signal primitives

Single-signal computations: statistics, spectral, entropy, fractal, etc.
"""

from .statistics import (
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
)

from .calculus import (
    derivative,
    integral,
    curvature,
)

from .entropy import (
    sample_entropy,
    permutation_entropy,
    approximate_entropy,
)

from .fractal import (
    hurst_exponent,
    dfa,
    hurst_r2,
)

__all__ = [
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
