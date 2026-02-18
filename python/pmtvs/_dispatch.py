"""
Per-function backend dispatch.

Each function earns Rust by proving BOTH:
  1. Parity OK — output matches Python within tolerance
  2. Speedup > 1.0x — actually faster

Functions failing either criterion use Python. No exceptions.
"""

RUST_VALIDATED = frozenset({
    'sample_entropy',
    'hurst_exponent',
    'hurst_r2',
    'dfa',
    'integral',
    'permutation_entropy',
    'optimal_delay',
    'skewness',
    'kurtosis',
    'derivative',
    'crest_factor',
    'partial_autocorrelation',
    'rate_of_change',
    'correlation',
    'autocorrelation',
    'covariance',
    'euclidean_distance',
    'pulsation_index',
    'manhattan_distance',
    'std',
    'curvature',
})

RUST_BENCHED = frozenset({
    'lyapunov_rosenstein',
    'lyapunov_kantz',
    'variance_growth',
    'dynamic_time_warping',
    'ftle_direct_perturbation',
    'ftle_local_linearization',
    'eigendecomposition',
    'time_delay_embedding',
    'optimal_dimension',
    'distance_matrix',
    'cross_correlation',
    'lag_at_max_xcorr',
    'hilbert_transform',
    'svd',
    'correlation_matrix',
    'covariance_matrix',
    'fft_magnitude',
    'envelope',
    'instantaneous_amplitude',
    'instantaneous_frequency',
})


def use_rust(func_name: str) -> bool:
    """Check if a function should use Rust backend."""
    return func_name in RUST_VALIDATED
