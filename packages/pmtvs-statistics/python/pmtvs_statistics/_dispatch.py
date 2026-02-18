"""
Per-function backend dispatch for pmtvs-statistics.

Each function earns Rust by proving BOTH:
  1. Parity OK — output matches Python within tolerance
  2. Speedup > 1.0x — actually faster

Functions failing either criterion use Python. No exceptions.
"""

RUST_VALIDATED = frozenset({
    'mean',
    'std',
    'variance',
    'min_max',
    'rms',
    'peak_to_peak',
    'skewness',
    'kurtosis',
    'crest_factor',
    'pulsation_index',
    'derivative',
    'integral',
    'curvature',
    'rate_of_change',
})


def use_rust(func_name: str) -> bool:
    """Check if a function should use Rust backend."""
    return func_name in RUST_VALIDATED
