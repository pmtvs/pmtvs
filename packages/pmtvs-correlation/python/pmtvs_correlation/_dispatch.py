"""
Per-function backend dispatch for pmtvs-correlation.

Each function earns Rust by proving BOTH:
  1. Parity OK — output matches Python within tolerance
  2. Speedup > 1.0x — actually faster

Functions failing either criterion use Python. No exceptions.
"""

RUST_VALIDATED = frozenset({
    'autocorrelation',          # 4.6x
    'partial_autocorrelation',  # 5.3x
    'correlation',              # 4.7x
    'covariance',               # 4.4x
})


def use_rust(func_name: str) -> bool:
    """Check if a function should use Rust backend."""
    return func_name in RUST_VALIDATED
