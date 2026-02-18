"""
Per-function backend dispatch for pmtvs-entropy.

Each function earns Rust by proving BOTH:
  1. Parity OK — output matches Python within tolerance
  2. Speedup > 1.0x — actually faster

Functions failing either criterion use Python. No exceptions.
"""

RUST_VALIDATED = frozenset({
    'sample_entropy',       # 1441x
    'permutation_entropy',  # 26x
})


def use_rust(func_name: str) -> bool:
    """Check if a function should use Rust backend."""
    return func_name in RUST_VALIDATED
