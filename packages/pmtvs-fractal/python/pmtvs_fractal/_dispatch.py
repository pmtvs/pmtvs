"""
Per-function backend dispatch for pmtvs-fractal.

Each function earns Rust by proving BOTH:
  1. Parity OK — output matches Python within tolerance
  2. Speedup > 1.0x — actually faster

Functions failing either criterion use Python. No exceptions.
"""

RUST_VALIDATED = frozenset({
    'hurst_exponent',   # 297x
    'hurst_r2',         # 289x
    'dfa',              # 258x
})


def use_rust(func_name: str) -> bool:
    """Check if a function should use Rust backend."""
    return func_name in RUST_VALIDATED
