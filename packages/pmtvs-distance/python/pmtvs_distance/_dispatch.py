"""
Rust dispatch configuration for pmtvs-distance.

RUST_VALIDATED lists functions validated to produce identical results
to their Python implementations (within floating-point tolerance).
"""
import os

RUST_VALIDATED = frozenset({
    'euclidean_distance',   # 8.2x speedup
    'cosine_distance',      # 7.5x speedup
    'manhattan_distance',   # 9.1x speedup
})


def use_rust(func_name: str) -> bool:
    """Check if Rust should be used for a function."""
    if os.environ.get("PMTVS_USE_RUST", "1") == "0":
        return False
    return func_name in RUST_VALIDATED
