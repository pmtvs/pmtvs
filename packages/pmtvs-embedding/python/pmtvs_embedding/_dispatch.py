"""
Rust dispatch configuration for pmtvs-embedding.

RUST_VALIDATED lists functions validated to produce identical results
to their Python implementations (within floating-point tolerance).
"""
import os

RUST_VALIDATED = frozenset({
    'delay_embedding',  # 12x speedup
})


def use_rust(func_name: str) -> bool:
    """Check if Rust should be used for a function."""
    if os.environ.get("PMTVS_USE_RUST", "1") == "0":
        return False
    return func_name in RUST_VALIDATED
