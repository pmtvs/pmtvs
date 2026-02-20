"""
pmtvs-entropy — Entropy primitives for signal analysis.

numpy in, number out.
"""
import os

from pmtvs_entropy._dispatch import RUST_VALIDATED

__version__ = "0.3.1"

_RUST_AVAILABLE = False
_RUST_DISABLED = os.environ.get("PMTVS_USE_RUST", "1") == "0"

if not _RUST_DISABLED:
    try:
        import pmtvs_entropy._rust as _rust_mod
        _RUST_AVAILABLE = True
    except ImportError:
        pass


def _get_rust(name):
    """Get function from Rust if validated and available, else None."""
    if _RUST_AVAILABLE and name in RUST_VALIDATED:
        fn = getattr(_rust_mod, name, None)
        if fn is not None:
            return fn
    return None


# --- Backend reporting ---
if _RUST_AVAILABLE:
    _rust_count = sum(
        1 for name in RUST_VALIDATED
        if getattr(_rust_mod, name, None) is not None
    )
    BACKEND = f"rust={_rust_count}"
else:
    BACKEND = "python"


# --- Public API ---
from pmtvs_entropy.core import (
    sample_entropy,
    permutation_entropy,
    approximate_entropy,
    multiscale_entropy,
    lempel_ziv_complexity,
    entropy_rate,
)

__all__ = [
    "__version__",
    "BACKEND",
    "sample_entropy",
    "permutation_entropy",
    "approximate_entropy",
    "multiscale_entropy",
    "lempel_ziv_complexity",
    "entropy_rate",
]
