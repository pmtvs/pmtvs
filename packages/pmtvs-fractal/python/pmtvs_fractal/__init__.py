"""
pmtvs-fractal — Fractal primitives for signal analysis.

numpy in, number out.
"""
import os

from pmtvs_fractal._dispatch import RUST_VALIDATED

__version__ = "0.1.0"

_RUST_AVAILABLE = False
_RUST_DISABLED = os.environ.get("PMTVS_USE_RUST", "1") == "0"

if not _RUST_DISABLED:
    try:
        import pmtvs_fractal._rust as _rust_mod
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
from pmtvs_fractal.core import (
    hurst_exponent,
    dfa,
    hurst_r2,
)

__all__ = [
    "__version__",
    "BACKEND",
    "hurst_exponent",
    "dfa",
    "hurst_r2",
]
