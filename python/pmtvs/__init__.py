"""
pmtvs — Rust-accelerated signal analysis primitives.

numpy in, number out.
"""
import os

from pmtvs._dispatch import RUST_VALIDATED

__version__ = "0.1.0"

_RUST_AVAILABLE = False
_RUST_DISABLED = os.environ.get("PMTVS_USE_RUST", "1") == "0"

if not _RUST_DISABLED:
    try:
        import pmtvs._rust as _rust_mod
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
    BACKEND = f"hybrid:rust={_rust_count},python=rest"
else:
    BACKEND = "python"


# --- Individual primitives (added by subsequent PRs) ---
from pmtvs.individual import (
    # Statistics
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
    # Calculus
    derivative,
    integral,
    curvature,
    # Entropy
    sample_entropy,
    permutation_entropy,
    approximate_entropy,
    # Fractal
    hurst_exponent,
    dfa,
    hurst_r2,
)

__all__ = [
    "__version__",
    "BACKEND",
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
