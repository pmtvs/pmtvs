"""
pmtvs-correlation — Correlation primitives for signal analysis.

numpy in, number out.
"""
import os

from pmtvs_correlation._dispatch import RUST_VALIDATED

__version__ = "0.3.3"

_RUST_AVAILABLE = False
_RUST_DISABLED = os.environ.get("PMTVS_USE_RUST", "1") == "0"

if not _RUST_DISABLED:
    try:
        import pmtvs_correlation._rust as _rust_mod
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
from pmtvs_correlation.individual import (
    autocorrelation,
    partial_autocorrelation,
    autocorrelation_function,
    acf_decay_time,
)

from pmtvs_correlation.pairwise import (
    correlation,
    covariance,
    cross_correlation,
    lag_at_max_xcorr,
    partial_correlation,
    coherence,
    cross_spectral_density,
    phase_spectrum,
    wavelet_coherence,
    spearman_correlation,
    kendall_tau,
)

__all__ = [
    "__version__",
    "BACKEND",
    # Individual
    "autocorrelation",
    "partial_autocorrelation",
    "autocorrelation_function",
    "acf_decay_time",
    # Pairwise
    "correlation",
    "covariance",
    "cross_correlation",
    "lag_at_max_xcorr",
    "partial_correlation",
    "coherence",
    "cross_spectral_density",
    "phase_spectrum",
    "wavelet_coherence",
    "spearman_correlation",
    "kendall_tau",
]
