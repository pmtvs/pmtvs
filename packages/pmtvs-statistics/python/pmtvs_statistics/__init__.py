"""
pmtvs-statistics — Statistics and calculus primitives for signal analysis.

numpy in, number out.
"""
import os

from pmtvs_statistics._dispatch import RUST_VALIDATED

__version__ = "0.3.2"

_RUST_AVAILABLE = False
_RUST_DISABLED = os.environ.get("PMTVS_USE_RUST", "1") == "0"

if not _RUST_DISABLED:
    try:
        import pmtvs_statistics._rust as _rust_mod
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
from pmtvs_statistics.statistics import (
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
    pulsation_index,
    zero_crossings,
    mean_crossings,
)

from pmtvs_statistics.calculus import (
    derivative,
    integral,
    curvature,
    rate_of_change,
)

from pmtvs_statistics.derivatives import (
    first_derivative,
    second_derivative,
    gradient,
    laplacian,
    finite_difference,
    velocity,
    acceleration,
    jerk,
    smoothed_derivative,
)

from pmtvs_statistics.normalization import (
    zscore_normalize,
    robust_normalize,
    mad_normalize,
    minmax_normalize,
    quantile_normalize,
    inverse_normalize,
    normalize,
    recommend_method,
)

__all__ = [
    "__version__",
    "BACKEND",
    # Statistics
    "mean",
    "std",
    "variance",
    "min_max",
    "percentiles",
    "skewness",
    "kurtosis",
    "rms",
    "peak_to_peak",
    "crest_factor",
    "pulsation_index",
    "zero_crossings",
    "mean_crossings",
    # Calculus
    "derivative",
    "integral",
    "curvature",
    "rate_of_change",
    # Derivatives
    "first_derivative",
    "second_derivative",
    "gradient",
    "laplacian",
    "finite_difference",
    "velocity",
    "acceleration",
    "jerk",
    "smoothed_derivative",
    # Normalization
    "zscore_normalize",
    "robust_normalize",
    "mad_normalize",
    "minmax_normalize",
    "quantile_normalize",
    "inverse_normalize",
    "normalize",
    "recommend_method",
]
