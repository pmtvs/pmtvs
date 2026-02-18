"""
Rust dispatch mechanism for pmtvs

This module handles the transparent switching between Python and Rust
implementations. Set PMTVS_USE_RUST=0 to force Python-only mode.

The RUST_VALIDATED set contains functions that have been validated
for numerical parity with Python. DO NOT ADD FUNCTIONS without
running the parity test suite.
"""

import os
from typing import Callable, Optional

# Frozen set of validated Rust functions
# DO NOT MODIFY without parity validation
RUST_VALIDATED = frozenset({
    # Entropy (PR 2)
    "sample_entropy",
    "permutation_entropy",
    "approximate_entropy",
    # Fractal (PR 2)
    "hurst_exponent",
    "dfa",
    # Embedding (PR 2)
    "time_delay_embedding",
    "optimal_delay",
    # Stats (PR 1) - to be added after validation
    # "skewness",
    # "kurtosis",
    # "rms",
    # "peak_to_peak",
    # "crest_factor",
    # "zero_crossings",
    # "mean_crossings",
    # Calculus (PR 1) - to be added after validation
    # "derivative",
    # "integral",
    # "curvature",
    # Correlation (PR 3) - to be added after validation
    # "autocorrelation",
    # "partial_autocorrelation",
    # "cross_correlation",
    # "correlation",
    # "covariance",
    # "lag_at_max_xcorr",
})

# Try to import Rust module
_rust_module = None
_rust_available = False
_use_rust = os.environ.get("PMTVS_USE_RUST", "1").lower() not in ("0", "false", "no")

if _use_rust:
    try:
        from pmtvs import _rust
        _rust_module = _rust
        _rust_available = True
    except ImportError:
        pass


def get_backend() -> str:
    """Return the current backend: 'rust' or 'python'"""
    if _rust_available and _use_rust:
        return "rust"
    return "python"


def is_rust_available() -> bool:
    """Check if Rust backend is available"""
    return _rust_available


def dispatch(
    name: str,
    python_fn: Callable,
    rust_fn_name: Optional[str] = None,
) -> Callable:
    """
    Create a dispatching wrapper that calls Rust or Python implementation.

    Args:
        name: Function name (must be in RUST_VALIDATED to dispatch to Rust)
        python_fn: Python implementation
        rust_fn_name: Rust function name (default: {name}_rs)

    Returns:
        Wrapper function that dispatches to appropriate backend
    """
    if rust_fn_name is None:
        rust_fn_name = f"{name}_rs"

    # Only dispatch to Rust if:
    # 1. Rust is available
    # 2. PMTVS_USE_RUST is enabled
    # 3. Function is in the validated set
    should_use_rust = (
        _rust_available
        and _use_rust
        and name in RUST_VALIDATED
    )

    if should_use_rust:
        rust_fn = getattr(_rust_module, rust_fn_name, None)
        if rust_fn is not None:
            # Return Rust implementation with Python fallback on error
            def wrapper(*args, **kwargs):
                try:
                    return rust_fn(*args, **kwargs)
                except Exception:
                    # Fallback to Python on any Rust error
                    return python_fn(*args, **kwargs)

            wrapper.__name__ = name
            wrapper.__doc__ = python_fn.__doc__
            wrapper._backend = "rust"
            return wrapper

    # Return Python implementation
    python_fn._backend = "python"
    return python_fn


def rust_only(name: str, rust_fn_name: Optional[str] = None) -> Callable:
    """
    Get Rust-only implementation (raises if not available).

    Use this for benchmarking to ensure you're testing Rust.
    """
    if rust_fn_name is None:
        rust_fn_name = f"{name}_rs"

    if not _rust_available:
        raise RuntimeError(f"Rust backend not available for {name}")

    if name not in RUST_VALIDATED:
        raise RuntimeError(f"{name} is not in RUST_VALIDATED set")

    fn = getattr(_rust_module, rust_fn_name, None)
    if fn is None:
        raise RuntimeError(f"Rust function {rust_fn_name} not found")

    return fn


def python_only(python_fn: Callable) -> Callable:
    """
    Mark a function as Python-only (no Rust dispatch).

    Use this for functions that don't have Rust implementations
    or where Rust wouldn't provide meaningful speedup.
    """
    python_fn._backend = "python"
    python_fn._python_only = True
    return python_fn
