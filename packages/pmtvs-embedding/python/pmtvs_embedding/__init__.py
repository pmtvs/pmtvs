"""
pmtvs-embedding — Time-delay embedding for dynamical systems analysis.

numpy in, array out.
"""
import os

from pmtvs_embedding._dispatch import RUST_VALIDATED

__version__ = "0.1.0"

_RUST_AVAILABLE = False
_RUST_DISABLED = os.environ.get("PMTVS_USE_RUST", "1") == "0"

if not _RUST_DISABLED:
    try:
        import pmtvs_embedding._rust as _rust_mod
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
from pmtvs_embedding.embedding import (
    delay_embedding,
    optimal_embedding_dimension,
    mutual_information_delay,
    false_nearest_neighbors,
)

__all__ = [
    "__version__",
    "BACKEND",
    "delay_embedding",
    "optimal_embedding_dimension",
    "mutual_information_delay",
    "false_nearest_neighbors",
]
