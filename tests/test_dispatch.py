"""Test the dispatch mechanism"""

import os
import pytest


def test_import():
    """Package imports without error"""
    import pmtvs
    assert hasattr(pmtvs, "__version__")
    assert hasattr(pmtvs, "BACKEND")
    assert pmtvs.BACKEND in ("rust", "python")


def test_backend_detection():
    """Backend is correctly detected"""
    import pmtvs
    from pmtvs._dispatch import is_rust_available, get_backend

    backend = get_backend()
    assert backend == pmtvs.BACKEND

    if is_rust_available():
        # If Rust is available and not disabled, should use Rust
        if os.environ.get("PMTVS_USE_RUST", "1").lower() not in ("0", "false", "no"):
            assert backend == "rust"
    else:
        assert backend == "python"


def test_rust_validated_frozen():
    """RUST_VALIDATED set is frozen and cannot be modified"""
    from pmtvs._dispatch import RUST_VALIDATED

    assert isinstance(RUST_VALIDATED, frozenset)

    with pytest.raises(AttributeError):
        RUST_VALIDATED.add("fake_function")


def test_dispatch_decorator():
    """dispatch() returns correct backend"""
    from pmtvs._dispatch import dispatch, RUST_VALIDATED, is_rust_available

    def python_impl(x):
        return x * 2

    # For a validated function
    if "sample_entropy" in RUST_VALIDATED:
        wrapped = dispatch("sample_entropy", python_impl)
        if is_rust_available():
            assert wrapped._backend == "rust"
        else:
            assert wrapped._backend == "python"

    # For a non-validated function
    wrapped = dispatch("not_in_validated_set", python_impl)
    assert wrapped._backend == "python"


def test_python_only_decorator():
    """python_only() marks functions correctly"""
    from pmtvs._dispatch import python_only

    @python_only
    def my_func(x):
        return x

    assert my_func._backend == "python"
    assert my_func._python_only is True
    assert my_func(5) == 5
