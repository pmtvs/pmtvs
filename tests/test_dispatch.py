"""Verify dispatch configuration."""
import pytest
from pmtvs._dispatch import RUST_VALIDATED, RUST_BENCHED


def test_no_overlap():
    overlap = RUST_VALIDATED & RUST_BENCHED
    assert not overlap, f"Functions in both sets: {overlap}"


def test_critical_functions_classified():
    critical = {
        'hurst_exponent', 'sample_entropy', 'permutation_entropy',
        'dfa', 'optimal_delay', 'lyapunov_rosenstein', 'lyapunov_kantz',
    }
    classified = RUST_VALIDATED | RUST_BENCHED
    unclassified = critical - classified
    assert not unclassified, f"Unclassified: {unclassified}"


def test_dynamics_not_in_rust():
    dangerous = {'lyapunov_rosenstein', 'lyapunov_kantz',
                 'ftle_direct_perturbation', 'ftle_local_linearization'}
    in_rust = dangerous & RUST_VALIDATED
    assert not in_rust, f"Unvalidated in RUST_VALIDATED: {in_rust}"


def test_backend_string():
    from pmtvs import BACKEND
    assert isinstance(BACKEND, str)
    assert BACKEND in ('python',) or BACKEND.startswith('hybrid:')
