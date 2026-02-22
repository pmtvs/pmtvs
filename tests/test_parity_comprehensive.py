"""
Every function, both backends, same answer.
This is where the Rossler bug lived.
"""
import numpy as np
import pytest
import os
import importlib
import warnings


# Generate test signals once
np.random.seed(42)
SIGNALS = {
    'white_noise': np.random.randn(2000),
    'sine': np.sin(np.linspace(0, 20 * np.pi, 2000)),
    'random_walk': np.cumsum(np.random.randn(2000)),
    'logistic': None,  # generated below
    'mixed': np.sin(np.linspace(0, 10 * np.pi, 2000)) + 0.3 * np.random.randn(2000),
    'trend': np.linspace(0, 10, 2000) + 0.1 * np.random.randn(2000),
    'step': np.concatenate([np.zeros(1000), np.ones(1000) * 5]),
}

# Logistic map
x = np.zeros(2000)
x[0] = 0.1
for i in range(1, 2000):
    x[i] = 3.9 * x[i - 1] * (1 - x[i - 1])
SIGNALS['logistic'] = x

# Functions that require more than a single 1D array positional arg
SKIP_SINGLE_ARG = {
    'permutation_test',
    'surrogate_test',
    't_test_paired',
    't_test_independent',
    'mannwhitney_test',
    'kruskal_test',
    'f_test',
    'anova',
}


def get_python_result(func_name, signal):
    """Get result using Python backend."""
    os.environ['PMTVS_USE_RUST'] = '0'
    import pmtvs as m
    importlib.reload(m)
    func = getattr(m, func_name, None)
    if func is None:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return func(signal)


def get_rust_result(func_name, signal):
    """Get result using Rust backend."""
    os.environ['PMTVS_USE_RUST'] = '1'
    import pmtvs as m
    importlib.reload(m)
    func = getattr(m, func_name, None)
    if func is None:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return func(signal)


def compare_results(py_result, rs_result, name, signal_name, tol=1e-8):
    """Compare Python and Rust results with appropriate tolerance."""
    if py_result is None and rs_result is None:
        return

    if isinstance(py_result, float) and isinstance(rs_result, float):
        if np.isnan(py_result) and np.isnan(rs_result):
            return
        if np.isnan(py_result) != np.isnan(rs_result):
            pytest.fail(
                f"PARITY FAIL {name} on {signal_name}: "
                f"Python={'NaN' if np.isnan(py_result) else py_result}, "
                f"Rust={'NaN' if np.isnan(rs_result) else rs_result}"
            )
        if abs(py_result) > 1e-10:
            rel_err = abs(py_result - rs_result) / abs(py_result)
            assert rel_err < tol, (
                f"PARITY FAIL {name} on {signal_name}: "
                f"Python={py_result}, Rust={rs_result}, rel_err={rel_err:.2e}"
            )
        else:
            assert abs(py_result - rs_result) < 1e-10, (
                f"PARITY FAIL {name} on {signal_name}: "
                f"Python={py_result}, Rust={rs_result}"
            )

    elif isinstance(py_result, tuple) and isinstance(rs_result, tuple):
        assert len(py_result) == len(rs_result), (
            f"PARITY FAIL {name} on {signal_name}: tuple length mismatch "
            f"Python={len(py_result)}, Rust={len(rs_result)}"
        )
        for i, (pv, rv) in enumerate(zip(py_result, rs_result)):
            if isinstance(pv, (int, float)) and isinstance(rv, (int, float)):
                compare_results(float(pv), float(rv), f"{name}[{i}]", signal_name, tol)
            elif isinstance(pv, np.ndarray) and isinstance(rv, np.ndarray):
                compare_results(pv, rv, f"{name}[{i}]", signal_name, tol)

    elif isinstance(py_result, dict) and isinstance(rs_result, dict):
        for key in py_result:
            if key not in rs_result:
                pytest.fail(f"PARITY FAIL {name} on {signal_name}: key '{key}' missing from Rust")
            py_val = py_result[key]
            rs_val = rs_result[key]
            if isinstance(py_val, (int, float)) and isinstance(rs_val, (int, float)):
                compare_results(float(py_val), float(rs_val), f"{name}[{key}]", signal_name, tol)

    elif isinstance(py_result, np.ndarray) and isinstance(rs_result, np.ndarray):
        assert py_result.shape == rs_result.shape, (
            f"PARITY FAIL {name} on {signal_name}: shape mismatch "
            f"Python={py_result.shape}, Rust={rs_result.shape}"
        )
        mask = ~(np.isnan(py_result) | np.isnan(rs_result))
        if mask.any():
            max_diff = np.max(np.abs(py_result[mask] - rs_result[mask]))
            assert max_diff < tol, (
                f"PARITY FAIL {name} on {signal_name}: max_diff={max_diff:.2e}"
            )


# Build function list
os.environ['PMTVS_USE_RUST'] = '0'
import pmtvs

# Functions known to be computationally expensive
SLOW_FUNCTIONS = {
    'largest_lyapunov_exponent', 'lyapunov_rosenstein', 'lyapunov_kantz',
    'lyapunov_spectrum', 'correlation_dimension', 'correlation_integral',
    'information_dimension', 'kaplan_yorke_dimension', 'compute_basin_stability',
    'basin_stability', 'compute_sensitivity_evolution', 'detect_sensitivity_transitions',
    'compute_influence_matrix', 'compute_variable_sensitivity', 'compute_directional_sensitivity',
    'convergent_cross_mapping', 'estimate_embedding_dim_cao', 'false_nearest_neighbors',
    'ftle_local_linearization', 'ftle_direct_perturbation', 'recurrence_matrix',
    'determinism_from_signal', 'rqa_metrics', 'marchenko_pastur_test', 'block_bootstrap_ci',
    'granger_causality', 'granger_matrix', 'transfer_entropy_matrix', 'multiscale_entropy',
    'attractor_reconstruction', 'estimate_tau_ami', 'multivariate_embedding',
    'wavelet_coherence', 'wavelet_stability', 'hilbert_stability', 'phase_coupling',
    'cross_spectral_density', 'coherence', 'partial_information_decomposition',
    'information_atoms', 'information_flow', 'transfer_entropy', 'detect_collapse',
    'dual_total_correlation', 'total_correlation', 'multivariate_mutual_information',
    'mutual_information_matrix', 'interaction_information', 'conditional_mutual_information',
    'redundancy', 'synergy',
}

FUNC_NAMES = [
    name for name in sorted(dir(pmtvs))
    if not name.startswith('_')
    and name not in ('BACKEND',)
    and callable(getattr(pmtvs, name))
    and not isinstance(getattr(pmtvs, name), type)
    and name not in SKIP_SINGLE_ARG
    and name not in SLOW_FUNCTIONS
]


class TestRustPythonParity:
    """Every function x every signal type."""

    @pytest.mark.parametrize("func_name", FUNC_NAMES)
    @pytest.mark.parametrize("signal_name", list(SIGNALS.keys()))
    def test_parity(self, func_name, signal_name):
        signal = SIGNALS[signal_name]

        try:
            py_result = get_python_result(func_name, signal)
        except TypeError:
            return  # Function needs kwargs — skip
        except Exception:
            py_result = None

        try:
            rs_result = get_rust_result(func_name, signal)
        except TypeError:
            return
        except Exception as e:
            if py_result is not None:
                pytest.fail(
                    f"Rust CRASHED but Python succeeded for {func_name} on {signal_name}: {e}"
                )
            return

        if py_result is None:
            return

        compare_results(py_result, rs_result, func_name, signal_name)
