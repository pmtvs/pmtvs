"""
Edge cases that break naive implementations.
Every function must handle these gracefully — return NaN, not crash.

Run: pytest tests/test_edge_cases.py -v --timeout=30 --tb=short
"""
import numpy as np
import pytest
import os
import warnings

os.environ['PMTVS_USE_RUST'] = '0'

import pmtvs

# Default per-test timeout (seconds)
pytestmark = pytest.mark.timeout(30)


# ════════════════════════════════════════════
# Auto-discover all callable public functions
# ════════════════════════════════════════════

# Functions that require more than a single 1D array positional arg
# (two-sample tests, functions needing a callable, etc.)
SKIP_SINGLE_ARG = {
    'permutation_test',       # (x, y, statistic)
    'surrogate_test',         # (signal, statistic) — statistic is required
    't_test_paired',          # (x, y)
    't_test_independent',     # (x, y)
    'mannwhitney_test',       # (x, y)
    'kruskal_test',           # (x, y)
    'f_test',                 # (x, y)
    'anova',                  # (*groups)
}

# Functions known to be computationally expensive (>10s per call)
# Tested separately in dedicated deep test files
SLOW_FUNCTIONS = {
    # Lyapunov / chaos (embedding + neighbor search)
    'largest_lyapunov_exponent',
    'lyapunov_rosenstein',
    'lyapunov_kantz',
    'lyapunov_spectrum',
    'correlation_dimension',
    'correlation_integral',
    'information_dimension',
    'kaplan_yorke_dimension',
    # Sensitivity / basin (many iterations)
    'compute_basin_stability',
    'basin_stability',
    'compute_sensitivity_evolution',
    'detect_sensitivity_transitions',
    'compute_influence_matrix',
    'compute_variable_sensitivity',
    'compute_directional_sensitivity',
    # Embedding / neighbor search
    'convergent_cross_mapping',
    'estimate_embedding_dim_cao',
    'false_nearest_neighbors',
    'ftle_local_linearization',
    'ftle_direct_perturbation',
    # Recurrence (O(n^2) matrix construction)
    'recurrence_matrix',
    'determinism_from_signal',
    'rqa_metrics',
    # Bootstrap (many resamples)
    'marchenko_pastur_test',
    'block_bootstrap_ci',
    # Granger (regression over many lags)
    'granger_causality',
    'granger_matrix',
    'transfer_entropy_matrix',
    # Multiscale (many scales)
    'multiscale_entropy',
    # Other slow functions (embedding/neighbor ops)
    'attractor_reconstruction',
    'estimate_tau_ami',
    'multivariate_embedding',
    'wavelet_coherence',
    'wavelet_stability',
    'hilbert_stability',
    'phase_coupling',
    'cross_spectral_density',
    'coherence',
    'partial_information_decomposition',
    'information_atoms',
    'information_flow',
    'transfer_entropy',
    'detect_collapse',
    # Information-theoretic (expensive for multi-dim)
    'dual_total_correlation',
    'total_correlation',
    'multivariate_mutual_information',
    'mutual_information_matrix',
    'interaction_information',
    'conditional_mutual_information',
    'redundancy',
    'synergy',
}

SCALAR_FUNCTIONS = []

for name in sorted(dir(pmtvs)):
    if name.startswith('_'):
        continue
    if name in ('BACKEND',):
        continue
    obj = getattr(pmtvs, name)
    if callable(obj) and not isinstance(obj, type) and name not in SKIP_SINGLE_ARG:
        SCALAR_FUNCTIONS.append((name, obj))

# Fast subset: exclude slow functions for intensive parametrized tests
FAST_FUNCTIONS = [(n, f) for n, f in SCALAR_FUNCTIONS if n not in SLOW_FUNCTIONS]


class TestEmptyInput:
    """Empty array should return NaN or empty, NEVER crash."""

    @pytest.mark.parametrize("name,func", SCALAR_FUNCTIONS, ids=[n for n, _ in SCALAR_FUNCTIONS])
    def test_empty_array(self, name, func):
        x = np.array([], dtype=float)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = func(x)
            if isinstance(result, float):
                assert np.isnan(result) or result == 0.0, \
                    f"{name} returned {result} on empty input"
        except (ValueError, IndexError, ZeroDivisionError) as e:
            pytest.fail(f"{name} CRASHED on empty input: {type(e).__name__}: {e}")
        except TypeError:
            pass  # Some functions need kwargs — skip


class TestSingleValue:
    """Single-element array — most statistics are undefined."""

    @pytest.mark.parametrize("name,func", SCALAR_FUNCTIONS, ids=[n for n, _ in SCALAR_FUNCTIONS])
    def test_single_value(self, name, func):
        x = np.array([42.0])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = func(x)
        except (ValueError, IndexError, ZeroDivisionError) as e:
            pytest.fail(f"{name} CRASHED on single value: {type(e).__name__}: {e}")
        except TypeError:
            pass


class TestConstantSignal:
    """Constant signal — zero variance, many things undefined."""

    @pytest.mark.parametrize("name,func", SCALAR_FUNCTIONS, ids=[n for n, _ in SCALAR_FUNCTIONS])
    def test_constant(self, name, func):
        x = np.ones(500) * 3.14
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = func(x)
            if isinstance(result, float):
                assert not np.isinf(result), \
                    f"{name} returned inf on constant signal"
            elif isinstance(result, dict):
                for k, v in result.items():
                    if isinstance(v, float):
                        assert not np.isinf(v), \
                            f"{name}['{k}'] returned inf on constant signal"
        except (ValueError, IndexError, ZeroDivisionError) as e:
            pytest.fail(f"{name} CRASHED on constant signal: {type(e).__name__}: {e}")
        except TypeError:
            pass


class TestAllNaN:
    """All-NaN input."""

    @pytest.mark.parametrize("name,func", SCALAR_FUNCTIONS, ids=[n for n, _ in SCALAR_FUNCTIONS])
    def test_all_nan(self, name, func):
        x = np.full(500, np.nan)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = func(x)
        except (ValueError, IndexError, ZeroDivisionError, FloatingPointError) as e:
            pytest.fail(f"{name} CRASHED on all-NaN: {type(e).__name__}: {e}")
        except TypeError:
            pass


class TestSparseNaN:
    """Signal with scattered NaNs — common in real sensor data."""

    @pytest.mark.parametrize("name,func", FAST_FUNCTIONS, ids=[n for n, _ in FAST_FUNCTIONS])
    def test_sparse_nan(self, name, func):
        np.random.seed(42)
        x = np.random.randn(500)
        x[np.random.choice(500, 50, replace=False)] = np.nan  # 10% NaN
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = func(x)
            if isinstance(result, float):
                assert not np.isinf(result), \
                    f"{name} returned inf with sparse NaN"
        except (ValueError, IndexError, ZeroDivisionError) as e:
            pytest.fail(f"{name} CRASHED with sparse NaN: {type(e).__name__}: {e}")
        except TypeError:
            pass


class TestInfInput:
    """Signal containing inf values."""

    @pytest.mark.parametrize("name,func", FAST_FUNCTIONS, ids=[n for n, _ in FAST_FUNCTIONS])
    def test_contains_inf(self, name, func):
        np.random.seed(42)
        x = np.random.randn(500)
        x[100] = np.inf
        x[200] = -np.inf
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = func(x)
        except (ValueError, IndexError, ZeroDivisionError, OverflowError) as e:
            pytest.fail(f"{name} CRASHED with inf input: {type(e).__name__}: {e}")
        except TypeError:
            pass


class TestVeryShort:
    """Minimum viable signal lengths — 2, 3, 5, 10 points."""

    @pytest.mark.parametrize("length", [2, 3, 5, 10])
    @pytest.mark.parametrize("name,func", FAST_FUNCTIONS, ids=[n for n, _ in FAST_FUNCTIONS])
    def test_short_signal(self, name, func, length):
        np.random.seed(42)
        x = np.random.randn(length)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = func(x)
        except (ValueError, IndexError, ZeroDivisionError) as e:
            pytest.fail(f"{name} CRASHED on length={length}: {type(e).__name__}: {e}")
        except TypeError:
            pass


# Extra exclusions for 100k signal (O(n^2) or worse)
_LONG_SKIP = SLOW_FUNCTIONS | {
    'adjacency_from_correlation', 'distance_matrix', 'average_path_length',
    'betweenness_centrality', 'closeness_centrality', 'clustering_coefficient',
    'community_detection', 'connected_components', 'degree_centrality',
    'eigenvector_centrality', 'graph_laplacian', 'modularity',
    'network_density', 'density', 'centrality_betweenness', 'centrality_eigenvector',
    'recurrence_rate', 'determinism', 'laminarity', 'trapping_time',
    'max_diagonal_line', 'divergence_rqa', 'entropy_recurrence', 'entropy_rqa',
    'correlation_matrix', 'covariance_matrix', 'partial_correlation',
    'approximate_entropy', 'sample_entropy',
}
LONG_FUNCTIONS = [(n, f) for n, f in SCALAR_FUNCTIONS if n not in _LONG_SKIP][:15]


class TestVeryLong:
    """Large signal — memory and performance check.
    Only tests functions known to be O(n) or O(n log n).
    """

    @pytest.mark.parametrize("name,func", LONG_FUNCTIONS, ids=[n for n, _ in LONG_FUNCTIONS])
    def test_long_signal(self, name, func):
        np.random.seed(42)
        x = np.random.randn(100_000)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = func(x)
        except MemoryError:
            pytest.fail(f"{name} ran out of memory on 100k signal")
        except TypeError:
            pass


class TestExtremeValues:
    """Very large and very small values — numerical stability."""

    @pytest.mark.parametrize("name,func", SCALAR_FUNCTIONS, ids=[n for n, _ in SCALAR_FUNCTIONS])
    def test_very_large(self, name, func):
        np.random.seed(42)
        x = np.random.randn(500) * 1e15
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = func(x)
            if isinstance(result, float):
                assert not np.isinf(result), \
                    f"{name} returned inf on large-scale signal"
        except (ValueError, OverflowError, FloatingPointError) as e:
            pytest.fail(f"{name} CRASHED on large values: {type(e).__name__}: {e}")
        except TypeError:
            pass

    @pytest.mark.parametrize("name,func", SCALAR_FUNCTIONS, ids=[n for n, _ in SCALAR_FUNCTIONS])
    def test_very_small(self, name, func):
        np.random.seed(42)
        x = np.random.randn(500) * 1e-15
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = func(x)
            if isinstance(result, float):
                assert not np.isinf(result), \
                    f"{name} returned inf on tiny-scale signal"
        except (ValueError, ZeroDivisionError) as e:
            pytest.fail(f"{name} CRASHED on tiny values: {type(e).__name__}: {e}")
        except TypeError:
            pass


class TestStepFunction:
    """Step function — discontinuity in the middle."""

    @pytest.mark.parametrize("name,func", SCALAR_FUNCTIONS, ids=[n for n, _ in SCALAR_FUNCTIONS])
    def test_step(self, name, func):
        x = np.concatenate([np.zeros(250), np.ones(250)])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = func(x)
        except (ValueError, IndexError, ZeroDivisionError) as e:
            pytest.fail(f"{name} CRASHED on step function: {type(e).__name__}: {e}")
        except TypeError:
            pass


class TestAlternating:
    """Rapidly alternating signal — worst case for many estimators."""

    @pytest.mark.parametrize("name,func", SCALAR_FUNCTIONS, ids=[n for n, _ in SCALAR_FUNCTIONS])
    def test_alternating(self, name, func):
        x = np.array([1.0, -1.0] * 250)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = func(x)
        except (ValueError, IndexError, ZeroDivisionError) as e:
            pytest.fail(f"{name} CRASHED on alternating: {type(e).__name__}: {e}")
        except TypeError:
            pass
