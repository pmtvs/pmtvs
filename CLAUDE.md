# CLAUDE.md

## Project Overview

**pmtvs** is a monorepo of Rust-accelerated signal analysis primitives. 14 micro-packages under `packages/`, one umbrella package (`pmtvs`) that re-exports everything as a flat API.

**numpy in, number out.**

244 total functions. 21 with validated Rust acceleration.

## Repository Structure

```
packages/
  pmtvs-entropy/       # Rust-accelerated (maturin)
  pmtvs-fractal/       # Rust-accelerated (maturin)
  pmtvs-statistics/    # Rust-accelerated (maturin)
  pmtvs-correlation/   # Rust-accelerated (maturin)
  pmtvs-distance/      # Rust-accelerated (maturin)
  pmtvs-embedding/     # Rust-accelerated (maturin)
  pmtvs-dynamics/      # Pure Python (setuptools)
  pmtvs-spectral/      # Pure Python (setuptools)
  pmtvs-matrix/        # Pure Python (setuptools)
  pmtvs-topology/      # Pure Python (setuptools)
  pmtvs-network/       # Pure Python (setuptools)
  pmtvs-information/   # Pure Python (setuptools)
  pmtvs-tests/         # Pure Python (setuptools)
  pmtvs-regression/    # Pure Python (setuptools)
  pmtvs/               # Umbrella package (re-exports all)
```

Each micro-package follows:
```
packages/pmtvs-<name>/
  pyproject.toml
  python/pmtvs_<name>/__init__.py   # __all__, __version__, BACKEND
  python/pmtvs_<name>/<modules>.py
  tests/test_<name>.py
  src/lib.rs                         # (Rust packages only)
  Cargo.toml                         # (Rust packages only)
```

## Flat API

Users import everything from the umbrella package:

```python
from pmtvs import sample_entropy, hurst_exponent, eigendecomposition
```

Never expose hierarchical imports (`from pmtvs.individual.entropy import ...`).
The umbrella `pmtvs/__init__.py` uses `from pmtvs_xxx import *` for all 14 packages.

## Coding Conventions

- Input normalization: `np.asarray(signal, dtype=np.float64).flatten()` for 1D inputs
- NaN filtering: `signal = signal[~np.isnan(signal)]` where appropriate
- Scalar returns: always `float(result)`, never bare numpy scalars
- Edge cases: return `np.nan` (not raise) for degenerate inputs (too short, constant, etc.)
- Docstrings: NumPy-style
- Every public function must be in `__all__` in `__init__.py`
- `__version__ = "0.3.2"` and `BACKEND = "python"` (or `"rust"`) in each `__init__.py`

## Rust Acceleration

Six packages have optional Rust backends via PyO3/maturin. The pattern:
1. `_dispatch.py` checks `PMTVS_USE_RUST` env var and tries importing `_rust` module
2. Each function calls `use_rust('fn_name')` — if True, delegates to Rust
3. Falls back to Python if Rust unavailable or `PMTVS_USE_RUST=0`

Each function earns Rust acceleration by proving:
1. **Parity** — Output matches Python within tolerance
2. **Speedup** — Actually faster than Python

Functions failing either criterion ship as Python-only. No exceptions.

Build Rust packages: `cd packages/pmtvs-<name> && maturin develop --release`

## Testing

```bash
# Run all tests
pytest packages/*/tests/ -v

# Run one package
pytest packages/pmtvs-fractal/tests/ -v

# Test Python fallback for Rust packages
PMTVS_USE_RUST=0 pytest packages/pmtvs-entropy/tests/ -v
```

Tests use `np.random.seed(42)` for reproducibility and pytest with class-based organization.

## Building & Installing

```bash
# From PyPI (users)
pip install pmtvs

# Development install (all packages)
python -m venv .venv && source .venv/bin/activate
pip install maturin pytest numpy scipy

# Rust packages
for pkg in packages/pmtvs-entropy packages/pmtvs-fractal packages/pmtvs-statistics packages/pmtvs-correlation packages/pmtvs-distance packages/pmtvs-embedding; do
  cd $pkg && maturin develop && cd ../..
done

# Pure Python packages
for pkg in packages/pmtvs-dynamics packages/pmtvs-spectral packages/pmtvs-matrix packages/pmtvs-topology packages/pmtvs-network packages/pmtvs-information packages/pmtvs-tests packages/pmtvs-regression; do
  pip install -e $pkg
done

# Umbrella
pip install -e packages/pmtvs
```

## Key Rules

- Never add cross-package imports (each micro-package is self-contained)
- scipy is an optional dependency — only pmtvs-information and pmtvs-tests require it
- The umbrella `pmtvs/__init__.py` uses `from pmtvs_xxx import *` for all 14 packages
- When adding functions: add to module, add to `__init__.py` imports and `__all__`, add tests
- Flat API only: `from pmtvs import X` — never expose submodule hierarchy to users
- pmtvs is pure math. It never orchestrates, windows, parallelizes, or does I/O.
  Manifold handles all of that.
