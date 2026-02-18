# pmtvs

**P**rimitives for **M**ultivariate **T**ime series and dynamical systems analysis, with optional **Rust** acceleration.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE.md)

## Overview

pmtvs is a standalone library of ~200 numerical primitives for time series and dynamical systems analysis. It provides:

- **Statistics**: mean, std, variance, skewness, kurtosis, percentiles, RMS, crest factor
- **Calculus**: derivatives, integrals, curvature
- **Spectral**: FFT, PSD, dominant frequency, spectral entropy, wavelets
- **Entropy**: sample entropy, permutation entropy, approximate entropy
- **Fractal**: Hurst exponent, DFA
- **Correlation**: auto/cross-correlation, partial correlation, Granger causality
- **Embedding**: time delay embedding, optimal delay/dimension (Cao's method)
- **Dynamical**: Lyapunov exponents, FTLE, recurrence quantification
- **Topology**: persistent homology, Betti numbers, Wasserstein distance
- **Network**: centrality, clustering, community detection
- **Information**: mutual information, transfer entropy, divergences
- **Statistical Tests**: t-tests, bootstrap, normalization, stationarity tests

## Installation

### Python-only (no Rust required)

```bash
pip install pmtvs
# or from source:
pip install -e .
```

### With Rust acceleration

Requires Rust toolchain (`rustup`).

```bash
pip install maturin
maturin develop --release -m pyproject-maturin.toml
```

## Quick Start

```python
import numpy as np
import pmtvs

# Check backend
print(f"Backend: {pmtvs.BACKEND}")  # 'rust' or 'python'

# Example: compute sample entropy
from pmtvs import sample_entropy
data = np.random.randn(1000)
se = sample_entropy(data, m=2, r=0.2 * np.std(data))
print(f"Sample entropy: {se:.4f}")

# Example: Hurst exponent
from pmtvs import hurst_exponent
h = hurst_exponent(data)
print(f"Hurst exponent: {h:.4f}")
```

## Forcing Python-only Mode

```bash
PMTVS_USE_RUST=0 python your_script.py
```

## Rust-Accelerated Functions

The following functions dispatch to Rust when available:

| Function | Module | Speedup |
|----------|--------|---------|
| `sample_entropy` | individual | ~50x |
| `permutation_entropy` | individual | ~20x |
| `approximate_entropy` | individual | ~50x |
| `hurst_exponent` | individual | ~10x |
| `dfa` | individual | ~10x |
| `time_delay_embedding` | embedding | ~5x |
| `optimal_delay` | embedding | ~10x |

## Development

```bash
# Clone and setup
git clone https://github.com/pmtvs/pmtvs.git
cd pmtvs

# Python-only development
pip install -e ".[dev]"

# With Rust
pip install maturin
maturin develop

# Run tests
pytest tests/ -v

# Run benchmarks
python benchmarks/bench_all.py
```

## License

MIT License - see [LICENSE.md](LICENSE.md)

## Credits

- pmtvs contributors
