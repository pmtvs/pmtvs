# pmtvs

Rust-accelerated signal analysis primitives.

**numpy in, number out.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE.md)

## Installation

```bash
# Install all packages
pip install pmtvs

# Or install individual packages
pip install pmtvs-entropy
pip install pmtvs-fractal
pip install pmtvs-statistics
```

## Quick Start

```python
import numpy as np
from pmtvs_entropy import sample_entropy, permutation_entropy
from pmtvs_fractal import hurst_exponent, dfa
from pmtvs_statistics import mean, std, skewness

# Generate sample signal
signal = np.random.randn(1000)

# Entropy measures
se = sample_entropy(signal)
pe = permutation_entropy(signal)

# Fractal analysis
h = hurst_exponent(signal)
alpha = dfa(signal)

# Basic statistics
m = mean(signal)
s = std(signal)
sk = skewness(signal)
```

## Packages

| Package | Functions | Rust | Description |
|---------|-----------|------|-------------|
| pmtvs-entropy | 3 | 2 | Sample entropy, permutation entropy |
| pmtvs-fractal | 3 | 3 | Hurst exponent, DFA |
| pmtvs-statistics | 17 | 14 | Statistics and calculus |
| pmtvs-correlation | 11 | 4 | Correlation, autocorrelation |
| pmtvs-distance | 4 | 3 | Distance metrics |
| pmtvs-embedding | 4 | 1 | Time delay embedding |
| pmtvs-dynamics | 17 | 0 | Lyapunov, FTLE, RQA |
| pmtvs-spectral | 12 | 0 | FFT, PSD, wavelets |
| pmtvs-matrix | 10 | 0 | SVD, covariance |
| pmtvs-topology | 6 | 0 | Persistent homology |
| pmtvs-network | 14 | 0 | Graph analysis |
| pmtvs-information | 13 | 0 | Information theory |
| pmtvs-tests | 18 | 0 | Statistical tests |

## Rust Acceleration

Each function earns Rust acceleration by proving:
1. **Parity** — Output matches Python within tolerance
2. **Speedup** — Actually faster than Python

Functions failing either criterion use Python. No exceptions.

Disable Rust globally:
```bash
export PMTVS_USE_RUST=0
```

## Development

```bash
# Clone and setup
git clone https://github.com/pmtvs/pmtvs.git
cd pmtvs

# Install a specific package in development mode
cd packages/pmtvs-entropy
pip install -e .

# With Rust (requires Rust toolchain)
maturin develop --release

# Run tests
pytest tests/ -v
```

## License

MIT License. Copyright (c) 2025 pmtvs contributors.
