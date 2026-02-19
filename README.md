# pmtvs

Rust-accelerated signal analysis primitives.

**numpy in, number out.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: PolyForm Noncommercial](https://img.shields.io/badge/License-PolyForm%20NC-blue.svg)](LICENSE)

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
from pmtvs_fractal import hurst_exponent, dfa, rescaled_range
from pmtvs_statistics import mean, std, skewness, first_derivative
from pmtvs_information import mutual_information, granger_causality
from pmtvs_regression import linear_regression

# Generate sample signal
signal = np.random.randn(1000)

# Entropy measures
se = sample_entropy(signal)
pe = permutation_entropy(signal)

# Fractal analysis
h = hurst_exponent(signal)
alpha = dfa(signal)
rs = rescaled_range(signal)

# Basic statistics
m = mean(signal)
s = std(signal)
sk = skewness(signal)
dx = first_derivative(signal)

# Pairwise analysis
slope, intercept, r2, se = linear_regression(signal[:500], signal[500:])
```

## Packages

| Package | Functions | Rust | Description |
|---------|-----------|------|-------------|
| pmtvs-entropy | 3 | 2 | Sample entropy, permutation entropy |
| pmtvs-fractal | 7 | 3 | Hurst exponent, DFA, rescaled range, long-range correlation |
| pmtvs-statistics | 38 | 14 | Statistics, calculus, derivatives, normalization |
| pmtvs-correlation | 15 | 4 | Correlation, autocorrelation, Spearman, Kendall |
| pmtvs-distance | 6 | 3 | Distance metrics, Earth mover's, cosine similarity |
| pmtvs-embedding | 4 | 1 | Time delay embedding |
| pmtvs-dynamics | 46 | 0 | Lyapunov, FTLE, RQA, saddle points, sensitivity |
| pmtvs-spectral | 9 | 0 | FFT, PSD, wavelets |
| pmtvs-matrix | 22 | 0 | SVD, covariance, DMD, geometry |
| pmtvs-topology | 6 | 0 | Persistent homology |
| pmtvs-network | 10 | 0 | Graph analysis, community detection |
| pmtvs-information | 25 | 0 | Information theory, causality, decomposition |
| pmtvs-tests | 27 | 0 | Statistical & hypothesis tests, stationarity |
| pmtvs-regression | 5 | 0 | Linear regression, signal arithmetic |

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

## Reporting Issues

The most useful bug report includes three things: a minimal code example that reproduces the problem, the value pmtvs returned, and the value you expected with a source (published paper, competing library, or analytical solution). "sample_entropy returns 2.14 but Richman-Moorman Table 2 gives 2.09 for this test vector" is a perfect bug report. We will investigate every well-sourced discrepancy. Edge cases are especially welcome — constant signals, very short signals, NaN-heavy data, extreme outliers. If you can break it, we want to know.

## License

PolyForm Noncommercial 1.0.0 with Additional Terms.

- **Students & individual researchers:** Free. Cite us.
- **Funded research labs (grants > $100K):** Academic Research License required. [Contact us](mailto:licensing@pmtvs.dev).
- **Commercial use:** Commercial License required. [Contact us](mailto:licensing@pmtvs.dev).

See [LICENSE](LICENSE) for full terms.
