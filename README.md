# pmtvs

Rust-accelerated signal analysis primitives.

**numpy in, number out.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: PolyForm Strict](https://img.shields.io/badge/License-PolyForm%20Strict-blue.svg)](LICENSE)

## Installation

```bash
pip install pmtvs
```

## Quick Start

```python
import numpy as np
from pmtvs import (
    sample_entropy, permutation_entropy,
    hurst_exponent, dfa,
    skewness, kurtosis,
    eigendecomposition, svd_decomposition,
    granger_causality, mutual_information,
)

signal = np.random.randn(5000)

# Entropy
print(sample_entropy(signal))       # ~2.19
print(permutation_entropy(signal))  # ~2.57

# Fractal
print(hurst_exponent(signal))       # ~0.50
print(dfa(signal))                  # ~0.50

# Statistics
print(skewness(signal))             # ~0.00
print(kurtosis(signal))             # ~3.00
```

## What's Inside

244 functions across 14 packages. One flat import.

| Package | Functions | Rust | Description |
|---------|-----------|------|-------------|
| pmtvs-entropy | 3 | 2 | Sample entropy, permutation entropy |
| pmtvs-fractal | 7 | 3 | Hurst exponent, DFA, rescaled range |
| pmtvs-statistics | 38 | 14 | Statistics, calculus, derivatives |
| pmtvs-correlation | 15 | 4 | Correlation, autocorrelation, Spearman |
| pmtvs-distance | 6 | 3 | Distance metrics, cosine similarity |
| pmtvs-embedding | 4 | 1 | Time delay embedding |
| pmtvs-dynamics | 46 | 0 | Lyapunov, FTLE, RQA, sensitivity |
| pmtvs-spectral | 9 | 0 | FFT, PSD, wavelets |
| pmtvs-matrix | 22 | 0 | SVD, covariance, DMD, geometry |
| pmtvs-topology | 6 | 0 | Persistent homology |
| pmtvs-network | 10 | 0 | Graph analysis, community detection |
| pmtvs-information | 25 | 0 | Information theory, causality |
| pmtvs-tests | 27 | 0 | Statistical & hypothesis tests |
| pmtvs-regression | 5 | 0 | Linear regression, signal arithmetic |

Full mathematical reference with LaTeX equations: [PRIMITIVES.md](PRIMITIVES.md)

## Rust Acceleration

21 functions have validated Rust backends (up to 1,441x on sample_entropy). Every Rust function must prove:

1. **Parity** — matches Python output within tolerance
2. **Speedup** — actually faster than Python

Functions failing either criterion use Python. No exceptions.

Disable Rust globally:
```bash
export PMTVS_USE_RUST=0
```

## Development

```bash
git clone https://github.com/pmtvs/pmtvs.git
cd pmtvs
pip install -e packages/pmtvs

# With Rust (requires Rust toolchain)
cd packages/pmtvs-entropy && maturin develop --release

# Run tests
pytest packages/*/tests/ -v
```

## Reporting Issues

The most useful bug report includes: a minimal code example, the value pmtvs returned, and the value you expected with a source (published paper, competing library, or analytical solution).

"sample_entropy returns 2.14 but Richman-Moorman Table 2 gives 2.09 for this test vector" is a perfect bug report.

Edge cases are especially welcome — constant signals, very short signals, NaN-heavy data, extreme outliers. If you can break it, we want to know.

Issues: issues@pmtvs.dev

## License

PolyForm Strict 1.0.0 with Additional Terms.

- **Students & individual researchers:** Free. Cite us.
- **Funded research labs (grants > $100K):** Academic Research License required.
- **Commercial use:** Commercial License required.
- **Institutional deployment:** Institutional License required.

Contact: licensing@eigenara.com

See [LICENSE](LICENSE) for full terms.
