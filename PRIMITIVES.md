# PRIMITIVES.md — Mathematical Reference for pmtvs

> **pmtvs** — *Precision Mathematical Toolkit for Validated Signals*
>
> 225 functions · 14 packages · Rust acceleration · Python fallback
>
> `pip install pmtvs`

---

## Purpose

This document is the complete mathematical reference for every function in the pmtvs ecosystem. Each entry includes the formal equation, a plain-language explanation of what is computed and why it matters, implementation notes, and edge-case behavior.

pmtvs exists because signal analysis requires traceable, validated mathematics. Every function documented here produces deterministic, reproducible output from deterministic input. There are no black boxes. Every calculation traces to a published algorithm with full mathematical justification.

**Design Philosophy:**
- **numpy in, number out.** Every function accepts numpy arrays and returns floats or arrays.
- **Accuracy before speed.** Rust acceleration ships only after validated parity with Python reference implementations.
- **Graceful degradation.** If Rust is unavailable, Python fallback is automatic and silent.
- **No domain assumptions.** The same `hurst_exponent` works on turbofan vibration, EEG signals, and financial time series.

**Citation:**
```
Rudder, J. & Rudder, A. (2025). pmtvs: Precision Mathematical Toolkit for Validated Signals.
https://github.com/pmtvs/pmtvs
```

---

## Table of Contents

1. [Statistics](#1-statistics-pmtvs-statistics)
2. [Calculus & Derivatives](#2-calculus--derivatives-pmtvs-statistics)
3. [Normalization](#3-normalization-pmtvs-statistics)
4. [Entropy & Complexity](#4-entropy--complexity-pmtvs-entropy)
5. [Fractal & Long-Range Dependence](#5-fractal--long-range-dependence-pmtvs-fractal)
6. [Correlation](#6-correlation-pmtvs-correlation)
7. [Distance & Similarity](#7-distance--similarity-pmtvs-distance)
8. [Embedding](#8-embedding-pmtvs-embedding)
9. [Dynamical Systems](#9-dynamical-systems-pmtvs-dynamics)
10. [Spectral Analysis](#10-spectral-analysis-pmtvs-spectral)
11. [Matrix & Geometry](#11-matrix--geometry-pmtvs-matrix)
12. [Information Theory](#12-information-theory-pmtvs-information)
13. [Network Analysis](#13-network-analysis-pmtvs-network)
14. [Topology](#14-topology-pmtvs-topology)
15. [Statistical Tests](#15-statistical-tests-pmtvs-tests)
16. [Regression & Pairwise Arithmetic](#16-regression--pairwise-arithmetic-pmtvs-regression)
17. [Rust Acceleration](#17-rust-acceleration)
18. [Validation & Parity](#18-validation--parity)

---

## 1. Statistics (`pmtvs-statistics`)

The foundation. Every signal analysis pipeline begins with descriptive statistics — they establish baseline behavior, identify anomalies, and provide the moments needed by higher-order analyses.

### 1.1 Arithmetic Mean

$$\bar{x} = \frac{1}{N} \sum_{i=1}^{N} x_i$$

**What it computes:** The central tendency of a signal — the "expected value" around which observations distribute.

**Why it matters:** The mean is subtracted before computing covariance matrices, autocorrelation, and spectral estimates. A drifting mean indicates non-stationarity.

**Edge cases:** Empty signal → `NaN`. Single value → that value. NaN values are stripped before computation.

```python
from pmtvs import mean
mu = mean(signal)  # float
```

---

### 1.2 Standard Deviation

$$\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_i - \bar{x})^2}$$

**What it computes:** The average distance of observations from the mean, in the same units as the signal.

**Why it matters:** Standard deviation normalizes tolerance parameters (e.g., `r = 0.2 * std` for sample entropy). It is the square root of the variance, which forms the diagonal of the covariance matrix. Dropping standard deviation in a multi-signal system precedes dimensional collapse.

**Implementation note:** Uses population standard deviation (`ddof=0`) by default, consistent with NumPy. Sample standard deviation (`ddof=1`) is available via parameter.

```python
from pmtvs import std
sigma = std(signal)
```

---

### 1.3 Variance

$$\sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \bar{x})^2$$

**What it computes:** The second central moment — the average squared deviation from the mean.

**Why it matters:** Variance is the building block of covariance matrices. The eigenvalues of a covariance matrix represent variance along principal directions. Total variance equals the trace of the covariance matrix, which equals the sum of all eigenvalues.

---

### 1.4 Skewness

$$\gamma_1 = \frac{1}{N} \sum_{i=1}^{N} \left( \frac{x_i - \bar{x}}{\sigma} \right)^3$$

**What it computes:** The third standardized moment — asymmetry of the distribution around the mean.

**Why it matters:** Positive skewness (right tail) indicates occasional large positive excursions. Negative skewness indicates occasional large negative excursions. In vibration analysis, skewness changes precede bearing degradation. In financial signals, skewness measures crash risk.

**Rust acceleration:** 10.9× speedup, validated parity.

---

### 1.5 Kurtosis

$$\gamma_2 = \frac{1}{N} \sum_{i=1}^{N} \left( \frac{x_i - \bar{x}}{\sigma} \right)^4 - 3$$

**What it computes:** The fourth standardized moment minus 3 (excess kurtosis). Measures the "tailedness" of the distribution relative to a Gaussian.

**Why it matters:** High kurtosis signals contain rare, extreme events — impulsive faults in machinery, spikes in neural data, flash crashes in markets. A kurtosis increase in rolling windows is a classic early warning of system degradation.

**Implementation note:** Returns *excess* kurtosis (Fischer's definition). A Gaussian distribution has excess kurtosis = 0.

**Rust acceleration:** 9.4× speedup, validated parity.

---

### 1.6 Root Mean Square (RMS)

$$x_{\text{rms}} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2}$$

**What it computes:** The quadratic mean — the square root of the mean of squared values.

**Why it matters:** RMS is the standard measure of signal energy in vibration analysis, audio engineering, and electrical engineering. It equals standard deviation when the mean is zero. ISO 10816 uses RMS velocity for machinery vibration severity classification.

---

### 1.7 Peak-to-Peak

$$x_{\text{pp}} = \max(x) - \min(x)$$

**What it computes:** The total range of the signal.

**Why it matters:** Peak-to-peak amplitude tracks the maximum excursion envelope. In rotating machinery, increasing peak-to-peak often indicates imbalance or looseness.

---

### 1.8 Crest Factor

$$CF = \frac{\max(|x|)}{x_{\text{rms}}}$$

**What it computes:** The ratio of peak absolute value to RMS — how "spiky" the signal is relative to its energy.

**Why it matters:** A crest factor of 1.0 means the signal is constant. A crest factor of $\sqrt{2} \approx 1.414$ is characteristic of a pure sine wave. Crest factors above 4–5 indicate impulsive content (bearing spalls, gear tooth cracks). ISO 13373 uses crest factor as a condition indicator.

**Rust acceleration:** 5.5× speedup, validated parity.

---

### 1.9 Pulsation Index

$$PI = \frac{\max(x) - \min(x)}{\bar{x}}$$

**What it computes:** The peak-to-peak amplitude normalized by the mean. Originally from hemodynamics (arterial blood flow pulsatility).

**Why it matters:** In any positive-valued signal (flow, pressure, current), the pulsation index separates "how much the signal varies" from "how large the signal is." A pump with high flow and low pulsation is healthy. A pump with low flow and high pulsation is failing.

**Rust acceleration:** 2.8× speedup, validated parity.

---

### 1.10 Zero Crossings

$$ZC = \sum_{i=1}^{N-1} \mathbb{1}\left[ x_i \cdot x_{i+1} < 0 \right]$$

**What it computes:** The number of times the signal crosses zero (sign changes between consecutive samples).

**Why it matters:** Zero-crossing rate is a proxy for dominant frequency. In speech processing, it distinguishes voiced from unvoiced segments. In vibration, a sudden increase in zero crossings indicates broadband noise (often a fault signature).

---

### 1.11 Mean Crossings

$$MC = \sum_{i=1}^{N-1} \mathbb{1}\left[ (x_i - \bar{x})(x_{i+1} - \bar{x}) < 0 \right]$$

**What it computes:** The number of times the signal crosses its own mean. Equivalent to zero crossings of the mean-centered signal.

**Why it matters:** More robust than zero crossings for signals with non-zero DC offset.

---

### 1.12 Percentiles

$$P_k = x_{(\lceil kN/100 \rceil)}$$

**What it computes:** The value below which $k\%$ of observations fall, using the sorted signal.

**Why it matters:** Percentiles define the interquartile range (IQR = P75 − P25), which is the basis of robust normalization. Percentile-based thresholds are resistant to outliers that corrupt mean-based statistics.

---

### 1.13 Min/Max

$$x_{\min} = \min_i(x_i), \quad x_{\max} = \max_i(x_i)$$

**What it computes:** The extreme values of the signal.

**Implementation note:** Returns a tuple `(min_val, max_val)`. Returns `(NaN, NaN)` for empty signals.

---

## 2. Calculus & Derivatives (`pmtvs-statistics`)

Numerical differentiation and integration. These functions estimate continuous-time properties from discrete samples.

### 2.1 First Derivative

$$x'_i = \frac{x_{i+1} - x_{i-1}}{2\Delta t}$$

**What it computes:** The central finite difference approximation to the first derivative. Forward/backward differences at boundaries.

**Why it matters:** The derivative captures the *rate of change* of a signal. In dynamical systems, derivatives construct the velocity field. In vibration, the derivative of displacement is velocity.

**Implementation note:** Uses `numpy.gradient` for consistent boundary handling.

**Rust acceleration:** 5.9× speedup (as `derivative`), validated parity.

---

### 2.2 Second Derivative

$$x''_i = \frac{x_{i+1} - 2x_i + x_{i-1}}{\Delta t^2}$$

**What it computes:** The central finite difference approximation to the second derivative.

**Why it matters:** The second derivative is acceleration (in mechanics), curvature (in geometry), and concavity (in optimization). In signal processing, it highlights inflection points and rapid transitions.

---

### 2.3 Gradient

$$\nabla x_i = \frac{x_{i+1} - x_{i-1}}{2\Delta t}$$

**What it computes:** Identical to the first derivative for 1D signals. Wrapper for `numpy.gradient` with configurable spacing.

---

### 2.4 Laplacian

$$\nabla^2 x_i = x''_i = \frac{x_{i+1} - 2x_i + x_{i-1}}{\Delta t^2}$$

**What it computes:** The discrete Laplacian — the second spatial derivative. For 1D signals, equivalent to the second derivative.

**Why it matters:** The Laplacian measures deviation from local linearity. In diffusion problems, it drives heat flow. In image processing, it detects edges.

---

### 2.5 Velocity, Acceleration, Jerk

$$v_i = x'_i, \quad a_i = x''_i, \quad j_i = x'''_i$$

**What it computes:** The first, second, and third derivatives of a position signal, interpreted as velocity, acceleration, and jerk respectively.

**Why it matters:** In rotating machinery, velocity (mm/s RMS) is the ISO standard vibration metric. Acceleration captures higher-frequency content. Jerk is the rate of change of acceleration — excessive jerk damages servo mechanisms and causes passenger discomfort in vehicles.

---

### 2.6 Curvature

$$\kappa_i = \frac{|x''_i|}{(1 + (x'_i)^2)^{3/2}}$$

**What it computes:** The curvature of the signal treated as a plane curve $(i, x_i)$.

**Why it matters:** Curvature quantifies how sharply the signal bends. High-curvature points correspond to rapid transitions, onsets, and regime changes. In the eigenvalue trajectory context, curvature of the leading eigenvalue signals the onset of dimensional collapse.

**Rust acceleration:** 2.1× speedup, validated parity.

---

### 2.7 Integral

$$\int x \, dt \approx \sum_{i=1}^{N-1} \frac{x_i + x_{i+1}}{2} \Delta t$$

**What it computes:** The trapezoidal rule approximation of the definite integral.

**Why it matters:** Integration accumulates signal energy over time. Integrating acceleration yields velocity; integrating velocity yields displacement. In thermodynamics, the integral of heat flow is total energy transferred.

**Rust acceleration:** 225× speedup, validated parity. This is the single largest speedup in pmtvs — the Rust implementation uses SIMD-friendly accumulation.

---

### 2.8 Rate of Change

$$\text{ROC}_i = \frac{x_{i+1} - x_i}{x_i}$$

**What it computes:** The fractional (relative) change between consecutive samples.

**Why it matters:** Unlike the absolute derivative, rate of change is scale-independent. A 10-unit change in a signal with mean 1000 is different from a 10-unit change in a signal with mean 10. Financial returns are rate-of-change values.

**Rust acceleration:** 5.1× speedup, validated parity.

---

### 2.9 Smoothed Derivative

$$x'_{\text{smooth}} = \frac{d}{dt}\left[ \text{SavGol}(x, w, p) \right]$$

**What it computes:** The derivative of the Savitzky-Golay smoothed signal. Equivalent to fitting a local polynomial of order $p$ over a window of width $w$ and evaluating its derivative.

**Why it matters:** Raw numerical derivatives amplify noise. Savitzky-Golay differentiation suppresses noise while preserving signal shape (unlike simple moving-average-then-differentiate, which distorts peaks).

---

## 3. Normalization (`pmtvs-statistics`)

Normalization transforms signals to standard scales, enabling comparison across different units, ranges, and distributions.

### 3.1 Z-Score Normalization

$$z_i = \frac{x_i - \bar{x}}{\sigma}$$

**What it computes:** Centers the signal to zero mean and scales to unit variance.

**Why it matters:** Z-score normalization is required before computing covariance matrices for eigendecomposition. Without it, variables with larger scales dominate the principal components. It is the default normalization for most statistical learning methods.

**Returns:** Tuple of `(normalized_signal, params)` where params contains `mean` and `std` for inverse transformation.

**Caution:** Z-score normalization makes total variance constant (= number of variables), which may lose dynamics information when applied to rolling windows. Consider whether the variance trend itself carries information before normalizing.

---

### 3.2 Min-Max Normalization

$$x'_i = \frac{x_i - x_{\min}}{x_{\max} - x_{\min}} (b - a) + a$$

**What it computes:** Linearly maps the signal to a target range $[a, b]$, typically $[0, 1]$.

**Why it matters:** Guarantees bounded output, which is useful for neural network inputs and visualization. Sensitive to outliers — a single extreme value compresses the rest of the range.

---

### 3.3 Robust Normalization

$$x'_i = \frac{x_i - \text{median}(x)}{\text{IQR}(x)}$$

**What it computes:** Centers by median and scales by interquartile range (P75 − P25).

**Why it matters:** Robust to outliers. A single spike that would distort z-score normalization has minimal effect on the median and IQR. Preferred for signals with impulsive noise or heavy-tailed distributions.

---

### 3.4 MAD Normalization

$$x'_i = \frac{x_i - \text{median}(x)}{\text{MAD}(x)}, \quad \text{MAD} = \text{median}(|x_i - \text{median}(x)|)$$

**What it computes:** Centers by median and scales by Median Absolute Deviation.

**Why it matters:** Even more robust than IQR-based normalization. MAD has a 50% breakdown point — up to half the data can be arbitrary without affecting the result.

---

### 3.5 Quantile Normalization

**What it computes:** Transforms the signal so that its distribution matches a reference distribution (typically uniform or Gaussian).

**Why it matters:** Forces signals with different distributional shapes onto a common scale while preserving rank ordering. Common in genomics and when comparing signals from different sensor types.

---

## 4. Entropy & Complexity (`pmtvs-entropy`)

Entropy measures quantify the *irregularity*, *unpredictability*, and *information content* of a signal. Low entropy means the signal is regular and predictable. High entropy means it is complex and hard to compress.

### 4.1 Sample Entropy

$$\text{SampEn}(m, r) = -\ln \frac{A}{B}$$

where:
- $B$ = number of template matches of length $m$ within tolerance $r$
- $A$ = number of template matches of length $m + 1$ within tolerance $r$
- Tolerance: $r = 0.2 \times \sigma$ (default)

**What it computes:** The negative natural logarithm of the conditional probability that sequences matching for $m$ consecutive points also match at the $(m+1)$-th point. Does not count self-matches (unlike approximate entropy).

**Why it matters:** Sample entropy is the standard measure of signal complexity in physiology (heart rate variability), neuroscience (EEG), and industrial monitoring. Healthy physiological signals have high sample entropy (complex, adaptive). Degrading systems lose complexity — entropy drops before failure.

**Reference:** Richman, J.S. & Moorman, J.R. (2000). "Physiological time-series analysis using approximate entropy and sample entropy." *American Journal of Physiology*.

**Rust acceleration:** 1,441× speedup, validated parity. The Rust implementation uses optimized distance calculations avoiding redundant comparisons.

```python
from pmtvs import sample_entropy
se = sample_entropy(signal, m=2, r=0.2)
```

---

### 4.2 Permutation Entropy

$$H_p = -\sum_{\pi \in S_m} p(\pi) \ln p(\pi)$$

where $S_m$ is the set of all permutations of order $m$, and $p(\pi)$ is the relative frequency of permutation pattern $\pi$ in the signal.

**What it computes:** The Shannon entropy of the distribution of ordinal patterns (permutations) extracted from the signal using sliding windows of length $m$.

**Why it matters:** Permutation entropy is robust to monotonic transformations — it captures the *ordering* structure, not the amplitude. It is fast to compute (no distance calculations), handles non-stationary data well, and has clear theoretical bounds: $0$ (perfectly deterministic) to $\ln(m!)$ (completely random).

**Reference:** Bandt, C. & Pompe, B. (2002). "Permutation entropy: a natural complexity measure for time series." *Physical Review Letters*, 88(17).

**Normalized form:** $H_p / \ln(m!)$ maps to $[0, 1]$.

**Rust acceleration:** 26× speedup, validated parity.

---

### 4.3 Approximate Entropy

$$\text{ApEn}(m, r) = \Phi^m(r) - \Phi^{m+1}(r)$$

where:

$$\Phi^m(r) = \frac{1}{N - m + 1} \sum_{i=1}^{N-m+1} \ln C_i^m(r)$$

and $C_i^m(r)$ counts template matches including self-matches.

**What it computes:** Similar to sample entropy but includes self-matches in the count, introducing a bias toward regularity.

**Why it matters:** Historically the first embedding-based entropy measure (Pincus, 1991). Sample entropy was developed specifically to address ApEn's bias and dependence on signal length. ApEn is included for backward compatibility and comparison with legacy studies.

---

### 4.4 Multiscale Entropy

$$\text{MSE}(\tau) = \text{SampEn}\left( y^{(\tau)} \right)$$

where:

$$y_j^{(\tau)} = \frac{1}{\tau} \sum_{i=(j-1)\tau+1}^{j\tau} x_i$$

**What it computes:** Sample entropy computed at multiple time scales $\tau$. At each scale, the signal is coarse-grained by averaging non-overlapping windows of size $\tau$.

**Why it matters:** A signal can appear complex at one scale but simple at another. Multiscale entropy reveals the *scale structure* of complexity. Healthy heart rate variability maintains high entropy across scales; pathological states show entropy loss at specific scales.

**Reference:** Costa, M., Goldberger, A.L. & Peng, C.-K. (2005). "Multiscale entropy analysis of biological signals." *Physical Review E*, 71(2).

---

### 4.5 Lempel-Ziv Complexity

**What it computes:** The number of distinct substrings encountered when scanning the binarized signal from left to right. The signal is binarized by thresholding at the median.

**Why it matters:** Lempel-Ziv complexity measures compressibility — how many "new patterns" the signal contains. It is related to the Kolmogorov complexity (which is uncomputable). Fast to compute and applicable to very short signals where entropy estimates are unreliable.

---

### 4.6 Entropy Rate

$$h = \lim_{m \to \infty} H(X_m | X_{m-1}, \ldots, X_1)$$

**What it computes:** The asymptotic rate at which the signal generates new information, estimated by the slope of block entropy as block length increases.

**Why it matters:** Entropy rate separates *complexity from randomness*. White noise has maximum entropy but is not complex — it is simply unpredictable. Complex systems generate information at an intermediate rate.

---

## 5. Fractal & Long-Range Dependence (`pmtvs-fractal`)

Fractal measures characterize *self-similarity* and *memory* — the tendency of past behavior to predict future behavior over long time horizons.

### 5.1 Hurst Exponent (R/S Analysis)

$$H = \frac{\ln(R/S)}{\ln(n)}$$

Estimated via linear regression of $\ln(R/S)$ vs. $\ln(n)$ across segment sizes $n$:

$$\frac{R(n)}{S(n)} \sim c \cdot n^H$$

where for each segment of size $n$:
- $R(n) = \max(Y_i) - \min(Y_i)$ (range of cumulative deviations)
- $S(n) = \text{std}$ of the segment
- $Y_i = \sum_{k=1}^{i}(x_k - \bar{x})$ (cumulative deviation from mean)

**What it computes:** The scaling exponent of the Rescaled Range statistic.

**Interpretation:**
| Hurst | Behavior |
|-------|----------|
| $H < 0.5$ | Anti-persistent (mean-reverting) — increases tend to be followed by decreases |
| $H = 0.5$ | Random walk — no memory |
| $H > 0.5$ | Persistent (trending) — increases tend to be followed by increases |

**Why it matters:** The Hurst exponent is the fundamental measure of long-range dependence. A system with $H > 0.5$ has memory that decays as a power law, not exponentially. This has profound implications for prediction — ARMA models assume exponential decay and underestimate persistence.

**Reference:** Mandelbrot, B.B. & Wallis, J.R. (1969). "Robustness of the rescaled range R/S in the measurement of noncyclic long run statistical dependence." *Water Resources Research*, 5(5).

**Rust acceleration:** 297× speedup, validated parity.

```python
from pmtvs import hurst_exponent
H = hurst_exponent(signal)  # float in [0, 1]
```

---

### 5.2 Detrended Fluctuation Analysis (DFA)

$$F(n) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left[ Y_i - Y_{n,i}^{\text{fit}} \right]^2} \sim n^\alpha$$

where:
- $Y_i = \sum_{k=1}^{i} (x_k - \bar{x})$ is the integrated (cumulative sum) signal
- $Y_{n,i}^{\text{fit}}$ is the local polynomial trend fit within each segment of size $n$
- $\alpha$ is estimated by log-log regression of $F(n)$ vs. $n$

**What it computes:** The scaling exponent of detrended fluctuations. Unlike R/S analysis, DFA removes polynomial trends before computing fluctuations, making it robust to non-stationarities.

**Interpretation:**
| $\alpha$ | Signal type |
|----------|-------------|
| $\alpha \approx 0.5$ | White noise (uncorrelated) |
| $\alpha \approx 1.0$ | $1/f$ noise (pink noise) |
| $\alpha \approx 1.5$ | Brownian motion (random walk) |

**Why it matters:** DFA is preferred over Hurst R/S for non-stationary signals. The detrending step removes slow drifts that would bias the Hurst estimate. Higher-order DFA (quadratic, cubic detrending) handles more complex trends.

**Reference:** Peng, C.-K. et al. (1994). "Mosaic organization of DNA nucleotides." *Physical Review E*, 49(2).

**Rust acceleration:** 258× speedup, validated parity.

---

### 5.3 Hurst R² (Goodness of Fit)

$$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

**What it computes:** The coefficient of determination of the log-log regression used to estimate the Hurst exponent. Measures how well the power-law scaling actually fits the data.

**Why it matters:** A high Hurst exponent with a low $R^2$ means the scaling relationship is poor — the signal doesn't actually follow a power law. Always report $H$ alongside $R^2$. A threshold of $R^2 > 0.9$ is typical for reliable estimates.

**Rust acceleration:** 289× speedup, validated parity.

---

### 5.4 Rescaled Range (Full)

**What it computes:** The full R/S statistic at a given segment size, returning the mean $R/S$ ratio across all non-overlapping segments. This is the building block of the Hurst exponent calculation.

---

### 5.5 Variance Growth

$$\text{Var}(x_\tau) \sim \tau^{2H}$$

**What it computes:** How the variance of aggregated values scales with aggregation window size $\tau$. The scaling exponent $\beta$ relates to Hurst as $H = \beta / 2$.

**Why it matters:** An alternative estimator for long-range dependence that avoids the R/S bias at small segment sizes.

---

### 5.6 Long-Range Correlation

**What it computes:** The autocorrelation function at long lags, fit to a power-law decay:

$$C(\tau) \sim \tau^{-\gamma}, \quad \gamma = 2 - 2H$$

**Why it matters:** Directly measures the memory structure. If $\gamma < 1$, the autocorrelation is non-summable (infinite memory), which is the hallmark of long-range dependence.

---

## 6. Correlation (`pmtvs-correlation`)

Correlation functions measure *linear dependence* between signals or between a signal and its own past.

### 6.1 Autocorrelation Function (ACF)

$$R(\tau) = \frac{1}{(N - \tau)\sigma^2} \sum_{i=1}^{N-\tau} (x_i - \bar{x})(x_{i+\tau} - \bar{x})$$

**What it computes:** The Pearson correlation of a signal with a time-lagged copy of itself.

**Why it matters:** The ACF reveals periodicity (oscillating ACF), persistence ($R(\tau) > 0$ for large $\tau$), and decorrelation timescale (the lag at which $R(\tau)$ first crosses zero). It is the inverse Fourier transform of the power spectral density (Wiener-Khinchin theorem).

**Rust acceleration:** 4.6× speedup, validated parity.

---

### 6.2 Partial Autocorrelation Function (PACF)

$$\alpha(\tau) = \text{Corr}(x_t, x_{t+\tau} | x_{t+1}, \ldots, x_{t+\tau-1})$$

**What it computes:** The correlation between $x_t$ and $x_{t+\tau}$ after removing the linear effect of all intermediate values. Estimated via the Durbin-Levinson algorithm.

**Why it matters:** PACF identifies the *direct* influence at each lag, stripping out indirect correlations mediated by intermediate lags. The PACF cuts off sharply at lag $p$ for an AR($p$) process, making it the standard tool for autoregressive model order selection.

**Rust acceleration:** 5.3× speedup, validated parity.

---

### 6.3 Pearson Correlation (Pairwise)

$$\rho_{xy} = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$$

**What it computes:** The linear correlation coefficient between two signals, bounded in $[-1, 1]$.

**Rust acceleration:** 4.7× speedup, validated parity.

---

### 6.4 Covariance (Pairwise)

$$\text{Cov}(x, y) = \frac{1}{N} \sum_{i=1}^{N} (x_i - \bar{x})(y_i - \bar{y})$$

**What it computes:** The unnormalized linear co-dependence between two signals.

**Rust acceleration:** 4.4× speedup, validated parity.

---

### 6.5 Cross-Correlation

$$R_{xy}(\tau) = \sum_{i} x_i \cdot y_{i+\tau}$$

**What it computes:** The full cross-correlation function, measuring similarity between $x$ and a time-shifted $y$.

**Why it matters:** The lag at maximum cross-correlation estimates the time delay between two signals (e.g., propagation delay between sensors). Used in sonar, radar, and distributed sensor systems.

---

### 6.6 Spearman Rank Correlation

$$\rho_s = 1 - \frac{6 \sum d_i^2}{N(N^2 - 1)}$$

where $d_i$ is the difference between the ranks of $x_i$ and $y_i$.

**What it computes:** Pearson correlation applied to the rank-transformed data. Measures monotonic (not just linear) association.

**Why it matters:** Robust to outliers and nonlinear relationships. If two signals have a monotonic but curved relationship, Pearson underestimates the dependence while Spearman captures it.

---

### 6.7 Kendall's Tau

$$\tau = \frac{C - D}{\binom{N}{2}}$$

where $C$ = concordant pairs, $D$ = discordant pairs.

**What it computes:** The probability of concordance minus the probability of discordance between all pairs of observations.

**Why it matters:** More interpretable than Spearman for small samples and handles ties naturally.

---

### 6.8 Coherence

$$C_{xy}(f) = \frac{|S_{xy}(f)|^2}{S_{xx}(f) S_{yy}(f)}$$

**What it computes:** The frequency-domain equivalent of correlation — how linearly related two signals are at each frequency.

**Why it matters:** Two signals can be uncorrelated in the time domain but highly coherent at a specific frequency (e.g., two sensors both sensing a 120 Hz vibration buried in different noise).

---

### 6.9 ACF Half-Life

$$t_{1/2}: \quad R(t_{1/2}) = 0.5$$

**What it computes:** The lag at which the autocorrelation decays to 0.5.

**Why it matters:** A single-number summary of memory timescale. Short half-life = fast-decaying memory. Long half-life = persistent signal.

---

## 7. Distance & Similarity (`pmtvs-distance`)

Distance functions quantify *dissimilarity* between signals or vectors.

### 7.1 Euclidean Distance

$$d(x, y) = \sqrt{\sum_{i=1}^{N} (x_i - y_i)^2}$$

**What it computes:** The L2 (straight-line) distance between two vectors.

**Rust acceleration:** 3.0× speedup, validated parity.

---

### 7.2 Manhattan Distance

$$d(x, y) = \sum_{i=1}^{N} |x_i - y_i|$$

**What it computes:** The L1 (taxicab) distance — the sum of absolute differences.

**Why it matters:** More robust to outliers than Euclidean distance. In high-dimensional spaces, L1 distance maintains better discrimination (the "curse of dimensionality" affects L2 more).

**Rust acceleration:** 2.8× speedup, validated parity.

---

### 7.3 Cosine Similarity

$$\cos(\theta) = \frac{x \cdot y}{\|x\| \|y\|}$$

**What it computes:** The cosine of the angle between two vectors. Ranges from $-1$ (opposite) to $1$ (identical direction), with $0$ indicating orthogonality.

**Why it matters:** Cosine similarity is insensitive to magnitude — it measures directional alignment only. Two signals can have very different amplitudes but identical cosine similarity if they vary in the same proportional pattern.

---

### 7.4 Dynamic Time Warping (DTW)

$$\text{DTW}(x, y) = \min_\pi \sqrt{\sum_{(i,j) \in \pi} (x_i - y_j)^2}$$

where $\pi$ is a warping path satisfying boundary, continuity, and monotonicity constraints.

**What it computes:** The minimum-cost alignment between two signals, allowing non-linear time warping. DTW "stretches" and "compresses" time to find the best match.

**Why it matters:** Euclidean distance fails when signals are similar but misaligned in time (e.g., the same gesture performed at different speeds). DTW is the standard similarity measure for time series classification, speech recognition, and gesture recognition.

---

### 7.5 Earth Mover's Distance

$$\text{EMD}(p, q) = \inf_{\gamma \in \Gamma(p,q)} \int |x - y| \, d\gamma(x, y)$$

**What it computes:** The minimum "work" required to transform one distribution into another, where work = mass × distance moved. Also known as the Wasserstein-1 distance.

**Why it matters:** EMD respects the geometry of the underlying space — it knows that moving probability mass from bin 1 to bin 2 costs less than moving it from bin 1 to bin 100. KL divergence and Jensen-Shannon divergence do not have this property.

---

## 8. Embedding (`pmtvs-embedding`)

Time-delay embedding reconstructs the state space of a dynamical system from a single scalar time series. This is the bridge between observed signals and geometric analysis.

### 8.1 Time-Delay Embedding

$$\mathbf{v}_i = (x_i, x_{i+\tau}, x_{i+2\tau}, \ldots, x_{i+(m-1)\tau})$$

**What it computes:** Constructs $m$-dimensional vectors from the scalar signal using delay $\tau$. The embedded vectors approximate the attractor geometry of the underlying dynamical system.

**Why it matters:** Takens' embedding theorem (1981) proves that for generic observations of a dynamical system, the time-delay embedding preserves the topology of the original attractor if $m \geq 2d + 1$, where $d$ is the attractor dimension. This means a single sensor can reconstruct the full state space geometry.

**Output shape:** $(N - (m-1)\tau, m)$

---

### 8.2 Optimal Delay ($\tau$)

**Method:** First minimum of the Average Mutual Information (AMI):

$$I(\tau) = \sum_{x_t, x_{t+\tau}} p(x_t, x_{t+\tau}) \log \frac{p(x_t, x_{t+\tau})}{p(x_t) p(x_{t+\tau})}$$

**What it computes:** The time delay $\tau$ at which the signal and its delayed copy share the least redundant information.

**Why it matters:** Too small a delay produces nearly identical coordinates (the embedding collapses onto the diagonal). Too large a delay produces nearly independent coordinates (the embedding fills space randomly). The first AMI minimum balances information content.

**Rust acceleration:** 17× speedup, validated parity.

---

### 8.3 Optimal Dimension ($m$)

**Method:** Cao's method — the ratio of distances in successive embedding dimensions:

$$E_1(d) = \frac{\bar{a}(d+1)}{\bar{a}(d)}, \quad \bar{a}(d) = \frac{1}{N_d} \sum_i \frac{\|v_i^{(d+1)} - v_{n(i)}^{(d+1)}\|}{\|v_i^{(d)} - v_{n(i)}^{(d)}\|}$$

where $n(i)$ is the nearest neighbor of point $i$ in $d$-dimensional embedding.

**What it computes:** The minimum embedding dimension $m$ at which nearest-neighbor relationships stabilize (no more "false neighbors" created by projection from a higher-dimensional space).

**Why it matters:** Underembedding (too few dimensions) creates spurious crossings in the reconstructed attractor. Overembedding wastes computation and dilutes signal with noise dimensions. The optimal dimension is the minimum $m$ where $E_1(d) \to 1$.

**Reference:** Cao, L. (1997). "Practical method for determining the minimum embedding dimension of a scalar time series." *Physica D*, 110(1-2).

---

### 8.4 Multivariate Embedding

$$\mathbf{v}_i = (x_{1,i}, x_{1,i+\tau_1}, \ldots, x_{2,i}, x_{2,i+\tau_2}, \ldots)$$

**What it computes:** Embeds multiple signals simultaneously, using potentially different delays and dimensions for each.

**Why it matters:** When multiple sensors observe the same system, multivariate embedding combines their complementary information into a single high-dimensional state space representation.

---

## 9. Dynamical Systems (`pmtvs-dynamics`)

These functions characterize the *dynamics* of a system — its stability, sensitivity to perturbation, and trajectory toward or away from failure.

### 9.1 Largest Lyapunov Exponent (Rosenstein)

$$\lambda_1 = \frac{1}{\Delta t} \frac{1}{M} \sum_{i=1}^{M} \ln \frac{d_i(k)}{d_i(0)}$$

where $d_i(k)$ is the distance between the $i$-th pair of initially close trajectories after $k$ time steps.

**What it computes:** The average exponential rate at which nearby trajectories diverge in state space.

**Interpretation:**
| $\lambda_1$ | Behavior |
|-------------|----------|
| $\lambda_1 > 0$ | Chaotic — nearby states diverge exponentially |
| $\lambda_1 \approx 0$ | Marginal — limit cycle or quasi-periodic |
| $\lambda_1 < 0$ | Stable — nearby states converge (dissipative) |

**Why it matters:** A positive Lyapunov exponent is the mathematical definition of chaos. In the Rudder Framework context, a decreasing Lyapunov exponent indicates the system is becoming more predictable — often because it is collapsing toward a lower-dimensional attractor (failure mode).

**Reference:** Rosenstein, M.T., Collins, J.J. & De Luca, C.J. (1993). "A practical method for calculating largest Lyapunov exponents from small data sets." *Physica D*, 65(1-2).

**Note:** Rust implementation exists (2,776× speedup) but is BENCHED pending parity validation. Currently runs as Python.

---

### 9.2 Largest Lyapunov Exponent (Kantz)

**What it computes:** Same quantity as Rosenstein, but using a different algorithm that averages over neighborhoods of reference points rather than tracking individual nearest neighbors.

**Why it matters:** Kantz's method provides a scaling region in the $\ln d(k)$ vs. $k$ plot. The existence of a linear scaling region is itself a diagnostic — non-chaotic signals do not produce one.

**Reference:** Kantz, H. (1994). "A robust method to estimate the maximal Lyapunov exponent of a time series." *Physics Letters A*, 185(1).

---

### 9.3 Lyapunov Spectrum

$$\lambda_1 \geq \lambda_2 \geq \ldots \geq \lambda_m$$

Computed via QR decomposition of the product of local Jacobians along the trajectory.

**What it computes:** All $m$ Lyapunov exponents, not just the largest. The full spectrum characterizes the expansion and contraction rates along every principal direction.

**Why it matters:** The sum of all Lyapunov exponents equals the rate of phase space volume contraction (Liouville's theorem). The Kaplan-Yorke dimension depends on the full spectrum. A system transitioning from chaos to failure loses positive exponents one by one.

---

### 9.4 Finite-Time Lyapunov Exponent (FTLE)

$$\text{FTLE}(x_0, T) = \frac{1}{2T} \ln \lambda_{\max}(\Delta^T \Delta)$$

where $\Delta$ is the deformation gradient tensor (Cauchy-Green tensor) computed along the trajectory over time horizon $T$.

**What it computes:** The maximum stretching rate experienced by an infinitesimal perturbation at point $x_0$ over a finite time window $T$.

**Why it matters:** FTLE fields reveal Lagrangian Coherent Structures (LCS) — the hidden barriers and channels that organize transport in dynamical systems. In multi-signal systems, FTLE ridges separate dynamically distinct regions. This is the function that maps the "failure landscape."

**Two methods:**
- **Local linearization:** Estimates the deformation gradient from neighboring trajectories in the embedded space.
- **Direct perturbation:** Computes stretching by tracking explicitly perturbed initial conditions.

---

### 9.5 Recurrence Quantification Analysis (RQA)

The recurrence matrix:

$$R_{ij} = \Theta(\epsilon - \|v_i - v_j\|)$$

where $\Theta$ is the Heaviside function and $\epsilon$ is the recurrence threshold.

**Recurrence Rate:**
$$RR = \frac{1}{N^2} \sum_{i,j} R_{ij}$$

The fraction of the state space that is recurrent — how often the system revisits previous states.

**Determinism:**
$$DET = \frac{\sum_{l=l_{\min}}^{N} l \cdot P(l)}{\sum_{i,j} R_{ij}}$$

The fraction of recurrent points forming diagonal lines. Diagonal lines indicate deterministic dynamics.

**Laminarity:**
$$LAM = \frac{\sum_{v=v_{\min}}^{N} v \cdot P(v)}{\sum_{i,j} R_{ij}}$$

The fraction of recurrent points forming vertical lines. Vertical lines indicate laminar (slowly changing) states.

**Trapping Time:**
$$TT = \frac{\sum_{v=v_{\min}}^{N} v \cdot P(v)}{\sum_{v=v_{\min}}^{N} P(v)}$$

The average length of vertical lines — how long the system stays trapped in a laminar state.

**Why RQA matters:** RQA provides a complete diagnostic toolkit for nonlinear dynamics. Increasing determinism + decreasing trapping time is a classic pre-failure signature: the system becomes more predictable while its dwell times shorten.

**Reference:** Marwan, N. et al. (2007). "Recurrence plots for the analysis of complex systems." *Physics Reports*, 438(5-6).

---

### 9.6 Saddle Point Analysis

**Jacobian Estimation:**

$$J_{ij} \approx \frac{\partial f_i}{\partial x_j} \bigg|_{x_0}$$

Estimated from local linear regression of nearby trajectory points.

**Eigenvalue Classification:**

| Eigenvalue pattern | Fixed point type |
|-------------------|-----------------|
| All Re($\lambda$) < 0 | Stable node/spiral |
| All Re($\lambda$) > 0 | Unstable node/spiral |
| Mixed signs | Saddle point |
| Re($\lambda$) = 0 | Center/Hopf bifurcation |

**Basin Stability:**

$$S_B = \frac{V_{\text{basin}}}{V_{\text{total}}}$$

The fraction of state space volume that converges to a given attractor.

**Why it matters:** Saddle points are the gatekeepers of regime transitions. A system near a saddle point is at a decision boundary — a small perturbation determines which basin of attraction it falls into. Detecting proximity to saddle points is early warning of critical transitions.

---

### 9.7 Sensitivity Analysis

**Variable Sensitivity:**

$$S_j = \left\| \frac{\partial \Phi^T}{\partial x_j} \right\|$$

The norm of the derivative of the flow map $\Phi^T$ with respect to the $j$-th state variable.

**Influence Matrix:**

$$M_{ij} = \frac{\partial x_i(T)}{\partial x_j(0)}$$

How perturbation of variable $j$ at time 0 affects variable $i$ at time $T$.

**Why it matters:** Sensitivity analysis identifies which variables drive system behavior. A variable with increasing sensitivity is becoming more influential — potentially a leading indicator of mode change.

---

### 9.8 Correlation Dimension

$$D_2 = \lim_{r \to 0} \frac{\ln C(r)}{\ln r}$$

where $C(r)$ is the correlation integral:

$$C(r) = \lim_{N \to \infty} \frac{2}{N(N-1)} \sum_{i<j} \Theta(r - \|v_i - v_j\|)$$

**What it computes:** The fractal dimension of the attractor, estimated from the scaling of the correlation integral.

**Why it matters:** The correlation dimension tells you the "effective number of active degrees of freedom." A turbofan with 21 sensors might have a correlation dimension of 3 — meaning only 3 independent modes are active. A decreasing correlation dimension means the system is losing degrees of freedom. **This is the fundamental insight of the Rudder Framework: systems lose effective dimensionality before failure.**

**Reference:** Grassberger, P. & Procaccia, I. (1983). "Characterization of strange attractors." *Physical Review Letters*, 50(5).

---

### 9.9 Kaplan-Yorke Dimension

$$D_{KY} = j + \frac{\sum_{i=1}^{j} \lambda_i}{|\lambda_{j+1}|}$$

where $j$ is the largest index such that $\sum_{i=1}^{j} \lambda_i \geq 0$.

**What it computes:** The information dimension of the attractor, estimated from the Lyapunov spectrum. Represents the boundary between expanding and contracting directions.

---

## 10. Spectral Analysis (`pmtvs-spectral`)

Spectral analysis decomposes signals into frequency components.

### 10.1 Power Spectral Density (PSD)

$$S_{xx}(f) = \lim_{T \to \infty} \frac{1}{T} |X(f)|^2$$

Estimated via Welch's method: average periodograms of overlapping windowed segments.

**What it computes:** The distribution of signal power across frequencies.

**Why it matters:** The PSD reveals the fundamental frequencies of the system, harmonic relationships, and broadband noise floors. In rotating machinery, peaks at shaft frequency and its multiples indicate imbalance, misalignment, and gear mesh frequencies.

---

### 10.2 Dominant Frequency

$$f_{\text{dom}} = \arg\max_f S_{xx}(f)$$

**What it computes:** The frequency with the highest spectral power.

---

### 10.3 Spectral Centroid

$$f_c = \frac{\sum_f f \cdot S_{xx}(f)}{\sum_f S_{xx}(f)}$$

**What it computes:** The "center of mass" of the power spectrum.

**Why it matters:** Spectral centroid tracks the overall frequency content. Machinery with developing faults shifts spectral centroid upward as high-frequency harmonics grow.

---

### 10.4 Spectral Bandwidth

$$BW = \sqrt{\frac{\sum_f (f - f_c)^2 \cdot S_{xx}(f)}{\sum_f S_{xx}(f)}}$$

**What it computes:** The spread of the power spectrum around the centroid.

---

### 10.5 Spectral Entropy

$$H_s = -\sum_f p(f) \log p(f), \quad p(f) = \frac{S_{xx}(f)}{\sum_k S_{xx}(k)}$$

**What it computes:** The Shannon entropy of the normalized power spectrum.

**Why it matters:** Low spectral entropy = power concentrated at few frequencies (tonal signal). High spectral entropy = power spread uniformly (broadband noise). A system transitioning from tonal to broadband behavior is losing coherent dynamics.

---

### 10.6 Hilbert Transform

$$\hat{x}(t) = \frac{1}{\pi} \text{P.V.} \int_{-\infty}^{\infty} \frac{x(\tau)}{t - \tau} d\tau$$

**What it computes:** The $90°$ phase-shifted version of the signal (quadrature component).

**Why it matters:** Together, $x(t)$ and $\hat{x}(t)$ form the analytic signal $z(t) = x(t) + i\hat{x}(t)$, from which instantaneous amplitude, phase, and frequency are derived.

---

### 10.7 Instantaneous Amplitude (Envelope)

$$A(t) = |z(t)| = \sqrt{x(t)^2 + \hat{x}(t)^2}$$

**What it computes:** The magnitude of the analytic signal — the slowly varying envelope.

**Why it matters:** The envelope extracts amplitude modulation. In vibration, amplitude modulation at the bearing fault frequency is the classic diagnostic signature (envelope analysis / high-frequency resonance technique).

---

### 10.8 Instantaneous Frequency

$$f(t) = \frac{1}{2\pi} \frac{d\phi(t)}{dt}, \quad \phi(t) = \arg(z(t))$$

**What it computes:** The rate of change of the instantaneous phase.

**Why it matters:** Instantaneous frequency tracks frequency modulation. Machinery with variable-speed operation produces frequency-modulated vibration that is invisible to Fourier analysis but clear in instantaneous frequency.

---

## 11. Matrix & Geometry (`pmtvs-matrix`)

Matrix functions operate on multi-signal systems, extracting the geometric structure of the state space.

### 11.1 Covariance Matrix

$$\Sigma_{ij} = \frac{1}{N} \sum_{k=1}^{N} (x_{i,k} - \bar{x}_i)(x_{j,k} - \bar{x}_j)$$

**What it computes:** The pairwise covariance between all signal pairs. Symmetric, positive semi-definite.

---

### 11.2 Eigendecomposition

$$\Sigma = V \Lambda V^T, \quad \Lambda = \text{diag}(\lambda_1, \ldots, \lambda_p)$$

**What it computes:** The eigenvalues $\lambda_i$ (variance along each principal direction) and eigenvectors $V$ (the principal directions themselves).

**Why it matters:** This is the core computation of the Rudder Framework. The eigenvalue spectrum encodes the dimensional structure of the system. The eigenvectors encode the coupling structure. Changes in the eigenvalue spectrum — tracked over time — predict remaining useful life.

---

### 11.3 Effective Dimension

$$D_{\text{eff}} = \frac{\left(\sum_i \lambda_i\right)^2}{\sum_i \lambda_i^2}$$

**What it computes:** The participation ratio of the eigenvalue spectrum — how many dimensions contribute meaningfully to the total variance.

**Why it matters:** **This is the single most important diagnostic quantity in the framework.** Effective dimension measures how many independent modes are active. A turbofan engine with 21 sensors might have $D_{\text{eff}} = 8$ during normal operation and $D_{\text{eff}} = 2$ just before failure. The trajectory of effective dimension over time is the primary predictor of remaining useful life. The key insight: **effective dimensionality (63% importance) predicts remaining useful life because systems collapse dimensionally before failure.**

---

### 11.4 Participation Ratio

$$PR = \frac{(\sum_i \lambda_i)^2}{\sum_i \lambda_i^2}$$

Identical formula to effective dimension. Sometimes called the "inverse participation ratio" when defined as $1/PR$.

---

### 11.5 Condition Number

$$\kappa = \frac{\lambda_{\max}}{\lambda_{\min}}$$

**What it computes:** The ratio of the largest to smallest eigenvalue.

**Why it matters:** High condition number = the system is numerically ill-conditioned and nearly singular. In the signal analysis context, a very high condition number means one mode dominates while others are nearly inactive — the system is approaching a lower-dimensional state.

---

### 11.6 Matrix Entropy (Von Neumann)

$$S = -\sum_i \hat{\lambda}_i \log \hat{\lambda}_i, \quad \hat{\lambda}_i = \frac{\lambda_i}{\sum_j \lambda_j}$$

**What it computes:** The Shannon entropy of the normalized eigenvalue distribution.

**Why it matters:** Maximum when all eigenvalues are equal (all dimensions equally active). Minimum (zero) when one eigenvalue dominates (the system is one-dimensional). Matrix entropy declining over time = dimensional collapse.

---

### 11.7 Dynamic Mode Decomposition (DMD)

$$X' = A X, \quad A \approx U_r \tilde{A} V_r^*$$

$$\tilde{A} = U_r^* X' V_r \Sigma_r^{-1}$$

**What it computes:** The best-fit linear operator that advances the state forward in time. DMD eigenvalues encode growth/decay rates and oscillation frequencies.

**Why it matters:** DMD provides a data-driven modal analysis — it finds the dynamic modes (spatial patterns) and their temporal evolution (frequency, growth rate) without assuming any model.

---

### 11.8 Explained Variance Ratio

$$\text{EVR}_i = \frac{\lambda_i}{\sum_j \lambda_j}$$

**What it computes:** The fraction of total variance captured by the $i$-th principal component.

---

### 11.9 Cumulative Variance Ratio

$$\text{CVR}_k = \sum_{i=1}^{k} \frac{\lambda_i}{\sum_j \lambda_j}$$

**What it computes:** The fraction of total variance captured by the first $k$ components.

**Why it matters:** The number of components needed to reach 95% CVR is another measure of effective dimensionality.

---

## 12. Information Theory (`pmtvs-information`)

Information-theoretic measures quantify *statistical dependence* without assuming linearity.

### 12.1 Shannon Entropy

$$H(X) = -\sum_i p(x_i) \log_2 p(x_i)$$

**What it computes:** The average information content (in bits) of a random variable.

---

### 12.2 Mutual Information

$$I(X; Y) = H(X) + H(Y) - H(X, Y)$$

**What it computes:** The reduction in uncertainty about $X$ given knowledge of $Y$. Captures both linear and nonlinear dependence.

---

### 12.3 Transfer Entropy

$$T_{Y \to X} = H(X_{t+1} | X_t^{(k)}) - H(X_{t+1} | X_t^{(k)}, Y_t^{(l)})$$

**What it computes:** The reduction in uncertainty about the future of $X$ given the past of $Y$, beyond what the past of $X$ already provides. A directional, nonlinear measure of information flow.

**Why it matters:** Transfer entropy detects causal influence between signals without assuming a linear model. Unlike Granger causality, it captures nonlinear coupling.

**Reference:** Schreiber, T. (2000). "Measuring information transfer." *Physical Review Letters*, 85(2).

---

### 12.4 KL Divergence

$$D_{KL}(P \| Q) = \sum_i p_i \log \frac{p_i}{q_i}$$

**What it computes:** The information lost when distribution $Q$ is used to approximate distribution $P$.

**Note:** Not symmetric. Use Jensen-Shannon divergence for a symmetric metric.

---

### 12.5 Jensen-Shannon Divergence

$$JSD(P, Q) = \frac{1}{2} D_{KL}(P \| M) + \frac{1}{2} D_{KL}(Q \| M), \quad M = \frac{P + Q}{2}$$

**What it computes:** A symmetric, bounded version of KL divergence. The square root is a true metric.

---

### 12.6 Rényi Entropy

$$H_\alpha(X) = \frac{1}{1 - \alpha} \log \sum_i p_i^\alpha$$

**What it computes:** A one-parameter family of entropies. Reduces to Shannon entropy as $\alpha \to 1$. $\alpha = 2$ gives the collision entropy; $\alpha = 0$ gives the Hartley entropy.

---

### 12.7 Tsallis Entropy

$$S_q(X) = \frac{1}{q - 1} \left(1 - \sum_i p_i^q\right)$$

**What it computes:** Non-extensive entropy used in complex systems with long-range interactions or power-law distributions.

---

### 12.8 Granger Causality

$$F = \frac{(\text{RSS}_{\text{restricted}} - \text{RSS}_{\text{full}}) / p}{\text{RSS}_{\text{full}} / (N - 2p - 1)}$$

**What it computes:** Whether past values of $Y$ significantly improve the prediction of $X$ beyond what past values of $X$ alone provide, using an F-test on nested linear models.

---

### 12.9 Convergent Cross Mapping (CCM)

**What it computes:** Tests for causal coupling in deterministic dynamical systems by checking whether the attractor reconstructed from $X$ can predict $Y$, and whether this prediction improves with library size.

**Reference:** Sugihara, G. et al. (2012). "Detecting causality in complex ecosystems." *Science*, 338(6106).

---

### 12.10 Partial Information Decomposition

**What it computes:** Decomposes the information that two source variables provide about a target into: *redundancy* (both sources carry), *unique information* (only one source carries), and *synergy* (only available from both sources together).

---

## 13. Network Analysis (`pmtvs-network`)

Network functions operate on *graphs* derived from signal systems — typically adjacency matrices constructed from correlation, mutual information, or recurrence.

### 13.1 Network Density

$$\rho = \frac{2|E|}{|V|(|V|-1)}$$

**What it computes:** The fraction of possible edges that exist.

---

### 13.2 Clustering Coefficient

$$C_i = \frac{2 T_i}{k_i (k_i - 1)}$$

where $T_i$ is the number of triangles through node $i$ and $k_i$ is its degree.

**What it computes:** The probability that two neighbors of a node are themselves connected.

---

### 13.3 Betweenness Centrality

$$g(v) = \sum_{s \neq v \neq t} \frac{\sigma_{st}(v)}{\sigma_{st}}$$

**What it computes:** The fraction of shortest paths between all pairs that pass through node $v$.

**Why it matters:** High betweenness = the signal is a critical bridge in the information flow network. Failure of a high-betweenness node disrupts system communication.

---

### 13.4 Modularity

$$Q = \frac{1}{2m} \sum_{ij} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$

**What it computes:** How well the network decomposes into communities compared to a random graph.

---

### 13.5 Community Detection

**What it computes:** Identifies groups of densely interconnected signals using the Louvain algorithm (greedy modularity maximization).

**Why it matters:** In a multi-sensor system, communities represent functionally coupled subsystems. A sensor migrating between communities indicates changing coupling — a potential early warning of mode change.

---

## 14. Topology (`pmtvs-topology`)

Topological data analysis captures the *shape* of data — features that are invariant to continuous deformation.

### 14.1 Persistence Diagram

**What it computes:** Birth-death pairs $(b_i, d_i)$ of topological features (connected components, loops, voids) across a filtration of distance thresholds.

**Why it matters:** Long-lived features (large $d_i - b_i$) represent robust topological structure. Short-lived features are noise. The persistence diagram is a complete descriptor of the multi-scale topology of the point cloud.

---

### 14.2 Betti Numbers

$$\beta_0 = \text{connected components}, \quad \beta_1 = \text{loops}, \quad \beta_2 = \text{voids}$$

**What it computes:** The count of topological features at each dimension.

---

### 14.3 Persistence Entropy

$$H_p = -\sum_i \frac{l_i}{L} \log \frac{l_i}{L}, \quad l_i = d_i - b_i, \quad L = \sum_i l_i$$

**What it computes:** The Shannon entropy of the persistence lifetimes.

---

### 14.4 Wasserstein Distance

$$W_p(D_1, D_2) = \left( \inf_\gamma \sum_{(x,y) \in \gamma} \|x - y\|^p \right)^{1/p}$$

**What it computes:** The optimal transport distance between two persistence diagrams.

---

### 14.5 Bottleneck Distance

$$d_B(D_1, D_2) = \inf_\gamma \sup_{(x,y) \in \gamma} \|x - y\|_\infty$$

**What it computes:** The worst-case mismatch in the optimal matching between two persistence diagrams.

---

## 15. Statistical Tests (`pmtvs-tests`)

Hypothesis tests, stationarity diagnostics, and bootstrap methods.

### 15.1 Augmented Dickey-Fuller Test

$$\Delta x_t = \alpha + \beta t + \gamma x_{t-1} + \sum_{i=1}^{p} \delta_i \Delta x_{t-i} + \epsilon_t$$

$H_0$: $\gamma = 0$ (unit root / non-stationary).

**What it computes:** Tests whether a signal has a unit root (is non-stationary).

**Why it matters:** Stationarity is a prerequisite for many time series methods. ADF is the standard first-pass test.

---

### 15.2 KPSS Test

$H_0$: The signal is stationary (opposite of ADF).

**What it computes:** Tests the null hypothesis of stationarity against the alternative of a unit root.

**Why it matters:** Use ADF and KPSS together. If ADF rejects and KPSS does not reject → stationary. If ADF does not reject and KPSS rejects → non-stationary. If both reject → difference-stationary. If neither rejects → trend-stationary.

---

### 15.3 Bootstrap Confidence Interval

$$\hat{\theta}^*_{\alpha/2} \leq \theta \leq \hat{\theta}^*_{1-\alpha/2}$$

**What it computes:** Non-parametric confidence intervals by resampling with replacement and computing the statistic on each resample.

**Why it matters:** No distributional assumptions. Works for any statistic (Hurst, entropy, dimension) where analytic confidence intervals don't exist.

---

### 15.4 Marchenko-Pastur Test

$$\lambda_{\pm} = \sigma^2 \left(1 \pm \sqrt{\frac{p}{N}}\right)^2$$

**What it computes:** Tests whether eigenvalues are consistent with a random matrix (null hypothesis of no structure).

**Why it matters:** Eigenvalues outside the Marchenko-Pastur bounds indicate genuine signal structure, not random noise. This is the significance test for eigendecomposition results.

---

### 15.5 ARCH Test

$$\epsilon_t^2 = \alpha_0 + \sum_{i=1}^{q} \alpha_i \epsilon_{t-i}^2 + u_t$$

**What it computes:** Tests for autoregressive conditional heteroscedasticity — whether the variance of the signal changes over time in a structured way.

**Why it matters:** ARCH effects indicate volatility clustering, which is common in financial data and in machinery approaching failure.

---

## 16. Regression & Pairwise Arithmetic (`pmtvs-regression`)

Simple pairwise operations between two signals.

### 16.1 Linear Regression

$$y = \beta_0 + \beta_1 x + \epsilon$$

**Returns:** `(slope, intercept, r_squared, std_error)`

---

### 16.2 Signal Arithmetic

$$\text{ratio}(x, y) = x_i / y_i, \quad \text{product} = x_i y_i, \quad \text{difference} = x_i - y_i, \quad \text{sum} = x_i + y_i$$

**What they compute:** Element-wise arithmetic operations between two signals.

**Why they matter:** Ratio signals (e.g., pressure/temperature) often reveal physics that individual signals don't. Product signals capture interaction effects. Difference signals are the simplest change detection.

---

## 17. Rust Acceleration

21 functions have validated Rust implementations with proven parity against Python references.

| Function | Speedup | Parity |
|----------|---------|--------|
| `sample_entropy` | 1,441× | OK |
| `hurst_exponent` | 297× | OK |
| `hurst_r2` | 289× | OK |
| `dfa` | 258× | OK |
| `integral` | 225× | OK |
| `permutation_entropy` | 26× | OK |
| `optimal_delay` | 17× | OK |
| `skewness` | 10.9× | OK |
| `kurtosis` | 9.4× | OK |
| `derivative` | 5.9× | OK |
| `crest_factor` | 5.5× | OK |
| `partial_autocorrelation` | 5.3× | OK |
| `rate_of_change` | 5.1× | OK |
| `correlation` | 4.7× | OK |
| `autocorrelation` | 4.6× | OK |
| `covariance` | 4.4× | OK |
| `euclidean_distance` | 3.0× | OK |
| `pulsation_index` | 2.8× | OK |
| `manhattan_distance` | 2.8× | OK |
| `std` | 2.5× | OK |
| `curvature` | 2.1× | OK |

**Dispatch rule:** A function earns Rust acceleration only after demonstrating BOTH:
1. **Parity OK** — output matches Python reference within defined tolerance tier
2. **Speedup > 1.0×** — actually faster than Python

Functions failing either criterion use Python. No exceptions.

**BENCHED functions** (Rust exists, parity unvalidated):
`lyapunov_rosenstein` (2,776×), `lyapunov_kantz` (1,437×), `variance_growth` (1,206×), `dynamic_time_warping` (121×) — these ship as Python until validated.

**Python-wins functions** (Rust is slower):
`distance_matrix` (0.015×), `cross_correlation` (0.1×), `hilbert_transform` (0.5×), `svd` (0.8×) — NumPy/SciPy's underlying Fortran/C (BLAS, LAPACK, FFTW) beats Rust for these patterns.

**Environment control:**
```bash
PMTVS_USE_RUST=0  # Force all-Python mode
```

---

## 18. Validation & Parity

### Tolerance Tiers

| Tier | Tolerance | Used for |
|------|-----------|----------|
| Exact | 0 | Integer results (optimal_delay) |
| Tight | $10^{-12}$ | Linear algebra (std, covariance) |
| Normal | $10^{-8}$ | Most functions |
| Relaxed | $10^{-4}$ | Iterative algorithms (entropy, DFA, Hurst) |

### Testing Philosophy

Every function is tested against:
1. **Known analytical values** — white noise, sine waves, random walks, Lorenz attractor, logistic map
2. **Edge cases** — empty signals, single values, constant signals, NaN-contaminated signals
3. **Published paper values** — where available, results are compared to values reported in the original publications
4. **Cross-implementation agreement** — results compared with nolds, antropy, neurokit2, and MATLAB implementations

### Reproducibility

Every function is deterministic for deterministic input. No function uses random initialization without a configurable seed. Results are reproducible across platforms, Python versions (3.9–3.12), and Rust/Python backend selection.

---

## License

PolyForm Noncommercial 1.0.0 with Additional Terms.

- **Students & individual researchers:** Free. Cite us.
- **Funded research labs (grants > $100K):** Academic Research License required. [Contact us](mailto:licensing@pmtvs.dev).
- **Commercial use:** Commercial License required. [Contact us](mailto:licensing@pmtvs.dev).

See [LICENSE](LICENSE) for full terms.

---

*pmtvs — because systems lose coherence before failure, and the math to detect it should be open, validated, and fast.*
