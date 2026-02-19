# pmtvs-tests

Statistical hypothesis testing primitives.

## Installation

```bash
pip install pmtvs-tests
```

## Functions

### Bootstrap Methods
- `bootstrap_mean(data)` - Bootstrap distribution of mean
- `bootstrap_confidence_interval(data, statistic)` - Bootstrap CI
- `bootstrap_ci(data, statistic)` - Bootstrap CI (alias)
- `bootstrap_std(data)` - Bootstrap standard error
- `block_bootstrap_ci(data, statistic)` - Block bootstrap for dependent data

### Non-parametric Tests
- `permutation_test(x, y)` - Two-sample permutation test
- `surrogate_test(signal, statistic)` - Surrogate data test
- `runs_test(signal)` - Runs test (randomness)
- `mann_kendall_test(signal)` - Trend test
- `mannwhitney_test(sample1, sample2)` - Mann-Whitney U test
- `kruskal_test(*samples)` - Kruskal-Wallis H-test

### Parametric Tests
- `t_test(sample)` - One-sample t-test
- `t_test_paired(sample1, sample2)` - Paired t-test
- `t_test_independent(sample1, sample2)` - Independent t-test
- `f_test(sample1, sample2)` - F-test for variance equality
- `chi_squared_test(observed)` - Chi-squared goodness-of-fit
- `anova(*samples)` - One-way ANOVA
- `shapiro_test(sample)` - Shapiro-Wilk normality test
- `levene_test(*samples)` - Levene's variance equality test

### Stationarity Tests
- `adf_test(signal)` - Augmented Dickey-Fuller
- `stationarity_test(signal)` - Rolling-window stationarity
- `kpss_test(signal)` - KPSS test
- `phillips_perron_test(signal)` - Phillips-Perron unit root test
- `trend(signal)` - Linear trend estimation
- `changepoints(signal)` - CUSUM changepoint detection

### Spectral Tests
- `marchenko_pastur_test(eigenvalues)` - Marchenko-Pastur law test
- `arch_test(data)` - Engle's ARCH test for heteroscedasticity

## Backend

Pure Python implementation. Requires scipy >= 1.7.

## License

PolyForm Noncommercial 1.0.0 with Additional Terms.

- **Students & individual researchers:** Free. Cite us.
- **Funded research labs (grants > $100K):** Academic Research License required. [Contact us](mailto:licensing@pmtvs.dev).
- **Commercial use:** Commercial License required. [Contact us](mailto:licensing@pmtvs.dev).

See [LICENSE](LICENSE) for full terms.
