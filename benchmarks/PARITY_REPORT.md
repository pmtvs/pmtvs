### Rust vs Python Parity — Full Disclosure

Every Rust-accelerated function is tested against the Python reference
implementation using identical inputs. The table below shows the maximum
absolute difference observed across all test signals. A function ships
Rust ONLY if parity is within tolerance AND speedup exceeds 1.0x.

| Function | Signal | Python | Rust | Max |delta| | Tolerance | Status |
|----------|--------|--------|------|------------|-----------|--------|
| `sample_entropy` | white_noise | 2.190637243 | 2.190637243 | 0.00e+00 | relaxed (1e-04) | PASS |
| `sample_entropy` | sine | 0.01027264528 | 0.01027264528 | 0.00e+00 | relaxed (1e-04) | PASS |
| `sample_entropy` | lorenz | 0.4787390971 | 0.4787390971 | 0.00e+00 | relaxed (1e-04) | PASS |
| `permutation_entropy` | white_noise | 0.999950202 | 0.999950202 | 0.00e+00 | relaxed (1e-04) | PASS |
| `permutation_entropy` | sine | 0.3956590946 | 0.3956590946 | 0.00e+00 | relaxed (1e-04) | PASS |
| `permutation_entropy` | lorenz | 0.4567246408 | 0.4567246408 | 0.00e+00 | relaxed (1e-04) | PASS |
| `hurst_exponent` | white_noise | 0.5600278979 | 0.5600278979 | 2.93e-14 | relaxed (1e-04) | PASS |
| `hurst_exponent` | random_walk | 1 | 1 | 0.00e+00 | relaxed (1e-04) | PASS |
| `hurst_exponent` | trending | 1 | 1 | 0.00e+00 | relaxed (1e-04) | PASS |
| `hurst_r2` | white_noise | 0.9942249091 | 0.9942249091 | 0.00e+00 | normal (1e-08) | PASS |
| `hurst_r2` | random_walk | 0.9989392827 | 0.9989392827 | 0.00e+00 | normal (1e-08) | PASS |
| `dfa` | white_noise | 0.5263830629 | 0.5263830629 | 1.11e-15 | relaxed (1e-04) | PASS |
| `dfa` | random_walk | 1.491461668 | 1.491461668 | 6.66e-16 | relaxed (1e-04) | PASS |
| `mean` | white_noise | 0.005601930088 | 0.005601930088 | 1.47e-17 | tight (1e-12) | PASS |
| `mean` | sine | 0 | 8.57353743e-18 | 8.57e-18 | tight (1e-12) | PASS |
| `mean` | constant | 3.14 | 3.14 | 2.04e-13 | tight (1e-12) | PASS |
| `std` | white_noise | 0.9964798288 | 0.9964798288 | 2.22e-16 | tight (1e-12) | PASS |
| `std` | sine | 0.7071067812 | 0.7071067812 | 7.77e-16 | tight (1e-12) | PASS |
| `std` | constant | 0 | 2.038573341e-13 | 2.04e-13 | tight (1e-12) | PASS |
| `variance` | white_noise | 0.9929720492 | 0.9929720492 | 5.55e-16 | tight (1e-12) | PASS |
| `variance` | sine | 0.5 | 0.5 | 9.99e-16 | tight (1e-12) | PASS |
| `variance` | constant | 0 | 4.155781266e-26 | 4.16e-26 | tight (1e-12) | PASS |
| `rms` | white_noise | 0.9963959235 | 0.9963959235 | 6.66e-16 | tight (1e-12) | PASS |
| `rms` | sine | 0.707036067 | 0.707036067 | 7.77e-16 | tight (1e-12) | PASS |
| `rms` | impulse | 0.6401045841 | 0.6401045841 | 2.22e-16 | tight (1e-12) | PASS |
| `peak_to_peak` | white_noise | 7.167505047 | 7.167505047 | 0.00e+00 | tight (1e-12) | PASS |
| `peak_to_peak` | sine | 1.999999901 | 1.999999901 | 0.00e+00 | tight (1e-12) | PASS |
| `peak_to_peak` | impulse | 10.34480429 | 10.34480429 | 0.00e+00 | tight (1e-12) | PASS |
| `skewness` | white_noise | -0.01192865021 | -0.01192865021 | 1.30e-16 | tight (1e-12) | PASS |
| `skewness` | skewed | 1.961484641 | 1.961484641 | 1.04e-14 | tight (1e-12) | PASS |
| `skewness` | uniform | 0.02173844496 | 0.02173844496 | 6.25e-17 | tight (1e-12) | PASS |
| `kurtosis` | white_noise | 0.04024425766 | 0.04024425766 | 4.00e-15 | tight (1e-12) | PASS |
| `kurtosis` | uniform | -1.16847618 | -1.16847618 | 2.22e-15 | tight (1e-12) | PASS |
| `kurtosis` | heavy_tail | 25.33092384 | 25.33092384 | 5.33e-14 | tight (1e-12) | PASS |
| `crest_factor` | white_noise | 3.940439351 | 3.940439351 | 2.66e-15 | tight (1e-12) | PASS |
| `crest_factor` | sine | 1.414354935 | 1.414354935 | 1.55e-15 | tight (1e-12) | PASS |
| `crest_factor` | impulse | 15.62244709 | 15.62244709 | 5.33e-15 | tight (1e-12) | PASS |
| `pulsation_index` | white_noise | 2.187773662 | 2.187773662 | 8.88e-16 | tight (1e-12) | PASS |
| `pulsation_index` | sine | 0.6110629868 | 0.6110629868 | 3.33e-16 | tight (1e-12) | PASS |
| `min_max` | white_noise | (-3.241267, 3.926238) | (-3.241267, 3.926238) | 0.00e+00 | tight (1e-12) | PASS |
| `min_max` | sine | (-1.000000, 1.000000) | (-1.000000, 1.000000) | 0.00e+00 | tight (1e-12) | PASS |
| `min_max` | impulse | (-0.344804, 10.000000) | (-0.344804, 10.000000) | 0.00e+00 | tight (1e-12) | PASS |
| `derivative` | sine | array(4999) | array(4999) | 0.00e+00 | tight (1e-12) | PASS |
| `derivative` | white_noise | array(4999) | array(4999) | 0.00e+00 | tight (1e-12) | PASS |
| `derivative` | quadratic | array(4999) | array(4999) | 0.00e+00 | tight (1e-12) | PASS |
| `integral` | sine | array(5000) | array(5000) | 0.00e+00 | tight (1e-12) | PASS |
| `integral` | white_noise | array(5000) | array(5000) | 0.00e+00 | tight (1e-12) | PASS |
| `integral` | constant | array(5000) | array(5000) | 0.00e+00 | tight (1e-12) | PASS |
| `curvature` | sine | array(5000) | array(5000) | 0.00e+00 | tight (1e-12) | PASS |
| `curvature` | quadratic | array(5000) | array(5000) | 0.00e+00 | tight (1e-12) | PASS |
| `rate_of_change` | sine | array(4999) | array(4999) | 0.00e+00 | tight (1e-12) | PASS |
| `rate_of_change` | white_noise | array(4999) | array(4999) | 0.00e+00 | tight (1e-12) | PASS |
| `autocorrelation` | white_noise | -0.01303733881 | -0.01303733881 | 3.47e-17 | normal (1e-08) | PASS |
| `autocorrelation` | sine | 0.999980253 | 0.999980253 | 1.33e-15 | normal (1e-08) | PASS |
| `autocorrelation` | ar1 | 0.9052696657 | 0.9052696657 | 0.00e+00 | normal (1e-08) | PASS |
| `partial_autocorrelation` | white_noise | array(11) | array(11) | 7.98e-17 | normal (1e-08) | PASS |
| `partial_autocorrelation` | ar1 | array(11) | array(11) | 3.57e-14 | normal (1e-08) | PASS |
| `partial_autocorrelation` | ar2 | array(11) | array(11) | 5.03e-15 | normal (1e-08) | PASS |
| `correlation` | correlated_pair | 0.9710740756 | 0.9710740756 | 2.11e-15 | tight (1e-12) | PASS |
| `covariance` | correlated_pair | 0.8343321016 | 0.8343321016 | 1.44e-15 | tight (1e-12) | PASS |
| `euclidean_distance` | correlated_pair | 20.1933727 | 20.1933727 | 3.55e-15 | tight (1e-12) | PASS |
| `cosine_distance` | correlated_pair | 0.02893626307 | 0.02893626307 | 3.77e-15 | tight (1e-12) | PASS |
| `manhattan_distance` | correlated_pair | 1138.345423 | 1138.345423 | 4.55e-13 | tight (1e-12) | PASS |

**Test conditions:** N=5,000 samples, seed=42, all platforms.
**Tolerance tiers:** Exact (0), Tight (1e-12), Normal (1e-8), Relaxed (1e-4)
