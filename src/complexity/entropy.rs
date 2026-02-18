//! Entropy computations
//!
//! Sample entropy, permutation entropy, and approximate entropy
//! are O(n^2) algorithms that benefit enormously from Rust.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Sample entropy - measures complexity/regularity of a time series
///
/// Args:
///     data: Input time series
///     m: Embedding dimension (pattern length)
///     r: Tolerance (typically 0.2 * std)
///
/// Returns:
///     Sample entropy value (higher = more complex/irregular)
#[pyfunction]
pub fn sample_entropy_rs(
    _py: Python,
    data: PyReadonlyArray1<f64>,
    m: usize,
    r: f64,
) -> PyResult<f64> {
    let arr = data.as_slice()?;
    Ok(sample_entropy_impl(arr, m, r))
}

fn sample_entropy_impl(data: &[f64], m: usize, r: f64) -> f64 {
    let n = data.len();
    if n <= m + 1 {
        return f64::NAN;
    }

    let count_matches = |dim: usize| -> usize {
        let mut count = 0;
        for i in 0..n - dim {
            for j in (i + 1)..n - dim {
                let mut is_match = true;
                for k in 0..dim {
                    if (data[i + k] - data[j + k]).abs() > r {
                        is_match = false;
                        break;
                    }
                }
                if is_match {
                    count += 1;
                }
            }
        }
        count
    };

    let a = count_matches(m + 1) as f64;
    let b = count_matches(m) as f64;

    if b == 0.0 || a == 0.0 {
        f64::NAN
    } else {
        -((a / b).ln())
    }
}

/// Permutation entropy - measures complexity via ordinal patterns
///
/// Args:
///     data: Input time series
///     order: Permutation order (typically 3-7)
///     delay: Time delay between samples
///
/// Returns:
///     Normalized permutation entropy (0 = regular, 1 = random)
#[pyfunction]
pub fn permutation_entropy_rs(
    _py: Python,
    data: PyReadonlyArray1<f64>,
    order: usize,
    delay: usize,
) -> PyResult<f64> {
    let arr = data.as_slice()?;
    Ok(permutation_entropy_impl(arr, order, delay))
}

fn permutation_entropy_impl(data: &[f64], order: usize, delay: usize) -> f64 {
    let n = data.len();
    let n_patterns = n.saturating_sub((order - 1) * delay);

    if n_patterns == 0 || order < 2 {
        return f64::NAN;
    }

    // Count permutation patterns
    let factorial: usize = (1..=order).product();
    let mut counts = vec![0usize; factorial];

    for i in 0..n_patterns {
        // Extract pattern indices
        let mut indices: Vec<usize> = (0..order).collect();
        indices.sort_by(|&a, &b| {
            let va = data[i + a * delay];
            let vb = data[i + b * delay];
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Convert to permutation index (Lehmer code)
        let mut perm_idx = 0;
        let mut remaining: Vec<usize> = (0..order).collect();
        for (pos, &idx) in indices.iter().enumerate() {
            let rank = remaining.iter().position(|&x| x == idx).unwrap();
            perm_idx += rank * (1..=(order - 1 - pos)).product::<usize>();
            remaining.remove(rank);
        }
        counts[perm_idx] += 1;
    }

    // Compute Shannon entropy
    let total = n_patterns as f64;
    let mut entropy = 0.0;
    for &c in &counts {
        if c > 0 {
            let p = c as f64 / total;
            entropy -= p * p.ln();
        }
    }

    // Normalize by max entropy
    let max_entropy = (factorial as f64).ln();
    if max_entropy > 0.0 {
        entropy / max_entropy
    } else {
        f64::NAN
    }
}

/// Approximate entropy - similar to sample entropy but with self-matches
///
/// Args:
///     data: Input time series
///     m: Embedding dimension
///     r: Tolerance
///
/// Returns:
///     Approximate entropy value
#[pyfunction]
pub fn approximate_entropy_rs(
    _py: Python,
    data: PyReadonlyArray1<f64>,
    m: usize,
    r: f64,
) -> PyResult<f64> {
    let arr = data.as_slice()?;
    Ok(approximate_entropy_impl(arr, m, r))
}

fn approximate_entropy_impl(data: &[f64], m: usize, r: f64) -> f64 {
    let n = data.len();
    if n <= m + 1 {
        return f64::NAN;
    }

    let phi = |dim: usize| -> f64 {
        let n_patterns = n - dim + 1;
        let mut total = 0.0;

        for i in 0..n_patterns {
            let mut count = 0;
            for j in 0..n_patterns {
                let mut is_match = true;
                for k in 0..dim {
                    if (data[i + k] - data[j + k]).abs() > r {
                        is_match = false;
                        break;
                    }
                }
                if is_match {
                    count += 1;
                }
            }
            total += (count as f64 / n_patterns as f64).ln();
        }
        total / n_patterns as f64
    };

    phi(m) - phi(m + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_entropy_constant() {
        // Constant signal should have low entropy
        let data: Vec<f64> = vec![1.0; 100];
        let se = sample_entropy_impl(&data, 2, 0.2);
        assert!(se.is_nan() || se < 0.1);
    }

    #[test]
    fn test_permutation_entropy_monotonic() {
        // Monotonic signal should have zero entropy
        let data: Vec<f64> = (0..100).map(|x| x as f64).collect();
        let pe = permutation_entropy_impl(&data, 3, 1);
        assert!(pe < 0.1);
    }
}
