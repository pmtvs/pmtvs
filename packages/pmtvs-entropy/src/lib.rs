use numpy::ndarray::ArrayView1;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Compute sample entropy.
#[pyfunction]
fn sample_entropy(py: Python<'_>, signal: PyReadonlyArray1<f64>, m: usize, r: f64) -> PyResult<f64> {
    let signal = signal.as_array();
    let n = signal.len();

    if r == 0.0 || n < m + 2 {
        return Ok(f64::NAN);
    }

    let count_matches = |dim: usize| -> usize {
        let mut count = 0usize;
        let n_templates = n - dim;

        for i in 0..n_templates {
            for j in (i + 1)..n_templates {
                let mut max_diff = 0.0f64;
                for k in 0..dim {
                    let diff = (signal[i + k] - signal[j + k]).abs();
                    if diff > max_diff {
                        max_diff = diff;
                    }
                }
                if max_diff < r {
                    count += 1;
                }
            }
        }
        count
    };

    let b = count_matches(m);
    let a = count_matches(m + 1);

    if b == 0 {
        return Ok(f64::NAN);
    }

    if a > 0 {
        Ok(-((a as f64) / (b as f64)).ln())
    } else {
        Ok(f64::NAN)
    }
}

/// Compute permutation entropy.
#[pyfunction]
fn permutation_entropy(
    py: Python<'_>,
    signal: PyReadonlyArray1<f64>,
    order: usize,
    delay: usize,
    normalize: bool,
) -> PyResult<f64> {
    let signal = signal.as_array();
    let n = signal.len();

    if n < order * delay {
        return Ok(f64::NAN);
    }

    let n_patterns = n - (order - 1) * delay;

    // Count permutation patterns
    let mut pattern_counts = std::collections::HashMap::new();

    for i in 0..n_patterns {
        // Extract pattern indices
        let mut indices: Vec<usize> = (0..order).collect();
        indices.sort_by(|&a, &b| {
            let va = signal[i + a * delay];
            let vb = signal[i + b * delay];
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        });

        *pattern_counts.entry(indices).or_insert(0usize) += 1;
    }

    // Compute entropy
    let total = n_patterns as f64;
    let mut entropy = 0.0;

    for &count in pattern_counts.values() {
        let p = count as f64 / total;
        if p > 0.0 {
            entropy -= p * p.log2();
        }
    }

    if normalize {
        let max_entropy = (factorial(order) as f64).log2();
        if max_entropy > 0.0 {
            entropy /= max_entropy;
        }
    }

    Ok(entropy)
}

fn factorial(n: usize) -> usize {
    (1..=n).product()
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sample_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(permutation_entropy, m)?)?;
    Ok(())
}
