//! Phase space embedding functions
//!
//! Time delay embedding for reconstructing attractors from scalar time series.

use numpy::{PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use ndarray::Array2;

/// Time delay embedding
///
/// Args:
///     data: Input time series
///     dim: Embedding dimension
///     delay: Time delay between coordinates
///
/// Returns:
///     Embedded trajectory as 2D array (n_points x dim)
#[pyfunction]
pub fn time_delay_embedding_rs<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<f64>,
    dim: usize,
    delay: usize,
) -> PyResult<&'py PyArray2<f64>> {
    let arr = data.as_slice()?;
    let result = time_delay_embedding_impl(arr, dim, delay);
    Ok(PyArray2::from_owned_array(py, result))
}

fn time_delay_embedding_impl(data: &[f64], dim: usize, delay: usize) -> Array2<f64> {
    let n = data.len();
    let n_points = n.saturating_sub((dim - 1) * delay);

    if n_points == 0 || dim == 0 {
        return Array2::zeros((0, dim.max(1)));
    }

    let mut result = Array2::zeros((n_points, dim));
    for i in 0..n_points {
        for d in 0..dim {
            result[[i, d]] = data[i + d * delay];
        }
    }
    result
}

/// Optimal delay via first minimum of auto-mutual information
///
/// Args:
///     data: Input time series
///     max_delay: Maximum delay to search
///     n_bins: Number of bins for MI estimation
///
/// Returns:
///     Optimal delay value
#[pyfunction]
#[pyo3(signature = (data, max_delay=50, n_bins=16))]
pub fn optimal_delay_rs(
    _py: Python,
    data: PyReadonlyArray1<f64>,
    max_delay: usize,
    n_bins: usize,
) -> PyResult<usize> {
    let arr = data.as_slice()?;
    Ok(optimal_delay_impl(arr, max_delay, n_bins))
}

fn optimal_delay_impl(data: &[f64], max_delay: usize, n_bins: usize) -> usize {
    let n = data.len();
    if n < max_delay + 1 || n_bins < 2 {
        return 1;
    }

    // Find data range for binning
    let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max_val - min_val;
    if range == 0.0 {
        return 1;
    }

    let bin_width = range / n_bins as f64;

    let to_bin = |x: f64| -> usize {
        let b = ((x - min_val) / bin_width) as usize;
        b.min(n_bins - 1)
    };

    // Compute mutual information for each delay
    let mut mi_values = Vec::with_capacity(max_delay);

    for delay in 1..=max_delay {
        let n_pairs = n - delay;
        if n_pairs == 0 {
            break;
        }

        // Joint and marginal histograms
        let mut joint = vec![vec![0usize; n_bins]; n_bins];
        let mut marginal_x = vec![0usize; n_bins];
        let mut marginal_y = vec![0usize; n_bins];

        for i in 0..n_pairs {
            let bx = to_bin(data[i]);
            let by = to_bin(data[i + delay]);
            joint[bx][by] += 1;
            marginal_x[bx] += 1;
            marginal_y[by] += 1;
        }

        // Compute MI
        let n_f = n_pairs as f64;
        let mut mi = 0.0;
        for i in 0..n_bins {
            for j in 0..n_bins {
                if joint[i][j] > 0 && marginal_x[i] > 0 && marginal_y[j] > 0 {
                    let p_xy = joint[i][j] as f64 / n_f;
                    let p_x = marginal_x[i] as f64 / n_f;
                    let p_y = marginal_y[j] as f64 / n_f;
                    mi += p_xy * (p_xy / (p_x * p_y)).ln();
                }
            }
        }
        mi_values.push(mi);
    }

    // Find first local minimum
    for i in 1..mi_values.len().saturating_sub(1) {
        if mi_values[i] < mi_values[i - 1] && mi_values[i] < mi_values[i + 1] {
            return i + 1; // delay is 1-indexed
        }
    }

    // If no minimum found, return delay at lowest MI
    mi_values
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i + 1)
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_dimensions() {
        let data: Vec<f64> = (0..100).map(|x| x as f64).collect();
        let result = time_delay_embedding_impl(&data, 3, 2);
        assert_eq!(result.shape(), &[96, 3]);
    }
}
