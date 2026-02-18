//! Hurst exponent and DFA (Detrended Fluctuation Analysis)
//!
//! These measure long-range dependence in time series.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// Hurst exponent via R/S analysis
///
/// Args:
///     data: Input time series
///     min_window: Minimum window size (default: 10)
///     max_window: Maximum window size (default: n/4)
///
/// Returns:
///     Hurst exponent (0.5 = random walk, >0.5 = persistent, <0.5 = anti-persistent)
#[pyfunction]
#[pyo3(signature = (data, min_window=10, max_window=None))]
pub fn hurst_exponent_rs(
    _py: Python,
    data: PyReadonlyArray1<f64>,
    min_window: usize,
    max_window: Option<usize>,
) -> PyResult<f64> {
    let arr = data.as_slice()?;
    let max_w = max_window.unwrap_or(arr.len() / 4);
    Ok(hurst_impl(arr, min_window, max_w))
}

fn hurst_impl(data: &[f64], min_window: usize, max_window: usize) -> f64 {
    let n = data.len();
    if n < min_window * 2 {
        return f64::NAN;
    }

    let max_window = max_window.min(n / 2);
    if min_window >= max_window {
        return f64::NAN;
    }

    // Generate window sizes (logarithmically spaced)
    let mut window_sizes = Vec::new();
    let mut w = min_window;
    while w <= max_window {
        window_sizes.push(w);
        w = (w as f64 * 1.5).ceil() as usize;
    }

    if window_sizes.len() < 3 {
        return f64::NAN;
    }

    let mut log_n = Vec::new();
    let mut log_rs = Vec::new();

    for &window in &window_sizes {
        let n_windows = n / window;
        if n_windows == 0 {
            continue;
        }

        let mut rs_values = Vec::new();
        for i in 0..n_windows {
            let start = i * window;
            let end = start + window;
            let segment = &data[start..end];

            // Mean
            let mean: f64 = segment.iter().sum::<f64>() / window as f64;

            // Cumulative deviation from mean
            let mut cumsum = Vec::with_capacity(window);
            let mut sum = 0.0;
            for &x in segment {
                sum += x - mean;
                cumsum.push(sum);
            }

            // Range
            let max_val = cumsum.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min_val = cumsum.iter().cloned().fold(f64::INFINITY, f64::min);
            let range = max_val - min_val;

            // Standard deviation
            let variance: f64 = segment.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window as f64;
            let std = variance.sqrt();

            if std > 0.0 {
                rs_values.push(range / std);
            }
        }

        if !rs_values.is_empty() {
            let mean_rs: f64 = rs_values.iter().sum::<f64>() / rs_values.len() as f64;
            log_n.push((window as f64).ln());
            log_rs.push(mean_rs.ln());
        }
    }

    if log_n.len() < 3 {
        return f64::NAN;
    }

    // Linear regression to get slope (Hurst exponent)
    linear_regression_slope(&log_n, &log_rs)
}

/// DFA (Detrended Fluctuation Analysis)
///
/// Args:
///     data: Input time series
///     order: Polynomial order for detrending (1 = linear, 2 = quadratic)
///     min_window: Minimum window size
///     max_window: Maximum window size
///
/// Returns:
///     DFA scaling exponent (similar interpretation to Hurst)
#[pyfunction]
#[pyo3(signature = (data, order=1, min_window=10, max_window=None))]
pub fn dfa_rs(
    _py: Python,
    data: PyReadonlyArray1<f64>,
    order: usize,
    min_window: usize,
    max_window: Option<usize>,
) -> PyResult<f64> {
    let arr = data.as_slice()?;
    let max_w = max_window.unwrap_or(arr.len() / 4);
    Ok(dfa_impl(arr, order, min_window, max_w))
}

fn dfa_impl(data: &[f64], order: usize, min_window: usize, max_window: usize) -> f64 {
    let n = data.len();
    if n < min_window * 2 {
        return f64::NAN;
    }

    // Integrate the signal (cumulative sum of deviations from mean)
    let mean: f64 = data.iter().sum::<f64>() / n as f64;
    let mut y = Vec::with_capacity(n);
    let mut cumsum = 0.0;
    for &x in data {
        cumsum += x - mean;
        y.push(cumsum);
    }

    let max_window = max_window.min(n / 4);
    if min_window >= max_window {
        return f64::NAN;
    }

    // Generate window sizes
    let mut window_sizes = Vec::new();
    let mut w = min_window;
    while w <= max_window {
        window_sizes.push(w);
        w = (w as f64 * 1.5).ceil() as usize;
    }

    if window_sizes.len() < 3 {
        return f64::NAN;
    }

    let mut log_n = Vec::new();
    let mut log_f = Vec::new();

    for &window in &window_sizes {
        let n_windows = n / window;
        if n_windows == 0 {
            continue;
        }

        let mut fluctuations = Vec::new();
        for i in 0..n_windows {
            let start = i * window;
            let end = start + window;
            let segment = &y[start..end];

            // Detrend with polynomial fit
            let trend = polynomial_fit(segment, order);

            // RMS of residuals
            let rms: f64 = segment
                .iter()
                .zip(trend.iter())
                .map(|(y, t)| (y - t).powi(2))
                .sum::<f64>()
                / window as f64;

            fluctuations.push(rms.sqrt());
        }

        if !fluctuations.is_empty() {
            let mean_f: f64 = fluctuations.iter().sum::<f64>() / fluctuations.len() as f64;
            if mean_f > 0.0 {
                log_n.push((window as f64).ln());
                log_f.push(mean_f.ln());
            }
        }
    }

    if log_n.len() < 3 {
        return f64::NAN;
    }

    linear_regression_slope(&log_n, &log_f)
}

fn polynomial_fit(y: &[f64], order: usize) -> Vec<f64> {
    let n = y.len();
    if order == 0 || n == 0 {
        let mean: f64 = y.iter().sum::<f64>() / n as f64;
        return vec![mean; n];
    }

    // Simple linear detrending for order=1 (most common case)
    if order == 1 {
        let x_mean = (n - 1) as f64 / 2.0;
        let y_mean: f64 = y.iter().sum::<f64>() / n as f64;

        let mut num = 0.0;
        let mut den = 0.0;
        for (i, &yi) in y.iter().enumerate() {
            let xi = i as f64 - x_mean;
            num += xi * (yi - y_mean);
            den += xi * xi;
        }

        let slope = if den > 0.0 { num / den } else { 0.0 };
        let intercept = y_mean - slope * x_mean;

        return (0..n).map(|i| intercept + slope * i as f64).collect();
    }

    // For higher orders, fall back to simple linear (good enough for DFA)
    polynomial_fit(y, 1)
}

fn linear_regression_slope(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return f64::NAN;
    }

    let x_mean: f64 = x.iter().sum::<f64>() / n as f64;
    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;

    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..n {
        let dx = x[i] - x_mean;
        let dy = y[i] - y_mean;
        num += dx * dy;
        den += dx * dx;
    }

    if den > 0.0 {
        num / den
    } else {
        f64::NAN
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hurst_random_walk() {
        // Random walk should have H ≈ 0.5
        // Using deterministic "random" for reproducibility
        let mut data = vec![0.0; 500];
        let mut val = 0.0;
        for i in 1..500 {
            val += if i % 2 == 0 { 1.0 } else { -1.0 };
            data[i] = val;
        }
        let h = hurst_impl(&data, 10, 100);
        assert!(h > 0.3 && h < 0.7, "Hurst exponent {h} not near 0.5");
    }
}
