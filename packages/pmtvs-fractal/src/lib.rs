use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

const DEFAULT_RS_MIN_K: usize = 10;
const DEFAULT_RS_MAX_K_RATIO: f64 = 0.25;
const DEFAULT_RS_MAX_K_CAP: usize = 500;
const DEFAULT_DFA_MIN_SCALE: usize = 4;
const DEFAULT_DFA_MAX_SCALE_RATIO: f64 = 0.25;
const DEFAULT_DFA_MAX_SCALE_CAP: usize = 256;
const DEFAULT_DFA_N_SCALES: usize = 20;
const DEFAULT_MIN_SAMPLES_DFA: usize = 64;

/// Compute Hurst exponent using R/S method.
#[pyfunction]
fn hurst_exponent(py: Python<'_>, signal: PyReadonlyArray1<f64>, method: &str) -> PyResult<f64> {
    let signal: Vec<f64> = signal.as_array().iter().filter(|x| x.is_finite()).copied().collect();
    let n = signal.len();

    if n < DEFAULT_RS_MIN_K {
        return Ok(f64::NAN);
    }

    if method == "dfa" {
        return dfa_internal(&signal, DEFAULT_DFA_MIN_SCALE, -1, 1);
    }

    // R/S method
    let max_k = std::cmp::min((n as f64 * DEFAULT_RS_MAX_K_RATIO) as usize, DEFAULT_RS_MAX_K_CAP);

    let mut log_k = Vec::new();
    let mut log_rs = Vec::new();

    for k in DEFAULT_RS_MIN_K..max_k {
        let n_subseries = n / k;
        let mut rs_sum = 0.0;

        for i in 0..n_subseries {
            let start = i * k;
            let end = start + k;
            let subseries = &signal[start..end];

            let mean: f64 = subseries.iter().sum::<f64>() / k as f64;

            // Cumulative deviation
            let mut y = vec![0.0; k];
            let mut cumsum = 0.0;
            for (j, &val) in subseries.iter().enumerate() {
                cumsum += val - mean;
                y[j] = cumsum;
            }

            let r = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                  - y.iter().cloned().fold(f64::INFINITY, f64::min);

            // Standard deviation (ddof=1)
            let variance: f64 = subseries.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (k - 1) as f64;
            let s = variance.sqrt();

            if s > 0.0 {
                rs_sum += r / s;
            }
        }

        if n_subseries > 0 {
            let rs_avg = rs_sum / n_subseries as f64;
            if rs_avg > 0.0 {
                log_k.push((k as f64).ln());
                log_rs.push(rs_avg.ln());
            }
        }
    }

    if log_k.len() < 3 {
        return Ok(f64::NAN);
    }

    // Linear fit
    let h = linear_fit_slope(&log_k, &log_rs);
    Ok(h.clamp(0.0, 1.0))
}

/// Compute DFA exponent.
#[pyfunction]
fn dfa(py: Python<'_>, signal: PyReadonlyArray1<f64>, min_scale: i32, max_scale: i32, order: i32) -> PyResult<f64> {
    let signal: Vec<f64> = signal.as_array().iter().filter(|x| x.is_finite()).copied().collect();
    dfa_internal(&signal, min_scale as usize, max_scale as i32, order as usize)
}

fn dfa_internal(signal: &[f64], min_scale: usize, max_scale_in: i32, order: usize) -> PyResult<f64> {
    let n = signal.len();

    if n < DEFAULT_MIN_SAMPLES_DFA {
        return Ok(f64::NAN);
    }

    // Integrate signal
    let mean: f64 = signal.iter().sum::<f64>() / n as f64;
    let mut y = vec![0.0; n];
    let mut cumsum = 0.0;
    for (i, &val) in signal.iter().enumerate() {
        cumsum += val - mean;
        y[i] = cumsum;
    }

    let max_scale = if max_scale_in < 0 {
        std::cmp::min((n as f64 * DEFAULT_DFA_MAX_SCALE_RATIO) as usize, DEFAULT_DFA_MAX_SCALE_CAP)
    } else {
        max_scale_in as usize
    };

    // Generate log-spaced scales
    let mut scales = Vec::new();
    let log_min = (min_scale as f64).log10();
    let log_max = (max_scale as f64).log10();

    for i in 0..DEFAULT_DFA_N_SCALES {
        let log_scale = log_min + (log_max - log_min) * i as f64 / (DEFAULT_DFA_N_SCALES - 1) as f64;
        let scale = 10f64.powf(log_scale) as usize;
        if scales.is_empty() || *scales.last().unwrap() != scale {
            scales.push(scale);
        }
    }

    let mut log_scales = Vec::new();
    let mut log_fluct = Vec::new();

    for &scale in &scales {
        let n_segments = n / scale;
        if n_segments < 2 {
            continue;
        }

        let mut f_sq_sum = 0.0;
        let mut count = 0;

        for i in 0..n_segments {
            let start = i * scale;
            let end = start + scale;
            let segment = &y[start..end];

            // Polynomial fit (order 1 = linear)
            let trend = polynomial_fit(segment, order);

            // Fluctuation
            let f_sq: f64 = segment.iter().zip(trend.iter())
                .map(|(y, t)| (y - t).powi(2))
                .sum::<f64>() / scale as f64;

            f_sq_sum += f_sq;
            count += 1;
        }

        if count > 0 {
            let f = (f_sq_sum / count as f64).sqrt();
            log_scales.push((scale as f64).ln());
            log_fluct.push(f.ln());
        }
    }

    if log_scales.len() < 3 {
        return Ok(f64::NAN);
    }

    Ok(linear_fit_slope(&log_scales, &log_fluct))
}

/// Compute R² of Hurst fit.
#[pyfunction]
fn hurst_r2(py: Python<'_>, signal: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let signal: Vec<f64> = signal.as_array().iter().filter(|x| x.is_finite()).copied().collect();
    let n = signal.len();

    if n < DEFAULT_RS_MIN_K {
        return Ok(f64::NAN);
    }

    let max_k = std::cmp::min((n as f64 * DEFAULT_RS_MAX_K_RATIO) as usize, DEFAULT_RS_MAX_K_CAP);

    let mut log_k = Vec::new();
    let mut log_rs = Vec::new();

    for k in DEFAULT_RS_MIN_K..max_k {
        let n_subseries = n / k;
        let mut rs_sum = 0.0;

        for i in 0..n_subseries {
            let start = i * k;
            let end = start + k;
            let subseries = &signal[start..end];

            let mean: f64 = subseries.iter().sum::<f64>() / k as f64;

            let mut y = vec![0.0; k];
            let mut cumsum = 0.0;
            for (j, &val) in subseries.iter().enumerate() {
                cumsum += val - mean;
                y[j] = cumsum;
            }

            let r = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                  - y.iter().cloned().fold(f64::INFINITY, f64::min);

            let variance: f64 = subseries.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (k - 1) as f64;
            let s = variance.sqrt();

            if s > 0.0 {
                rs_sum += r / s;
            }
        }

        if n_subseries > 0 {
            let rs_avg = rs_sum / n_subseries as f64;
            if rs_avg > 0.0 {
                log_k.push((k as f64).ln());
                log_rs.push(rs_avg.ln());
            }
        }
    }

    if log_k.len() < 3 {
        return Ok(f64::NAN);
    }

    // Compute R²
    let (slope, intercept) = linear_fit(&log_k, &log_rs);
    let mean_y: f64 = log_rs.iter().sum::<f64>() / log_rs.len() as f64;

    let ss_res: f64 = log_k.iter().zip(log_rs.iter())
        .map(|(x, y)| (y - (slope * x + intercept)).powi(2))
        .sum();

    let ss_tot: f64 = log_rs.iter().map(|y| (y - mean_y).powi(2)).sum();

    if ss_tot == 0.0 {
        return Ok(f64::NAN);
    }

    Ok(1.0 - ss_res / ss_tot)
}

fn linear_fit_slope(x: &[f64], y: &[f64]) -> f64 {
    let (slope, _) = linear_fit(x, y);
    slope
}

fn linear_fit(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    let sum_xx: f64 = x.iter().map(|a| a * a).sum();

    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    let intercept = (sum_y - slope * sum_x) / n;

    (slope, intercept)
}

fn polynomial_fit(y: &[f64], order: usize) -> Vec<f64> {
    let n = y.len();
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();

    if order == 1 {
        // Linear fit
        let (slope, intercept) = linear_fit(&x, y);
        x.iter().map(|&xi| slope * xi + intercept).collect()
    } else {
        // For simplicity, just use linear for now
        // Full polynomial would need matrix operations
        let (slope, intercept) = linear_fit(&x, y);
        x.iter().map(|&xi| slope * xi + intercept).collect()
    }
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hurst_exponent, m)?)?;
    m.add_function(wrap_pyfunction!(dfa, m)?)?;
    m.add_function(wrap_pyfunction!(hurst_r2, m)?)?;
    Ok(())
}
