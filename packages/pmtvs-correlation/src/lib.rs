use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Compute autocorrelation at specified lag.
#[pyfunction]
fn autocorrelation(py: Python<'_>, signal: PyReadonlyArray1<f64>, lag: usize) -> PyResult<f64> {
    let signal: Vec<f64> = signal.as_array().iter().filter(|x| x.is_finite()).copied().collect();
    let n = signal.len();

    if n < lag + 2 {
        return Ok(f64::NAN);
    }

    let x = &signal[..n - lag];
    let y = &signal[lag..];

    let x_mean: f64 = x.iter().sum::<f64>() / x.len() as f64;
    let y_mean: f64 = y.iter().sum::<f64>() / y.len() as f64;

    let num: f64 = x.iter().zip(y.iter()).map(|(a, b)| (a - x_mean) * (b - y_mean)).sum();
    let den_x: f64 = x.iter().map(|a| (a - x_mean).powi(2)).sum();
    let den_y: f64 = y.iter().map(|b| (b - y_mean).powi(2)).sum();
    let den = (den_x * den_y).sqrt();

    if den == 0.0 {
        return Ok(f64::NAN);
    }

    Ok(num / den)
}

/// Compute partial autocorrelation function.
#[pyfunction]
fn partial_autocorrelation<'py>(py: Python<'py>, signal: PyReadonlyArray1<f64>, max_lag: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let signal: Vec<f64> = signal.as_array().iter().filter(|x| x.is_finite()).copied().collect();
    let n = signal.len();

    if n < max_lag + 2 {
        let result = vec![f64::NAN; max_lag + 1];
        return Ok(PyArray1::from_vec(py, result));
    }

    // Compute autocorrelations
    let mut acf = vec![0.0; max_lag + 1];
    acf[0] = 1.0;

    for k in 1..=max_lag {
        let x = &signal[..n - k];
        let y = &signal[k..];

        let x_mean: f64 = x.iter().sum::<f64>() / x.len() as f64;
        let y_mean: f64 = y.iter().sum::<f64>() / y.len() as f64;

        let num: f64 = x.iter().zip(y.iter()).map(|(a, b)| (a - x_mean) * (b - y_mean)).sum();
        let den_x: f64 = x.iter().map(|a| (a - x_mean).powi(2)).sum();
        let den_y: f64 = y.iter().map(|b| (b - y_mean).powi(2)).sum();
        let den = (den_x * den_y).sqrt();

        acf[k] = if den == 0.0 { f64::NAN } else { num / den };
    }

    // Durbin-Levinson recursion
    let mut pacf = vec![0.0; max_lag + 1];
    pacf[0] = 1.0;

    let mut phi = vec![vec![0.0; max_lag + 1]; max_lag + 1];

    for k in 1..=max_lag {
        if k == 1 {
            phi[1][1] = acf[1];
        } else {
            let mut num = acf[k];
            for j in 1..k {
                num -= phi[k-1][j] * acf[k - j];
            }

            let mut den = 1.0;
            for j in 1..k {
                den -= phi[k-1][j] * acf[j];
            }

            phi[k][k] = if den.abs() < 1e-10 { 0.0 } else { num / den };

            for j in 1..k {
                phi[k][j] = phi[k-1][j] - phi[k][k] * phi[k-1][k - j];
            }
        }
        pacf[k] = phi[k][k];
    }

    Ok(PyArray1::from_vec(py, pacf))
}

/// Compute Pearson correlation coefficient.
#[pyfunction]
fn correlation(py: Python<'_>, x: PyReadonlyArray1<f64>, y: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let x_arr = x.as_array();
    let y_arr = y.as_array();

    if x_arr.len() != y_arr.len() || x_arr.len() < 2 {
        return Ok(f64::NAN);
    }

    // Filter out NaN pairs
    let pairs: Vec<(f64, f64)> = x_arr.iter()
        .zip(y_arr.iter())
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .map(|(a, b)| (*a, *b))
        .collect();

    let n = pairs.len();
    if n < 2 {
        return Ok(f64::NAN);
    }

    let x_mean: f64 = pairs.iter().map(|(a, _)| a).sum::<f64>() / n as f64;
    let y_mean: f64 = pairs.iter().map(|(_, b)| b).sum::<f64>() / n as f64;

    let num: f64 = pairs.iter().map(|(a, b)| (a - x_mean) * (b - y_mean)).sum();
    let den_x: f64 = pairs.iter().map(|(a, _)| (a - x_mean).powi(2)).sum();
    let den_y: f64 = pairs.iter().map(|(_, b)| (b - y_mean).powi(2)).sum();
    let den = (den_x * den_y).sqrt();

    if den == 0.0 {
        return Ok(f64::NAN);
    }

    Ok(num / den)
}

/// Compute covariance.
#[pyfunction]
fn covariance(py: Python<'_>, x: PyReadonlyArray1<f64>, y: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let x_arr = x.as_array();
    let y_arr = y.as_array();

    if x_arr.len() != y_arr.len() || x_arr.len() < 2 {
        return Ok(f64::NAN);
    }

    // Filter out NaN pairs
    let pairs: Vec<(f64, f64)> = x_arr.iter()
        .zip(y_arr.iter())
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .map(|(a, b)| (*a, *b))
        .collect();

    let n = pairs.len();
    if n < 2 {
        return Ok(f64::NAN);
    }

    let x_mean: f64 = pairs.iter().map(|(a, _)| a).sum::<f64>() / n as f64;
    let y_mean: f64 = pairs.iter().map(|(_, b)| b).sum::<f64>() / n as f64;

    let cov: f64 = pairs.iter().map(|(a, b)| (a - x_mean) * (b - y_mean)).sum::<f64>() / (n - 1) as f64;

    Ok(cov)
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(autocorrelation, m)?)?;
    m.add_function(wrap_pyfunction!(partial_autocorrelation, m)?)?;
    m.add_function(wrap_pyfunction!(correlation, m)?)?;
    m.add_function(wrap_pyfunction!(covariance, m)?)?;
    Ok(())
}
