use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Compute mean.
#[pyfunction]
fn mean(py: Python<'_>, signal: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let signal: Vec<f64> = signal.as_array().iter().filter(|x| x.is_finite()).copied().collect();
    if signal.is_empty() {
        return Ok(f64::NAN);
    }
    Ok(signal.iter().sum::<f64>() / signal.len() as f64)
}

/// Compute standard deviation.
#[pyfunction]
#[pyo3(name = "std")]
fn std_dev(py: Python<'_>, signal: PyReadonlyArray1<f64>, ddof: usize) -> PyResult<f64> {
    let signal: Vec<f64> = signal.as_array().iter().filter(|x| x.is_finite()).copied().collect();
    let n = signal.len();
    if n < ddof + 1 {
        return Ok(f64::NAN);
    }
    let mean = signal.iter().sum::<f64>() / n as f64;
    let variance: f64 = signal.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - ddof) as f64;
    Ok(variance.sqrt())
}

/// Compute variance.
#[pyfunction]
fn variance(py: Python<'_>, signal: PyReadonlyArray1<f64>, ddof: usize) -> PyResult<f64> {
    let signal: Vec<f64> = signal.as_array().iter().filter(|x| x.is_finite()).copied().collect();
    let n = signal.len();
    if n < ddof + 1 {
        return Ok(f64::NAN);
    }
    let mean = signal.iter().sum::<f64>() / n as f64;
    Ok(signal.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - ddof) as f64)
}

/// Compute min and max.
#[pyfunction]
fn min_max(py: Python<'_>, signal: PyReadonlyArray1<f64>) -> PyResult<(f64, f64)> {
    let signal: Vec<f64> = signal.as_array().iter().filter(|x| x.is_finite()).copied().collect();
    if signal.is_empty() {
        return Ok((f64::NAN, f64::NAN));
    }
    let min = signal.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = signal.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    Ok((min, max))
}

/// Compute RMS.
#[pyfunction]
fn rms(py: Python<'_>, signal: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let signal: Vec<f64> = signal.as_array().iter().filter(|x| x.is_finite()).copied().collect();
    if signal.is_empty() {
        return Ok(f64::NAN);
    }
    let sum_sq: f64 = signal.iter().map(|x| x * x).sum();
    Ok((sum_sq / signal.len() as f64).sqrt())
}

/// Compute peak-to-peak.
#[pyfunction]
fn peak_to_peak(py: Python<'_>, signal: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let signal: Vec<f64> = signal.as_array().iter().filter(|x| x.is_finite()).copied().collect();
    if signal.is_empty() {
        return Ok(f64::NAN);
    }
    let min = signal.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = signal.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    Ok(max - min)
}

/// Compute skewness.
#[pyfunction]
fn skewness(py: Python<'_>, signal: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let signal: Vec<f64> = signal.as_array().iter().filter(|x| x.is_finite()).copied().collect();
    let n = signal.len();
    if n < 3 {
        return Ok(f64::NAN);
    }

    let mean = signal.iter().sum::<f64>() / n as f64;
    let variance: f64 = signal.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let std = variance.sqrt();

    if std == 0.0 {
        return Ok(f64::NAN);
    }

    let skew: f64 = signal.iter().map(|x| ((x - mean) / std).powi(3)).sum::<f64>() / n as f64;
    Ok(skew)
}

/// Compute kurtosis.
#[pyfunction]
fn kurtosis(py: Python<'_>, signal: PyReadonlyArray1<f64>, fisher: bool) -> PyResult<f64> {
    let signal: Vec<f64> = signal.as_array().iter().filter(|x| x.is_finite()).copied().collect();
    let n = signal.len();
    if n < 4 {
        return Ok(f64::NAN);
    }

    let mean = signal.iter().sum::<f64>() / n as f64;
    let variance: f64 = signal.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let std = variance.sqrt();

    if std == 0.0 {
        return Ok(f64::NAN);
    }

    let mut kurt: f64 = signal.iter().map(|x| ((x - mean) / std).powi(4)).sum::<f64>() / n as f64;
    if fisher {
        kurt -= 3.0;
    }
    Ok(kurt)
}

/// Compute crest factor.
#[pyfunction]
fn crest_factor(py: Python<'_>, signal: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let signal: Vec<f64> = signal.as_array().iter().filter(|x| x.is_finite()).copied().collect();
    if signal.is_empty() {
        return Ok(f64::NAN);
    }

    let sum_sq: f64 = signal.iter().map(|x| x * x).sum();
    let rms_val = (sum_sq / signal.len() as f64).sqrt();

    if rms_val == 0.0 {
        return Ok(f64::NAN);
    }

    let peak = signal.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
    Ok(peak / rms_val)
}

/// Compute pulsation index.
#[pyfunction]
fn pulsation_index(py: Python<'_>, signal: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let signal: Vec<f64> = signal.as_array().iter().filter(|x| x.is_finite()).copied().collect();
    if signal.is_empty() {
        return Ok(f64::NAN);
    }

    let mean = signal.iter().sum::<f64>() / signal.len() as f64;
    if mean == 0.0 {
        return Ok(f64::NAN);
    }

    let min = signal.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = signal.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    Ok((max - min) / mean.abs())
}

/// Compute derivative.
#[pyfunction]
fn derivative<'py>(py: Python<'py>, signal: PyReadonlyArray1<f64>, dt: f64) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let signal = signal.as_array();
    let n = signal.len();

    if n < 2 {
        return Ok(PyArray1::from_vec_bound(py, vec![f64::NAN]));
    }

    let mut result = Vec::with_capacity(n - 1);
    for i in 0..(n - 1) {
        result.push((signal[i + 1] - signal[i]) / dt);
    }

    Ok(PyArray1::from_vec_bound(py, result))
}

/// Compute integral (trapezoidal).
#[pyfunction]
fn integral<'py>(py: Python<'py>, signal: PyReadonlyArray1<f64>, dt: f64) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let signal = signal.as_array();
    let n = signal.len();

    if n < 2 {
        return Ok(PyArray1::from_vec_bound(py, vec![0.0]));
    }

    let mut result = vec![0.0; n];
    for i in 1..n {
        result[i] = result[i - 1] + 0.5 * (signal[i] + signal[i - 1]) * dt;
    }

    Ok(PyArray1::from_vec_bound(py, result))
}

/// Compute curvature.
#[pyfunction]
fn curvature<'py>(py: Python<'py>, signal: PyReadonlyArray1<f64>, dt: f64) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let signal = signal.as_array();
    let n = signal.len();

    if n < 3 {
        return Ok(PyArray1::from_vec_bound(py, vec![f64::NAN]));
    }

    let mut dy = vec![0.0; n];
    let mut d2y = vec![0.0; n];

    // First derivative (central difference)
    for i in 1..(n - 1) {
        dy[i] = (signal[i + 1] - signal[i - 1]) / (2.0 * dt);
    }
    dy[0] = (signal[1] - signal[0]) / dt;
    dy[n - 1] = (signal[n - 1] - signal[n - 2]) / dt;

    // Second derivative
    for i in 1..(n - 1) {
        d2y[i] = (signal[i + 1] - 2.0 * signal[i] + signal[i - 1]) / (dt * dt);
    }
    d2y[0] = d2y[1];
    d2y[n - 1] = d2y[n - 2];

    // Curvature
    let result: Vec<f64> = dy.iter().zip(d2y.iter())
        .map(|(&dy_i, &d2y_i)| d2y_i.abs() / (1.0 + dy_i * dy_i).powf(1.5))
        .collect();

    Ok(PyArray1::from_vec_bound(py, result))
}

/// Compute rate of change.
#[pyfunction]
fn rate_of_change<'py>(py: Python<'py>, signal: PyReadonlyArray1<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let signal = signal.as_array();
    let n = signal.len();

    if n < 2 {
        return Ok(PyArray1::from_vec_bound(py, vec![f64::NAN]));
    }

    let mut result = Vec::with_capacity(n - 1);
    for i in 0..(n - 1) {
        if signal[i] != 0.0 {
            result.push((signal[i + 1] - signal[i]) / signal[i].abs());
        } else {
            result.push(f64::NAN);
        }
    }

    Ok(PyArray1::from_vec_bound(py, result))
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mean, m)?)?;
    m.add_function(wrap_pyfunction!(std_dev, m)?)?;
    m.add_function(wrap_pyfunction!(variance, m)?)?;
    m.add_function(wrap_pyfunction!(min_max, m)?)?;
    m.add_function(wrap_pyfunction!(rms, m)?)?;
    m.add_function(wrap_pyfunction!(peak_to_peak, m)?)?;
    m.add_function(wrap_pyfunction!(skewness, m)?)?;
    m.add_function(wrap_pyfunction!(kurtosis, m)?)?;
    m.add_function(wrap_pyfunction!(crest_factor, m)?)?;
    m.add_function(wrap_pyfunction!(pulsation_index, m)?)?;
    m.add_function(wrap_pyfunction!(derivative, m)?)?;
    m.add_function(wrap_pyfunction!(integral, m)?)?;
    m.add_function(wrap_pyfunction!(curvature, m)?)?;
    m.add_function(wrap_pyfunction!(rate_of_change, m)?)?;
    Ok(())
}
