use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// Compute Euclidean (L2) distance between two signals.
#[pyfunction]
fn euclidean_distance(py: Python<'_>, x: PyReadonlyArray1<f64>, y: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let x_arr = x.as_array();
    let y_arr = y.as_array();

    if x_arr.len() != y_arr.len() {
        return Ok(f64::NAN);
    }

    // Filter out NaN pairs and compute distance
    let sum_sq: f64 = x_arr.iter()
        .zip(y_arr.iter())
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .map(|(a, b)| (a - b).powi(2))
        .sum();

    let valid_count = x_arr.iter()
        .zip(y_arr.iter())
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .count();

    if valid_count == 0 {
        return Ok(f64::NAN);
    }

    Ok(sum_sq.sqrt())
}

/// Compute cosine distance between two signals.
#[pyfunction]
fn cosine_distance(py: Python<'_>, x: PyReadonlyArray1<f64>, y: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let x_arr = x.as_array();
    let y_arr = y.as_array();

    if x_arr.len() != y_arr.len() {
        return Ok(f64::NAN);
    }

    // Filter valid pairs
    let pairs: Vec<(f64, f64)> = x_arr.iter()
        .zip(y_arr.iter())
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .map(|(a, b)| (*a, *b))
        .collect();

    if pairs.is_empty() {
        return Ok(f64::NAN);
    }

    let dot: f64 = pairs.iter().map(|(a, b)| a * b).sum();
    let norm_x: f64 = pairs.iter().map(|(a, _)| a.powi(2)).sum::<f64>().sqrt();
    let norm_y: f64 = pairs.iter().map(|(_, b)| b.powi(2)).sum::<f64>().sqrt();

    if norm_x == 0.0 || norm_y == 0.0 {
        return Ok(f64::NAN);
    }

    let cosine_similarity = (dot / (norm_x * norm_y)).clamp(-1.0, 1.0);
    Ok(1.0 - cosine_similarity)
}

/// Compute Manhattan (L1) distance between two signals.
#[pyfunction]
fn manhattan_distance(py: Python<'_>, x: PyReadonlyArray1<f64>, y: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let x_arr = x.as_array();
    let y_arr = y.as_array();

    if x_arr.len() != y_arr.len() {
        return Ok(f64::NAN);
    }

    let sum_abs: f64 = x_arr.iter()
        .zip(y_arr.iter())
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .map(|(a, b)| (a - b).abs())
        .sum();

    let valid_count = x_arr.iter()
        .zip(y_arr.iter())
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .count();

    if valid_count == 0 {
        return Ok(f64::NAN);
    }

    Ok(sum_abs)
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(euclidean_distance, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_distance, m)?)?;
    m.add_function(wrap_pyfunction!(manhattan_distance, m)?)?;
    Ok(())
}
