use numpy::{PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;

/// Construct time-delay embedding matrix.
#[pyfunction]
fn delay_embedding<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<f64>,
    dim: usize,
    tau: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let signal: Vec<f64> = signal.as_array().iter().filter(|x| x.is_finite()).copied().collect();
    let n = signal.len();

    if dim < 1 || tau < 1 {
        return Ok(PyArray2::from_vec2_bound(py, &[vec![f64::NAN]]).unwrap());
    }

    let n_vectors = n.saturating_sub((dim - 1) * tau);
    if n_vectors < 1 {
        return Ok(PyArray2::from_vec2_bound(py, &[vec![f64::NAN]]).unwrap());
    }

    // Build embedding matrix
    let mut embedding = Vec::with_capacity(n_vectors);
    for i in 0..n_vectors {
        let mut row = Vec::with_capacity(dim);
        for j in 0..dim {
            row.push(signal[i + j * tau]);
        }
        embedding.push(row);
    }

    Ok(PyArray2::from_vec2_bound(py, &embedding).unwrap())
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(delay_embedding, m)?)?;
    Ok(())
}
