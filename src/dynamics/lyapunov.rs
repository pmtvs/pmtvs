//! Lyapunov exponent estimation
//!
//! NOT REGISTERED - awaiting parity validation

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// Rosenstein algorithm for largest Lyapunov exponent
///
/// This function exists but is NOT exposed to Python until
/// numerical parity with the Python implementation is validated.
#[pyfunction]
#[allow(dead_code)]
pub fn lyapunov_rosenstein_rs(
    _py: Python,
    _data: PyReadonlyArray1<f64>,
    _dim: usize,
    _delay: usize,
    _min_tsep: usize,
) -> PyResult<f64> {
    // Placeholder - implement when ready for validation
    Ok(f64::NAN)
}
