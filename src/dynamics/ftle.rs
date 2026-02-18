//! Finite-Time Lyapunov Exponent
//!
//! NOT REGISTERED - awaiting parity validation

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// FTLE computation
///
/// This function exists but is NOT exposed to Python until
/// numerical parity with the Python implementation is validated.
#[pyfunction]
#[allow(dead_code)]
pub fn ftle_rs(
    _py: Python,
    _trajectory: PyReadonlyArray1<f64>,
    _dt: f64,
) -> PyResult<f64> {
    // Placeholder - implement when ready for validation
    Ok(f64::NAN)
}
