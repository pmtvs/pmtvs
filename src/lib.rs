use pyo3::prelude::*;

// Module declarations — uncomment as implementations land
// mod complexity;
// mod individual;
// mod embedding;

/// Rust-accelerated signal analysis primitives.
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Functions registered here as Rust implementations land.
    // See python/pmtvs/_dispatch.py for which functions are validated.
    Ok(())
}
