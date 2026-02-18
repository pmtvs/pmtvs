//! pmtvs Rust accelerators
//!
//! This module provides high-performance Rust implementations of
//! computationally intensive primitives. Functions are registered
//! here and dispatched from Python via _dispatch.py.

use pyo3::prelude::*;

mod complexity;
mod embedding;
// mod dynamics;  // Not registered until parity validated

/// The Rust extension module for pmtvs
#[pymodule]
fn _rust(_py: Python, m: &PyModule) -> PyResult<()> {
    // Complexity functions (entropy, hurst)
    m.add_function(wrap_pyfunction!(complexity::sample_entropy_rs, m)?)?;
    m.add_function(wrap_pyfunction!(complexity::permutation_entropy_rs, m)?)?;
    m.add_function(wrap_pyfunction!(complexity::approximate_entropy_rs, m)?)?;
    m.add_function(wrap_pyfunction!(complexity::hurst_exponent_rs, m)?)?;
    m.add_function(wrap_pyfunction!(complexity::dfa_rs, m)?)?;

    // Embedding functions
    m.add_function(wrap_pyfunction!(embedding::time_delay_embedding_rs, m)?)?;
    m.add_function(wrap_pyfunction!(embedding::optimal_delay_rs, m)?)?;

    // Statistics functions (to be added in PR 1)
    // m.add_function(wrap_pyfunction!(stats::skewness_rs, m)?)?;
    // m.add_function(wrap_pyfunction!(stats::kurtosis_rs, m)?)?;
    // etc.

    Ok(())
}
