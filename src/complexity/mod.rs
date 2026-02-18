//! Complexity measures: entropy and fractal analysis
//!
//! These are the "crown jewels" - the functions where Rust provides
//! the most significant speedup over pure Python/NumPy.

mod entropy;
mod hurst;

pub use entropy::{sample_entropy_rs, permutation_entropy_rs, approximate_entropy_rs};
pub use hurst::{hurst_exponent_rs, dfa_rs};
