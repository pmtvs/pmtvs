//! Dynamical systems analysis
//!
//! FTLE, Lyapunov exponents, etc.
//!
//! NOTE: These functions are NOT registered in lib.rs until parity
//! with Python implementations is validated. The code compiles but
//! is not exposed to Python.

mod lyapunov;
mod ftle;

// Not re-exported until validated
// pub use lyapunov::lyapunov_rosenstein_rs;
// pub use ftle::ftle_rs;
