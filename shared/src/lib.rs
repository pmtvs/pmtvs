/// Shared utilities for pmtvs Rust crates.
/// NaN handling, array validation, common patterns.

/// Clean a signal by removing NaN and infinite values.
pub fn clean_signal(data: &[f64]) -> Vec<f64> {
    data.iter().filter(|x| x.is_finite()).copied().collect()
}

/// Compute mean of a slice.
pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Compute standard deviation of a slice (sample std, ddof=1).
pub fn std(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return f64::NAN;
    }
    let m = mean(data);
    let variance = data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
    variance.sqrt()
}
