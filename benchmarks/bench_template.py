"""
Benchmark template for pmtvs functions

Usage:
    python benchmarks/bench_template.py

This template shows how to benchmark Rust vs Python implementations.
Actual benchmarks will be added as functions are ported.
"""

import time
import numpy as np
from typing import Callable, Tuple


def benchmark(
    func: Callable,
    args: tuple,
    n_runs: int = 100,
    warmup: int = 5,
) -> Tuple[float, float]:
    """
    Benchmark a function.

    Returns:
        (mean_time_ms, std_time_ms)
    """
    # Warmup
    for _ in range(warmup):
        func(*args)

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    return np.mean(times), np.std(times)


def parity_check(
    python_func: Callable,
    rust_func: Callable,
    args: tuple,
    rtol: float = 1e-10,
    atol: float = 1e-10,
) -> bool:
    """
    Check numerical parity between Python and Rust implementations.

    Returns:
        True if results match within tolerance
    """
    py_result = python_func(*args)
    rs_result = rust_func(*args)

    if isinstance(py_result, np.ndarray):
        return np.allclose(py_result, rs_result, rtol=rtol, atol=atol)
    elif isinstance(py_result, float):
        if np.isnan(py_result) and np.isnan(rs_result):
            return True
        return np.isclose(py_result, rs_result, rtol=rtol, atol=atol)
    else:
        return py_result == rs_result


def format_result(name: str, py_time: float, rs_time: float, parity: bool) -> str:
    """Format benchmark result for display"""
    speedup = py_time / rs_time if rs_time > 0 else float('inf')
    parity_str = "OK" if parity else "FAIL"
    return f"{name:30} Python: {py_time:8.3f}ms  Rust: {rs_time:8.3f}ms  Speedup: {speedup:6.1f}x  Parity: {parity_str}"


if __name__ == "__main__":
    print("pmtvs Benchmark Template")
    print("=" * 70)
    print("\nActual benchmarks will be added as functions are ported.")
    print("\nExample usage:")
    print("""
    from pmtvs._dispatch import rust_only
    from pmtvs.individual.entropy import sample_entropy as py_sample_entropy

    # Get Rust implementation
    rs_sample_entropy = rust_only("sample_entropy")

    # Generate test data
    data = np.random.randn(1000)
    args = (data, 2, 0.2)

    # Benchmark
    py_mean, py_std = benchmark(py_sample_entropy, args)
    rs_mean, rs_std = benchmark(rs_sample_entropy, args)

    # Check parity
    parity = parity_check(py_sample_entropy, rs_sample_entropy, args)

    print(format_result("sample_entropy", py_mean, rs_mean, parity))
    """)
