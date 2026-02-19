"""
Rust vs Python parity benchmark.
Generates the full disclosure table for PRIMITIVES.md

Since dispatch is cached at import time, we use subprocess calls
to run Python-mode measurements in a clean process.
"""
import numpy as np
import subprocess
import sys
import json
import os
import textwrap

PYTHON = sys.executable

# ============================================================
# Test signal generators
# ============================================================

def generate_signals(N=5000, seed=42):
    rng = np.random.RandomState(seed)
    signals = {}

    signals["white_noise"] = rng.randn(N)

    t = np.linspace(0, 10 * np.pi, N)
    signals["sine"] = np.sin(t)

    signals["random_walk"] = np.cumsum(rng.randn(N))
    signals["trending"] = np.cumsum(rng.randn(N) + 0.01)
    signals["skewed"] = rng.exponential(2.0, N)
    signals["uniform"] = rng.uniform(-1, 1, N)
    signals["heavy_tail"] = rng.standard_t(3, N)

    impulse = rng.randn(N) * 0.1
    impulse[rng.randint(0, N, 20)] = 10.0
    signals["impulse"] = impulse

    signals["constant"] = np.ones(N) * 3.14
    signals["quadratic"] = np.linspace(0, 10, N) ** 2

    ar1 = np.zeros(N)
    ar1[0] = rng.randn()
    for i in range(1, N):
        ar1[i] = 0.9 * ar1[i - 1] + rng.randn()
    signals["ar1"] = ar1

    ar2 = np.zeros(N)
    ar2[0] = rng.randn()
    ar2[1] = rng.randn()
    for i in range(2, N):
        ar2[i] = 0.5 * ar2[i - 1] - 0.3 * ar2[i - 2] + rng.randn()
    signals["ar2"] = ar2

    # Lorenz attractor x-component
    from scipy.integrate import odeint
    def lorenz(state, t, sigma=10, rho=28, beta=8 / 3):
        x, y, z = state
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
    t_lorenz = np.linspace(0, 50, N)
    sol = odeint(lorenz, [1, 1, 1], t_lorenz)
    signals["lorenz"] = sol[:, 0]

    # Correlated pair
    x = rng.randn(N)
    y = 0.8 * x + 0.2 * rng.randn(N)
    signals["correlated_x"] = x
    signals["correlated_y"] = y

    return signals


# ============================================================
# Function definitions for testing
# ============================================================

FUNCTIONS = [
    # (name, module, call_template, tolerance, signals, return_type)
    # return_type: "scalar", "array", "tuple", "int"
    # call_template uses {sig} for signal, {sig2} for second signal

    # --- Entropy ---
    ("sample_entropy", "pmtvs_entropy", "{fn}({sig}, m=2, r=0.2)", "relaxed",
     ["white_noise", "sine", "lorenz"], "scalar"),
    ("permutation_entropy", "pmtvs_entropy", "{fn}({sig}, order=3, delay=1)", "relaxed",
     ["white_noise", "sine", "lorenz"], "scalar"),

    # --- Fractal ---
    ("hurst_exponent", "pmtvs_fractal", "{fn}({sig})", "relaxed",
     ["white_noise", "random_walk", "trending"], "scalar"),
    ("hurst_r2", "pmtvs_fractal", "{fn}({sig})", "normal",
     ["white_noise", "random_walk"], "scalar"),
    ("dfa", "pmtvs_fractal", "{fn}({sig})", "relaxed",
     ["white_noise", "random_walk"], "scalar"),

    # --- Statistics (scalars) ---
    ("mean", "pmtvs_statistics", "{fn}({sig})", "tight",
     ["white_noise", "sine", "constant"], "scalar"),
    ("std", "pmtvs_statistics", "{fn}({sig})", "tight",
     ["white_noise", "sine", "constant"], "scalar"),
    ("variance", "pmtvs_statistics", "{fn}({sig})", "tight",
     ["white_noise", "sine", "constant"], "scalar"),
    ("rms", "pmtvs_statistics", "{fn}({sig})", "tight",
     ["white_noise", "sine", "impulse"], "scalar"),
    ("peak_to_peak", "pmtvs_statistics", "{fn}({sig})", "tight",
     ["white_noise", "sine", "impulse"], "scalar"),
    ("skewness", "pmtvs_statistics", "{fn}({sig})", "tight",
     ["white_noise", "skewed", "uniform"], "scalar"),
    ("kurtosis", "pmtvs_statistics", "{fn}({sig})", "tight",
     ["white_noise", "uniform", "heavy_tail"], "scalar"),
    ("crest_factor", "pmtvs_statistics", "{fn}({sig})", "tight",
     ["white_noise", "sine", "impulse"], "scalar"),
    ("pulsation_index", "pmtvs_statistics", "{fn}(np.abs({sig}) + 1)", "tight",
     ["white_noise", "sine"], "scalar"),
    ("min_max", "pmtvs_statistics", "{fn}({sig})", "tight",
     ["white_noise", "sine", "impulse"], "tuple"),

    # --- Statistics (arrays) ---
    ("derivative", "pmtvs_statistics", "{fn}({sig})", "tight",
     ["sine", "white_noise", "quadratic"], "array"),
    ("integral", "pmtvs_statistics", "{fn}({sig})", "tight",
     ["sine", "white_noise", "constant"], "array"),
    ("curvature", "pmtvs_statistics", "{fn}({sig})", "tight",
     ["sine", "quadratic"], "array"),
    ("rate_of_change", "pmtvs_statistics", "{fn}({sig})", "tight",
     ["sine", "white_noise"], "array"),

    # --- Correlation ---
    ("autocorrelation", "pmtvs_correlation", "{fn}({sig}, lag=1)", "normal",
     ["white_noise", "sine", "ar1"], "scalar"),
    ("partial_autocorrelation", "pmtvs_correlation", "{fn}({sig}, max_lag=10)", "normal",
     ["white_noise", "ar1", "ar2"], "array"),
    ("correlation", "pmtvs_correlation", "{fn}({sig}, {sig2})", "tight",
     ["correlated"], "scalar"),
    ("covariance", "pmtvs_correlation", "{fn}({sig}, {sig2})", "tight",
     ["correlated"], "scalar"),

    # --- Distance ---
    ("euclidean_distance", "pmtvs_distance", "{fn}({sig}, {sig2})", "tight",
     ["correlated"], "scalar"),
    ("cosine_distance", "pmtvs_distance", "{fn}({sig}, {sig2})", "tight",
     ["correlated"], "scalar"),
    ("manhattan_distance", "pmtvs_distance", "{fn}({sig}, {sig2})", "tight",
     ["correlated"], "scalar"),
]

TOL_VALUES = {"exact": 0, "tight": 1e-12, "normal": 1e-8, "relaxed": 1e-4}


def build_worker_script(func_name, module, call_template, sig_name, return_type, use_rust):
    """Build a self-contained Python script that runs one test case."""
    rust_flag = "1" if use_rust else "0"
    is_pairwise = "{sig2}" in call_template

    if is_pairwise:
        sig_line1 = "signal = signals['correlated_x']"
        sig_line2 = "signal2 = signals['correlated_y']"
    else:
        sig_line1 = f"signal = signals['{sig_name}']"
        sig_line2 = ""

    call_str = call_template.replace("{fn}", func_name).replace("{sig2}", "signal2").replace("{sig}", "signal")

    lines = [
        f'import os',
        f'os.environ["PMTVS_USE_RUST"] = "{rust_flag}"',
        f'import numpy as np',
        f'import json',
        f'from scipy.integrate import odeint',
        f'',
        f'def generate_signals(N=5000, seed=42):',
        f'    rng = np.random.RandomState(seed)',
        f'    signals = {{}}',
        f'    signals["white_noise"] = rng.randn(N)',
        f'    t = np.linspace(0, 10 * np.pi, N)',
        f'    signals["sine"] = np.sin(t)',
        f'    signals["random_walk"] = np.cumsum(rng.randn(N))',
        f'    signals["trending"] = np.cumsum(rng.randn(N) + 0.01)',
        f'    signals["skewed"] = rng.exponential(2.0, N)',
        f'    signals["uniform"] = rng.uniform(-1, 1, N)',
        f'    signals["heavy_tail"] = rng.standard_t(3, N)',
        f'    impulse = rng.randn(N) * 0.1',
        f'    impulse[rng.randint(0, N, 20)] = 10.0',
        f'    signals["impulse"] = impulse',
        f'    signals["constant"] = np.ones(N) * 3.14',
        f'    signals["quadratic"] = np.linspace(0, 10, N) ** 2',
        f'    ar1 = np.zeros(N)',
        f'    ar1[0] = rng.randn()',
        f'    for i in range(1, N):',
        f'        ar1[i] = 0.9 * ar1[i-1] + rng.randn()',
        f'    signals["ar1"] = ar1',
        f'    ar2 = np.zeros(N)',
        f'    ar2[0] = rng.randn()',
        f'    ar2[1] = rng.randn()',
        f'    for i in range(2, N):',
        f'        ar2[i] = 0.5 * ar2[i-1] - 0.3 * ar2[i-2] + rng.randn()',
        f'    signals["ar2"] = ar2',
        f'    def lorenz(state, t, sigma=10, rho=28, beta=8/3):',
        f'        x, y, z = state',
        f'        return [sigma*(y-x), x*(rho-z)-y, x*y-beta*z]',
        f'    t_lorenz = np.linspace(0, 50, N)',
        f'    sol = odeint(lorenz, [1, 1, 1], t_lorenz)',
        f'    signals["lorenz"] = sol[:, 0]',
        f'    x = rng.randn(N)',
        f'    y = 0.8 * x + 0.2 * rng.randn(N)',
        f'    signals["correlated_x"] = x',
        f'    signals["correlated_y"] = y',
        f'    return signals',
        f'',
        f'signals = generate_signals()',
        f'from {module} import {func_name}',
        sig_line1,
        sig_line2,
        f'result = {call_str}',
        f'',
        f'if isinstance(result, np.ndarray):',
        f'    print(json.dumps({{"type": "array", "data": result.tolist()}}))',
        f'elif isinstance(result, tuple):',
        f'    print(json.dumps({{"type": "tuple", "data": [float(v) for v in result]}}))',
        f'elif isinstance(result, (int, np.integer)):',
        f'    print(json.dumps({{"type": "int", "data": int(result)}}))',
        f'else:',
        f'    print(json.dumps({{"type": "scalar", "data": float(result)}}))',
    ]
    script = "\n".join(lines)
    return script


def run_one(func_name, module, call_template, sig_name, return_type, use_rust):
    """Run one test case in a subprocess, return parsed result."""
    script = build_worker_script(func_name, module, call_template, sig_name, return_type, use_rust)
    proc = subprocess.run(
        [PYTHON, "-c", script],
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{'Rust' if use_rust else 'Python'} subprocess failed:\n{proc.stderr.strip()}")
    output = proc.stdout.strip().split("\n")[-1]
    return json.loads(output)


def compare(py_result, rust_result, tolerance_tier):
    """Compare Python and Rust results, return diff metrics."""
    tol_val = TOL_VALUES[tolerance_tier]

    if py_result["type"] == "array":
        py_arr = np.array(py_result["data"])
        rust_arr = np.array(rust_result["data"])
        if py_arr.shape != rust_arr.shape:
            return {
                "max_abs": float("inf"), "mean_abs": float("inf"),
                "max_rel": float("inf"), "passed": False,
                "py_repr": f"array({len(py_arr)})",
                "rust_repr": f"array({len(rust_arr)})",
            }
        # Both NaN at same position = agreement; filter those out
        both_nan = np.isnan(py_arr) & np.isnan(rust_arr)
        nan_mismatch = np.isnan(py_arr) != np.isnan(rust_arr)
        if np.any(nan_mismatch):
            return {
                "max_abs": float("inf"), "mean_abs": float("inf"),
                "max_rel": float("inf"), "passed": False,
                "py_repr": f"array({len(py_arr)})",
                "rust_repr": f"array({len(rust_arr)})",
            }
        mask = ~both_nan
        if not np.any(mask):
            # All NaN in both — perfect agreement
            return {
                "max_abs": 0.0, "mean_abs": 0.0, "max_rel": 0.0,
                "passed": True,
                "py_repr": f"array({len(py_arr)})",
                "rust_repr": f"array({len(rust_arr)})",
            }
        abs_diff = np.abs(py_arr[mask] - rust_arr[mask])
        safe_denom = np.abs(py_arr[mask]) + 1e-30
        return {
            "max_abs": float(np.max(abs_diff)),
            "mean_abs": float(np.mean(abs_diff)),
            "max_rel": float(np.max(abs_diff / safe_denom)),
            "passed": float(np.max(abs_diff)) <= tol_val,
            "py_repr": f"array({len(py_arr)})",
            "rust_repr": f"array({len(rust_arr)})",
        }
    elif py_result["type"] == "tuple":
        py_vals = py_result["data"]
        rust_vals = rust_result["data"]
        diffs = [abs(p - r) for p, r in zip(py_vals, rust_vals)]
        max_abs = max(diffs)
        mean_abs = sum(diffs) / len(diffs)
        max_rel = max(abs(d / (abs(p) + 1e-30)) for d, p in zip(diffs, py_vals))
        return {
            "max_abs": max_abs, "mean_abs": mean_abs, "max_rel": max_rel,
            "passed": max_abs <= tol_val,
            "py_repr": f"({', '.join(f'{v:.6f}' for v in py_vals)})",
            "rust_repr": f"({', '.join(f'{v:.6f}' for v in rust_vals)})",
        }
    elif py_result["type"] == "int":
        diff = abs(py_result["data"] - rust_result["data"])
        return {
            "max_abs": float(diff), "mean_abs": float(diff),
            "max_rel": float(diff), "passed": diff == 0,
            "py_repr": str(py_result["data"]),
            "rust_repr": str(rust_result["data"]),
        }
    else:
        py_val = py_result["data"]
        rust_val = rust_result["data"]
        abs_diff = abs(py_val - rust_val)
        rel_diff = abs_diff / (abs(py_val) + 1e-30)
        return {
            "max_abs": abs_diff, "mean_abs": abs_diff, "max_rel": rel_diff,
            "passed": abs_diff <= tol_val,
            "py_repr": f"{py_val:.10g}",
            "rust_repr": f"{rust_val:.10g}",
        }


def generate_markdown_table(all_results):
    lines = []
    lines.append("### Rust vs Python Parity — Full Disclosure")
    lines.append("")
    lines.append("Every Rust-accelerated function is tested against the Python reference")
    lines.append("implementation using identical inputs. The table below shows the maximum")
    lines.append("absolute difference observed across all test signals. A function ships")
    lines.append("Rust ONLY if parity is within tolerance AND speedup exceeds 1.0x.")
    lines.append("")
    lines.append("| Function | Signal | Python | Rust | Max |delta| | Tolerance | Status |")
    lines.append("|----------|--------|--------|------|------------|-----------|--------|")

    for r in all_results:
        status = "PASS" if r["passed"] else "FAIL"
        lines.append(
            f"| `{r['function']}` "
            f"| {r['signal']} "
            f"| {r['py_repr']} "
            f"| {r['rust_repr']} "
            f"| {r['max_abs']:.2e} "
            f"| {r['tolerance']} ({r['tol_val']:.0e}) "
            f"| {status} |"
        )

    lines.append("")
    lines.append("**Test conditions:** N=5,000 samples, seed=42, all platforms.")
    lines.append("**Tolerance tiers:** Exact (0), Tight (1e-12), Normal (1e-8), Relaxed (1e-4)")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 60)
    print("PMTVS Rust vs Python Parity Benchmark")
    print("=" * 60)

    all_results = []

    for func_name, module, call_template, tol_tier, sig_names, ret_type in FUNCTIONS:
        for sig_name in sig_names:
            display_sig = sig_name if sig_name != "correlated" else "correlated_pair"
            label = f"{func_name} / {display_sig}"
            print(f"  {label} ...", end=" ", flush=True)

            try:
                py = run_one(func_name, module, call_template, sig_name, ret_type, use_rust=False)
                rust = run_one(func_name, module, call_template, sig_name, ret_type, use_rust=True)
                cmp = compare(py, rust, tol_tier)

                status = "PASS" if cmp["passed"] else "FAIL"
                print(f"{status}  (max |delta| = {cmp['max_abs']:.2e})")

                all_results.append({
                    "function": func_name,
                    "signal": display_sig,
                    "py_repr": cmp["py_repr"],
                    "rust_repr": cmp["rust_repr"],
                    "max_abs": cmp["max_abs"],
                    "mean_abs": cmp["mean_abs"],
                    "max_rel": cmp["max_rel"],
                    "tolerance": tol_tier,
                    "tol_val": TOL_VALUES[tol_tier],
                    "passed": cmp["passed"],
                })
            except Exception as e:
                print(f"ERROR: {e}")
                all_results.append({
                    "function": func_name,
                    "signal": display_sig,
                    "py_repr": "ERROR",
                    "rust_repr": "ERROR",
                    "max_abs": float("nan"),
                    "mean_abs": float("nan"),
                    "max_rel": float("nan"),
                    "tolerance": tol_tier,
                    "tol_val": TOL_VALUES[tol_tier],
                    "passed": False,
                })

    # Summary
    passed = sum(1 for r in all_results if r["passed"])
    total = len(all_results)
    print()
    print("=" * 60)
    print(f"PARITY SUMMARY: {passed}/{total} passed")
    print("=" * 60)

    for r in all_results:
        if not r["passed"]:
            print(f"  FAIL: {r['function']} / {r['signal']} — max |delta| = {r['max_abs']:.2e}")

    # Write markdown
    md = generate_markdown_table(all_results)
    report_path = os.path.join(os.path.dirname(__file__), "PARITY_REPORT.md")
    with open(report_path, "w") as f:
        f.write(md)
    print(f"\nReport written to {report_path}")

    # Also write raw JSON for later use
    json_path = os.path.join(os.path.dirname(__file__), "parity_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Raw data written to {json_path}")
