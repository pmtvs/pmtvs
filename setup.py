"""
Fallback setup.py for Python-only installation.

For Rust-accelerated builds, use: maturin develop --release
For Python-only builds, use: pip install -e . --no-build-isolation
"""

from setuptools import setup, find_packages

setup(
    name="pmtvs",
    version="0.1.0",
    description="High-performance primitives for time series and dynamical systems analysis",
    author="Avery Rudder",
    author_email="avery@rudder.io",
    license="MIT",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-benchmark>=4.0",
        ],
        "full": [
            "scikit-learn>=1.2",
            "statsmodels>=0.14",
            "giotto-tda>=0.6",
            "networkx>=3.0",
        ],
    },
)
