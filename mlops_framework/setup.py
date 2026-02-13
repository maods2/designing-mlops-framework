"""Setup script for MLOps Framework.

The package directory (mlops_framework/) is the framework repo root.
Contains setup.py and mlops_framework/ (the Python package).
"""

from setuptools import setup, find_packages
from pathlib import Path

_readme = Path(__file__).resolve().parent / "README.md"
_long_desc = _readme.read_text(encoding="utf-8") if _readme.exists() else "MLOps Framework"

setup(
    name="mlops-framework",
    version="0.1.0",
    author="MLOps Framework Team",
    description="A minimal but extensible MLOps framework for ML model development",
    long_description=_long_desc,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/mlops-framework",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=["PyYAML>=5.4.1", "pydantic>=2.0"],
    extras_require={
        "mlflow": ["mlflow>=1.20.0"],
        "gcs": ["google-cloud-storage>=2.0.0"],
        "vertex": ["google-cloud-aiplatform>=1.38.0"],
    },
    entry_points={
        "console_scripts": ["mlops=mlops_framework.cli.main:main"],
    },
)
