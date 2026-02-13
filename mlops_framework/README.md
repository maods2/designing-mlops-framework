# MLOps Framework

A minimal but extensible MLOps framework for ML model development. Step-based pipelines, pluggable backends (local/GCS/Vertex), same code runs locally or in cloud.

**Framework and projects are independent repos.** This directory is the framework repo root (package directory). It contains `setup.py` and `mlops_framework/` (the Python package).

## Installation

From this directory (mlops_framework/):

```bash
cd mlops_framework
pip install -e .
```

## Project Structure

Projects (separate repos) depend on `mlops-framework` and run from their own root:

```
your-project/
├── config.yaml
├── pipeline.yaml
├── steps/
│   ├── preprocess.py
│   ├── train.py
│   └── inference.py
└── model.py
```

## Usage

From a project repo:

```bash
mlops run preprocess
mlops run train [--tracking]   # --tracking persists metrics to ./runs
mlops run inference
mlops compile pipeline/pipeline.yaml -o dags/xxx.py
```

**Commands**: `mlops run <step> [--env dev|qa|prod] [--tracking]` | `mlops compile pipeline/pipeline.yaml -o dags/xxx.py`

See main README.md for full documentation.
