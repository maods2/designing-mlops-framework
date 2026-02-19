# Virtual Environment Setup and Testing

## Monorepo Structure

- **Root**: mlplatform framework, template_model (and other models)
- **template_model**: Model project with config, pipeline, artifacts, dist
- **base_path**: Injected by orchestrator (bucket or root folder) - NOT in config

## Create and Activate Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -e mlplatform
```

## Run Tests with template_model

```bash
export PYTHONPATH="$(pwd)"
python scripts/test_framework.py
# or: ./scripts/run_test.sh
```

## Run CLI (project_root defaults to template_model)

```bash
export PYTHONPATH="$(pwd)"

# Run pipeline - artifacts go to template_model/artifacts/
# Paths relative to project-root (default: template_model)
mlplatform run --dag train_infer --env dev \
  --train-data data/sample_train.csv \
  --inference-data data/sample_inference.csv
```

## Spark/Dataproc Execution (Local Testing)

All outputs under template_model/:

```bash
# 1. Build root.zip -> template_model/dist/
mlplatform build-package --project-root .

# 2. Run step (run_config.json and root.zip in template_model/dist/)
mlplatform run-spark-main --config template_model/dist/run_config.json \
  --packages template_model/dist/root.zip --step-name train \
  --input-path template_model/data/sample_train.csv
```

**Dataproc format** (base_path injected by orchestrator):
```
spark-submit main.py --py-files root.zip -- --config gs://bucket/config.json
```

**Environment configs** are defined per step in `pipeline/steps/*.yaml` under `envs:` (dev, local_spark, prod, etc.). base_path is injected by orchestrator.

## Run from Python/Notebook

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))  # project root

from mlplatform.local import load_pipeline_config, run_pipeline_local
import pandas as pd
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100, n_features=5, random_state=42)
config = load_pipeline_config(
    dag_path="template_model/pipeline/dags/train_infer.yaml",
    steps_dir="template_model/pipeline/steps",
    env="dev",
)
results = run_pipeline_local(
    config,
    project_root="template_model",
    base_path="template_model/artifacts",
    train={"train_data": {"X": pd.DataFrame(X), "y": pd.Series(y)}},
    inference={"inference_data": pd.DataFrame(X[:10])},
)
```
