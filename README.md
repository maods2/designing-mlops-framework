# mlplatform

MLOps framework for training and batch prediction pipelines. Supports local execution and cloud deployment (Dataproc/VertexAI via PySpark).

## First-Time Environment Setup

### 1. Python

Requires **Python 3.8+**. Verify your version:

```bash
python3 --version
```

### 2. Install framework dependencies

From the repository root:

```bash
pip install -r mlplatform/requirements.txt
```

This installs the core framework dependencies (pyyaml, pandas, scikit-learn, joblib, etc.).

### 3. Install model dependencies

Each model has its own `requirements.txt`. For the included example model:

```bash
pip install -r example_model/requirements.txt
```

### 4. (Optional) PySpark for distributed batch prediction

Only needed if you want to run PySpark-based batch prediction locally:

```bash
pip install pyspark pyarrow
```

### 5. Verify the setup

Run the example model training to confirm everything works:

```bash
python3 example_model/train.py
```

You should see log output with an accuracy metric. Artifacts are written to `./artifacts/`.

### How imports work

Model files (`train.py`, `predict.py`) include a small `sys.path` bootstrap at the top that adds both the repo root and the `mlplatform/` directory to the Python path. This means you can run any model file directly -- no `pip install -e`, no `PYTHONPATH` export, no virtual environment tricks required.

If you prefer the traditional approach, you can still set `PYTHONPATH` explicitly:

```bash
export PYTHONPATH="$(pwd):$(pwd)/mlplatform"
```

## Project Structure

```
my_project/
  template_training_dag.yaml      # Training DAG config
  template_prediction_dag.yaml    # Prediction DAG config
  my_model/                       # Your model package
    __init__.py
    constants.py
    requirements.txt              # Model-specific Python dependencies
    train.py                      # Implements BaseTrainer
    predict.py                    # Implements BasePredictor
    data/                         # Local sample data for development
  mlplatform/                     # Framework package
    mlplatform/                   # Python package source
    requirements.txt              # Framework dependencies
    pyproject.toml
```

## DAG Configuration

All pipelines are driven by a single YAML file. There are two types: training and prediction.

### Training DAG

```yaml
workflow_name: my_workflow
execution_mode: sequential        # sequential or parallel
pipeline_type: training
feature_name: my_feature          # Groups artifacts under this feature
config_version: 2
log_level: INFO                   # Optional: DEBUG, INFO, WARNING, ERROR
models:
  - model_name: my_model_v1      # Unique model identifier
    compute: s                    # xs, s, m, l (cloud compute size)
    training_platform: VertexAI   # VertexAI or Dataproc
    module: "my_model.train"      # Python module containing your BaseTrainer subclass
    optional_configs:
      test_size: 0.2
      hyperparameters:
        max_iter: 1000
```

### Prediction DAG

```yaml
workflow_name: my_workflow_prediction
execution_mode: sequential
pipeline_type: prediction
feature_name: my_feature
config_version: 2
models:
  - model_name: my_model_v1
    serving_platform: VertexAI
    module: "my_model.predict"    # Python module containing your BasePredictor subclass
    model_version: "latest"
    optional_configs:
      prediction_threshold: 0.5
```

### Key Fields

| Field | Description |
|---|---|
| `pipeline_type` | `training` or `prediction` -- determines which flow runs |
| `feature_name` | Top-level grouping for artifacts (e.g., team or feature area) |
| `model_name` | Unique identifier; artifacts are stored under `{feature_name}/{model_name}/{version}/` |
| `module` | Dotted Python import path to the module containing your trainer or predictor class |
| `optional_configs` | Free-form dict passed to your code via `ctx.optional_configs` |
| `log_level` | Optional (default `INFO`). Controls framework log verbosity |

## Writing Model Code

### Trainer (training)

Create a file that extends `BaseTrainer` and implements `train()`. Include the path bootstrap at the top and a `__main__` block at the bottom for direct execution:

```python
"""Training: MyTrainer."""

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
for _p in [str(_repo_root), str(_repo_root / "mlplatform")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mlplatform.core.trainer import BaseTrainer

class MyTrainer(BaseTrainer):
    def _load_data(self) -> tuple:
        """DS is responsible for data loading (CSV, Parquet, BigQuery, GCS, etc.)."""
        data = load_my_data()
        return data.drop(columns=["target"]), data["target"]

    def train(self) -> None:
        ctx = self.context
        X, y = self._load_data()

        model = SomeModel()
        model.fit(X, y)

        ctx.save_artifact("model.pkl", model)
        ctx.log_metrics({"accuracy": 0.95})
        ctx.log_params({"model_type": "SomeModel"})
        ctx.register_model({"accuracy": 0.95})
        ctx.log.info("Training complete")

if __name__ == "__main__":
    from mlplatform.runner import dev_context
    ctx = dev_context("template_training_dag.yaml")
    trainer = MyTrainer()
    trainer.context = ctx
    trainer.train()
```

### Predictor (prediction)

Create a file that extends `BasePredictor` and implements `load_model()` and `predict_chunk()`:

```python
"""Prediction: MyPredictor."""

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
for _p in [str(_repo_root), str(_repo_root / "mlplatform")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
from mlplatform.core.predictor import BasePredictor

class MyPredictor(BasePredictor):
    def load_model(self):
        self._model = self.context.load_artifact("model.pkl")

    def predict_chunk(self, data) -> pd.DataFrame:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        predictions = self._model.predict(df)
        return df.assign(prediction=predictions)

if __name__ == "__main__":
    from mlplatform.runner import dev_context
    ctx = dev_context("template_prediction_dag.yaml")
    predictor = MyPredictor()
    predictor.context = ctx
    predictor.load_model()
    result = predictor.predict_chunk(load_input_data())
    print(result)
```

### Creating a new model

To create a new model package, replicate the `example_model/` structure:

```
my_model/
  __init__.py
  constants.py              # Artifact names, feature columns
  requirements.txt          # Model-specific dependencies (pandas, scikit-learn, etc.)
  train.py                  # Subclass of BaseTrainer
  predict.py                # Subclass of BasePredictor
  data/                     # Local sample data for development
    sample_train.csv
    sample_inference.csv
```

Create a DAG YAML pointing to your module (`module: "my_model.train"`) and install your model dependencies:

```bash
pip install -r my_model/requirements.txt
```

### ExecutionContext API

Your code receives `self.context` (an `ExecutionContext`) with these fields and helpers:

| Field / Method | Description |
|---|---|
| `ctx.feature_name` | From the DAG config |
| `ctx.model_name` | From the DAG config |
| `ctx.version` | Auto-generated or passed via CLI |
| `ctx.optional_configs` | The `optional_configs` dict from the DAG |
| `ctx.log` | Python `logging.Logger` instance |
| `ctx.save_artifact(name, obj)` | Save an artifact (path handled by framework) |
| `ctx.load_artifact(name)` | Load an artifact for the current model/version |
| `ctx.load_artifact_from(model, version, name)` | Load from a different model or version |
| `ctx.log_metrics(dict)` | Log metrics to the experiment tracker |
| `ctx.log_params(dict)` | Log parameters to the experiment tracker |
| `ctx.register_model(metadata)` | Register the model version in the artifact store |

Artifacts are stored at `{base_path}/{feature_name}/{model_name}/{version}/{artifact_name}`. You never construct this path yourself.

## Running Locally

### Via CLI

```bash
# Training
mlplatform run --dag template_training_dag.yaml

# Training with explicit version and artifact path
mlplatform run --dag template_training_dag.yaml --version v1.0 --base-path ./my_artifacts

# Prediction
mlplatform run --dag template_prediction_dag.yaml --version v1.0 --base-path ./my_artifacts
```

### Via Python

```python
from mlplatform.runner import run_workflow

# Training
results = run_workflow("template_training_dag.yaml", version="v1.0", base_path="./artifacts")

# Prediction (use the same version and base_path as training)
results = run_workflow("template_prediction_dag.yaml", version="v1.0", base_path="./artifacts")
```

### Direct execution (debug mode)

You can run any trainer or predictor file directly with `python` for debugging. Every model file needs two things:

**1. Path bootstrap** at the top of the file (before any `mlplatform` imports):

```python
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
for _p in [str(_repo_root), str(_repo_root / "mlplatform")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
```

**2. `__main__` block** at the bottom that uses `dev_context()`:

```python
if __name__ == "__main__":
    from mlplatform.runner import dev_context

    ctx = dev_context("template_training_dag.yaml")
    trainer = MyTrainer()
    trainer.context = ctx
    trainer.train()
```

Then run or debug directly from the repo root:

```bash
python example_model/train.py
python example_model/predict.py
```

`dev_context()` reads your DAG YAML, builds a local `ExecutionContext` with version `"dev"`, and points artifacts at `./artifacts`. You get the full framework (logging, `save_artifact`, `load_artifact`, metrics tracking) without needing the CLI or an external orchestrator. Set breakpoints anywhere in your `train()` or `predict_chunk()` method and debug normally.

Parameters: `dev_context(dag_path, model_index=0, profile="local", version="dev", base_path=None)`

### Prediction must use the same version and base_path as training

The predictor loads artifacts from `{base_path}/{feature_name}/{model_name}/{version}/`. If you trained with `--version v1.0 --base-path ./artifacts`, you must predict with the same values so `load_artifact` finds the saved model.

## PySpark Batch Prediction

For distributed prediction using Spark's `mapInPandas`, the framework provides `spark/main.py` as the unified entry point for both local and cloud execution.

### Step 1: Train a model first

```bash
mlplatform run --dag template_training_dag.yaml --version v1.0 --base-path ./artifacts
```

### Step 2: Generate a config JSON for Spark

```python
from mlplatform.config.loader import load_workflow_config
from mlplatform.spark.config_serializer import write_workflow_config

workflow = load_workflow_config("template_prediction_dag.yaml")
model_cfg = workflow.models[0]
write_workflow_config(workflow, model_cfg, "dist/spark_config.json",
                      base_path="./artifacts", version="v1.0")
```

### Step 3: Run locally

```bash
python mlplatform/mlplatform/spark/main.py \
  --config dist/spark_config.json \
  --input-path data/input.csv \
  --output-path dist/predictions.parquet
```

### Step 4 (cloud): Build root.zip and submit

```bash
# Build the deployment package
mlplatform build-package --model-package my_model --project-root .

# Submit to Dataproc / VertexAI
spark-submit \
  mlplatform/mlplatform/spark/main.py \
  --py-files dist/root.zip \
  -- \
  --config gs://bucket/spark_config.json \
  --input-path gs://bucket/input.parquet \
  --output-path gs://bucket/predictions.parquet
```

The `root.zip` contains your model package and the `mlplatform` framework so Spark workers can import both.

## Cloud Deployment (Dataproc / VertexAI)

For cloud execution the pattern is:

1. **Package**: `mlplatform build-package` creates `dist/root.zip` containing your model code + the framework
2. **Serialize config**: `write_workflow_config()` produces a JSON that `spark/main.py` reads
3. **Upload**: Put `main.py`, `root.zip`, and the config JSON to GCS
4. **Submit**: Use `gcloud dataproc jobs submit pyspark` or VertexAI custom training with `main.py` as the entry point and `root.zip` as `--py-files`

The same `spark/main.py` handles both training (runs on driver) and prediction (distributes via `mapInPandas`).

## Artifact Layout

After training, artifacts are stored as:

```
artifacts/
  {feature_name}/
    {model_name}/
      {version}/
        model.pkl
        scaler.pkl
        ...
  model_registry.json       # Tracks all registered model versions
  metrics.json               # Experiment tracking data
```

## Running Tests

Quick smoke test (trains + predicts using the example model):

```bash
python3 example_model/train.py
python3 example_model/predict.py
```

Full framework test suite:

```bash
PYTHONPATH="$(pwd):$(pwd)/mlplatform" python3 scripts/test_example_model.py
```

This runs:
- In-process training and prediction
- `root.zip` packaging
- Config serialization
- PySpark local batch prediction via `mapInPandas`

## Extending the Framework

- **New storage backends** (e.g., GCS): implement `Storage` ABC in `mlplatform/storage/`
- **New artifact stores** (e.g., MLflow): implement `ArtifactStore` ABC in `mlplatform/artifacts/`
- **New experiment trackers** (e.g., Weights & Biases): implement `ExperimentTracker` ABC in `mlplatform/tracking/`
- **Shared utilities**: add reusable functions to `mlplatform/utils/` for cross-model use
- **New cloud profiles**: extend the `_create_infra()` factory in `mlplatform/runner.py`
