# Part-Failure Classification - Example Project

Template project for the MLOps framework. In production, this would live in a **separate repository** from the framework. Install `mlops-framework` and run from this project root.

## Project Structure

```
part_failure_model/
├── pipeline/
│   ├── pipeline.yaml   # DAG definition (steps, dependencies)
│   └── config.yaml     # Per-step config + env overrides (dev, qa, prod)
├── custom/             # Data scientist business logic
│   ├── __init__.py
│   ├── feature_engineering.py   # build_features()
│   ├── evaluation.py            # compute_metrics()
│   └── data_loader.py           # load_raw_data(), synthetic data
├── run.py              # Simple entry point for local dev
├── requirements.txt    # mlops-framework, pandas, scikit-learn, ...
├── model.py            # PartFailureModel (Random Forest classifier)
├── steps/
│   ├── preprocess.py   # PartFailurePreprocess
│   ├── train.py        # PartFailureTrain
│   ├── inference.py    # PartFailureInference
│   ├── data_drift.py   # PartFailureDataDrift
│   └── model_monitor.py # PartFailureModelMonitor
├── tests/              # Unit and integration tests
└── data/               # (optional) data files
```

Steps import and call custom module functions:
```python
from custom.feature_engineering import build_features
from custom.evaluation import compute_metrics
from custom.data_loader import load_raw_data
```

## Setup

From project root:

```bash
pip install -r requirements.txt
# Or: pip install mlops-framework pandas scikit-learn pydantic
```

## Commands

| Command | Description |
|---------|-------------|
| `mlops run <step> [--env dev\|qa\|prod] [--tracking] [--tracking-backend local\|vertex]` | Run a pipeline step |
| `mlops compile pipeline/pipeline.yaml -o dags/part_failure_dag.py` | Generate Airflow DAG |
| `python run.py <step> [--env dev\|qa\|prod]` | Project entry point |
| `python -m steps.<step>` | Run step directly (e.g. `steps.preprocess`) |

## Running Locally

```bash
cd part_failure_model   # or your project root
mlops run preprocess
mlops run train
mlops run inference
```

With experiment tracking (persist to `./runs`):

```bash
mlops run train --tracking
```

With Vertex AI tracking (requires GCP creds):

```bash
mlops run train --tracking --tracking-backend vertex
```

## Direct Debugging

```bash
python -m steps.preprocess
python -m steps.train
python -m steps.inference
```

## Testing

```bash
cd part_failure_model
pip install -r requirements.txt
pytest tests/ -v
```

Tests cover:
- **Unit**: `custom.drift.compute_drift`, `custom.monitoring.compute_model_health`
- **Integration**: `PartFailureDataDrift` and `PartFailureModelMonitor` steps

## Configuration

Edit `pipeline/config.yaml`. Optional top-level `tracking_backend: local` or `vertex`.

## Output

- **Artifacts**: `./artifacts/*.pkl` (train_data, model, predictions)
- **Tracking**: `./runs/{run_id}/` (params, metrics) when `--tracking` or `tracking_backend: local`

## Custom Modules

Data scientists extend `custom/` with business logic. Steps stay thin and delegate:

- `custom/feature_engineering.py`: `build_features(df)` — feature transforms
- `custom/evaluation.py`: `compute_metrics(y_true, y_pred, y_proba)` — metrics
- `custom/data_loader.py`: `load_raw_data(path)`, `create_synthetic_*` — data loading

## Cloud

Same step code runs in the cloud. Backend (GCS, Vertex) is switched via RunContext. Generate Airflow DAG:

```bash
mlops compile pipeline/pipeline.yaml -o dags/part_failure_dag.py
```
