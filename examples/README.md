# mlplatform examples

Runnable examples for the released sub-packages:
`mlplatform[utils]`, `mlplatform[config]`, `mlplatform[artifacts]`, and workflow APIs.

## 1. Install

### From PyPI / JFrog (once published)

```bash
pip install mlplatform[utils]    # serialisation + GCS upload helpers
pip install mlplatform[config]   # Pydantic pipeline config models
pip install mlplatform[core]    # both of the above in one command
```

### From this repository (local development)

```bash
# Editable install
pip install -e "mlplatform/[core]"

# If editable install fails, use:
pip install -e "mlplatform/[core]" --no-build-isolation
```

### For train/prediction examples (06, 07)

```bash
pip install -r example_model/requirements.txt   # sklearn
```

---

## 2. Examples

| File | Sub-package | What it shows |
|---|---|---|
| `01_utils_serialization.py` | `mlplatform[utils]` | `sanitize` and `to_serializable` |
| `02_utils_storage.py` | `mlplatform[utils]` | `save_plot` and `save_html` with `LocalFileSystem` |
| `03_config_direct.py` | `mlplatform[config]` | Build `TrainingConfig` / `PredictionConfig` directly in Python |
| `04_config_from_yaml.py` | `mlplatform[config]` | Load a `PipelineConfig` from a YAML pipeline file |
| `05_artifacts_standalone.py` | `mlplatform[core]` | `create_artifacts` standalone and config-driven modes |
| `06_train_standalone.py` | `mlplatform[core]` | `dev_train`, `dev_context`, `BaseTrainer` — one-liner and manual training |
| `07_prediction_standalone.py` | `mlplatform[core]` | `dev_predict` — prediction with DataFrame or from config input_path |
| `08_experiment_tracking.py` | `mlplatform[core]` | `LocalJsonTracker`, `ExecutionContext` — experiment tracking |

Run any example from the **repository root**:

```bash
# Serialisation helpers
python examples/01_utils_serialization.py

# Storage helpers (writes to examples/output/ — safe to delete)
python examples/02_utils_storage.py

# Config: direct construction
python examples/03_config_direct.py

# Config: load from YAML
python examples/04_config_from_yaml.py

# Artifacts: create_artifacts standalone and config-driven
python examples/05_artifacts_standalone.py

# Training: dev_train and dev_context
python examples/06_train_standalone.py

# Prediction: dev_predict (run 06 first to train a model)
python examples/07_prediction_standalone.py

# Experiment tracking: LocalJsonTracker, ExecutionContext
python examples/08_experiment_tracking.py
```

All examples print their output to stdout. Output artifacts are written to
`examples/output/` and can be deleted freely.
