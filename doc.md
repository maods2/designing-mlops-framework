# mlplatform — Main Functionalities and How It Works

This document explains the main functionalities of the mlplatform package and how they work together.

---

## Overview

**mlplatform** is a production-ready MLOps platform with pluggable backends. The current public API (v0.1.x) provides:

1. **Config** — Pydantic-validated configuration models for training and prediction
2. **Artifacts** — Save and load ML artifacts (models, metrics, plots) with a consistent path convention
3. **Storage** — Backends for local filesystem and Google Cloud Storage (GCS)
4. **Utils** — Serialization helpers, HTML reports, and storage utilities

---

## 1. Artifacts — Save and Load ML Artifacts

### What It Does

The artifact system lets you save and load model artifacts (models, metrics, plots, HTML reports) with a consistent path structure and automatic format handling.

### Path Convention

All artifacts are stored under:

```
{feature}/{model_name}/{version}/{artifact_name}
```

Example: `churn/churn_model/v1/model.pkl`, `churn/churn_model/v1/metrics.json`

### How to Use

**Option A: `Artifact()` — convenience constructor**

```python
from mlplatform import Artifact

artifact = Artifact(
    model_name="churn_model",
    feature="churn",
    version="v1",
    base_path="./artifacts",
    backend="local",
)

artifact.save("model.pkl", trained_model)
artifact.save("metrics.json", {"accuracy": 0.95})
artifact.save("report/loss.png", fig)  # matplotlib Figure → PNG

loaded = artifact.load("model.pkl")
metrics = artifact.load("metrics.json")  # returns dict
```

**Option B: `create_artifacts()` — explicit params**

```python
from mlplatform.artifacts import create_artifacts

artifacts = create_artifacts(
    backend="local",
    base_path="./artifacts",
    feature_name="churn",
    model_name="model",
    version="v1",
)
```

### Format Dispatch (Automatic)

| Path extension | Object type | Behavior |
|----------------|-------------|----------|
| `.pkl`, `.joblib` | Any | joblib serialization |
| `.json` | dict | JSON (sanitized for numpy/NaN) |
| `.png`, `.jpg` | matplotlib/plotly Figure | PNG image bytes |
| `.html` | str/bytes | Raw HTML bytes |

- **Save**: Format is inferred from the path extension and object type.
- **Load**: JSON files are parsed to dict; others are returned as-is (bytes or joblib objects).

### Storage Backends

- **local** — `LocalFileSystem` writes to a directory on disk
- **gcs** — `GCSStorage` writes to `gs://bucket/prefix`. Requires `bucket` or `base_bucket` and optionally `project_id`

---

## 2. Config — Training and Prediction Configuration

### What It Does

Config models validate and normalize parameters for training and prediction. They accept a kwargs dict or keyword args, and provide `to_artifact_kwargs()` for building `Artifact` instances.

### Config Types

| Config | Purpose |
|--------|---------|
| `TrainingConfig` | Training job — model_name, feature, version, base_path, backend, etc. |
| `PredictionConfig` | Prediction job — same base + input_path, output_path, BigQuery fields |
| `RunConfig` | Minimal config for artifact runs |

### How to Use

**From kwargs dict (e.g. from CLI or YAML):**

```python
def train(kwargs):
    cfg = TrainingConfig(kwargs)
    artifact = Artifact(**cfg.to_artifact_kwargs())
    artifact.save("model.pkl", model)

def predict(kwargs):
    cfg = PredictionConfig(kwargs)
    artifact = Artifact(**cfg.to_artifact_kwargs())
    model = artifact.load("model.pkl")
```

**From keyword args:**

```python
from mlplatform.config import TrainingConfig

cfg = TrainingConfig(
    model_name="churn_model",
    feature="churn",
    version="v1",
    base_path="./artifacts",
)
artifact = Artifact(**cfg.to_artifact_kwargs())
```

**Aliases:** `feature_name` → `feature`, `model_version` → `version`

### Config Profiles (YAML)

Load and merge YAML profiles for environment-specific config:

```python
from mlplatform.config import load_config_profiles, TrainingConfig

merged = load_config_profiles(
    ["global", "dev"],  # dev overrides global
    config_dir="my_model/config",
)
cfg = TrainingConfig(merged)
```

Profiles are loaded from `config_dir/{name}.yaml` and deep-merged. Later profiles override earlier ones.

---

## 3. Storage — Backends for Artifact Persistence

### What It Does

Storage backends abstract where artifacts are stored. The artifact system uses them internally; you can also use them directly for custom workflows.

### Implementations

| Backend | Class | Use case |
|---------|-------|----------|
| Local | `LocalFileSystem` | Development, single-machine |
| GCS | `GCSStorage` | Production, cloud |

### Interface

```python
from mlplatform.storage import LocalFileSystem

store = LocalFileSystem("./artifacts")
store.save("path/to/model.pkl", obj)
loaded = store.load("path/to/model.pkl")
store.save_bytes("data.json", b'{"a": 1}')
store.exists("path")
store.list_artifacts("prefix")
store.delete("path")
```

---

## 4. Utils — Serialization and Reports

### Serialization

- **`sanitize(obj)`** — Recursively converts to JSON-safe types (numpy → int/float, NaN/Inf → None, datetime → ISO string). Used internally when saving dicts to `.json`; you don’t need to call it before `artifact.save("metrics.json", dict)`.
- **`to_serializable(obj)`** — Converts dataclasses, Pydantic models, and objects with `__dict__` to plain dict/list.

### HTML Reports

```python
from mlplatform.utils import HTMLReport

report = HTMLReport(title="Model Report", feature_name="churn")
report.add_metric("accuracy", 0.95)
report.add_plot("loss", "report/loss.png")
artifact.save("report.html", report.to_html())
```

### Storage Helpers

- **`save_plot(fig, path, storage)`** — Save matplotlib/plotly figure to storage
- **`save_html(html, path, storage)`** — Save HTML string/bytes to storage

---

## 5. End-to-End Flow

### Train and Predict

```python
from mlplatform import Artifact
from mlplatform.config import TrainingConfig

CONFIG = TrainingConfig(
    model_name="sample_model",
    feature="demo",
    version="v1",
    base_path="./artifacts",
    backend="local",
)

def train(config):
    artifact = Artifact(**config.to_artifact_kwargs())
    model = train_model()
    artifact.save("model.pkl", model)
    artifact.save("metrics.json", {"accuracy": 0.95})
    return artifact

def predict(config):
    artifact = Artifact(**config.to_artifact_kwargs())
    model = artifact.load("model.pkl")
    return model.predict(X)

train(CONFIG)
predict(CONFIG)
```

---

## 6. Installation

```bash
# Base (config, artifacts, storage)
pip install mlplatform

# With config (Pydantic models)
pip install mlplatform[config]

# With GCS storage
pip install mlplatform[storage]

# With utils (matplotlib, save_plot, save_html)
pip install mlplatform[utils]

# Full public API
pip install mlplatform[core]
```

---

## 7. Module Layout

```
mlplatform/
├── artifacts/          # ArtifactRegistry, Artifact(), create_artifacts()
│   ├── core.py         # Artifact, create_artifacts, ArtifactConfig
│   └── registry.py     # ArtifactRegistry, save/load logic
├── config/             # TrainingConfig, PredictionConfig, load_config_profiles
│   ├── loader.py       # load_config_profiles, _load_config_profiles, _deep_merge
│   └── models.py       # TrainingConfig, PredictionConfig, RunConfig
├── storage/            # LocalFileSystem, GCSStorage
│   ├── base.py         # Storage ABC
│   ├── local.py        # LocalFileSystem
│   └── gcs.py          # GCSStorage
└── utils/              # sanitize, to_serializable, HTMLReport, save_plot, save_html
    ├── serialization.py
    ├── reports.py
    └── storage_helpers.py
```
