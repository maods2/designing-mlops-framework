# ML Platform Migration Proposal v3

> **Supersedes**: MIGRATION_PROPOSAL_V2.md
>
> **Goal**: Migrate `plat/` into `mlplatform/`, rethinking and simplifying abstractions.
> Remove DAG YAML dependency — the framework receives validated config from orchestrator args
> or from a Python API for local development. **Additionally**, improve the model-code
> developer experience so that data scientists define artifact identity once, config templates
> drive all loading, and the same code path works for local dev and cloud deployment.

---

## 1. Current State Analysis

### What exists in `mlplatform/` (v0.1.x — ~1,270 LOC)

| Module | Key Exports | Status |
|--------|------------|--------|
| `config` | `TrainingConfig`, `PredictionConfig`, `RunConfig`, `load_config_profiles` | Solid, Pydantic v2 |
| `artifacts` | `ArtifactRegistry`, `Artifact`, `create_artifacts` | Solid, format dispatch |
| `storage` | `Storage` ABC, `LocalFileSystem`, `GCSStorage`, `NoneStorage` | Solid, lazy imports |
| `utils` | `sanitize`, `to_serializable`, `save_plot`, `save_html`, `HTMLReport`, `get_logger` | Solid |

### What exists in `plat/` (legacy — ~3,550 LOC)

| Module | Key Exports | Will migrate? |
|--------|------------|--------------|
| `core` | `ExecutionContext`, `BaseTrainer`, `BasePredictor`, `ArtifactRegistry`, `PredictionInputSchema` | Yes — rethink |
| `config` | `load_workflow_config`, `ModelConfig`, `WorkflowConfig`, `PipelineConfig` | **Redesign** — no more DAG parsing |
| `inference` | `InProcessInference`, `SparkBatchInference`, `FastAPIInference` | Yes |
| `profiles` | `Profile`, `get_profile`, `register_profile` + 6 profiles | Yes |
| `runner` | `run_workflow`, `resolve_class`, `dev_train`, `dev_predict` | **Redesign** — no DAG orchestration |
| `tracking` | `ExperimentTracker`, `NoneTracker`, `LocalJsonTracker`, `VertexAITracker` | Yes — provider-agnostic |
| `data` | `load_prediction_input`, `write_prediction_output` | Yes |
| `spark` | Spark entry point, packager, config serializer | Yes |
| `cli` | `mlplatform run`, `mlplatform build-package` | **Redesign** — args → config model |

### What exists in `model_code/` (example DS code)

| File | Current Pattern | Issue |
|------|----------------|-------|
| `constants.py` | `MODEL_ARTIFACT`, `SCALER_ARTIFACT`, `FEATURE_COLUMNS` | No artifact identity (model_name, feature) |
| `train.py` | `BaseTrainer` subclass, reads `self.config` | Config injected by framework; opaque |
| `predict.py` | `BasePredictor` subclass, reads `self.config` | Same injection pattern |
| `config/*.yaml` | `global.yaml`, `dev.yaml`, `gcs.yaml` | No `model_name`/`feature` in config |
| `__main__` blocks | `PipelineConfig.from_dict({...})` | Identity scattered across `__main__` and YAML |

---

## 2. Design Principles

1. **No DAG YAML** — the framework receives validated config from orchestrator args or Python API
2. **Define once** — artifact identity (`model_name`, `feature`) declared in one place
3. **Config templates drive everything** — YAML profiles merged automatically; no manual wiring
4. **Two API layers** — function-based for simple models, class-based for complex models (lifecycle hooks, tracking)
5. **Local = deployment** — same config shape, same code path; only profile values differ
6. **Convention + override** — sensible defaults (`./config` dir, `["global", "dev"]` profiles), overridable via args or env

---

## 3. Architectural Changes

### 3.1 No more DAG YAML parsing

**Original**: The framework reads a DAG YAML file to discover models, config profiles, and pipeline type.

**New**: The framework receives **already-resolved arguments** from the orchestrator
or from user code. These args are validated into a **frozen Pydantic config model**.

```
┌──────────────────┐     args (dict/kwargs)     ┌──────────────────┐
│   Orchestrator   │ ─────────────────────────► │  PipelineConfig  │
│  (VertexAI, DBX) │                            │  (frozen model)  │
└──────────────────┘                            └────────┬─────────┘
                                                         │
┌──────────────────┐     Python constructor              │
│   Local dev      │ ─────────────────────────►          │
│  (script/notebook│                                     │
└──────────────────┘                            ┌────────▼─────────┐
                                                │   Runner / CLI    │
┌──────────────────┐     CLI args (parsed)      │                  │
│   CLI            │ ─────────────────────────► │  execute(config) │
│  mlplatform run  │                            └──────────────────┘
└──────────────────┘
```

**What this means:**
- `load_workflow_config` is **removed** (no DAG parsing)
- `load_config_profiles` from v0.1.x is **kept** as a utility
- Config profiles are merged by `load_model_config()` — a new convenience helper

### 3.2 Builder pattern for config (frozen output)

A `PipelineConfigBuilder` validates incrementally and produces a **frozen** `PipelineConfig`.

```python
from mlplatform.config import PipelineConfigBuilder

# From orchestrator args
config = (
    PipelineConfigBuilder()
    .identity(model_name="churn_model", feature="churn", version="v1.2")
    .infra(backend="gcs", bucket="ml-artifacts", project_id="my-project")
    .pipeline(pipeline_type="training", profile="cloud-train")
    .configs(["global", "train-prod"], config_dir="./config")
    .build()
)

# For local dev (minimal)
config = (
    PipelineConfigBuilder()
    .identity(model_name="churn_model", feature="churn")
    .pipeline(pipeline_type="training")
    .build()  # defaults: backend=local, base_path=./artifacts, profile=local, version=dev
)
```

### 3.3 Implicit artifact definition via config template

DS defines artifact identity **once** in `model_code/config/global.yaml`:

```yaml
# model_code/config/global.yaml
model_name: churn_model
feature: churn
base_path: ./artifacts
backend: local
log_level: INFO
```

Or in `model_code/constants.py` (for code-only projects):

```python
ARTIFACT_IDENTITY = {
    "model_name": "churn_model",
    "feature": "churn",
}
```

The `load_model_config()` helper merges profiles and returns a dict that contains
all fields needed for `PipelineConfig` construction, including artifact identity.

### 3.4 Two-layer model-code API

**Layer 1 — Function-based (simple models, no framework inheritance):**

```python
# model_code/train.py
from mlplatform.config import load_model_config, TrainingConfig
from mlplatform import Artifact

def train(config: dict | None = None):
    cfg = config or load_model_config()
    training_cfg = TrainingConfig(**cfg)
    artifact = Artifact(**training_cfg.to_artifact_kwargs())

    df = pd.read_csv(cfg.get("train_data_path", "data/train.csv"))
    model = LogisticRegression().fit(X, y)

    artifact.save("model.pkl", model)
    artifact.save("metrics.json", metrics)

if __name__ == "__main__":
    train()
```

**Layer 2 — Class-based (complex models, lifecycle hooks, tracking):**

```python
# model_code/train.py
from mlplatform.core.trainer import BaseTrainer
import model_code.constants as cons

class ChurnTrainer(BaseTrainer):
    def train(self):
        df = pd.read_csv(self.config.get("train_data_path"))
        model = LogisticRegression().fit(X, y)

        self.artifacts.save(cons.MODEL_ARTIFACT, model)
        self.tracker.log_metrics(metrics)
        self.tracker.log_params({"model_type": "LogisticRegression"})

if __name__ == "__main__":
    from mlplatform.runner import dev_train
    dev_train(
        model_name="churn_model",
        feature="churn",
        trainer_class=ChurnTrainer,
    )
```

Both layers share the same config shape and artifact conventions. The class-based
layer adds: lifecycle hooks (`setup`/`teardown`), experiment tracking (`self.tracker`),
and framework-managed context injection.

### 3.5 Unified ArtifactRegistry

**Original**: Two `ArtifactRegistry` classes — one in `mlplatform/artifacts/` (format dispatch)
and one in `plat/core/` (cross-model loading, storage property).

**New**: **Merge into one**. The existing `mlplatform/artifacts/registry.py` gains:
- `load(name, *, model_name=None, version=None)` for cross-model loading
- `storage` property exposing the underlying backend

### 3.6 Provider-agnostic tracking

Same ABC but designed for future providers:

```python
class ExperimentTracker(ABC):
    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None: ...
    @abstractmethod
    def log_metrics(self, metrics: dict[str, float]) -> None: ...
    @abstractmethod
    def log_artifact(self, name: str, artifact: Any) -> None: ...
    def start_run(self, run_name: str | None = None) -> None: ...   # optional, no-op default
    def end_run(self) -> None: ...                                    # optional, no-op default
```

The `start_run`/`end_run` hooks enable providers like MLflow that have explicit run lifecycle.
Context manager protocol (`with tracker:`) scopes runs naturally.

### 3.7 Simplified runner (no DAG orchestration)

`execute(config: PipelineConfig)` — takes a frozen config, resolves profile,
builds context, runs training or prediction. Single model per invocation.

```python
from mlplatform.runner import execute, dev_train, dev_predict

# Production (orchestrator calls this)
result = execute(config)

# Local dev shortcuts
ctx = dev_train(
    model_name="churn_model",
    feature="churn",
    trainer_class=MyTrainer,
)

predictions = dev_predict(
    model_name="churn_model",
    feature="churn",
    predictor_class=MyPredictor,
    version="v1.2",
)
```

---

## 4. Config System Design

### 4.1 `load_model_config()` — the DS entry point

```python
def load_model_config(
    config_list: list[str] | None = None,
    config_dir: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and merge YAML config profiles for model code.

    Resolution order:
    1. config_list: from arg → env MLPLATFORM_CONFIG → default ["global", "dev"]
    2. config_dir:  from arg → env MLPLATFORM_CONFIG_DIR → default "./config"
    3. Load each {config_dir}/{name}.yaml and deep-merge in order
    4. Apply overrides dict on top (if provided)

    Returns a flat dict suitable for TrainingConfig(**cfg) or PipelineConfig(**cfg).
    """
```

**Convention + override behavior:**

| Source | Default | Override |
|--------|---------|----------|
| `config_list` | `["global", "dev"]` | Arg or `MLPLATFORM_CONFIG=global,prod` |
| `config_dir` | `./config` (cwd-relative) | Arg or `MLPLATFORM_CONFIG_DIR=/path/to/config` |
| Values | From merged YAML | `overrides` dict applied last |

### 4.2 `PipelineConfig` — the frozen contract

```python
class PipelineConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    # Identity
    model_name: str
    feature: str
    version: str = "dev"

    # Infrastructure
    base_path: str = "./artifacts"
    base_bucket: str | None = None
    backend: Literal["local", "gcs"] = "local"
    project_id: str | None = None

    # Pipeline
    pipeline_type: Literal["training", "prediction"]
    profile: str = "local"
    platform: str = "VertexAI"

    # Module resolution
    module: str = ""  # e.g. "my_package.train:MyTrainer"

    # Merged user config (from YAML profiles or direct dict)
    user_config: dict[str, Any] = Field(default_factory=dict)

    # Metadata
    commit_hash: str | None = None
    log_level: str = "INFO"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PipelineConfig":
        """Construct from orchestrator JSON/dict payload."""
        return cls(**payload)

    @classmethod
    def from_model_config(
        cls,
        config_list: list[str] | None = None,
        config_dir: str | None = None,
        pipeline_type: str = "training",
        **overrides,
    ) -> "PipelineConfig":
        """Construct from model config templates + overrides.

        Loads config profiles, extracts identity/infra fields,
        remaining keys go into user_config.
        """
        merged = load_model_config(config_list, config_dir)
        merged.update(overrides)
        # Split known PipelineConfig fields from user_config
        known_fields = cls.model_fields.keys()
        pipeline_kwargs = {k: v for k, v in merged.items() if k in known_fields}
        user_cfg = {k: v for k, v in merged.items() if k not in known_fields}
        pipeline_kwargs.setdefault("pipeline_type", pipeline_type)
        pipeline_kwargs.setdefault("user_config", {})
        pipeline_kwargs["user_config"] = {**pipeline_kwargs["user_config"], **user_cfg}
        return cls(**pipeline_kwargs)
```

### 4.3 `PipelineConfigBuilder` — ergonomic construction

```python
class PipelineConfigBuilder:
    def identity(self, *, model_name, feature, version="dev") -> Self: ...
    def infra(self, *, backend="local", base_path="./artifacts",
              base_bucket=None, project_id=None) -> Self: ...
    def pipeline(self, *, pipeline_type, profile="local",
                 platform="VertexAI", module="") -> Self: ...
    def configs(self, profile_names, config_dir="./config") -> Self: ...
    def user_config(self, config: dict) -> Self: ...
    def metadata(self, *, commit_hash=None, log_level="INFO") -> Self: ...
    def build(self) -> PipelineConfig: ...
```

**Validation rules in `build()`:**
- `backend=gcs` requires `base_bucket` (or `base_bucket` in `user_config`)
- `profile` starting with `cloud-` requires `backend=gcs`
- `pipeline_type=prediction` requires `module` (must have predictor to run)
- `model_name` and `feature` are always required

### 4.4 Config template structure

```
model_code/
├── config/
│   ├── global.yaml       # model_name, feature, base_path, backend, log_level
│   ├── dev.yaml           # train_data_path, hyperparameters, base_path override
│   ├── prod.yaml          # backend: gcs, base_bucket, project_id
│   └── predict-dev.yaml   # input_path, output_path for local prediction
├── constants.py           # MODEL_ARTIFACT, SCALER_ARTIFACT, FEATURE_COLUMNS
├── train.py
├── predict.py
└── evaluate.py
```

**`global.yaml` — identity + defaults (always loaded first):**
```yaml
model_name: churn_model
feature: churn
base_path: ./artifacts
backend: local
log_level: INFO
```

**`dev.yaml` — local development overrides:**
```yaml
log_level: DEBUG
base_path: ./dev_artifacts
train_data_path: model_code/data/sample_train.csv
hyperparameters:
  max_iter: 1000
  random_state: 42
test_size: 0.2
```

**`prod.yaml` — production/cloud overrides:**
```yaml
backend: gcs
base_bucket: ml-artifacts-prod
project_id: my-gcp-project
profile: cloud-train
log_level: INFO
train_data_path: gs://ml-data-prod/churn/train.parquet
hyperparameters:
  max_iter: 5000
  random_state: 42
```

### 4.5 Local vs Deployment parity

| Concern | Local | Deployment |
|---------|-------|------------|
| Config source | `load_model_config()` → `["global","dev"]` | Orchestrator passes `PipelineConfig` |
| Artifact | From constants + config merge | Same; overrides from orchestrator |
| Storage | `base_path=./dev_artifacts` | `base_bucket`, `project_id` from prod |
| Config list | Default or `MLPLATFORM_CONFIG` env | Orchestrator sets `config_list` |
| Tracking | `NoneTracker` / `LocalJsonTracker` | `VertexAITracker` via profile |
| Entry point | `python train.py` or `dev_train()` | `execute(config)` from orchestrator |

---

## 5. Model Code Patterns (After Migration)

### 5.1 Function-based train (simple models)

```python
# model_code/train.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from mlplatform.config import load_model_config, TrainingConfig
from mlplatform import Artifact

import model_code.constants as cons
from model_code.evaluate import evaluate


def train(config: dict | None = None):
    """Train a churn model. Config auto-loaded from model_code/config/ if not provided."""
    cfg = config or load_model_config()
    training_cfg = TrainingConfig(**cfg)
    artifact = Artifact(**training_cfg.to_artifact_kwargs())

    # 1. Load data
    data_path = cfg.get("train_data_path", "model_code/data/sample_train.csv")
    df = pd.read_csv(data_path)
    X = df[cons.FEATURE_COLUMNS]
    y = df["target"]

    # 2. Train
    hyperparams = cfg.get("hyperparameters", {})
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(
        max_iter=hyperparams.get("max_iter", 1000),
        random_state=42,
    ).fit(scaler.transform(X), y)

    # 3. Save
    artifact.save(cons.MODEL_ARTIFACT, model)
    artifact.save(cons.SCALER_ARTIFACT, scaler)
    artifact.save("metrics.json", evaluate(model, scaler, df))

    return artifact


if __name__ == "__main__":
    train()  # uses config/global.yaml + config/dev.yaml by default
```

### 5.2 Class-based train (complex models with tracking)

```python
# model_code/train.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from mlplatform.core.trainer import BaseTrainer

import model_code.constants as cons
from model_code.evaluate import evaluate


class ChurnTrainer(BaseTrainer):
    """Train a churn model with full lifecycle and tracking."""

    def train(self):
        # 1. Load data
        data_path = self.config.get("train_data_path", "model_code/data/sample_train.csv")
        df = pd.read_csv(data_path)
        X = df[cons.FEATURE_COLUMNS]
        y = df["target"]

        # 2. Train
        hyperparams = self.config.get("hyperparameters", {})
        scaler = StandardScaler().fit(X)
        model = LogisticRegression(
            max_iter=hyperparams.get("max_iter", 1000),
            random_state=42,
        ).fit(scaler.transform(X), y)

        # 3. Evaluate and track
        metrics = evaluate(model, scaler, df)
        self.log.info("Validation metrics: %s", metrics)
        self.tracker.log_metrics(metrics)
        self.tracker.log_params({
            "model_type": "LogisticRegression",
            "max_iter": hyperparams.get("max_iter", 1000),
        })

        # 4. Save artifacts
        self.artifacts.save(cons.MODEL_ARTIFACT, model)
        self.artifacts.save(cons.SCALER_ARTIFACT, scaler)


if __name__ == "__main__":
    from mlplatform.runner import dev_train
    dev_train(
        model_name="churn_model",
        feature="churn",
        trainer_class=ChurnTrainer,
    )
```

### 5.3 Function-based predict

```python
# model_code/predict.py
import pandas as pd

from mlplatform.config import load_model_config, PredictionConfig
from mlplatform import Artifact

import model_code.constants as cons


def predict(data: pd.DataFrame | None = None, config: dict | None = None):
    """Run prediction. Config auto-loaded from model_code/config/ if not provided."""
    cfg = config or load_model_config(config_list=["global", "predict-dev"])
    pred_cfg = PredictionConfig(**cfg)
    artifact = Artifact(**pred_cfg.to_artifact_kwargs())

    model = artifact.load(cons.MODEL_ARTIFACT)
    scaler = artifact.load(cons.SCALER_ARTIFACT)

    if data is None:
        input_path = cfg.get("input_path", "model_code/data/sample_input.csv")
        data = pd.read_csv(input_path)

    X = data[cons.FEATURE_COLUMNS]
    X_scaled = scaler.transform(X)
    return data.assign(prediction=model.predict(X_scaled))


if __name__ == "__main__":
    result = predict()
    print(result)
```

### 5.4 Class-based predict

```python
# model_code/predict.py
import pandas as pd

from mlplatform.core.predictor import BasePredictor

import model_code.constants as cons


class ChurnPredictor(BasePredictor):
    """Load model and scaler, run predictions on data."""

    def load_model(self):
        self._model = self.artifacts.load(cons.MODEL_ARTIFACT)
        self._scaler = self.artifacts.load(cons.SCALER_ARTIFACT)
        return self._model

    def predict(self, data):
        df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
        X = df[cons.FEATURE_COLUMNS]
        X_scaled = self._scaler.transform(X)
        return df.assign(prediction=self._model.predict(X_scaled))


if __name__ == "__main__":
    from mlplatform.runner import dev_predict
    result = dev_predict(
        model_name="churn_model",
        feature="churn",
        predictor_class=ChurnPredictor,
        version="dev",
    )
    print(result)
```

### 5.5 Updated `constants.py`

```python
# model_code/constants.py
"""Model constants — artifact identity and feature definitions."""

# Artifact identity (single source of truth for model_name + feature)
ARTIFACT_IDENTITY = {
    "model_name": "churn_model",
    "feature": "churn",
}

# Artifact file names
MODEL_ARTIFACT = "model.pkl"
SCALER_ARTIFACT = "scaler.pkl"

# Feature columns
FEATURE_COLUMNS = ["f0", "f1", "f2", "f3", "f4"]
```

---

## 6. Architecture Diagrams

### 6.1 Training Flow

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                 Cloud / Orchestrator                     │
                    │  ┌──────────┐                                           │
                    │  │Orchestrat│──Config──►┌──────────────────────────────┐│
                    │  │   or     │           │    Base Trainer Wrapper      ││
                    │  └──────────┘           │                              ││
YAML configs ─────────────────────────────►  │  ┌─────┐  ┌──────────────┐  ││
(merged by user)                             │  │ CLI │  │  Train Code   │  ││
                    │                        │  └─────┘  │  (user impl)  │  ││
                    │                        │           └──────────────┘  ││
                    │                        └──────────────────────────────┘│
                    │                              │              │          │
                    │                    ┌─────────▼──┐   ┌──────▼───────┐  │
                    │                    │  Artifact   │   │  Tracking    │  │
                    │                    │  Registry   │   │  Interface   │  │
                    │                    └──────┬──────┘   └──────┬───────┘  │
                    │                           │                 │          │
                    │                    ┌──────▼──────┐   ┌──────▼───────┐  │
                    │                    │  Storage    │   │  Tracker     │  │
                    │                    │  Interface  │   │  Impls       │  │
                    │                    └──────┬──────┘   └──────────────┘  │
                    │                           │                            │
                    │                 ┌─────────┴─────────┐                  │
                    │                 │                    │                  │
                    │          ┌──────▼─────┐  ┌──────────▼──┐              │
                    │          │Local Storage│  │ GCS Storage  │              │
                    │          └────────────┘  └─────────────┘              │
                    │                                                        │
                    │        ┌────────┐  ┌───────┐                          │
                    │        │ Config │  │ Utils │                          │
                    │        └────────┘  └───────┘                          │
                    └─────────────────────────────────────────────────────────┘
                              ▲
              Config ─────────┘
              (local dev)
```

### 6.2 Config Loading Flow

```
┌─────────────────────────────────────────────────────────┐
│                    Config Resolution                     │
│                                                          │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────┐  │
│  │ global.yaml │───►│              │    │ Orchestrator│  │
│  │ (identity + │    │ load_model_  │    │   args      │  │
│  │  defaults)  │    │ config()     │    └──────┬─────┘  │
│  └─────────────┘    │              │           │         │
│                     │  deep-merge  │           │         │
│  ┌─────────────┐    │  in order    │           │         │
│  │  dev.yaml   │───►│              │           │         │
│  │ (overrides) │    └──────┬───────┘           │         │
│  └─────────────┘           │                   │         │
│                            ▼                   ▼         │
│                    ┌───────────────┐   ┌──────────────┐  │
│                    │  merged dict  │   │  PipelineConf │  │
│                    └───────┬───────┘   │  .from_dict() │  │
│                            │           └──────┬───────┘  │
│                            ▼                  │          │
│                    ┌───────────────┐          │          │
│                    │ PipelineConfig│◄─────────┘          │
│                    │ .from_model_  │                     │
│                    │  config()     │                     │
│                    └───────┬───────┘                     │
│                            │                             │
│                            ▼                             │
│                    ┌───────────────┐                     │
│                    │ Frozen config │                     │
│                    │ (immutable)   │                     │
│                    └───────────────┘                     │
└─────────────────────────────────────────────────────────┘
```

### 6.3 Two-Layer API Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      Model Code API Layers                       │
│                                                                  │
│  Layer 1: Function-based (simple)                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  train(config=None)                                       │   │
│  │    cfg = load_model_config()          ← auto-loads YAML   │   │
│  │    training_cfg = TrainingConfig(**cfg)                    │   │
│  │    artifact = Artifact(**cfg.to_artifact_kwargs())         │   │
│  │    ... train, save ...                                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Layer 2: Class-based (complex)                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  class ChurnTrainer(BaseTrainer):                         │   │
│  │      def train(self):                                     │   │
│  │          self.config      ← user_config from PipelineConf │   │
│  │          self.artifacts   ← ArtifactRegistry from context │   │
│  │          self.tracker     ← ExperimentTracker from profile│   │
│  │          self.log         ← Logger                        │   │
│  │          ... train, save, track ...                       │   │
│  │                                                           │   │
│  │  Entry: dev_train(model_name, feature, trainer_class)     │   │
│  │    → builds PipelineConfig from args + load_model_config  │   │
│  │    → builds ExecutionContext from config + profile         │   │
│  │    → runs setup() → train() → teardown()                  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 6.4 Module Dependency Graph

```
                            ┌───────────────────────────┐
                            │        CLI (thin)          │
                            │  args → PipelineConfig     │
                            └─────────────┬─────────────┘
                                          │
                            ┌─────────────▼─────────────┐
                            │         Runner             │
                            │  execute(config)           │
                            │  dev_train / dev_predict   │
                            └──┬──────────┬──────────┬──┘
                               │          │          │
                ┌──────────────▼┐  ┌──────▼──────┐  ┌▼─────────────┐
                │    Profiles    │  │    Core     │  │  Inference   │
                │  get_profile   │  │ ExecContext │  │  Strategy    │
                │  Profile{}     │  │ BaseTrainer │  │  InProcess   │
                │                │  │ BasePredict │  │  SparkBatch  │
                │                │  │ ArtifactReg │  │  FastAPI     │
                └───┬───────┬───┘  └──┬──────┬───┘  └──┬───────┬──┘
                    │       │         │      │         │       │
           ┌────────▼───┐ ┌─▼─────────▼┐  ┌─▼─────────▼─┐  ┌─▼──────┐
           │  Tracking   │ │  Storage    │  │   Config     │  │  Data  │
           │ Tracker ABC │ │ Storage ABC │  │ PipelineConf │  │  I/O   │
           │ None/Local  │ │ Local/GCS   │  │ Builder      │  │        │
           │ VertexAI    │ │ None        │  │ Training/    │  │        │
           │ (MLflow...) │ │             │  │ Prediction   │  │        │
           └─────────────┘ └─────────────┘  │ load_model_  │  └────────┘
                                  │         │  config()    │
                            ┌─────▼─────────┴─────────────┘
                            │         Utils             │
                            │  serialization, logging   │
                            │  storage_helpers, reports  │
                            └───────────────────────────┘
                                  │
                            ┌─────▼─────────────────────┐
                            │      Artifacts             │
                            │  (format dispatch layer)   │
                            └───────────────────────────┘
```

---

## 7. Migration Phases

### Phase 1: Config Redesign (foundation — breaking change)

**What changes:**
- **Remove** `load_workflow_config` (DAG parser) from migration target
- **Keep** `load_config_profiles` as utility (users call it themselves)
- **Keep** `TrainingConfig`, `PredictionConfig`, `RunConfig` from v0.1.x
- **Add** `PipelineConfig` — frozen model built from orchestrator/CLI args
- **Add** `PipelineConfigBuilder` — builder with incremental validation
- **Add** `load_model_config()` — convenience helper for DS config loading

**PipelineConfig fields:**

```python
class PipelineConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    # Identity
    model_name: str
    feature: str
    version: str = "dev"

    # Infrastructure
    base_path: str = "./artifacts"
    base_bucket: str | None = None
    backend: Literal["local", "gcs"] = "local"
    project_id: str | None = None

    # Pipeline
    pipeline_type: Literal["training", "prediction"]
    profile: str = "local"
    platform: str = "VertexAI"

    # Module resolution
    module: str = ""

    # Merged user config (from YAML profiles or direct dict)
    user_config: dict[str, Any] = Field(default_factory=dict)

    # Metadata
    commit_hash: str | None = None
    log_level: str = "INFO"
```

**`load_model_config()` implementation:**

```python
def load_model_config(
    config_list: list[str] | None = None,
    config_dir: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_list = config_list or _env_list("MLPLATFORM_CONFIG", ["global", "dev"])
    resolved_dir = config_dir or os.environ.get("MLPLATFORM_CONFIG_DIR", "./config")
    merged = load_config_profiles(resolved_list, resolved_dir)
    if overrides:
        merged = _deep_merge(merged, overrides)
    return merged

def _env_list(var: str, default: list[str]) -> list[str]:
    val = os.environ.get(var)
    return val.split(",") if val else default
```

**Files:**
- Modify: `mlplatform/config/models.py` — add `PipelineConfig`
- New: `mlplatform/config/builder.py` — `PipelineConfigBuilder`
- Modify: `mlplatform/config/loader.py` — add `load_model_config()` (keep `load_config_profiles`)

---

### Phase 2: Tracking (provider-agnostic)

**What changes:**
- Move `plat/tracking/` → `mlplatform/tracking/`
- Add `start_run`/`end_run` lifecycle hooks (no-op defaults)
- Context manager protocol

**ExperimentTracker ABC:**

```python
class ExperimentTracker(ABC):
    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None: ...
    @abstractmethod
    def log_metrics(self, metrics: dict[str, float]) -> None: ...
    @abstractmethod
    def log_artifact(self, name: str, artifact: Any) -> None: ...
    def start_run(self, run_name: str | None = None) -> None: pass
    def end_run(self) -> None: pass
    def __enter__(self) -> "ExperimentTracker":
        self.start_run()
        return self
    def __exit__(self, *exc) -> None:
        self.end_run()
```

**Implementations:**

| Tracker | `log_params` | `log_metrics` | `log_artifact` | `start_run`/`end_run` |
|---------|-------------|--------------|----------------|----------------------|
| `NoneTracker` | no-op | no-op | no-op | no-op |
| `LocalJsonTracker` | JSON file under base_path | JSON file | delegates to storage | no-op |
| `VertexAITracker` | Vertex Experiments | Vertex Experiments | Vertex Experiments | creates/closes run |

**Files:**
- New: `mlplatform/tracking/__init__.py`, `base.py`, `none.py`, `local.py`, `vertexai.py`

---

### Phase 3: Core (rethink + simplify)

**What changes:**
- Merge two `ArtifactRegistry` classes into one (extend `mlplatform/artifacts/registry.py`)
- Move `ExecutionContext` — uses merged registry, adds `from_config` factory
- Keep `BaseTrainer` and `BasePredictor` (class-based layer)
- Move `PredictionInputSchema`

**ArtifactRegistry additions** (to existing `mlplatform/artifacts/registry.py`):

```python
class ArtifactRegistry:
    # ... existing save/load with format dispatch ...

    @property
    def storage(self) -> Storage:
        return self._storage

    def load(self, name, *, model_name=None, version=None) -> Any:
        """Override model_name/version for cross-model loading."""
        if model_name or version:
            mn = model_name or self._model_name
            ver = version or self._version
            path = _resolve_path(self._feature_name, mn, ver, name)
            raw = self._storage.load(path)
            return _deserialize_for_load(name, raw, _ext(name))
        path = _resolve_path(self._feature_name, self._model_name, self._version, name)
        raw = self._storage.load(path)
        return _deserialize_for_load(name, raw, _ext(name))
```

**ExecutionContext — `from_config` replaces `from_profile`:**

```python
@dataclass
class ExecutionContext:
    artifacts: ArtifactRegistry
    experiment_tracker: ExperimentTracker = field(default_factory=NoneTracker)
    feature_name: str = ""
    model_name: str = ""
    version: str = ""
    user_config: dict[str, Any] = field(default_factory=dict)  # renamed from optional_configs
    log: logging.Logger = field(default_factory=lambda: logging.getLogger("mlplatform"))
    pipeline_type: str = ""
    commit_hash: str | None = None

    @classmethod
    def from_config(
        cls,
        config: PipelineConfig,
        profile: Profile,
        extra_overrides: dict[str, Any] | None = None,
    ) -> "ExecutionContext": ...
        # Single canonical constructor
        # Takes PipelineConfig instead of scattered args
```

> **Key simplification**: `from_profile` took 10 args. `from_config` takes a frozen
> `PipelineConfig` + a `Profile`. All the scattered args are already validated in the config.

**BaseTrainer / BasePredictor**: Same contract as current `plat/core/`, no changes needed.

**Files:**
- Modify: `mlplatform/artifacts/registry.py` — add `storage` property + cross-model load
- New: `mlplatform/core/__init__.py`, `context.py`, `trainer.py`, `predictor.py`, `prediction_schema.py`

---

### Phase 4: Profiles

**What changes:**
- Move `plat/profiles/` → `mlplatform/profiles/`
- Same `Profile` dataclass + registry pattern
- Same lazy factory helpers (no cloud imports at module load)

No architectural changes needed.

**Files:**
- New: `mlplatform/profiles/__init__.py`, `registry.py`

---

### Phase 5: Inference Strategies

**What changes:**
- Move `plat/inference/` → `mlplatform/inference/`
- `InferenceStrategy.invoke()` receives `PipelineConfig` instead of `ModelConfig`

```python
class InferenceStrategy(ABC):
    @abstractmethod
    def invoke(self, predictor, context, config: PipelineConfig) -> Any: ...
```

**Files:**
- New: `mlplatform/inference/__init__.py`, `base.py`, `in_process.py`, `spark_batch.py`, `fastapi_serving.py`

---

### Phase 6: Data I/O

**What changes:**
- Move `plat/data/` → `mlplatform/data/`
- Adapt `load_prediction_input` / `write_prediction_output` to use `PipelineConfig`

**Files:**
- New: `mlplatform/data/__init__.py`, `io.py`

---

### Phase 7: Runner (simplified)

**What changes:**
- **Remove**: `run_workflow` (DAG orchestration is the orchestrator's job)
- **Add**: `execute(config: PipelineConfig)` — single-model execution from a frozen config
- **Redesign**: `dev_train`, `dev_predict` — config-driven, support both API layers
- **Keep**: `resolve_class` for module:Class resolution

```python
def execute(config: PipelineConfig) -> dict[str, str]:
    """Execute a single model training or prediction from a frozen config."""
    profile = get_profile(config.profile)
    ctx = ExecutionContext.from_config(config, profile)

    if config.pipeline_type == "training":
        trainer_cls = resolve_class(config.module, BaseTrainer)
        trainer = trainer_cls()
        trainer.context = ctx
        trainer.setup()
        try:
            trainer.train()
        finally:
            trainer.teardown()
        return {"status": "ok"}
    else:
        inference = profile.inference_strategy_factory()
        predictor_cls = resolve_class(config.module, BasePredictor)
        predictor = predictor_cls()
        predictor.context = ctx
        predictor.setup()
        try:
            inference.invoke(predictor, ctx, config)
        finally:
            predictor.teardown()
        return {"status": "ok"}


def dev_train(
    model_name: str | None = None,
    feature: str | None = None,
    trainer_class: type[BaseTrainer] | None = None,
    *,
    version: str = "dev",
    base_path: str = "./artifacts",
    user_config: dict[str, Any] | None = None,
    config_list: list[str] | None = None,
    config_dir: str | None = None,
) -> ExecutionContext:
    """Local training — no CLI, no config files needed.

    If model_name/feature not provided, loads from config template.
    If trainer_class not provided, uses function-based flow.
    """
    cfg = load_model_config(config_list, config_dir)
    cfg.update(user_config or {})

    resolved_name = model_name or cfg.get("model_name")
    resolved_feature = feature or cfg.get("feature")

    config = (
        PipelineConfigBuilder()
        .identity(model_name=resolved_name, feature=resolved_feature, version=version)
        .infra(base_path=cfg.get("base_path", base_path))
        .pipeline(pipeline_type="training")
        .user_config(cfg)
        .build()
    )
    profile = get_profile("local")
    ctx = ExecutionContext.from_config(config, profile)

    if trainer_class:
        trainer = trainer_class()
        trainer.context = ctx
        trainer.setup()
        trainer.train()
        trainer.teardown()

    return ctx
```

**Files:**
- New: `mlplatform/runner/__init__.py`, `execute.py`, `resolve.py`, `dev.py`

---

### Phase 8: Spark

**What changes:**
- Move `plat/spark/` → `mlplatform/spark/`
- Spark entry point receives serialized `PipelineConfig` (JSON) instead of DAG path
- `config_serializer` converts frozen config ↔ JSON dict

**Files:**
- New: `mlplatform/spark/__init__.py`, `main.py`, `packager.py`, `config_serializer.py`

---

### Phase 9: CLI (thin wrapper)

**What changes:**
- CLI parses args → builds `PipelineConfig` via builder → calls `execute(config)`

```
mlplatform run \
  --model-name churn_model \
  --feature churn \
  --version v1.2 \
  --pipeline-type training \
  --profile local \
  --base-path ./artifacts \
  --config global,dev \
  --config-dir ./config \
  --module my_package.train:ChurnTrainer
```

**Files:**
- New: `mlplatform/cli/__init__.py`, `main.py`
- Modify: `pyproject.toml` — add `[project.scripts] mlplatform = "mlplatform.cli.main:main"`

---

### Phase 10: Model Code Migration

**What changes:**
- Update `model_code/config/global.yaml` to include `model_name` and `feature`
- Add `ARTIFACT_IDENTITY` to `model_code/constants.py`
- Refactor `model_code/train.py` and `model_code/predict.py` to use new patterns
- Update `__main__` blocks to use `load_model_config()` instead of hardcoded dicts

This is the **final phase** — it validates that all framework changes work end-to-end
from the DS perspective. Both function-based and class-based patterns should work.

**Files:**
- Modify: `model_code/config/global.yaml` — add identity fields
- Modify: `model_code/constants.py` — add `ARTIFACT_IDENTITY`
- Modify: `model_code/train.py` — use new config loading
- Modify: `model_code/predict.py` — use new config loading

---

## 8. Migration Order & Dependencies

```
Phase 1: Config ─────────┐
                          ▼
Phase 2: Tracking ──► Phase 3: Core ──► Phase 4: Profiles
                                              │
                          ┌───────────────────┤
                          ▼                   ▼
                   Phase 5: Inference   Phase 6: Data I/O
                          │                   │
                          └────────┬──────────┘
                                   ▼
                            Phase 7: Runner
                                   │
                          ┌────────┴──────────┐
                          ▼                   ▼
                   Phase 8: Spark      Phase 9: CLI
                                   │
                                   ▼
                          Phase 10: Model Code
```

Each phase can be **independently tested** before moving to the next.

---

## 9. Files to Create / Modify

### New files (~15 framework files)

```
mlplatform/
├── config/
│   └── builder.py                    # PipelineConfigBuilder
├── core/
│   ├── __init__.py
│   ├── context.py                    # ExecutionContext (from_config factory)
│   ├── trainer.py                    # BaseTrainer (same contract)
│   ├── predictor.py                  # BasePredictor (same contract)
│   └── prediction_schema.py         # PredictionInputSchema
├── tracking/
│   ├── __init__.py
│   ├── base.py                       # ExperimentTracker ABC
│   ├── none.py                       # NoneTracker
│   ├── local.py                      # LocalJsonTracker
│   └── vertexai.py                   # VertexAITracker
├── inference/
│   ├── __init__.py
│   ├── base.py                       # InferenceStrategy ABC
│   ├── in_process.py                 # InProcessInference
│   ├── spark_batch.py                # SparkBatchInference
│   └── fastapi_serving.py           # FastAPIInference
├── data/
│   ├── __init__.py
│   └── io.py                         # load_prediction_input, write_prediction_output
├── profiles/
│   ├── __init__.py
│   └── registry.py                   # Profile, get_profile, register_profile
├── runner/
│   ├── __init__.py
│   ├── execute.py                    # execute(config)
│   ├── resolve.py                    # resolve_class
│   └── dev.py                        # dev_train, dev_predict, dev_context
├── spark/
│   ├── __init__.py
│   ├── main.py                       # Spark entry point
│   ├── packager.py                   # build_root_zip, build_model_package
│   └── config_serializer.py         # PipelineConfig ↔ JSON
└── cli/
    ├── __init__.py
    └── main.py                       # mlplatform run, mlplatform build-package
```

### Modified files (~5 files)

| File | Change |
|------|--------|
| `mlplatform/artifacts/registry.py` | Add `storage` property + `load(..., model_name, version)` |
| `mlplatform/config/models.py` | Add `PipelineConfig` frozen model |
| `mlplatform/config/loader.py` | Add `load_model_config()` convenience helper |
| `model_code/config/global.yaml` | Add `model_name`, `feature` identity fields |
| `pyproject.toml` | Add CLI entry point + optional deps |

---

## 10. What Gets Deleted from plat/

After migration is complete and all tests pass:

| plat/ module | Action |
|-------------|--------|
| `plat/config/loader.py` | **Delete** — DAG parsing removed |
| `plat/config/models.py` | **Delete** — replaced by PipelineConfig |
| `plat/core/*` | **Delete** — moved to mlplatform/core/ |
| `plat/tracking/*` | **Delete** — moved to mlplatform/tracking/ |
| `plat/inference/*` | **Delete** — moved to mlplatform/inference/ |
| `plat/profiles/*` | **Delete** — moved to mlplatform/profiles/ |
| `plat/runner/*` | **Delete** — replaced by mlplatform/runner/execute |
| `plat/data/*` | **Delete** — moved to mlplatform/data/ |
| `plat/spark/*` | **Delete** — moved to mlplatform/spark/ |
| `plat/cli/*` | **Delete** — replaced by mlplatform/cli/ |
| `plat/artifacts/*` | **Delete** — merged into mlplatform/artifacts/ |

---

## 11. pyproject.toml Optional Extras

```toml
[project.optional-dependencies]
# Existing
utils = ["matplotlib"]
config = ["pydantic>=2.0"]
storage = ["google-cloud-storage>=2.0"]

# New
tracking = ["google-cloud-aiplatform>=1.25"]
spark = ["pyspark>=3.2"]
serving = ["fastapi>=0.100", "uvicorn>=0.20"]
bigquery = ["google-cloud-bigquery>=3.0", "pandas-gbq>=0.17"]
parquet = ["pyarrow>=10.0"]

# Meta
core = ["mlplatform[utils]", "mlplatform[config]"]
all = ["mlplatform[core]", "mlplatform[tracking]", "mlplatform[spark]",
       "mlplatform[serving]", "mlplatform[bigquery]", "mlplatform[parquet]"]

[project.scripts]
mlplatform = "mlplatform.cli.main:main"
```

---

## 12. Key Simplifications vs. Original (`plat/`)

| Area | Original (`plat/`) | This Proposal |
|------|-------------------|---------------|
| **DAG parsing** | `load_workflow_config` with 2 YAML formats | **Remove** — config comes from args/builder |
| **Config model** | Flat `WorkflowConfig` with mixed concerns | **Builder → frozen `PipelineConfig`** with validation |
| **Config loading (DS)** | Framework auto-discovers from DAG | **`load_model_config()`** — convention + override |
| **Artifact identity** | Scattered across DAG, config, code | **Once** — in `config/global.yaml` or `constants.py` |
| **Model code API** | BaseTrainer/BasePredictor only | **Two layers** — function-based + class-based |
| **ArtifactRegistry** | Two classes to merge | **Extend existing** v0.1.x registry (add 2 features) |
| **Runner** | `run_workflow` iterates models from DAG | **`execute(config)`** — single model |
| **Local vs prod** | Different code paths | **Same** — only profile values differ |
| **Config profiles** | Framework loads YAMLs from DAG config: key | **DS loads** via `load_model_config()` |

---

## 13. Recommendations & Notes

### R1: Config profiles are now opt-in but convenient
The `load_model_config()` helper provides sensible defaults (`["global", "dev"]` from `./config`).
Users who want more control can pass explicit `config_list` and `config_dir`. Orchestrators
bypass YAML entirely by passing `PipelineConfig.from_dict(payload)`.

### R2: Single-model execution
The runner no longer iterates over models. Each `execute(config)` call handles one model.
The orchestrator handles multi-model flows. This is a significant simplification.

### R3: PipelineConfig is the contract
Every component receives the same frozen `PipelineConfig`. No more scattered args passed
through 3-4 function signatures.

### R4: Backward compatibility
The v0.1.x public API (`TrainingConfig`, `PredictionConfig`, `RunConfig`, `create_artifacts`,
`ArtifactRegistry`, all storage/utils) remains **unchanged**. The new modules are additive.

### R5: Tracking extensibility
The `ExperimentTracker` ABC with `start_run`/`end_run` supports both stateless trackers
(LocalJson, None) and stateful ones (VertexAI, MLflow). The context manager protocol
makes it natural to scope runs.

### R6: `PipelineConfig.from_model_config()` bridges the two worlds
For local dev, `PipelineConfig.from_model_config()` auto-loads YAML templates and splits
the merged dict into PipelineConfig fields + user_config. This means the same
`config/global.yaml` that defines `model_name` and `feature` also feeds the pipeline config.

### R7: Function-based layer is a "fast path"
DS who don't need tracking, lifecycle hooks, or inference strategies can use plain functions.
They get `load_model_config()` → `TrainingConfig` → `Artifact` and that's it. No framework
inheritance, no context injection. When they outgrow this, they graduate to `BaseTrainer`.

### R8: Test strategy
Each phase should include:
- Unit tests for the new module
- Integration test showing it works with existing v0.1.x code
- A "smoke test" that runs `dev_train` / `dev_predict` end-to-end with a toy model

### R9: Migration order for model code
Phase 10 (model code) is intentionally last. The framework must be fully migrated before
updating model code, so we can validate that the new patterns work end-to-end.

---

## 14. Success Criteria

- [ ] DS defines artifact identity once (in `config/global.yaml` or `constants.py`)
- [ ] Config template (YAML profiles) drives all loading via `load_model_config()`
- [ ] Same code path for local dev and deployment
- [ ] No DAG YAML dependency anywhere in the framework
- [ ] Train and predict are simpler — both function-based and class-based work
- [ ] `PipelineConfig` is the single frozen contract between all components
- [ ] All existing v0.1.x tests pass (backward compatibility)
- [ ] End-to-end smoke test: `python model_code/train.py` works with auto-loaded config
