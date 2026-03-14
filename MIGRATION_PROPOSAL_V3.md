# ML Platform Migration Proposal v3 — Unified

> **Aggregates**: MIGRATION\_PROPOSAL\_V2 (architecture + migration phases) with Model Code
> Improvement (simpler train/predict, implicit artifact definition, config-template auto-loading).
>
> **Goal**: Migrate `plat/` into `mlplatform/`, remove DAG YAML, and simultaneously simplify
> model code so that data scientists define artifact identity once, config templates drive all
> loading, and the same code path works for local dev and cloud deployment.

---

## Table of Contents

1. [Current State](#1-current-state)
2. [Design Principles](#2-design-principles)
3. [Architectural Changes](#3-architectural-changes)
4. [Model Code Improvements](#4-model-code-improvements)
5. [Migration Phases](#5-migration-phases)
6. [Architecture Diagrams](#6-architecture-diagrams)
7. [File Inventory](#7-file-inventory)
8. [pyproject.toml Changes](#8-pyprojecttoml-changes)
9. [What Gets Deleted](#9-what-gets-deleted)
10. [Recommendations](#10-recommendations)
11. [Success Criteria](#11-success-criteria)

---

## 1. Current State

### `mlplatform/` (v0.1.x stable — ~1,270 LOC)

| Module | Key Exports | Status |
|--------|------------|--------|
| `config` | `TrainingConfig`, `PredictionConfig`, `RunConfig`, `load_config_profiles` | Solid, Pydantic v2 |
| `artifacts` | `ArtifactRegistry`, `Artifact`, `create_artifacts` | Solid, format dispatch |
| `storage` | `Storage` ABC, `LocalFileSystem`, `GCSStorage`, `NoneStorage` | Solid, lazy imports |
| `utils` | `sanitize`, `to_serializable`, `save_plot`, `save_html`, `HTMLReport`, `get_logger` | Solid |

### `plat/` (legacy — ~3,550 LOC)

| Module | Key Exports | Action |
|--------|------------|--------|
| `core` | `ExecutionContext`, `BaseTrainer`, `BasePredictor`, `ArtifactRegistry`, `PredictionInputSchema` | Rethink + move |
| `config` | `load_workflow_config`, `ModelConfig`, `WorkflowConfig`, `PipelineConfig` | **Remove** DAG parsing; redesign config |
| `inference` | `InProcessInference`, `SparkBatchInference`, `FastAPIInference` | Move |
| `profiles` | `Profile`, `get_profile`, `register_profile` + 6 profiles | Move |
| `runner` | `run_workflow`, `resolve_class`, `dev_train`, `dev_predict` | **Redesign** — no DAG orchestration |
| `tracking` | `ExperimentTracker`, `NoneTracker`, `LocalJsonTracker`, `VertexAITracker` | Move; add lifecycle hooks |
| `data` | `load_prediction_input`, `write_prediction_output` | Move |
| `spark` | Spark entry point, packager, config serializer | Move; use PipelineConfig |
| `cli` | `mlplatform run`, `mlplatform build-package` | **Redesign** — args to config model |

### `model_code/` (current example)

| File | Current Pattern | Problem |
|------|----------------|---------|
| `constants.py` | `MODEL_ARTIFACT`, `SCALER_ARTIFACT`, `FEATURE_COLUMNS` | No artifact identity (model\_name, feature) |
| `train.py` | `MyTrainer(BaseTrainer)` with `self.config`, `self.artifacts`, `self.tracker` | Context injected by framework; requires DAG/runner setup |
| `predict.py` | `MyPredictor(BasePredictor)` with `self.artifacts.load()` | Same injection dependency |
| `config/` | `global.yaml`, `dev.yaml`, `gcs.yaml` | Not auto-loaded; profile not tied to identity |

---

## 2. Design Principles

1. **Define once** — Artifact identity (model\_name, feature) declared in one place; everything else loads automatically.
2. **Config templates drive loading** — YAML profiles in `model_code/config/` are merged; no DAG YAML.
3. **Simpler train and predict** — Receive config and use artifact explicitly; no magic injection.
4. **Unified local and deployment** — Same config shape for dev and prod; only profile values differ.
5. **Single frozen contract** — `PipelineConfig` is the one config model passed everywhere.
6. **Builder validates early** — Invalid combinations fail at construction, not at runtime.
7. **Single-model execution** — `execute(config)` handles one model; orchestrators handle DAGs.

---

## 3. Architectural Changes

### 3.1 No More DAG YAML Parsing

**Before**: Framework reads DAG YAML (`train.yaml`, `predict.yaml`) to discover models, config
profiles, and pipeline type. Two YAML formats (legacy + Databricks).

**After**: Framework receives **already-resolved arguments** from orchestrator or Python API.
Config validated into a **frozen Pydantic model**. Same model constructable in Python for local dev.

```
                           args (dict/kwargs)
 ┌───────────────┐    ──────────────────────────►  ┌──────────────────┐
 │  Orchestrator  │                                 │  PipelineConfig  │
 │ (VertexAI,DBX) │                                 │  (frozen model)  │
 └───────────────┘                                  └────────┬─────────┘
                                                             │
 ┌───────────────┐    Python constructor / from_dict         │
 │  Local dev     │    ──────────────────────────►           │
 │ (script/notebk)│                                          │
 └───────────────┘                                  ┌────────▼─────────┐
                                                    │  execute(config) │
 ┌───────────────┐    CLI args (parsed)             │  or dev_train()  │
 │  CLI           │    ──────────────────────────►  │                  │
 │ mlplatform run │                                 └──────────────────┘
 └───────────────┘
```

**Implications:**
- `load_workflow_config` is **removed** (no DAG parsing)
- `load_config_profiles` from v0.1.x is **kept** as a utility — users call it themselves
- Config profiles are the **user's responsibility** — they merge YAMLs and pass the result

### 3.2 PipelineConfig + Builder (Frozen Output)

**Before**: Flat `WorkflowConfig` mixing CLI args, infra, and pipeline concerns.

**After**: `PipelineConfigBuilder` validates incrementally; `.build()` produces a frozen `PipelineConfig`.

```python
from mlplatform.config import PipelineConfigBuilder

# From orchestrator (explicit)
config = (
    PipelineConfigBuilder()
    .identity(model_name="churn_model", feature="churn", version="v1.2")
    .infra(backend="gcs", bucket="ml-artifacts", project_id="my-project")
    .pipeline(pipeline_type="training", profile="cloud-train")
    .configs(["global", "train-prod"], config_dir="./config")
    .build()  # -> frozen PipelineConfig
)

# For local dev (minimal)
config = (
    PipelineConfigBuilder()
    .identity(model_name="churn_model", feature="churn")
    .pipeline(pipeline_type="training")
    .build()  # defaults: backend=local, base_path=./artifacts, profile=local, version=dev
)

# From dict (orchestrator JSON payload)
config = PipelineConfig.from_dict({
    "model_name": "churn_model",
    "feature": "churn",
    "pipeline_type": "training",
    "config_list": ["global", "dev"],
})
```

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
    module: str = ""  # e.g. "model_code.train:MyTrainer"

    # Config loading
    config_list: list[str] = Field(default_factory=lambda: ["global", "dev"])
    config_dir: str = "./config"

    # Merged user config (from YAML profiles or direct dict)
    user_config: dict[str, Any] = Field(default_factory=dict)

    # Metadata
    commit_hash: str | None = None
    log_level: str = "INFO"
```

**Builder validation rules:**
- `backend=gcs` requires `base_bucket` (or `base_bucket` in `user_config`)
- `profile` starting with `cloud-` requires `backend=gcs`
- `model_name` and `feature` are always required

### 3.3 Unified ArtifactRegistry

**Before**: Two `ArtifactRegistry` classes — `mlplatform/artifacts/` (format dispatch) and
`plat/core/` (path conventions, cross-model loading, storage property).

**After**: Merge into one. The existing `mlplatform/artifacts/registry.py` gains:
- `storage` property exposing the underlying backend
- `load(name, *, model_name=None, version=None)` for cross-model loading

### 3.4 Provider-Agnostic Tracking

Same `ExperimentTracker` ABC with added `start_run`/`end_run` lifecycle hooks (no-op defaults).
Context manager protocol for scoped runs.

```python
class ExperimentTracker(ABC):
    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None: ...
    @abstractmethod
    def log_metrics(self, metrics: dict[str, float]) -> None: ...
    @abstractmethod
    def log_artifact(self, name: str, artifact: Any) -> None: ...

    def start_run(self, run_name: str | None = None) -> None: ...   # no-op default
    def end_run(self) -> None: ...                                    # no-op default

    def __enter__(self): ...
    def __exit__(self, *exc): ...
```

### 3.5 Simplified Runner

**Before**: `run_workflow` parses DAG, iterates models, builds contexts.

**After**: `execute(config: PipelineConfig)` — takes a frozen config, resolves profile, builds
context, runs training or prediction. Single model per invocation.

```python
from mlplatform.runner import execute, dev_train, dev_predict

result = execute(config)

# Local dev shortcuts
ctx = dev_train(
    model_name="churn_model",
    feature="churn",
    trainer_class=MyTrainer,
)
```

### 3.6 ExecutionContext.from\_config

**Before**: `from_profile` takes 10+ scattered args.

**After**: `from_config(config: PipelineConfig, profile: Profile)` — all args already
validated in the frozen config.

```python
@classmethod
def from_config(
    cls,
    config: PipelineConfig,
    profile: Profile,
    extra_overrides: dict[str, Any] | None = None,
) -> "ExecutionContext": ...
```

---

## 4. Model Code Improvements

This is the **key addition** over V2 — making model code simpler for data scientists.

### 4.1 Implicit Artifact Definition

DS defines artifact identity **once** in `model_code/constants.py`:

```python
# model_code/constants.py
MODEL_ARTIFACT = "model.pkl"
SCALER_ARTIFACT = "scaler.pkl"
FEATURE_COLUMNS = ["f0", "f1", "f2", "f3", "f4"]

# NEW: artifact identity defined once
ARTIFACT_IDENTITY = {
    "model_name": "churn_model",
    "feature": "churn",
}
# version defaults to "dev" locally; overridden by orchestrator in prod
```

Or equivalently in `model_code/config/global.yaml`:

```yaml
# model_code/config/global.yaml
model_name: churn_model
feature: churn
base_path: ./artifacts
backend: local
log_level: INFO
profile: local
```

**Rule**: Identity is defined in exactly one place. `load_model_config()` reads it from
the config template. The orchestrator can override version, backend, etc. at deployment time.

### 4.2 Config Template Auto-Loading

A new helper `load_model_config()` handles the common case:

```python
def load_model_config(
    config_list: list[str] | None = None,
    config_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Load and merge model config from YAML profiles.

    Resolution order:
    1. config_list param (if provided)
    2. MLPLATFORM_CONFIG env var (e.g. "global,dev")
    3. Default: ["global", "dev"]

    config_dir resolution:
    1. config_dir param (if provided)
    2. MLPLATFORM_CONFIG_DIR env var
    3. Default: "./config" relative to caller or model_code/config/
    """
    ...
```

**Config profile structure:**

```
model_code/
├── config/
│   ├── global.yaml   # model_name, feature, base_path, backend, log_level
│   ├── dev.yaml      # train_data_path, hyperparameters, base_path override
│   ├── prod.yaml     # backend: gcs, base_bucket, project_id
│   └── local.yaml    # local overrides
```

**Merge chain**: `global.yaml` -> `dev.yaml` (deep merge, later overrides earlier).

The merged config dict is compatible with both `TrainingConfig(**cfg)` /
`PredictionConfig(**cfg)` and `PipelineConfig.from_dict(cfg)`.

### 4.3 Simpler Train Flow

**Option A — Function-based (no BaseTrainer):**

For simple models where DS doesn't need the full framework:

```python
# model_code/train.py (function-based)
from mlplatform.config import TrainingConfig, load_model_config
from mlplatform.artifacts import Artifact

def train(config: dict | None = None):
    """Train a model. Works locally and in deployment."""
    cfg = config or load_model_config()
    training_cfg = TrainingConfig(cfg)
    artifact = Artifact(**training_cfg.to_artifact_kwargs())

    # Load data
    data_path = cfg.get("train_data_path", "model_code/data/sample_train.csv")
    df = pd.read_csv(data_path)
    X = df[FEATURE_COLUMNS]
    y = df["target"]

    # Train
    model = LogisticRegression(**cfg.get("hyperparameters", {}))
    model.fit(X, y)

    # Save
    artifact.save("model.pkl", model)
    return artifact

if __name__ == "__main__":
    train()  # auto-loads ["global", "dev"] from model_code/config/
```

**Option B — BaseTrainer with config-driven context (recommended for complex models):**

```python
# model_code/train.py (class-based, config-driven)
from mlplatform.core.trainer import BaseTrainer
import model_code.constants as cons

class MyTrainer(BaseTrainer):
    def train(self) -> None:
        # self.config comes from merged config template + orchestrator overrides
        data_path = self.config.get("train_data_path", "model_code/data/sample_train.csv")
        df = pd.read_csv(data_path)
        X = df[cons.FEATURE_COLUMNS]
        y = df["target"]

        hyperparams = self.config.get("hyperparameters", {})
        model = LogisticRegression(**hyperparams)
        model.fit(X, y)

        self.tracker.log_params({"model_type": "LogisticRegression", **hyperparams})
        self.artifacts.save(cons.MODEL_ARTIFACT, model)

if __name__ == "__main__":
    from mlplatform.config import PipelineConfig
    from mlplatform.runner import dev_train

    # Config built from template — no DAG needed
    config = PipelineConfig.from_dict({
        "model_name": "churn_model",
        "feature": "churn",
        "pipeline_type": "training",
        "config_list": ["global", "dev"],
    })
    dev_train(config)
```

### 4.4 Simpler Predict Flow

Same patterns as train:

**Function-based:**

```python
# model_code/predict.py (function-based)
from mlplatform.config import PredictionConfig, load_model_config
from mlplatform.artifacts import Artifact

def predict(data, config: dict | None = None):
    """Load model and predict. Works locally and in deployment."""
    cfg = config or load_model_config()
    pred_cfg = PredictionConfig(cfg)
    artifact = Artifact(**pred_cfg.to_artifact_kwargs())

    model = artifact.load("model.pkl")
    scaler = artifact.load("scaler.pkl")

    df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
    X_scaled = scaler.transform(df[FEATURE_COLUMNS])
    return df.assign(prediction=model.predict(X_scaled))
```

**Class-based (recommended):**

```python
# model_code/predict.py (class-based, config-driven)
from mlplatform.core.predictor import BasePredictor
import model_code.constants as cons

class MyPredictor(BasePredictor):
    def load_model(self):
        self._model = self.artifacts.load(cons.MODEL_ARTIFACT)
        self._scaler = self.artifacts.load(cons.SCALER_ARTIFACT)
        return self._model

    def predict(self, data):
        df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
        X_scaled = self._scaler.transform(df[cons.FEATURE_COLUMNS])
        return df.assign(prediction=self._model.predict(X_scaled))

if __name__ == "__main__":
    from mlplatform.config import PipelineConfig
    from mlplatform.runner import dev_predict

    config = PipelineConfig.from_dict({
        "model_name": "churn_model",
        "feature": "churn",
        "version": "v1.0",
        "pipeline_type": "prediction",
        "config_list": ["global", "dev"],
    })
    result = dev_predict(config)
```

### 4.5 Config Template Structure (Target State)

```
model_code/
├── config/
│   ├── global.yaml       # model_name, feature, base_path, backend, log_level
│   ├── dev.yaml           # train_data_path, hyperparameters, base_path override
│   ├── prod.yaml          # backend: gcs, base_bucket, project_id
│   └── predict-dev.yaml   # prediction-specific local overrides
├── constants.py           # ARTIFACT_IDENTITY, FEATURE_COLUMNS, artifact names
├── train.py               # MyTrainer or train() function
├── predict.py             # MyPredictor or predict() function
├── evaluate.py            # evaluation helpers
├── utils.py               # model-specific utilities
├── data/                  # sample data for local dev
└── tests/                 # model tests
```

### 4.6 Local vs Deployment Parity

| Concern | Local | Deployment |
|---------|-------|------------|
| Config source | `load_model_config()` -> `["global","dev"]` | Orchestrator passes `PipelineConfig` (with `config_list=["global","prod"]`) |
| Artifact | From constants + config merge | Same; version/backend overridden by orchestrator |
| Storage | `base_path=./dev_artifacts` (local) | `base_bucket`, `project_id` from prod profile |
| Config list | Default `["global","dev"]` or `MLPLATFORM_CONFIG` env | Orchestrator sets `config_list` in PipelineConfig |
| Profile | `local` (default) | `cloud-train`, `cloud-batch`, etc. |
| Version | `dev` (default) | Orchestrator sets explicit version |

**Key insight**: The DS writes code once. The only things that change between local and
deployment are config profile values — the code path is identical.

---

## 5. Migration Phases

### Phase 1: Config Redesign (foundation)

**What changes:**
- **Remove** `load_workflow_config` (DAG parser)
- **Keep** `load_config_profiles` as utility
- **Keep** `TrainingConfig`, `PredictionConfig`, `RunConfig` from v0.1.x
- **Add** `PipelineConfig` — frozen model built from orchestrator/CLI args
- **Add** `PipelineConfigBuilder` — builder with incremental validation
- **Add** `load_model_config()` — convenience wrapper around `load_config_profiles` with env var + defaults

**PipelineConfigBuilder:**

```python
class PipelineConfigBuilder:
    def identity(self, *, model_name: str, feature: str,
                 version: str = "dev") -> Self: ...
    def infra(self, *, backend: str = "local", base_path: str = "./artifacts",
              base_bucket: str | None = None, project_id: str | None = None) -> Self: ...
    def pipeline(self, *, pipeline_type: str, profile: str = "local",
                 platform: str = "VertexAI", module: str = "") -> Self: ...
    def configs(self, profile_names: list[str],
                config_dir: str = "./config") -> Self: ...
    def user_config(self, config: dict[str, Any]) -> Self: ...
    def metadata(self, *, commit_hash: str | None = None,
                 log_level: str = "INFO") -> Self: ...
    def build(self) -> PipelineConfig: ...
```

**`PipelineConfig.from_dict` classmethod:**

```python
@classmethod
def from_dict(cls, payload: dict[str, Any]) -> "PipelineConfig":
    """Construct from a flat dict (e.g. orchestrator JSON payload).

    If 'config_list' and 'config_dir' are present, loads and merges YAML
    profiles into user_config before constructing the frozen model.
    """
    data = dict(payload)
    config_list = data.pop("config_list", None)
    config_dir = data.pop("config_dir", "./config")
    if config_list:
        merged = load_config_profiles(config_list, config_dir)
        # Identity fields in YAML feed into top-level fields
        for key in ("model_name", "feature", "version", "base_path",
                     "base_bucket", "backend", "project_id", "profile",
                     "log_level"):
            if key in merged and key not in data:
                data[key] = merged.pop(key)
        data.setdefault("user_config", {}).update(merged)
    return cls(**data)
```

**`load_model_config` helper:**

```python
# mlplatform/config/loader.py (addition)
def load_model_config(
    config_list: list[str] | None = None,
    config_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Load and merge model config from YAML profiles.

    Resolves config_list from param > MLPLATFORM_CONFIG env > ["global", "dev"].
    Resolves config_dir from param > MLPLATFORM_CONFIG_DIR env > "./config".
    """
    import os
    resolved_list = config_list or os.environ.get(
        "MLPLATFORM_CONFIG", "global,dev"
    ).split(",")
    resolved_dir = config_dir or os.environ.get(
        "MLPLATFORM_CONFIG_DIR", "./config"
    )
    return load_config_profiles(
        [n.strip() for n in resolved_list],
        resolved_dir,
    )
```

**Files:**
- Modify: `mlplatform/config/models.py` — add `PipelineConfig`
- New: `mlplatform/config/builder.py` — `PipelineConfigBuilder`
- Modify: `mlplatform/config/loader.py` — add `load_model_config`

---

### Phase 2: Tracking (provider-agnostic)

**What changes:**
- Move `plat/tracking/` -> `mlplatform/tracking/`
- Add `start_run`/`end_run` lifecycle hooks (no-op defaults)
- Context manager protocol

**Implementations:**

| Tracker | `log_params` | `log_metrics` | `start_run`/`end_run` |
|---------|-------------|--------------|----------------------|
| `NoneTracker` | no-op | no-op | no-op |
| `LocalJsonTracker` | JSON file | JSON file | no-op |
| `VertexAITracker` | Vertex Experiments | Vertex Experiments | creates/closes run |

**Files:**
- New: `mlplatform/tracking/__init__.py`, `base.py`, `none.py`, `local.py`, `vertexai.py`

---

### Phase 3: Core (rethink + simplify)

**What changes:**
- Merge two `ArtifactRegistry` classes into one (extend `mlplatform/artifacts/registry.py`)
- Move `ExecutionContext` with `from_config` factory (replaces `from_profile`)
- Keep `BaseTrainer` / `BasePredictor` (same contract)
- Move `PredictionInputSchema`

**ArtifactRegistry additions:**

```python
class ArtifactRegistry:
    # ... existing save/load with format dispatch ...

    @property
    def storage(self) -> Storage:
        return self._storage

    def load(self, name: str, *, model_name: str | None = None,
             version: str | None = None) -> Any:
        """Load artifact. Override model_name/version for cross-model loading."""
        ...
```

**ExecutionContext:**

```python
@dataclass
class ExecutionContext:
    artifacts: ArtifactRegistry
    experiment_tracker: ExperimentTracker
    feature_name: str
    model_name: str
    version: str
    user_config: dict[str, Any]       # renamed from optional_configs
    log: logging.Logger
    pipeline_type: str = ""
    commit_hash: str | None = None

    @classmethod
    def from_config(cls, config: PipelineConfig, profile: Profile,
                    extra_overrides: dict[str, Any] | None = None) -> "ExecutionContext":
        """Single canonical constructor. Replaces from_profile."""
        ...
```

**BaseTrainer / BasePredictor**: Same contract. Keep `self.artifacts`, `self.tracker`,
`self.config`, `self.log`. The `self.config` property now returns `user_config` from the
`ExecutionContext`, which is the merged config template + any orchestrator overrides.

**Files:**
- Modify: `mlplatform/artifacts/registry.py`
- New: `mlplatform/core/__init__.py`, `context.py`, `trainer.py`, `predictor.py`, `prediction_schema.py`

---

### Phase 4: Profiles

Move `plat/profiles/` -> `mlplatform/profiles/`. No architectural changes — the profile
system is well-designed.

**Files:**
- New: `mlplatform/profiles/__init__.py`, `registry.py`

---

### Phase 5: Inference Strategies

Move `plat/inference/` -> `mlplatform/inference/`. Signature change: receives `PipelineConfig`
instead of `ModelConfig`.

```python
class InferenceStrategy(ABC):
    @abstractmethod
    def invoke(self, predictor: BasePredictor, context: ExecutionContext,
               config: PipelineConfig) -> Any: ...
```

**Files:**
- New: `mlplatform/inference/__init__.py`, `base.py`, `in_process.py`, `spark_batch.py`, `fastapi_serving.py`

---

### Phase 6: Data I/O

Move `plat/data/` -> `mlplatform/data/`. Adapt to use `PipelineConfig`.

**Files:**
- New: `mlplatform/data/__init__.py`, `io.py`

---

### Phase 7: Runner (simplified)

**What changes:**
- **Remove** `run_workflow` (DAG orchestration is the orchestrator's job)
- **Add** `execute(config: PipelineConfig)` — single-model execution
- **Redesign** `dev_train`, `dev_predict` — accept `PipelineConfig` or use builder internally
- **Keep** `resolve_class` for module:Class resolution

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


def dev_train(config: PipelineConfig) -> ExecutionContext:
    """Local dev training from a PipelineConfig."""
    ...

def dev_predict(config: PipelineConfig, data=None) -> Any:
    """Local dev prediction from a PipelineConfig."""
    ...
```

**Integration with model code auto-loading:**

```python
# dev_train can also accept minimal args and build config from template
def dev_train(
    config: PipelineConfig | None = None,
    *,
    model_name: str | None = None,
    feature: str | None = None,
    trainer_class: type[BaseTrainer] | None = None,
    version: str = "dev",
    config_list: list[str] | None = None,
    config_dir: str = "./config",
) -> ExecutionContext:
    """Convenience for local training.

    If config is provided, uses it directly.
    Otherwise, builds from args + config template auto-loading.
    """
    if config is None:
        merged = load_model_config(config_list, config_dir)
        config = PipelineConfig.from_dict({
            "model_name": model_name or merged.get("model_name"),
            "feature": feature or merged.get("feature"),
            "version": version,
            "pipeline_type": "training",
            "user_config": merged,
        })
    ...
```

**Files:**
- New: `mlplatform/runner/__init__.py`, `execute.py`, `resolve.py`, `dev.py`

---

### Phase 8: Spark

Move `plat/spark/` -> `mlplatform/spark/`. Spark entry point receives serialized
`PipelineConfig` (JSON) instead of DAG path.

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="JSON PipelineConfig")
    args = parser.parse_args()
    config = PipelineConfig.model_validate_json(args.config)
    # ... Spark session, mapInPandas, etc.
```

**Files:**
- New: `mlplatform/spark/__init__.py`, `main.py`, `packager.py`, `config_serializer.py`

---

### Phase 9: CLI (thin wrapper)

CLI parses args -> builds `PipelineConfig` via builder -> calls `execute(config)`.

```
mlplatform run \
  --model-name churn_model \
  --feature churn \
  --version v1.2 \
  --pipeline-type training \
  --profile local \
  --config global,dev \
  --config-dir ./config \
  --module model_code.train:MyTrainer
```

**Files:**
- New: `mlplatform/cli/__init__.py`, `main.py`
- Modify: `pyproject.toml` — add `[project.scripts]`

---

### Phase 10: Model Code Refactor

**New phase** (after framework migration is complete) — update `model_code/` to use
the new patterns.

**What changes:**
- Add `ARTIFACT_IDENTITY` to `model_code/constants.py`
- Update `model_code/config/global.yaml` with artifact identity fields
- Refactor `model_code/train.py` to use config template (no DAG)
- Refactor `model_code/predict.py` similarly
- Update `model_code/config/dev.yaml` with hyperparameters section
- Add `model_code/config/prod.yaml` for cloud deployment

**Updated `model_code/constants.py`:**

```python
"""Model constants — artifact identity and feature definitions."""

MODEL_ARTIFACT = "model.pkl"
SCALER_ARTIFACT = "scaler.pkl"
FEATURE_COLUMNS = ["f0", "f1", "f2", "f3", "f4"]

# Artifact identity — defined once, used everywhere
ARTIFACT_IDENTITY = {
    "model_name": "churn_model",
    "feature": "churn",
}
```

**Updated `model_code/config/global.yaml`:**

```yaml
# Global config — baseline for all environments
model_name: churn_model
feature: churn
base_path: ./artifacts
backend: local
log_level: INFO
profile: local
```

**Updated `model_code/config/dev.yaml`:**

```yaml
# Dev config — local development overrides
log_level: DEBUG
base_path: ./dev_artifacts
profile: local
train_data_path: model_code/data/sample_train.csv

hyperparameters:
  max_iter: 1000
  random_state: 42
```

**New `model_code/config/prod.yaml`:**

```yaml
# Prod config — cloud deployment settings
backend: gcs
base_bucket: ml-artifacts-prod
project_id: my-gcp-project
profile: cloud-train
log_level: INFO

hyperparameters:
  max_iter: 2000
  random_state: 42
```

**Updated `model_code/train.py`:**

```python
"""Training: MyTrainer — config-driven, no DAG dependency."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mlplatform.core.trainer import BaseTrainer
import model_code.constants as cons
from model_code.evaluate import evaluate


class MyTrainer(BaseTrainer):
    """Train a model from merged config template."""

    def train(self) -> None:
        # Data
        train_data = self.config.get("train_data")
        if train_data is not None:
            df = pd.concat(
                [train_data["X"], train_data["y"].rename("target")], axis=1
            )
        else:
            data_path = self.config.get(
                "train_data_path", "model_code/data/sample_train.csv"
            )
            df = pd.read_csv(data_path)

        X = df[cons.FEATURE_COLUMNS]
        y = df["target"]

        # Split + scale
        test_size = self.config.get("test_size", 0.2)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Train
        hyperparams = self.config.get("hyperparameters", {})
        max_iter = hyperparams.get("max_iter", 1000)
        model = LogisticRegression(max_iter=max_iter, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Evaluate + log
        val_df = pd.DataFrame(X_val, columns=cons.FEATURE_COLUMNS)
        val_df["target"] = y_val.values
        metrics = evaluate(model, scaler, val_df)
        self.log.info("Validation metrics: %s", metrics)
        self.tracker.log_metrics(metrics)
        self.tracker.log_params(
            {"model_type": "LogisticRegression", "max_iter": max_iter}
        )

        # Save
        self.artifacts.save(cons.MODEL_ARTIFACT, model)
        self.artifacts.save(cons.SCALER_ARTIFACT, scaler)


if __name__ == "__main__":
    from mlplatform.config import PipelineConfig
    from mlplatform.runner import dev_train

    # Config from template — no DAG, no scattered args
    config = PipelineConfig.from_dict({
        "model_name": "churn_model",
        "feature": "churn",
        "pipeline_type": "training",
        "config_list": ["global", "dev"],
        "config_dir": "model_code/config",
    })
    dev_train(config)
```

**Files:**
- Modify: `model_code/constants.py`
- Modify: `model_code/config/global.yaml`
- Modify: `model_code/config/dev.yaml`
- New: `model_code/config/prod.yaml`
- Modify: `model_code/train.py`
- Modify: `model_code/predict.py`

---

## 6. Architecture Diagrams

### 6.1 Training Flow

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                 Cloud / Orchestrator                     │
                    │  ┌──────────┐                                           │
                    │  │Orchestrat│──PipelineConfig──►┌────────────────────┐  │
                    │  │   or     │                    │  BaseTrainer       │  │
                    │  └──────────┘                    │  ┌──────────────┐  │  │
Config template ─────────────────────────────────►    │  │  Train Code  │  │  │
(YAML profiles)                                       │  │ (user impl)  │  │  │
                    │                                 │  └──────────────┘  │  │
                    │                                 └────────────────────┘  │
                    │                              │              │           │
                    │                    ┌─────────▼──┐   ┌──────▼───────┐   │
                    │                    │  Artifact   │   │  Tracking    │   │
                    │                    │  Registry   │   │  Interface   │   │
                    │                    └──────┬──────┘   └──────┬───────┘   │
                    │                           │                 │           │
                    │                    ┌──────▼──────┐   ┌──────▼───────┐   │
                    │                    │  Storage    │   │  Tracker     │   │
                    │                    │  Interface  │   │  Impls       │   │
                    │                    └──────┬──────┘   └──────────────┘   │
                    │                 ┌─────────┴─────────┐                   │
                    │          ┌──────▼─────┐  ┌──────────▼──┐               │
                    │          │Local Stor. │  │ GCS Storage  │               │
                    │          └────────────┘  └─────────────┘               │
                    └─────────────────────────────────────────────────────────┘
                              ▲
              PipelineConfig  │  (local dev: from_dict + config template)
              ────────────────┘
```

### 6.2 Config Flow (NEW — model code perspective)

```
 ┌─────────────────────┐
 │ model_code/config/   │
 │  global.yaml         │  ──► load_model_config()  ──┐
 │  dev.yaml            │      or load_config_profiles │
 │  prod.yaml           │                              │
 └─────────────────────┘                              │
                                                       ▼
 ┌─────────────────────┐                    ┌─────────────────────┐
 │ model_code/          │                    │  Merged config dict │
 │  constants.py        │  ARTIFACT_IDENTITY │  (flat key-value)   │
 │  (identity defined)  │  ────────────────► │                     │
 └─────────────────────┘                    └──────────┬──────────┘
                                                       │
                                            ┌──────────▼──────────┐
                                            │  PipelineConfig     │
                          PipelineConfig    │  .from_dict()       │
  ┌──────────────┐      .from_dict() or    │  or Builder         │
  │ Orchestrator │  ──► builder overrides  │                     │
  │ (prod only)  │  ──────────────────────►│  (frozen, validated)│
  └──────────────┘                          └──────────┬──────────┘
                                                       │
                                            ┌──────────▼──────────┐
                                            │  execute(config)    │
                                            │  or dev_train()     │
                                            └─────────────────────┘
```

### 6.3 Module Dependency Graph

```
                            ┌───────────────────────────┐
                            │        CLI (thin)          │
                            │  args -> PipelineConfig    │
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
                │  get_profile   │  │ ExecContext │  │  InProcess   │
                │  Profile{}     │  │ BaseTrainer │  │  SparkBatch  │
                │                │  │ BasePredict │  │  FastAPI     │
                └───┬───────┬───┘  └──┬──────┬───┘  └──┬───────┬──┘
                    │       │         │      │         │       │
           ┌────────▼───┐ ┌─▼─────────▼┐  ┌─▼─────────▼─┐  ┌─▼──────┐
           │  Tracking   │ │  Storage    │  │   Config     │  │  Data  │
           │ Tracker ABC │ │ Storage ABC │  │ PipelineConf │  │  I/O   │
           │ None/Local  │ │ Local/GCS   │  │ Builder      │  │        │
           │ VertexAI    │ │ None        │  │ Training/    │  │        │
           │ (MLflow...) │ │             │  │ Prediction   │  │        │
           └─────────────┘ └─────────────┘  └──────────────┘  └────────┘
                                 │                │
                           ┌─────▼────────────────▼───┐
                           │         Utils             │
                           │  serialization, logging   │
                           │  storage_helpers, reports  │
                           └───────────────────────────┘
                                 │
                           ┌─────▼─────────────────────┐
                           │      Artifacts             │
                           │  (unified registry)        │
                           └───────────────────────────┘
```

---

## 7. File Inventory

### New Files (~17 files)

```
mlplatform/
├── config/
│   └── builder.py                    # PipelineConfigBuilder
├── core/
│   ├── __init__.py
│   ├── context.py                    # ExecutionContext (from_config)
│   ├── trainer.py                    # BaseTrainer
│   ├── predictor.py                  # BasePredictor
│   └── prediction_schema.py          # PredictionInputSchema
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
│   └── fastapi_serving.py            # FastAPIInference
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
│   └── config_serializer.py          # PipelineConfig <-> JSON
└── cli/
    ├── __init__.py
    └── main.py                       # mlplatform run, build-package
```

### Modified Files (~5 files)

| File | Change |
|------|--------|
| `mlplatform/artifacts/registry.py` | Add `storage` property + `load(..., model_name, version)` |
| `mlplatform/config/models.py` | Add `PipelineConfig` frozen model |
| `mlplatform/config/loader.py` | Add `load_model_config()` helper |
| `model_code/constants.py` | Add `ARTIFACT_IDENTITY` |
| `pyproject.toml` | CLI entry point + optional deps |

### Model Code Updates (Phase 10)

| File | Change |
|------|--------|
| `model_code/constants.py` | Add `ARTIFACT_IDENTITY` dict |
| `model_code/config/global.yaml` | Add `model_name`, `feature` fields |
| `model_code/config/dev.yaml` | Add `hyperparameters` section |
| `model_code/config/prod.yaml` | **New** — cloud deployment config |
| `model_code/train.py` | Use `PipelineConfig.from_dict` in `__main__` |
| `model_code/predict.py` | Use `PipelineConfig.from_dict` in `__main__` |

---

## 8. pyproject.toml Changes

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
all = [
    "mlplatform[core]", "mlplatform[tracking]", "mlplatform[spark]",
    "mlplatform[serving]", "mlplatform[bigquery]", "mlplatform[parquet]",
]

[project.scripts]
mlplatform = "mlplatform.cli.main:main"
```

---

## 9. What Gets Deleted

After migration completes and all tests pass:

| plat/ module | Action |
|-------------|--------|
| `plat/config/loader.py` | **Delete** — DAG parsing removed |
| `plat/config/models.py` | **Delete** — replaced by PipelineConfig |
| `plat/core/*` | **Delete** — moved to mlplatform/core/ |
| `plat/tracking/*` | **Delete** — moved to mlplatform/tracking/ |
| `plat/inference/*` | **Delete** — moved to mlplatform/inference/ |
| `plat/profiles/*` | **Delete** — moved to mlplatform/profiles/ |
| `plat/runner/*` | **Delete** — replaced by mlplatform/runner/ |
| `plat/data/*` | **Delete** — moved to mlplatform/data/ |
| `plat/spark/*` | **Delete** — moved to mlplatform/spark/ |
| `plat/cli/*` | **Delete** — replaced by mlplatform/cli/ |
| `plat/artifacts/*` | **Delete** — merged into mlplatform/artifacts/ |

---

## 10. Recommendations

### R1: Config profiles are opt-in
Users who want YAML config merging call `load_config_profiles` or `load_model_config`
themselves. The framework doesn't auto-discover config directories.

### R2: Single-model execution
`execute(config)` handles one model. Orchestrators handle multi-model DAGs. The framework
does one thing well.

### R3: PipelineConfig is the contract
Every component receives the same frozen `PipelineConfig`. No more scattered args through
3-4 function signatures. Easier to test (just construct a config).

### R4: Backward compatibility
v0.1.x public API (`TrainingConfig`, `PredictionConfig`, `RunConfig`, `create_artifacts`,
`ArtifactRegistry`, all storage/utils) remains **unchanged**. New modules are additive.

### R5: load\_model\_config for DS ergonomics
The `load_model_config()` helper with env-var fallbacks makes it trivial for DS to load
config in scripts and notebooks without importing the builder.

### R6: ARTIFACT\_IDENTITY as the source of truth
By putting identity in `constants.py`, it's importable from tests, notebooks, and scripts.
The config template (`global.yaml`) can also carry it — `load_model_config` returns both.

### R7: Function-based and class-based coexist
Simple models use function-based train/predict (no BaseTrainer). Complex models keep
BaseTrainer/BasePredictor for lifecycle hooks, tracker integration, and inference strategies.
Both patterns use the same `PipelineConfig` and artifact system.

### R8: Test strategy
Each phase should include:
- Unit tests for the new module
- Integration test with existing v0.1.x code
- Smoke test: `dev_train` / `dev_predict` end-to-end with a toy model

---

## 11. Success Criteria

| Criterion | How to verify |
|-----------|--------------|
| DS defines artifact identity once | `ARTIFACT_IDENTITY` in constants.py or `model_name`/`feature` in global.yaml; no other source of identity |
| Config template drives all loading | `load_model_config()` returns everything needed; no DAG YAML |
| Same code path for local and deployment | `train.py` / `predict.py` identical; only config profile values differ |
| No DAG YAML dependency | `load_workflow_config` deleted; no YAML with `pipeline:` or `models:` keys |
| Train and predict are simpler | Model code uses `PipelineConfig.from_dict` or `load_model_config()`; no framework bootstrap boilerplate |
| Builder validates early | `PipelineConfigBuilder.build()` catches invalid combinations before execution |
| Frozen config everywhere | `PipelineConfig` is immutable; no mutation after construction |
| v0.1.x backward compatible | Existing `TrainingConfig`, `Artifact`, storage, utils unchanged |

---

## Migration Order & Dependencies

```
Phase 1: Config ─────────────┐
                              ▼
Phase 2: Tracking ────► Phase 3: Core ────► Phase 4: Profiles
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
                                       ┌──────────┘
                                       ▼
                                Phase 10: Model Code Refactor
```

Each phase can be independently tested before moving to the next.
