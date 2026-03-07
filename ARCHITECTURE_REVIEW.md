# MLOps SDK — Architecture Review

> **Date:** 2026-03-07
> **Scope:** Full architectural review of the `mlplatform` SDK — domain boundaries, modularization, storage/logging design, runtime abstractions, and incremental release strategy.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Domain Definition](#2-domain-definition)
3. [Proposed Architectural Layers](#3-proposed-architectural-layers)
4. [Recommended Module Structure](#4-recommended-module-structure)
5. [Storage and Logging Architecture](#5-storage-and-logging-architecture)
6. [Execution and Runtime Abstractions](#6-execution-and-runtime-abstractions)
7. [Incremental Release Strategy](#7-incremental-release-strategy)
8. [Potential Anti-Patterns](#8-potential-anti-patterns)
9. [Concrete Recommendations](#9-concrete-recommendations)

---

## 1. Executive Summary

The `mlplatform` SDK is a well-intentioned MLOps framework that aims to let data scientists write code once and run it across local, batch, distributed (Spark/Dataproc), and online (REST) environments. The codebase already demonstrates several good architectural instincts:

- **Abstract interfaces** for Storage, ExperimentTracker, and InvocationStrategy
- **Profile-based infrastructure resolution** that decouples user code from deployment targets
- **Clean separation** between user-facing abstractions (`BaseTrainer`, `BasePredictor`) and infrastructure plumbing
- **Optional-dependency extras** in `pyproject.toml` for incremental installation

However, the current structure has several areas that would benefit from refinement before the SDK matures beyond v0.1.x. This review identifies domain boundaries, proposes a layered architecture, and provides concrete recommendations to improve modularity, testability, and extensibility.

---

## 2. Domain Definition

### 2.1 Core Domain

The core domain of this SDK is **ML workflow orchestration** — the coordination of training and prediction workflows across heterogeneous execution environments. This is the unique value proposition: same user code, any runtime.

The core domain comprises:

| Concept | Current Location | Role |
|---|---|---|
| **Trainer contract** | `core/trainer.py` | Defines what "training" means to the framework |
| **Predictor contract** | `core/predictor.py` | Defines what "prediction" means to the framework |
| **Execution context** | `core/context.py` | Unified service bag injected into user code |
| **Artifact registry** | `core/artifact_registry.py` | Path convention + persistence delegation |
| **Workflow orchestration** | `runner.py` | Resolves profiles, builds contexts, runs workflows |

### 2.2 Supporting Domains

| Domain | Purpose | Current Modules |
|---|---|---|
| **Configuration** | YAML loading, validation, profile merging | `config/` |
| **Storage** | Artifact persistence backends | `storage/` |
| **Experiment Tracking** | Parameter/metric/artifact logging | `tracking/` |
| **Invocation** | Execution strategy for prediction | `invocation/` |
| **Runtime / Spark** | Distributed execution, packaging | `spark/` |
| **Profiles** | Infrastructure bundle resolution | `profiles/` |
| **Data I/O** | Prediction data loading/writing | `data_io.py` |
| **Schema Validation** | Input validation for prediction | `schema.py` |

### 2.3 Utility Domain

| Concern | Current Location |
|---|---|
| Serialization helpers | `utils/serialization.py` |
| Storage upload helpers | `utils/storage_helpers.py` |
| Framework logging setup | `log.py` |

### 2.4 Domain Boundary Assessment

The domain boundaries are **mostly sound**, but several cross-cutting concerns leak across boundaries:

1. **`runner.py`** conflates orchestration logic with context construction and class resolution. It sits at the root level despite being the most important module in the core domain.
2. **`data_io.py`** is a data ingestion concern used only by `invocation/in_process.py`, yet lives at the package root.
3. **`schema.py`** is a prediction-specific validation concern that lives at the root level rather than within `core/` or a dedicated `validation/` subpackage.
4. **`log.py`** is a thin wrapper over `logging.getLogger` — its placement at the root is fine for a utility, but it conflates Python logging configuration with the SDK's experiment tracking "logging" (parameters, metrics, artifacts).

---

## 3. Proposed Architectural Layers

The SDK should be organized into four clean layers, with dependencies flowing strictly downward:

```
┌─────────────────────────────────────────────────────────┐
│                    User / Data Scientist                │
│   (BaseTrainer subclass, BasePredictor subclass, CLI)   │
└─────────────────────┬───────────────────────────────────┘
                      │ uses
┌─────────────────────▼───────────────────────────────────┐
│              APPLICATION LAYER (Orchestration)          │
│                                                         │
│  runner.py — workflow orchestration                      │
│  cli/      — command-line entry points                  │
│  profiles/ — infrastructure bundle resolution           │
└─────────────────────┬───────────────────────────────────┘
                      │ depends on
┌─────────────────────▼───────────────────────────────────┐
│              CORE DOMAIN LAYER (Contracts)              │
│                                                         │
│  core/trainer.py      — BaseTrainer ABC                 │
│  core/predictor.py    — BasePredictor ABC               │
│  core/context.py      — ExecutionContext                │
│  core/artifact_registry.py — ArtifactRegistry           │
│  core/schema.py       — PredictionInputSchema           │
│  config/              — Configuration models + loading  │
└─────────────────────┬───────────────────────────────────┘
                      │ depends on
┌─────────────────────▼───────────────────────────────────┐
│           PLATFORM SERVICES LAYER (Abstractions)        │
│                                                         │
│  storage/base.py     — Storage ABC                      │
│  tracking/base.py    — ExperimentTracker ABC            │
│  invocation/base.py  — InvocationStrategy ABC           │
│  data_io.py          — Data loading/writing             │
└─────────────────────┬───────────────────────────────────┘
                      │ depends on
┌─────────────────────▼───────────────────────────────────┐
│          INFRASTRUCTURE ADAPTERS (Implementations)      │
│                                                         │
│  storage/local.py, storage/gcs.py                       │
│  tracking/local.py, tracking/vertexai.py, tracking/none │
│  invocation/in_process.py, spark_batch.py, fastapi.py   │
│  spark/   — packaging, config serialization, entry point│
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│              UTILITIES (Cross-cutting)                  │
│                                                         │
│  utils/serialization.py   — sanitize, to_serializable   │
│  utils/storage_helpers.py — save_plot, save_html        │
│  log.py                   — Python logging config       │
└─────────────────────────────────────────────────────────┘
```

### Layer Rules

1. **Core Domain** depends only on Python standard library and abstract interfaces. No concrete backends.
2. **Platform Services** define abstract contracts (ABCs). No implementation details.
3. **Infrastructure Adapters** implement platform service contracts against specific technologies.
4. **Application Layer** wires everything together — it is the only layer that knows about profiles, concrete adapters, and orchestration.
5. **Utilities** are dependency-free helpers available to all layers.

---

## 4. Recommended Module Structure

### 4.1 Current Structure (As-Is)

```
mlplatform/
├── __init__.py
├── _version.py
├── log.py              ← root-level utility
├── runner.py           ← root-level orchestrator (should be in core or app layer)
├── schema.py           ← root-level prediction schema (should be in core)
├── data_io.py          ← root-level data I/O (should be in a data/ or invocation/ module)
├── cli/
│   └── main.py
├── config/
│   ├── schema.py       ← naming collision with root schema.py
│   ├── models.py
│   └── loader.py
├── core/
│   ├── trainer.py
│   ├── predictor.py
│   ├── context.py
│   └── artifact_registry.py
├── storage/
│   ├── base.py
│   ├── local.py
│   └── gcs.py
├── tracking/
│   ├── base.py
│   ├── local.py
│   ├── vertexai.py
│   └── none.py
├── invocation/
│   ├── base.py
│   ├── in_process.py
│   ├── spark_batch.py
│   └── fastapi_serving.py
├── profiles/
│   └── registry.py
├── spark/
│   ├── main.py
│   ├── packager.py
│   └── config_serializer.py
└── utils/
    ├── serialization.py
    └── storage_helpers.py
```

### 4.2 Proposed Structure (To-Be)

```
mlplatform/
├── __init__.py              ← public API surface
├── _version.py
│
├── core/                    ← CORE DOMAIN (zero external deps)
│   ├── __init__.py
│   ├── trainer.py           ← BaseTrainer ABC
│   ├── predictor.py         ← BasePredictor ABC
│   ├── context.py           ← ExecutionContext
│   ├── artifact_registry.py ← ArtifactRegistry
│   └── schema.py            ← PredictionInputSchema (moved from root)
│
├── config/                  ← CONFIGURATION (Pydantic models + YAML loader)
│   ├── __init__.py
│   ├── models.py            ← TrainingConfig, PredictionConfig, PipelineConfig
│   ├── schema.py            ← ModelConfig, WorkflowConfig (dataclass-based)
│   └── loader.py            ← YAML loading + profile merging
│
├── storage/                 ← PLATFORM SERVICE: storage
│   ├── __init__.py
│   ├── base.py              ← Storage ABC
│   ├── local.py             ← LocalFileSystem
│   └── gcs.py               ← GCSStorage
│
├── tracking/                ← PLATFORM SERVICE: experiment tracking
│   ├── __init__.py
│   ├── base.py              ← ExperimentTracker ABC
│   ├── local.py             ← LocalJsonTracker
│   ├── vertexai.py          ← VertexAITracker
│   └── none.py              ← NoneTracker
│
├── invocation/              ← PLATFORM SERVICE: execution strategies
│   ├── __init__.py
│   ├── base.py              ← InvocationStrategy ABC
│   ├── in_process.py        ← InProcessInvocation
│   ├── spark_batch.py       ← SparkBatchInvocation
│   └── fastapi_serving.py   ← FastAPIInvocation
│
├── data/                    ← PLATFORM SERVICE: data I/O (renamed from data_io.py)
│   ├── __init__.py
│   └── io.py                ← load_prediction_input, write_prediction_output
│
├── runner/                  ← APPLICATION LAYER: orchestration (promoted from single file)
│   ├── __init__.py          ← re-exports run_workflow, dev_context, dev_predict
│   ├── workflow.py          ← run_workflow, _run_training, _run_prediction
│   └── dev.py               ← dev_context, dev_predict (development helpers)
│
├── profiles/                ← APPLICATION LAYER: infrastructure bundles
│   ├── __init__.py
│   └── registry.py
│
├── spark/                   ← INFRASTRUCTURE: Spark/Dataproc support
│   ├── __init__.py
│   ├── main.py              ← Spark entry point
│   ├── packager.py          ← build_root_zip, build_model_package
│   └── config_serializer.py
│
├── cli/                     ← APPLICATION LAYER: CLI
│   └── main.py
│
└── utils/                   ← UTILITIES: cross-cutting helpers
    ├── __init__.py
    ├── serialization.py     ← sanitize, to_serializable
    ├── storage_helpers.py   ← save_plot, save_html
    └── logging.py           ← get_logger (moved from root log.py)
```

### 4.3 Key Changes

| Change | Rationale |
|---|---|
| Move `schema.py` → `core/schema.py` | Prediction input validation is a core domain concern. Eliminates naming collision with `config/schema.py`. |
| Move `data_io.py` → `data/io.py` | Data ingestion deserves its own subpackage for future expansion (e.g., BigQuery reader, streaming sources). |
| Promote `runner.py` → `runner/` package | The orchestrator is large enough (212 lines) and has enough distinct responsibilities (workflow execution, dev helpers, class resolution) to warrant a subpackage. |
| Move `log.py` → `utils/logging.py` | Framework logging configuration is a utility concern, not a core domain concern. |
| Keep `profiles/` as application layer | Profiles wire together concrete adapters — this is composition root logic, not core domain logic. |

---

## 5. Storage and Logging Architecture

### 5.1 Current State

The current design uses **context-mediated access** — `ExecutionContext` holds references to both `ArtifactRegistry` (which wraps `Storage`) and `ExperimentTracker`, and user code accesses them via `self.context`:

```python
class MyTrainer(BaseTrainer):
    def train(self):
        ctx = self.context
        ctx.save_artifact("model.pkl", model)     # → ArtifactRegistry → Storage
        ctx.log_metrics({"accuracy": 0.95})        # → ExperimentTracker
        ctx.log_params({"lr": 0.01})               # → ExperimentTracker
```

**Assessment:** This is a reasonable design for the current scope. The `ExecutionContext` acts as a service locator, which is acceptable for an SDK where the composition root (`runner.py` / `profiles/`) is framework-controlled rather than user-controlled.

### 5.2 Issues with Current Design

#### 5.2.1 Context Is Set via Attribute Assignment

```python
trainer = MyTrainer()
trainer.context = ctx    # ← implicit, no enforcement
trainer.train()
```

If `trainer.train()` is called before `trainer.context` is set, the result is an `AttributeError` on `self.context`. There is no compile-time or initialization-time safety.

**Recommendation:** Pass `context` to the constructor or to `train()`/`predict()` directly:

```python
# Option A: Constructor injection (preferred)
class BaseTrainer(ABC):
    def __init__(self, context: ExecutionContext) -> None:
        self.context = context

    @abstractmethod
    def train(self) -> None: ...

# Option B: Method injection
class BaseTrainer(ABC):
    @abstractmethod
    def train(self, context: ExecutionContext) -> None: ...
```

Option A is better because the context is available in `__init__` for any setup the trainer needs. However, this is a **breaking change** — consider introducing it in the next major version, while keeping backward compatibility in v0.x via a deprecation path.

#### 5.2.2 ExperimentTracker Is Optional (Silently Swallowed)

```python
def log_params(self, params: dict[str, Any]) -> None:
    if self.experiment_tracker:          # ← silent no-op if None
        self.experiment_tracker.log_params(params)
```

This means calling `ctx.log_params(...)` might do nothing, with no indication to the user. The `NoneTracker` already implements the null-object pattern — use it consistently.

**Recommendation:** Make `experiment_tracker` non-optional; default to `NoneTracker()`:

```python
@dataclass
class ExecutionContext:
    experiment_tracker: ExperimentTracker  # always set, defaults to NoneTracker
```

This eliminates the `if self.experiment_tracker` guard and makes the API more predictable.

#### 5.2.3 Storage Interface Is Too Narrow

The current `Storage` ABC has only `save(path, obj)` and `load(path)` — it uses joblib serialization internally but this is hidden. There is no way to:

- Save raw bytes (except by passing `bytes` to `joblib.dump`, which wraps them)
- List artifacts in a directory
- Check if an artifact exists
- Delete an artifact

**Recommendation:** Extend the `Storage` ABC incrementally:

```python
class Storage(ABC):
    @abstractmethod
    def save(self, path: str, obj: Any) -> None: ...

    @abstractmethod
    def load(self, path: str) -> Any: ...

    def exists(self, path: str) -> bool:
        """Check if an artifact exists. Default: try to load, catch errors."""
        try:
            self.load(path)
            return True
        except Exception:
            return False

    def list(self, prefix: str) -> list[str]:
        """List artifacts under a prefix. Subclasses should override."""
        raise NotImplementedError

    def save_bytes(self, path: str, data: bytes) -> None:
        """Save raw bytes. Default: delegates to save()."""
        self.save(path, data)
```

Adding methods with default implementations preserves backward compatibility.

### 5.3 Recommended Logging Architecture

Separate the two meanings of "logging" clearly:

| Concern | Mechanism | Interface |
|---|---|---|
| **Python logging** (debug/info/warning/error) | `logging.getLogger()` | `utils/logging.py → get_logger()` |
| **Experiment logging** (params, metrics, artifacts) | `ExperimentTracker` implementations | `tracking/base.py → ExperimentTracker` |

The current conflation of "log" (Python logging) and "log_params/log_metrics" (experiment tracking) in `ExecutionContext` is confusing. Consider renaming the experiment tracking methods:

```python
# Current (ambiguous):
ctx.log_params(...)
ctx.log_metrics(...)
ctx.log.info(...)       # ← different kind of "log"

# Proposed (explicit):
ctx.track_params(...)   # or ctx.tracker.log_params(...)
ctx.track_metrics(...)  # or ctx.tracker.log_metrics(...)
ctx.log.info(...)       # Python logging — clear and distinct
```

Alternatively, expose the tracker directly and let users call methods on it:

```python
ctx.tracker.log_params({"lr": 0.01})
ctx.tracker.log_metrics({"accuracy": 0.95})
```

This makes the dependency explicit and avoids proxy methods on `ExecutionContext`.

---

## 6. Execution and Runtime Abstractions

### 6.1 Current Architecture

The execution model is structured around three key abstractions:

1. **`Profile`** — bundles `storage_factory`, `tracker_factory`, and `invocation_strategy_factory`
2. **`InvocationStrategy`** — defines how a predictor is invoked (in-process, Spark, FastAPI)
3. **`runner.run_workflow()`** — orchestrates the full pipeline

```
Profile ──► Storage factory      ──► LocalFileSystem / GCSStorage
         ├─► Tracker factory     ──► LocalJsonTracker / VertexAITracker
         └─► Invocation factory  ──► InProcess / SparkBatch / FastAPI
```

**Assessment:** This is a solid pattern. The profile system acts as a composition root that decouples user code from infrastructure choices. The `InvocationStrategy` pattern cleanly separates "what" (predict) from "how" (in-process vs. distributed).

### 6.2 Issues

#### 6.2.1 Training Has No InvocationStrategy

Currently, training always runs in-process. There is no `TrainingInvocationStrategy` — the runner directly calls `trainer.train()`. If training ever needs to be distributed (e.g., distributed PyTorch, Spark MLlib), the current design would require a parallel abstraction.

**Recommendation:** Introduce a symmetric `TrainingStrategy` or generalize `InvocationStrategy` to handle both:

```python
class ExecutionStrategy(ABC):
    @abstractmethod
    def execute_training(self, trainer: BaseTrainer, context: ExecutionContext) -> None: ...

    @abstractmethod
    def execute_prediction(self, predictor: BasePredictor, context: ExecutionContext, model_cfg: ModelConfig) -> Any: ...
```

Or keep them separate but add a `TrainingStrategy`:

```python
class TrainingStrategy(ABC):
    @abstractmethod
    def execute(self, trainer: BaseTrainer, context: ExecutionContext) -> None: ...

class InProcessTraining(TrainingStrategy):
    def execute(self, trainer, context):
        trainer.train()
```

This maintains symmetry and opens the door for distributed training strategies.

#### 6.2.2 Spark Worker Context Reconstruction Is Fragile

In `SparkBatchInvocation._build_partition_fn()`, the worker reconstructs an `ExecutionContext` from serialized kwargs. This duplicates the context construction logic from `runner._build_context()` and hard-codes `LocalFileSystem` + `NoneTracker`:

```python
# spark_batch.py:120 — hard-coded storage and tracker on workers
storage = LocalFileSystem(base_path=base)
...
experiment_tracker=NoneTracker(),
```

This means cloud Spark workers always use local filesystem storage, which only works if artifacts are pre-distributed. If the driver uses `GCSStorage`, the worker should too.

**Recommendation:** Serialize the profile name and reconstruct via `get_profile()` on the worker:

```python
ctx_kwargs = {
    ...
    "profile": profile_name,   # serialize the profile name
}

# In the partition function:
prof = get_profile(ctx_kwargs["profile"])
storage = prof.storage_factory(ctx_kwargs["storage_base"])
tracker = prof.tracker_factory(ctx_kwargs["storage_base"])
```

#### 6.2.3 Profile Registry Uses Eager Imports

`profiles/registry.py` imports **all** adapter classes at module load time, including `GCSStorage`, `VertexAITracker`, and `FastAPIInvocation`. While the factories themselves are lazy (lambda), the imports at the top of the file mean that `import mlplatform.profiles.registry` will fail if `google-cloud-storage` isn't installed.

The current code actually avoids this by having the cloud classes imported at the top of `registry.py`:

```python
from mlplatform.storage.gcs import GCSStorage          # ← fails without google-cloud-storage
from mlplatform.tracking.vertexai import VertexAITracker  # ← fails without google-cloud-aiplatform
```

**Recommendation:** Use lazy imports inside the factory lambdas:

```python
register_profile(Profile(
    name="cloud-batch",
    storage_factory=lambda bp: _lazy_gcs_storage(bp),
    tracker_factory=lambda bp: _lazy_vertex_tracker(bp),
    invocation_strategy_factory=lambda: SparkBatchInvocation(),
))

def _lazy_gcs_storage(bp: str) -> Storage:
    from mlplatform.storage.gcs import GCSStorage
    return GCSStorage(bp)
```

Or better, split profiles into `profiles/local.py` and `profiles/cloud.py`, with the cloud module only imported when a cloud profile is requested.

### 6.3 Recommended Runtime Architecture

```
┌──────────────────┐
│   User Code      │
│  (Trainer /      │
│   Predictor)     │
└────────┬─────────┘
         │ receives ExecutionContext
┌────────▼─────────┐
│ ExecutionContext  │──── ArtifactRegistry ──── Storage (abstract)
│                  │──── ExperimentTracker (abstract)
│                  │──── Logger
└────────┬─────────┘
         │ executed by
┌────────▼─────────────────────────┐
│ ExecutionStrategy                │
│  ├─ InProcessExecution           │  ← local dev, simple batch
│  ├─ SparkDistributedExecution    │  ← Dataproc, Spark clusters
│  ├─ FastAPIServingExecution      │  ← online inference
│  └─ [future: DistributedTraining]│  ← distributed PyTorch, etc.
└──────────────────────────────────┘
         │ configured by
┌────────▼─────────┐
│     Profile      │  ← bundles Storage + Tracker + Strategy factories
└──────────────────┘
```

---

## 7. Incremental Release Strategy

### 7.1 Current Extras Structure

The `pyproject.toml` already defines a good extras structure:

```
utils     → serialization + storage helpers
config    → Pydantic config models
core      → utils + config (meta-extra)
tracking  → Vertex AI tracker
spark     → PySpark support
serving   → FastAPI inference
bigquery  → BigQuery I/O
all       → everything
```

### 7.2 Recommended Release Phases

#### Phase 1: `mlplatform[utils]` + `mlplatform[config]` (Current — v0.1.x)

**Contents:**
- `utils/serialization.py` — `sanitize()`, `to_serializable()`
- `utils/storage_helpers.py` — `save_plot()`, `save_html()`
- `config/models.py` — `TrainingConfig`, `PredictionConfig`, `PipelineConfig`
- `config/loader.py` — `load_workflow_config()`
- `config/schema.py` — `ModelConfig`, `WorkflowConfig`
- `storage/base.py` + `storage/local.py` — `Storage` ABC, `LocalFileSystem`

**Key principle:** These modules must have **zero dependency on core, tracking, invocation, or spark**. Currently satisfied.

#### Phase 2: `mlplatform[core]` (Next release — v0.2.x)

**Contents:**
- `core/trainer.py` — `BaseTrainer`
- `core/predictor.py` — `BasePredictor`
- `core/context.py` — `ExecutionContext`
- `core/artifact_registry.py` — `ArtifactRegistry`
- `core/schema.py` — `PredictionInputSchema`
- `runner/` — `run_workflow()`, `dev_context()`, `dev_predict()`
- `profiles/` — profile registry with local profiles
- `tracking/base.py` + `tracking/local.py` + `tracking/none.py`
- `invocation/base.py` + `invocation/in_process.py`
- `data/io.py`
- `cli/main.py`

**Key principle:** This release introduces the full local workflow. Users can train and predict locally without any cloud dependencies.

#### Phase 3: `mlplatform[spark]` + `mlplatform[tracking]` + `mlplatform[serving]` (v0.3.x+)

**Contents:**
- `spark/` — Spark entry point, packager, config serializer
- `tracking/vertexai.py` — Vertex AI experiment tracking
- `invocation/spark_batch.py` — Spark distributed prediction
- `invocation/fastapi_serving.py` — REST inference server
- `storage/gcs.py` — GCS storage backend
- Cloud profiles in `profiles/`

**Key principle:** Cloud and distributed features are opt-in via extras. No user code changes required — only the profile selection changes.

#### Phase 4: `mlplatform[bigquery]` + Future Integrations (v0.4.x+)

- BigQuery data source/sink
- Model registry integration
- DAG orchestrator integration (Airflow, Dagster, etc.)

### 7.3 Module Independence Rules

To ensure modules can be released incrementally:

1. **No circular imports.** Currently clean — maintain this.
2. **ABCs in separate files from implementations.** Currently done — `base.py` pattern is good.
3. **Lazy imports for optional dependencies.** Cloud adapters (GCS, Vertex AI, PySpark) must be imported lazily. The adapters themselves already do this (e.g., `GCSStorage.__init__` imports `google.cloud.storage`), but `profiles/registry.py` imports them eagerly. Fix this.
4. **`__init__.py` only re-exports public API.** Don't import implementation modules in `__init__.py` unless they're part of the public API for that release phase.
5. **Tests must be runnable per-extra.** `pip install mlplatform[utils]` + run utils tests should work without `pydantic` installed.

---

## 8. Potential Anti-Patterns

### 8.1 Implicit Context via Attribute Assignment

**Risk:** `trainer.context = ctx` is set externally after construction. If forgotten, `self.context` raises `AttributeError`. There's no type-checker or runtime enforcement.

**Severity:** Medium. The framework controls this today, but if users instantiate trainers/predictors directly, they'll hit this.

**Mitigation:** Constructor injection (see Section 5.2.1).

### 8.2 Class Resolution via Module Scanning

```python
def _resolve_class(module_path: str, base_class: type) -> type:
    mod = importlib.import_module(module_path)
    for attr_name in dir(mod):
        attr = getattr(mod, attr_name)
        if isinstance(attr, type) and issubclass(attr, base_class) and attr is not base_class:
            return attr
```

**Risk:** This scans all attributes in a module looking for the first subclass. If a module imports `BaseTrainer` from another module, or defines multiple trainers, the result is undefined (whichever `dir()` returns first).

**Severity:** Medium. Fragile — especially as codebases grow.

**Mitigation:** Use an explicit class path in the YAML config (e.g., `module: "example_model.train:MyTrainer"`) or a class registry decorator.

### 8.3 Duplicated Context Construction Logic

Context construction logic is duplicated in:
- `runner._build_context()` (line 140)
- `spark/main.py:_build_context_from_config()` (line 66)
- `invocation/spark_batch.py:_build_partition_fn()` (line 108)

**Risk:** Changes to context construction must be synchronized across three locations. Bugs in one won't be caught by fixing another.

**Severity:** High. Already a maintenance burden.

**Mitigation:** Centralize context construction in a single factory function in `core/context.py`:

```python
@classmethod
def from_profile(
    cls,
    profile: Profile,
    feature_name: str,
    model_name: str,
    version: str,
    base_path: str = "./artifacts",
    **kwargs,
) -> ExecutionContext:
    storage = profile.storage_factory(base_path)
    tracker = profile.tracker_factory(base_path)
    registry = ArtifactRegistry(storage=storage, ...)
    return cls(artifacts=registry, experiment_tracker=tracker, ...)
```

### 8.4 `ModelConfig` Overloaded for Training and Prediction

`ModelConfig` (in `config/schema.py`) has 17 fields, many of which are prediction-specific (`prediction_dataset_name`, `prediction_table_name`, `predicted_label_column_name`, etc.). Training uses only ~5 of these fields.

**Risk:** The God Object anti-pattern. Adding training-specific fields will further bloat this class.

**Severity:** Low-to-Medium. The Pydantic `TrainingConfig` / `PredictionConfig` in `config/models.py` already address this by providing typed views, but the underlying `ModelConfig` dataclass is still monolithic.

**Mitigation:** Accept this for v0.x. In a future version, split `ModelConfig` into `BaseModelConfig`, `TrainingModelConfig`, and `PredictionModelConfig` at the dataclass level.

### 8.5 Hard-coded `sys.path` Manipulation in User Code

`example_model/train.py` includes:

```python
_repo_root = Path(__file__).resolve().parent.parent
for _p in [str(_repo_root), str(_repo_root / "mlplatform")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
```

**Risk:** Fragile path manipulation that breaks if the directory structure changes. Users will copy this pattern.

**Severity:** Medium. This is a DX concern.

**Mitigation:** Once the SDK is installable via `pip install -e .`, this is unnecessary. Provide clear documentation and a `pyproject.toml` example for model packages. Remove `sys.path` manipulation from example code.

### 8.6 Profile Registry Eager-Imports Cloud Dependencies

As discussed in Section 6.2.3.

### 8.7 No Lifecycle Hooks

The `BaseTrainer` and `BasePredictor` have minimal lifecycle:
- Trainer: `train()`
- Predictor: `load_model()`, `predict()`

There are no hooks for:
- Pre/post-training (e.g., data validation, model registration)
- Pre/post-prediction (e.g., input validation, output transformation)
- Cleanup/teardown (e.g., closing connections, ending experiment runs)

**Severity:** Low for now. Will become a pain point as usage grows.

**Mitigation:** Add optional lifecycle hooks with default no-op implementations:

```python
class BaseTrainer(ABC):
    def setup(self, context: ExecutionContext) -> None:
        """Called before train(). Override for custom initialization."""
        pass

    @abstractmethod
    def train(self) -> None: ...

    def teardown(self) -> None:
        """Called after train(). Override for cleanup."""
        pass
```

---

## 9. Concrete Recommendations

### Priority 1 — Do Now (Before Next Release)

| # | Recommendation | Effort | Impact |
|---|---|---|---|
| R1 | **Lazy-import cloud adapters in `profiles/registry.py`** — move `from mlplatform.storage.gcs import GCSStorage` etc. inside factory lambdas. This is a blocking issue for users who install `mlplatform[utils]` without cloud deps. | Small | High |
| R2 | **Move `schema.py` → `core/schema.py`** — eliminates naming confusion with `config/schema.py` and places the validation concern where it belongs. | Small | Medium |
| R3 | **Move `data_io.py` → `data/io.py`** — gives data I/O its own namespace for future expansion. | Small | Low |
| R4 | **Move `log.py` → `utils/logging.py`** — clarifies that this is a utility, not experiment tracking. | Small | Low |
| R5 | **Default `experiment_tracker` to `NoneTracker()`** instead of `Optional[None]` — removes silent no-op guards in `ExecutionContext`. | Small | Medium |

### Priority 2 — Do Soon (Next Minor Release)

| # | Recommendation | Effort | Impact |
|---|---|---|---|
| R6 | **Centralize context construction** into `ExecutionContext.from_profile()` factory method. Eliminate duplication across `runner.py`, `spark/main.py`, and `spark_batch.py`. | Medium | High |
| R7 | **Fix Spark worker context reconstruction** to use the profile system instead of hard-coding `LocalFileSystem` + `NoneTracker`. | Medium | High |
| R8 | **Make class resolution explicit** — support `module: "example_model.train:MyTrainer"` syntax in YAML, falling back to scanning for backward compatibility. | Medium | Medium |
| R9 | **Promote `runner.py` to `runner/` package** — split orchestration, dev helpers, and class resolution into separate files. | Medium | Medium |
| R10 | **Remove `sys.path` manipulation from example code** — ensure `pip install -e .` is the documented development workflow. | Small | Medium |

### Priority 3 — Do Later (Major Version)

| # | Recommendation | Effort | Impact |
|---|---|---|---|
| R11 | **Constructor injection for context** — pass `ExecutionContext` to `BaseTrainer.__init__()` and `BasePredictor.__init__()`. Deprecate attribute assignment. | Medium | High |
| R12 | **Add lifecycle hooks** — `setup()`, `teardown()` for both trainer and predictor. | Medium | Medium |
| R13 | **Extend `Storage` ABC** with `exists()`, `list()`, `delete()`, `save_bytes()` methods with default implementations. | Medium | Medium |
| R14 | **Introduce `TrainingStrategy`** symmetric to `InvocationStrategy`, to support distributed training in the future. | Medium | Medium |
| R15 | **Split `ModelConfig`** into training-specific and prediction-specific dataclasses at the base level. | Large | Medium |

### Priority 4 — Strategic / Long-term

| # | Recommendation | Effort | Impact |
|---|---|---|---|
| R16 | **Plugin system for profiles** — allow third parties to register custom profiles via entry points (`[project.entry-points]`). | Large | High |
| R17 | **Event system** — emit events (training_started, training_completed, artifact_saved, etc.) that can be subscribed to by tracking, monitoring, or notification services. | Large | High |
| R18 | **Schema-driven configuration validation** — validate pipeline YAML against a JSON Schema or Pydantic model before executing, with clear error messages. | Medium | Medium |

---

## Appendix: Dependency Graph (Current)

```
cli/main.py ──► runner.py ──► config/loader.py ──► config/schema.py
                    │              └──► yaml
                    ├──► core/context.py ──► core/artifact_registry.py ──► storage/base.py
                    │         └──► tracking/base.py
                    ├──► core/trainer.py
                    ├──► core/predictor.py
                    ├──► invocation/base.py
                    ├──► profiles/registry.py ──► storage/{local,gcs}.py
                    │         │                ├── tracking/{local,vertexai,none}.py
                    │         │                └── invocation/{in_process,spark_batch,fastapi}.py
                    └──► log.py

invocation/in_process.py ──► data_io.py ──► config/schema.py (TYPE_CHECKING)
                          └──► schema.py

invocation/spark_batch.py ──► core/context.py, core/artifact_registry.py
                           ├── storage/local.py, tracking/none.py
                           └── schema.py, log.py

spark/main.py ──► core/{context,artifact_registry,trainer,predictor}
               ├── config/schema.py
               ├── invocation/spark_batch.py
               ├── profiles/registry.py
               └── log.py

utils/storage_helpers.py ──► storage/base.py
config/models.py ──► config/loader.py ──► config/schema.py
```

All dependency arrows flow downward through the layers, with no circular dependencies. This is a healthy foundation to build on.
