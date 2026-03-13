# ML Platform Migration Proposal v2

> **Goal**: Migrate `plat/` into `mlplatform/`, rethinking and simplifying abstractions.
> Remove DAG YAML dependency — the framework receives validated config from orchestrator args
> or from a Python API for local development.

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

---

## 2. Architectural Changes from Original Prompt

### 2.1 No more DAG YAML parsing

**Original**: The framework reads a DAG YAML file (`train.yaml`, `predict.yaml`) to discover
models, config profiles, and pipeline type. Two formats supported (legacy + Databricks).

**New**: The framework receives **already-resolved arguments** from the orchestrator
(Vertex AI, Databricks, Airflow) or from user code. These args are validated into a
**frozen Pydantic config model**. The same model can be constructed in Python for local
dev/debugging.

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
- `load_config_profiles` from v0.1.x is **kept** as a utility for users who want YAML merging
- Config profiles are now the **user's responsibility** — they merge YAMLs themselves and pass
  the result to the config builder

### 2.2 Builder pattern for config (frozen output)

**Original**: Flat `WorkflowConfig` mixing CLI args, infra, and pipeline concerns.

**New**: A `PipelineConfigBuilder` that validates incrementally and produces a **frozen**
`PipelineConfig`.

```python
from mlplatform.config import PipelineConfigBuilder

# From orchestrator args
config = (
    PipelineConfigBuilder()
    .identity(model_name="churn_model", feature="churn", version="v1.2")
    .infra(backend="gcs", bucket="ml-artifacts", project_id="my-project")
    .pipeline(pipeline_type="training", profile="cloud-train")
    .configs(["global", "train-prod"], config_dir="./config")  # optional YAML merge
    .build()  # → frozen PipelineConfig
)

# For local dev (minimal)
config = (
    PipelineConfigBuilder()
    .identity(model_name="churn_model", feature="churn")
    .pipeline(pipeline_type="training")
    .build()  # defaults: backend=local, base_path=./artifacts, profile=local, version=dev
)

# From CLI (thin wrapper)
# mlplatform run --model-name churn_model --feature churn --pipeline-type training
```

The builder validates at each step and `.build()` produces an **immutable** Pydantic model.
Invalid combinations (e.g., `backend=gcs` without `bucket`) fail early with clear errors.

### 2.3 Unified ArtifactRegistry

**Original**: Two `ArtifactRegistry` classes — one in `mlplatform/artifacts/` (format dispatch)
and one in `plat/core/` (cross-model loading, storage property).

**New**: **Merge into one**. The existing `mlplatform/artifacts/registry.py` gains:
- `load(name, *, model_name=None, version=None)` for cross-model loading
- `storage` property exposing the underlying backend
- No other changes — format dispatch stays as-is

### 2.4 Provider-agnostic tracking

**Original**: `ExperimentTracker` ABC with VertexAI as the only cloud impl.

**New**: Same ABC but designed for future providers. The interface stays slim:

```python
class ExperimentTracker(ABC):
    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None: ...
    @abstractmethod
    def log_metrics(self, metrics: dict[str, float]) -> None: ...
    @abstractmethod
    def log_artifact(self, name: str, artifact: Any) -> None: ...
    def start_run(self, run_name: str | None = None) -> None: ...   # optional
    def end_run(self) -> None: ...                                    # optional
```

The `start_run`/`end_run` hooks enable providers like MLflow that have explicit run lifecycle.
Default implementations are no-ops so existing code doesn't break.

### 2.5 Simplified runner (no DAG orchestration)

**Original**: `run_workflow` parses DAG, iterates models, builds contexts.

**New**: `execute(config: PipelineConfig)` — takes a frozen config, resolves profile,
builds context, runs training or prediction. Single model per invocation (the orchestrator
handles multi-model DAGs).

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

## 3. Architecture Diagrams

### 3.1 Training Flow (matches attached diagram — top section)

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

### 3.2 Prediction Flow (matches attached diagram — bottom section)

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                 Cloud / Orchestrator                     │
                    │  ┌──────────┐                                           │
                    │  │Orchestrat│──Config──►┌──────────────────────────────┐│
                    │  │   or     │           │   Base Predictor Wrapper     ││
                    │  └──────────┘           │                              ││
YAML configs ─────────────────────────────►  │  ┌─────┐  ┌──────────────┐  ││
(merged by user)                             │  │ CLI │  │Prediction Code│  ││
                    │                        │  └─────┘  │ (user impl)   │  ││
                    │                        │           └──────────────┘  ││
                    │                        └──────────────────────────────┘│
                    │                    │              │            │       │
                    │          ┌─────────▼──┐   ┌──────▼────┐  ┌───▼────┐  │
                    │          │  Artifact   │   │  Runner    │  │Config  │  │
                    │          │  Registry   │   │           │  └────────┘  │
                    │          └──────┬──────┘   └──────┬────┘              │
                    │                 │                  │                   │
                    │          ┌──────▼──────┐   ┌──────▼─────────────────┐ │
                    │          │  Storage    │   │  Inference Strategy     │ │
                    │          │  Interface  │   │  ┌─────────────────┐   │ │
                    │          └──────┬──────┘   │  │ PySpark batch   │   │ │
                    │                 │          │  │ Service API     │   │ │
                    │       ┌─────────┴──────┐   │  │ In-process      │   │ │
                    │       │                │   │  └─────────────────┘   │ │
                    │ ┌─────▼─────┐ ┌────────▼┐  └───────────────────────┘ │
                    │ │Local Stor.│ │GCS Stor.│                            │
                    │ └───────────┘ └─────────┘        ┌───────┐           │
                    │                                   │ Utils │           │
                    │                                   └───────┘           │
                    └─────────────────────────────────────────────────────────┘
```

### 3.3 Module Dependency Graph

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
                            │  (format dispatch layer)   │
                            └───────────────────────────┘
```

---

## 4. Migration Phases

### Phase 1: Config Redesign (foundation — breaking change)

**What changes:**
- **Remove** `load_workflow_config` (DAG parser) from migration target
- **Keep** `load_config_profiles` as utility (users call it themselves)
- **Keep** `TrainingConfig`, `PredictionConfig`, `RunConfig` from v0.1.x
- **Add** `PipelineConfig` — the new frozen model built from orchestrator/CLI args
- **Add** `PipelineConfigBuilder` — builder with incremental validation

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
    module: str = ""  # e.g. "my_package.train:MyTrainer"

    # Merged user config (from YAML profiles or direct dict)
    user_config: dict[str, Any] = Field(default_factory=dict)

    # Metadata
    commit_hash: str | None = None
    log_level: str = "INFO"
```

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
        # Calls load_config_profiles → stores in user_config

    def user_config(self, config: dict[str, Any]) -> Self: ...
        # Direct dict merge into user_config

    def metadata(self, *, commit_hash: str | None = None,
                 log_level: str = "INFO") -> Self: ...

    def build(self) -> PipelineConfig: ...
        # Validates all fields, returns frozen model
        # Raises ValueError for invalid combinations
```

**Validation rules in `build()`:**
- `backend=gcs` requires `base_bucket` (or `base_bucket` in `user_config`)
- `profile` starting with `cloud-` requires `backend=gcs`
- `pipeline_type=prediction` requires `module` (must have predictor to run)
- `model_name` and `feature` are always required

**Files:**
- Modify: `mlplatform/config/models.py` — add `PipelineConfig`
- New: `mlplatform/config/builder.py` — `PipelineConfigBuilder`
- Keep: `mlplatform/config/loader.py` — `load_config_profiles` only (remove or deprecate DAG loader)

---

### Phase 2: Tracking (provider-agnostic)

**What changes:**
- Move `plat/tracking/` → `mlplatform/tracking/`
- Add `start_run`/`end_run` lifecycle hooks (no-op defaults)
- Design for future MLflow, W&B, etc. without building them now

**ExperimentTracker ABC:**

```python
class ExperimentTracker(ABC):
    """Provider-agnostic experiment tracking interface.

    Implementations: NoneTracker, LocalJsonTracker, VertexAITracker.
    Future: MLflowTracker, WandbTracker (community/plugin).
    """

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None: ...

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float]) -> None: ...

    @abstractmethod
    def log_artifact(self, name: str, artifact: Any) -> None: ...

    def start_run(self, run_name: str | None = None) -> None:
        """Begin a tracking run. No-op by default."""
        pass

    def end_run(self) -> None:
        """End the current tracking run. No-op by default."""
        pass

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
| `VertexAITracker` | Vertex Experiments | Vertex Experiments | Vertex Experiments | creates/closes experiment run |

**Files:**
- New: `mlplatform/tracking/__init__.py`, `base.py`, `none.py`, `local.py`, `vertexai.py`

---

### Phase 3: Core (rethink + simplify)

**What changes:**
- Merge two `ArtifactRegistry` classes into one (extend `mlplatform/artifacts/registry.py`)
- Move `ExecutionContext` — uses the merged registry
- Keep `BaseTrainer` and `BasePredictor` separate (per your preference)
- Move `PredictionInputSchema`

**ArtifactRegistry additions** (to existing `mlplatform/artifacts/registry.py`):

```python
class ArtifactRegistry:
    # ... existing save/load with format dispatch ...

    @property
    def storage(self) -> Storage:
        """Direct access to the underlying Storage backend."""
        return self._storage

    def load(
        self,
        name: str,
        *,
        model_name: str | None = None,
        version: str | None = None,
    ) -> Any:
        """Load artifact. Override model_name/version for cross-model loading."""
        if model_name or version:
            # Build alternate path
            feat = self._feature_name
            mn = model_name or self._model_name
            ver = version or self._version
            path = f"{feat}/{mn}/{ver}/{name}"
            return self._storage.load(path)
        return self._storage.load(self._resolve_path(name))
```

**ExecutionContext** stays largely the same but uses the merged registry:

```python
@dataclass(frozen=False)
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
    def from_config(
        cls,
        config: PipelineConfig,
        profile: Profile,
        extra_overrides: dict[str, Any] | None = None,
    ) -> "ExecutionContext": ...
        # Single canonical constructor
        # Replaces from_profile — takes PipelineConfig instead of scattered args
```

> **Key simplification**: `from_profile` took 10 args. `from_config` takes a frozen
> `PipelineConfig` + a `Profile`. All the scattered args are already validated in the config.

**BaseTrainer / BasePredictor**: Same contract as current `plat/core/`, no changes needed.
They keep their typed property accessors (`self.artifacts`, `self.tracker`, `self.config`,
`self.log`) and lifecycle hooks (`setup`, `teardown`).

**Files:**
- Modify: `mlplatform/artifacts/registry.py` — add `storage` property + cross-model load
- New: `mlplatform/core/__init__.py`, `context.py`, `trainer.py`, `predictor.py`, `prediction_schema.py`

---

### Phase 4: Profiles

**What changes:**
- Move `plat/profiles/` → `mlplatform/profiles/`
- Same `Profile` dataclass + registry pattern
- Same lazy factory helpers (no cloud imports at module load)

No architectural changes needed — the profile system is already well-designed.

**Files:**
- New: `mlplatform/profiles/__init__.py`, `registry.py`

---

### Phase 5: Inference Strategies

**What changes:**
- Move `plat/inference/` → `mlplatform/inference/`
- `InferenceStrategy.invoke()` signature change: receives `PipelineConfig` instead of `ModelConfig`

```python
class InferenceStrategy(ABC):
    @abstractmethod
    def invoke(
        self,
        predictor: BasePredictor,
        context: ExecutionContext,
        config: PipelineConfig,        # was: ModelConfig
    ) -> Any: ...
```

**Implementations stay the same**: `InProcessInference`, `SparkBatchInference`, `FastAPIInference`.

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
- **Keep**: `dev_train`, `dev_predict`, `dev_context` as convenience functions
- **Keep**: `resolve_class` for module:Class resolution

```python
def execute(config: PipelineConfig) -> dict[str, str]:
    """Execute a single model training or prediction from a frozen config.

    This is the main entry point for both orchestrator and CLI invocations.
    """
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
    model_name: str,
    feature: str,
    trainer_class: type[BaseTrainer],
    *,
    version: str = "dev",
    base_path: str = "./artifacts",
    user_config: dict[str, Any] | None = None,
) -> ExecutionContext:
    """Convenience for local training — no CLI, no config files needed."""
    config = (
        PipelineConfigBuilder()
        .identity(model_name=model_name, feature=feature, version=version)
        .infra(base_path=base_path)
        .pipeline(pipeline_type="training")
        .user_config(user_config or {})
        .build()
    )
    profile = get_profile("local")
    ctx = ExecutionContext.from_config(config, profile)
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
- `config_serializer` converts frozen config → JSON dict → frozen config

```python
# Spark worker entry point
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="JSON-serialized PipelineConfig")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    config = PipelineConfig.model_validate_json(args.config)
    # ... Spark session setup, mapInPandas, etc.
```

**Files:**
- New: `mlplatform/spark/__init__.py`, `main.py`, `packager.py`, `config_serializer.py`

---

### Phase 9: CLI (thin wrapper)

**What changes:**
- CLI parses args → builds `PipelineConfig` via builder → calls `execute(config)`
- Two commands: `run` and `build-package`

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

```python
def main():
    parser = argparse.ArgumentParser(prog="mlplatform")
    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run")
    run_parser.add_argument("--model-name", required=True)
    run_parser.add_argument("--feature", required=True)
    run_parser.add_argument("--version", default="dev")
    run_parser.add_argument("--pipeline-type", choices=["training", "prediction"], required=True)
    run_parser.add_argument("--profile", default="local")
    run_parser.add_argument("--module", required=True)
    run_parser.add_argument("--backend", default="local", choices=["local", "gcs"])
    run_parser.add_argument("--base-path", default="./artifacts")
    run_parser.add_argument("--base-bucket")
    run_parser.add_argument("--project-id")
    run_parser.add_argument("--config", help="Comma-separated config profile names")
    run_parser.add_argument("--config-dir", default="./config")
    run_parser.add_argument("--commit-hash")

    args = parser.parse_args()

    builder = PipelineConfigBuilder()
    builder.identity(model_name=args.model_name, feature=args.feature, version=args.version)
    builder.infra(backend=args.backend, base_path=args.base_path,
                  base_bucket=args.base_bucket, project_id=args.project_id)
    builder.pipeline(pipeline_type=args.pipeline_type, profile=args.profile,
                     module=args.module)
    if args.config:
        builder.configs(args.config.split(","), config_dir=args.config_dir)
    if args.commit_hash:
        builder.metadata(commit_hash=args.commit_hash)

    config = builder.build()
    execute(config)
```

**Files:**
- New: `mlplatform/cli/__init__.py`, `main.py`
- Modify: `pyproject.toml` — add `[project.scripts] mlplatform = "mlplatform.cli.main:main"`

---

## 5. Key Simplifications vs. Original Prompt

| Area | Original Prompt | This Proposal |
|------|----------------|---------------|
| **DAG parsing** | Keep `load_workflow_config` with 2 YAML formats | **Remove** — config comes from args/builder |
| **Config model** | Flat `WorkflowConfig` with mixed concerns | **Builder → frozen `PipelineConfig`** with validation |
| **ArtifactRegistry** | Two classes to merge | **Extend existing** v0.1.x registry (add 2 features) |
| **Runner** | `run_workflow` iterates models from DAG | **`execute(config)`** — single model, orchestrator handles DAG |
| **Config profiles** | Framework loads YAMLs from DAG config: key | **User loads YAMLs** (utility kept), passes dict to builder |
| **WorkflowConfig** | ~10 fields, some type-unsafe | **`PipelineConfig`** — strictly typed, frozen, validated |
| **Spark entry** | Receives serialized DAG + args | Receives serialized **PipelineConfig** (JSON) |

---

## 6. Files to Create / Modify

### New files (~15 files)

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

### Modified files (~3 files)

| File | Change |
|------|--------|
| `mlplatform/artifacts/registry.py` | Add `storage` property + `load(..., model_name, version)` |
| `mlplatform/config/models.py` | Add `PipelineConfig` frozen model |
| `pyproject.toml` | Add CLI entry point + optional deps (`tracking`, `spark`, `serving`, `bigquery`) |

---

## 7. pyproject.toml Optional Extras

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
all = ["mlplatform[core]", "mlplatform[tracking]", "mlplatform[spark]", "mlplatform[serving]", "mlplatform[bigquery]", "mlplatform[parquet]"]

[project.scripts]
mlplatform = "mlplatform.cli.main:main"
```

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
```

Each phase can be **independently tested** before moving to the next.

---

## 9. What Gets Deleted from plat/

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

## 10. Recommendations & Notes

### R1: Config profiles are now opt-in
Users who want YAML config merging call `load_config_profiles` themselves and pass the
result to the builder. This is simpler than the framework auto-discovering config directories.

### R2: Single-model execution
The runner no longer iterates over models. Each `execute(config)` call handles one model.
The orchestrator (Vertex AI pipeline, Databricks job, Airflow DAG) handles multi-model flows.
This is a significant simplification — the framework does one thing well.

### R3: PipelineConfig is the contract
Every component receives the same frozen `PipelineConfig`. No more scattered args passed
through 3-4 function signatures. This makes the framework easier to test (just construct a config).

### R4: Backward compatibility
The v0.1.x public API (`TrainingConfig`, `PredictionConfig`, `RunConfig`, `create_artifacts`,
`ArtifactRegistry`, all storage/utils) remains **unchanged**. Users of v0.1.x are not affected.
The new modules are additive.

### R5: Tracking extensibility
The `ExperimentTracker` ABC with `start_run`/`end_run` supports both stateless trackers
(LocalJson, None) and stateful ones (VertexAI, MLflow). The context manager protocol
(`with tracker:`) makes it natural to scope runs.

### R6: Consider a `PipelineConfig.from_dict()` classmethod
For orchestrators that pass JSON/dict payloads, a `from_dict` factory avoids going through
the builder:

```python
config = PipelineConfig.from_dict({
    "model_name": "churn_model",
    "feature": "churn",
    "pipeline_type": "training",
    ...
})
```

Since `PipelineConfig` is a Pydantic model, this is essentially `PipelineConfig(**payload)`
with validation. The builder is for ergonomic construction; `from_dict` is for deserialization.

### R7: Test strategy
Each phase should include:
- Unit tests for the new module
- Integration test showing it works with existing v0.1.x code
- A "smoke test" that runs `dev_train` / `dev_predict` end-to-end with a toy model
