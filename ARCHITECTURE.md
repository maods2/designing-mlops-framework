# MLPlatform Architecture

Architectural overview of the mlplatform framework: profiles, flows, and components.

---

## Quick Reference: Local vs Cloud

| Aspect | Local Development | Cloud |
|--------|-------------------|-------|
| **Storage** | `LocalFileSystem(base_path)` — e.g. `./artifacts` | `GCSStorage(gs://bucket/prefix)` |
| **Tracker** | `LocalJsonTracker` — JSON under base_path | `VertexAITracker` — Vertex AI Experiments |
| **Training** | `profile=local` → InProcess | `profile=cloud-train` → InProcess + GCS |
| **Prediction** | `profile=local` → InProcess (CSV/Parquet) | `profile=cloud-batch` → SparkBatch (Dataproc) |
| **Online serving** | N/A | `profile=cloud-online` → FastAPI |
| **Config** | `config/global.yaml`, `dev.yaml` — base_path, train_data_path | `config/gcs.yaml` — bucket, gcp_project |
| **Orchestrator injection** | Optional | `extra_overrides` — gcp_project, bucket |

---

## 1. Profile Matrix

A **Profile** bundles Storage, ExperimentTracker, and InferenceStrategy for an execution environment.

```mermaid
flowchart TB
    subgraph profiles [Profile Registry]
        local[local]
        localSpark[local-spark]
        cloudBatchEmu[cloud-batch-emulated]
        cloudTrain[cloud-train]
        cloudBatch[cloud-batch]
        cloudOnline[cloud-online]
    end

    subgraph storage [Storage]
        LocalFS[LocalFileSystem]
        GCS[GCSStorage]
    end

    subgraph tracker [Experiment Tracker]
        LocalJson[LocalJsonTracker]
        VertexAI[VertexAITracker]
    end

    subgraph inference [Inference Strategy]
        InProcess[InProcessInference]
        SparkBatch[SparkBatchInference]
        FastAPI[FastAPIInference]
    end

    local --> LocalFS
    local --> LocalJson
    local --> InProcess

    localSpark --> LocalFS
    localSpark --> LocalJson
    localSpark --> SparkBatch

    cloudBatchEmu --> LocalFS
    cloudBatchEmu --> LocalJson
    cloudBatchEmu --> SparkBatch

    cloudTrain --> GCS
    cloudTrain --> VertexAI
    cloudTrain --> InProcess

    cloudBatch --> GCS
    cloudBatch --> VertexAI
    cloudBatch --> SparkBatch

    cloudOnline --> GCS
    cloudOnline --> VertexAI
    cloudOnline --> FastAPI
```

| Profile | Storage | Tracker | Inference | Use Case |
|---------|---------|---------|------------|----------|
| `local` | LocalFileSystem | LocalJsonTracker | InProcess | Dev: train/predict in-process, artifacts to disk |
| `local-spark` | LocalFileSystem | LocalJsonTracker | SparkBatch | Dev: Spark batch prediction locally |
| `cloud-batch-emulated` | LocalFileSystem | LocalJsonTracker | SparkBatch | Dev: Spark flow without GCS |
| `cloud-train` | GCSStorage | VertexAITracker | InProcess | Cloud: training on Vertex AI |
| `cloud-batch` | GCSStorage | VertexAITracker | SparkBatch | Cloud: batch prediction on Dataproc/Vertex |
| `cloud-online` | GCSStorage | VertexAITracker | FastAPI | Cloud: REST serving |

---

## 2. Local Development Flow

```mermaid
flowchart TB
    subgraph entry [Entry Points]
        CLI[mlplatform run --dag train.yaml]
        DevTrain[dev_train]
        DevCtx[dev_context]
        DevPredict[dev_predict]
    end

    subgraph config [Config Layer]
        DAG[DAG YAML]
        ConfigProfiles[config/global.yaml, dev.yaml]
        Loader[load_workflow_config]
        WorkflowConfig[WorkflowConfig]
        ModelConfig[ModelConfig]
    end

    subgraph profile [Profile Resolution]
        GetProfile[get_profile]
        Profile[Profile: local]
    end

    subgraph context [Context Build]
        ResolvePath[_resolve_base_path]
        FromProfile[ExecutionContext.from_profile]
        Ctx[ExecutionContext]
    end

    subgraph infra [Infrastructure - local profile]
        LocalFS[LocalFileSystem]
        LocalJson[LocalJsonTracker]
        InProcess[InProcessInference]
    end

    subgraph execution [Execution]
        Trainer[BaseTrainer]
        Predictor[BasePredictor]
        ArtifactReg[ArtifactRegistry]
    end

    CLI --> Loader
    DevTrain --> Loader
    DevCtx --> Loader
    DevPredict --> Loader

    DAG --> Loader
    ConfigProfiles --> Loader
    Loader --> WorkflowConfig
    WorkflowConfig --> ModelConfig

    WorkflowConfig --> GetProfile
    GetProfile --> Profile

    ModelConfig --> ResolvePath
    Profile --> ResolvePath
    ResolvePath --> FromProfile
    Profile --> FromProfile
    FromProfile --> Ctx

    Profile --> LocalFS
    Profile --> LocalJson
    Profile --> InProcess

    Ctx --> ArtifactReg
    Ctx --> Trainer
    Ctx --> Predictor

    Trainer --> ArtifactReg
    Predictor --> ArtifactReg
```

**Local flow summary:**
- **Storage**: `LocalFileSystem(base_path)` — e.g. `./artifacts` or `./dev_artifacts`
- **Tracker**: `LocalJsonTracker` — params/metrics to JSON under base_path
- **Artifacts**: `{base_path}/{feature}/{model}/{version}/model.pkl`
- **Data**: CSV/Parquet from `input_path` / `output_path` in config

---

## 3. Cloud Flow

```mermaid
flowchart TB
    subgraph orchestrator [Orchestrator - Vertex AI / Databricks]
        Inject[extra_overrides: gcp_project, bucket]
        RunWorkflow[run_workflow]
    end

    subgraph config [Config]
        GCSConfig[gcs.yaml: bucket, artifact_prefix, gcp_project]
        Merge[Config merge]
    end

    subgraph profile [Profile - cloud-train / cloud-batch / cloud-online]
        GCSStorage[GCSStorage]
        VertexTracker[VertexAITracker]
        Infer[InferenceStrategy]
    end

    subgraph path [Base Path Resolution]
        Resolve[_resolve_base_path]
        GSPath["gs://bucket/prefix"]
    end

    subgraph context [ExecutionContext]
        Ctx[ExecutionContext]
        ArtifactReg[ArtifactRegistry]
    end

    subgraph storage [GCS]
        Bucket[GCS Bucket]
    end

    Inject --> RunWorkflow
    GCSConfig --> Merge
    Merge --> Resolve

    profile --> Resolve
    Resolve --> GSPath
    GSPath --> GCSStorage
    GCSStorage --> Bucket

    RunWorkflow --> profile
    profile --> Ctx
    Ctx --> ArtifactReg
    ArtifactReg --> Bucket
```

**Cloud flow summary:**
- **Storage**: `GCSStorage(gs://bucket/prefix)` — project from `extra_overrides` or env
- **Tracker**: `VertexAITracker` — experiments in Vertex AI
- **Base path**: `gs://{bucket}/{artifact_prefix}` from config (`gcs.yaml`) or `extra_overrides`
- **Orchestrator** injects `gcp_project`, `bucket` via `extra_overrides` when config is not sufficient

---

## 4. Base Path Resolution

```mermaid
flowchart LR
    subgraph inputs [Inputs]
        Override[base_path_override]
        ProfileName[profile name]
        OptionalConfigs[optional_configs]
    end

    subgraph logic [Logic]
        CheckOverride{override?}
        IsGCS{cloud profile?}
        HasBucket{bucket in config?}
        BuildGCS["gs://bucket/prefix"]
        UseConfig[base_path from config]
        Default["./artifacts"]
    end

    Override --> CheckOverride
    ProfileName --> IsGCS
    OptionalConfigs --> HasBucket
    OptionalConfigs --> UseConfig

    CheckOverride -->|yes| BuildGCS
    CheckOverride -->|no| IsGCS
    IsGCS -->|cloud-batch, cloud-online, cloud-train| HasBucket
    IsGCS -->|local*| UseConfig
    HasBucket -->|yes| BuildGCS
    HasBucket -->|no| Default
    UseConfig -->|present| UseConfig
    UseConfig -->|absent| Default
```

| Scenario | Result |
|----------|--------|
| `--base-path ./my_artifacts` | `./my_artifacts` |
| `profile=local`, config `base_path: ./dev_artifacts` | `./dev_artifacts` |
| `profile=cloud-train`, config `bucket: my-bucket`, `artifact_prefix: models` | `gs://my-bucket/models` |
| `profile=cloud-train`, no bucket | `./artifacts` (fallback) |

---

## 5. Training Flow (Detailed)

```mermaid
sequenceDiagram
    participant User
    participant Runner
    participant Config
    participant Profile
    participant Context
    participant Trainer
    participant Storage
    participant Tracker

    User->>Runner: run_workflow(dag_path, profile="local")
    Runner->>Config: load_workflow_config(dag_path)
    Config-->>Runner: WorkflowConfig, ModelConfig

    Runner->>Profile: get_profile("local")
    Profile-->>Runner: Profile(storage_factory, tracker_factory)

    Runner->>Context: _build_context(workflow, model_cfg, profile, ...)
    Context->>Profile: storage_factory(base_path, extra)
    Profile-->>Context: LocalFileSystem
    Context->>Profile: tracker_factory(base_path, extra)
    Profile-->>Context: LocalJsonTracker
    Context-->>Runner: ExecutionContext

    Runner->>Trainer: trainer.context = ctx; trainer.train()
    Trainer->>Storage: artifacts.save("model.pkl", model)
    Trainer->>Tracker: tracker.log_metrics(metrics)
    Trainer->>Tracker: tracker.log_params(params)
```

---

## 6. Prediction Flow by Inference Strategy

```mermaid
flowchart TB
    subgraph inprocess [InProcessInference - local, cloud-train]
        IP1[load_model]
        IP2[load_prediction_input]
        IP3[predict]
        IP4[write_prediction_output]
        IP1 --> IP2 --> IP3 --> IP4
    end

    subgraph spark [SparkBatchInference - local-spark, cloud-batch]
        S1[SparkSession]
        S2[Read CSV/Parquet/BigQuery]
        S3[mapInPandas: load_model + predict per partition]
        S4[Write Parquet/BigQuery]
        S1 --> S2 --> S3 --> S4
    end

    subgraph fastapi [FastAPIInference - cloud-online]
        F1[load_model]
        F2[Start FastAPI server]
        F3[POST /predict]
        F1 --> F2 --> F3
    end
```

| Strategy | Data source | Output | Used by |
|----------|-------------|--------|---------|
| InProcess | `input_path` CSV/Parquet or BigQuery | `output_path` or BigQuery | local, cloud-train |
| SparkBatch | `input_path` or BQ table | Parquet or BQ table | local-spark, cloud-batch |
| FastAPI | HTTP request body | HTTP response | cloud-online |

---

## 7. Standalone Artifacts (No Framework)

```mermaid
flowchart LR
    subgraph standalone [Standalone - scripts, notebooks]
        Create[create_artifacts]
        Mode1[Config-driven: workflow, model_cfg, version]
        Mode2[Standalone: backend, bucket, feature_name, model_name, version]
    end

    subgraph backend [Backend]
        Local[LocalFileSystem]
        GCS[GCSStorage]
    end

    subgraph registry [ArtifactRegistry]
        Save[save]
        Load[load]
        ResolvePath[resolve_path]
    end

    Create --> Mode1
    Create --> Mode2
    Mode1 --> Local
    Mode1 --> GCS
    Mode2 --> Local
    Mode2 --> GCS

    Local --> Registry
    GCS --> Registry
```

**Usage:**
```python
# Standalone local
artifacts = create_artifacts(
    backend="local",
    base_path="./artifacts",
    feature_name="demo",
    model_name="model",
    version="v1",
)

# Standalone GCS
artifacts = create_artifacts(
    backend="gcs",
    bucket="my-bucket",
    prefix="models",
    project="my-project",
    feature_name="demo",
    model_name="model",
    version="v1",
)
```

---

## 8. Component Dependency Overview

```mermaid
flowchart TB
    subgraph public [Public API]
        config[config]
        storage[storage]
        utils[utils]
        artifacts[artifacts]
    end

    subgraph core [Core - framework]
        context[ExecutionContext]
        artifact_reg[ArtifactRegistry]
        trainer[BaseTrainer]
        predictor[BasePredictor]
    end

    subgraph infra [Infrastructure]
        profiles[profiles]
        inference[inference]
        tracking[tracking]
    end

    subgraph data [Data]
        data_io[data.io]
    end

    config --> profiles
    storage --> context
    profiles --> context
    context --> artifact_reg
    context --> trainer
    context --> predictor

    inference --> predictor
    inference --> data_io
    data_io --> config

    tracking --> context
    artifacts --> storage
    artifacts --> config
```

---

## 9. Config Profile Merge Flow

```mermaid
flowchart LR
    DAG[DAG YAML] --> Tasks[tasks with config: key]
    Tasks --> Names[config: global, dev]
    Names --> Load1[Load global.yaml]
    Load1 --> Load2[Load dev.yaml]
    Load2 --> Merge[Deep merge]
    Merge --> TaskMerge[Merge into task entry]
    TaskMerge --> ModelConfig[ModelConfig]
    ModelConfig --> OptionalConfigs[optional_configs]
```

Config profiles supply: `base_path`, `bucket`, `artifact_prefix`, `gcp_project`, `train_data_path`, `input_path`, `output_path`, etc.
