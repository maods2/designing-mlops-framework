# ML Platform Framework — Architecture Diagrams

Mermaid diagrams at different abstraction levels to understand the framework.

---

## Level 1: High-Level Overview

Three layers: orchestration config, the framework, and user model code.

```mermaid
flowchart TB
    subgraph Config["Configuration"]
        DAG["DAG YAML\n(training / prediction)"]
    end

    subgraph Framework["mlplatform Framework"]
        Runner["Runner\n(run_workflow / dev_context)"]
        Context["ExecutionContext"]
        Storage["Storage"]
        Artifacts["ArtifactStore"]
        Tracker["ExperimentTracker"]
    end

    subgraph ModelCode["User Model Code"]
        Trainer["MyTrainer\n(extends BaseTrainer)"]
        Predictor["MyPredictor\n(extends BasePredictor)"]
    end

    DAG --> Runner
    Runner --> Context
    Context --> Storage
    Context --> Artifacts
    Context --> Tracker
    Context --> Trainer
    Context --> Predictor
```

---

## Level 2: Module Map

Package layout showing every module and its role.

```mermaid
flowchart LR
    subgraph mlplatform["mlplatform/"]
        runner["runner.py\nOrchestrator"]
        log["log.py\nLogging"]

        subgraph config["config/"]
            schema["schema.py\nWorkflowConfig\nModelConfig"]
            loader["loader.py\nload_workflow_config()"]
        end

        subgraph core["core/"]
            context["context.py\nExecutionContext"]
            trainer["trainer.py\nBaseTrainer ABC"]
            predictor["predictor.py\nBasePredictor ABC"]
        end

        subgraph storage["storage/"]
            storage_base["base.py\nStorage ABC"]
            storage_local["local.py\nLocalFileSystem"]
        end

        subgraph artifacts["artifacts/"]
            art_base["base.py\nArtifactStore ABC"]
            art_local["local.py\nLocalArtifactStore"]
        end

        subgraph tracking["tracking/"]
            trk_base["base.py\nExperimentTracker ABC"]
            trk_local["local.py\nLocalJsonTracker"]
            trk_none["none.py\nNoneTracker"]
        end

        subgraph spark["spark/"]
            spark_main["main.py\nSpark entry point"]
            serializer["config_serializer.py"]
            packager["packager.py\nbuild_root_zip()"]
        end

        subgraph cli["cli/"]
            cli_main["main.py\nmlplatform run\nmlplatform build-package"]
        end

        utils["utils/\nShared helpers"]
    end
```

---

## Level 3: Pluggable Backend Hierarchy

Abstract base classes and their concrete implementations.

```mermaid
classDiagram
    class Storage {
        <<ABC>>
        +save(path, obj)
        +load(path)
    }
    class LocalFileSystem {
        +save(path, obj)
        +load(path)
    }
    Storage <|-- LocalFileSystem

    class ArtifactStore {
        <<ABC>>
        +register_model(metadata)
        +resolve_model(model_name)
    }
    class LocalArtifactStore {
        -registry_path: str
        +register_model(metadata)
        +resolve_model(model_name)
    }
    ArtifactStore <|-- LocalArtifactStore

    class ExperimentTracker {
        <<ABC>>
        +log_params(params)
        +log_metrics(metrics)
        +log_artifact(name, obj)
    }
    class LocalJsonTracker {
        +log_params(params)
        +log_metrics(metrics)
        +log_artifact(name, obj)
    }
    class NoneTracker {
        +log_params(params)
        +log_metrics(metrics)
        +log_artifact(name, obj)
    }
    ExperimentTracker <|-- LocalJsonTracker
    ExperimentTracker <|-- NoneTracker

    class BaseTrainer {
        <<ABC>>
        +context: ExecutionContext
        +train()*
    }
    class BasePredictor {
        <<ABC>>
        +context: ExecutionContext
        +load_model()*
        +predict_chunk(data)*
    }
```

---

## Level 4: Training Data Flow

From DAG YAML through the runner to the user's trainer.

```mermaid
flowchart TB
    YAML["template_training_dag.yaml"] -->|load_workflow_config| WC["WorkflowConfig\n+ ModelConfig[]"]
    WC -->|run_workflow / dev_context| Build["_build_context()"]

    Build --> Infra["Create infrastructure"]
    Infra --> S["LocalFileSystem"]
    Infra --> A["LocalArtifactStore"]
    Infra --> T["LocalJsonTracker"]

    Build --> CTX["ExecutionContext\nfeature_name, model_name, version\noptional_configs, log"]

    CTX --> Resolve["_resolve_class(module)\nimport + find subclass"]
    Resolve --> Inst["trainer = MyTrainer()\ntrainer.context = ctx"]
    Inst --> Train["trainer.train()"]

    Train --> UserCode["User code runs"]
    UserCode -->|ctx.save_artifact| S
    UserCode -->|ctx.log_metrics| T
    UserCode -->|ctx.register_model| A
```

---

## Level 5: Prediction Data Flow

From DAG YAML through the runner to the user's predictor.

```mermaid
flowchart TB
    YAML["template_prediction_dag.yaml"] -->|load_workflow_config| WC["WorkflowConfig\n+ ModelConfig[]"]
    WC -->|run_workflow / dev_context| Build["_build_context()"]

    Build --> CTX["ExecutionContext"]

    CTX --> Resolve["_resolve_class(module)"]
    Resolve --> Inst["predictor = MyPredictor()\npredictor.context = ctx"]
    Inst --> Load["predictor.load_model()"]
    Load -->|ctx.load_artifact| Storage["Storage"]

    Load --> Predict["predictor.predict_chunk(data)"]
    Predict --> Result["DataFrame with predictions"]
```

---

## Level 6: Entry Points

Three ways to run the framework.

```mermaid
flowchart TB
    subgraph DirectExec["Direct Execution (debug)"]
        PyFile["python example_model/train.py"]
        DevCtx["dev_context(dag_path)"]
        PyFile --> DevCtx
    end

    subgraph CLI["CLI"]
        Cmd["mlplatform run --dag dag.yaml"]
        RunWF1["run_workflow()"]
        Cmd --> RunWF1
    end

    subgraph PythonAPI["Python API"]
        Import["from mlplatform.runner import run_workflow"]
        RunWF2["run_workflow()"]
        Import --> RunWF2
    end

    subgraph SparkEntry["Spark Entry Point"]
        SparkCmd["spark-submit main.py\n--py-files root.zip\n-- --config config.json"]
        SparkMain["spark/main.py"]
        SparkCmd --> SparkMain
    end

    DevCtx --> BuildCtx["_build_context()"]
    RunWF1 --> BuildCtx
    RunWF2 --> BuildCtx
    SparkMain --> BuildCtx2["_build_context_from_config()"]

    BuildCtx --> CTX["ExecutionContext"]
    BuildCtx2 --> CTX
```

---

## Level 7: Spark Distributed Prediction (mapInPandas)

How batch prediction works on a Spark cluster.

```mermaid
flowchart TB
    subgraph Prep["Preparation"]
        TrainFirst["1. Train model\n(artifacts saved)"]
        Serialize["2. write_workflow_config()\n→ spark_config.json"]
        Package["3. build_root_zip()\n→ root.zip"]
    end

    subgraph Driver["Driver Node (main.py)"]
        LoadCfg["Load spark_config.json"]
        CreateSpark["Create SparkSession"]
        ReadInput["spark.read input data"]
        BuildCtx["Build ExecutionContext"]
        Resolve["Resolve predictor class"]
        MapIP["sdf.mapInPandas(predict_fn)"]
        WriteOut["Write output parquet"]
    end

    subgraph Workers["Worker Nodes"]
        W1["Partition 1"]
        W2["Partition 2"]
        WN["Partition N"]
    end

    subgraph PerPartition["Per-Partition Logic"]
        LM["predictor.load_model()"]
        Iter["for batch in iterator"]
        PC["predictor.predict_chunk(batch)"]
        Yield["yield DataFrame"]
    end

    TrainFirst --> Serialize
    Serialize --> Package
    Package --> LoadCfg

    LoadCfg --> CreateSpark --> ReadInput --> BuildCtx --> Resolve --> MapIP
    MapIP --> W1 & W2 & WN
    W1 --> LM --> Iter --> PC --> Yield
    MapIP --> WriteOut
```

---

## Level 8: Artifact Layout

Where artifacts are stored on disk (or cloud storage).

```mermaid
flowchart TB
    Base["base_path\n(./artifacts or gs://bucket)"]
    Base --> Feature["feature_name/\n(e.g. eds)"]
    Feature --> Model["model_name/\n(e.g. lr_p708)"]
    Model --> Version["version/\n(e.g. v1.0 or dev)"]
    Version --> M["model.pkl"]
    Version --> S["scaler.pkl"]
    Version --> Other["..."]

    Base --> Registry["model_registry.json"]
    Base --> Metrics["metrics.json"]

    style Base fill:#f0f0f0,stroke:#333
    style Registry fill:#fff3cd,stroke:#856404
    style Metrics fill:#fff3cd,stroke:#856404
```

---

## Level 9: ExecutionContext API

What the user's code interacts with.

```mermaid
classDiagram
    class ExecutionContext {
        +storage: Storage
        +artifact_store: ArtifactStore
        +experiment_tracker: ExperimentTracker
        +feature_name: str
        +model_name: str
        +version: str
        +optional_configs: dict
        +log: Logger
        +artifact_base_path: str
        +save_artifact(name, obj)
        +load_artifact(name)
        +load_artifact_from(model, version, name)
        +log_params(params)
        +log_metrics(metrics)
        +register_model(metadata)
    }

    class BaseTrainer {
        +context: ExecutionContext
        +train()*
    }

    class BasePredictor {
        +context: ExecutionContext
        +load_model()*
        +predict_chunk(data)*
    }

    ExecutionContext --> BaseTrainer : injected
    ExecutionContext --> BasePredictor : injected
    ExecutionContext --> Storage : delegates
    ExecutionContext --> ArtifactStore : delegates
    ExecutionContext --> ExperimentTracker : delegates
```

---

## Level 10: Local vs Cloud Execution

Same framework code, different infrastructure.

```mermaid
flowchart TB
    DAG["DAG YAML"] --> Runner["Runner"]

    Runner --> EnvCheck{Profile?}

    EnvCheck -->|local| LocalPath
    EnvCheck -->|cloud| CloudPath

    subgraph LocalPath["Local Execution"]
        LS["LocalFileSystem\n(./artifacts)"]
        LA["LocalArtifactStore\n(model_registry.json)"]
        LT["LocalJsonTracker\n(metrics.json)"]
        LP["python train.py\nor mlplatform run"]
    end

    subgraph CloudPath["Cloud Execution (Dataproc / VertexAI)"]
        GCS["GCS Storage\n(gs://bucket/...)"]
        CA["Cloud ArtifactStore\n(Vertex AI Model Registry)"]
        CT["Cloud Tracker\n(MLflow / W&B)"]
        SS["spark-submit main.py\n--py-files root.zip"]
    end

    style CloudPath fill:#f8f9fa,stroke:#6c757d,stroke-dasharray: 5 5
```

> Dashed border: cloud backends are extension points, not yet implemented. The framework architecture supports them via the ABC interfaces.
