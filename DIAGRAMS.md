# ML Platform Framework — Architecture Diagrams

Diagrams at different abstraction levels to understand the framework.

---

## Level 1: Component Overview

High-level separation of responsibilities.

```mermaid
flowchart TB
    subgraph Orchestrator["Orchestrator (Airflow)"]
        DAG[DAG / Pipeline Config]
        Trigger[Trigger Step]
    end

    subgraph MLFramework["ML Framework Lib"]
        Runner[Runner]
        Storage[Storage]
        ETB[Experiment Tracker]
        Steps[Step Abstractions]
    end

    subgraph ModelCode["Model Code"]
        Train[MyTrain]
        Predict[MyInference]
    end

    DAG --> Trigger
    Trigger --> Runner
    Runner --> Steps
    Steps --> Storage
    Steps --> ETB
    Steps --> Train
    Steps --> Predict
```

---

## Level 2: Primitives & Registry

Pluggable backends and how they connect.

```mermaid
flowchart LR
    subgraph Runners["Runners"]
        LocalRunner[LocalRunner]
        LocalSpark[LocalSparkRunner]
        Dataproc[DataprocSparkRunner]
    end

    subgraph Storage["Storage"]
        LocalFS[LocalFileSystem]
        GCS[GCS]
    end

    subgraph ETB["Experiment Tracking"]
        NoneETB[NoneTracker]
        LocalJson[LocalJsonTracker]
        MLflow[MLflow]
    end

    subgraph Steps["Step Types"]
        TrainStep[TrainStep]
        InferenceStep[InferenceStep]
    end

    RunConfig[RunConfig] --> Runners
    RunConfig --> Storage
    RunConfig --> ETB
    RunConfig --> Steps
```

---

## Level 3: Local vs Cloud Execution Paths

When and how execution happens.

```mermaid
flowchart TB
    Start[Pipeline Run] --> EnvCheck{Environment?}
    
    EnvCheck -->|dev / local_spark| Local[Local Path]
    EnvCheck -->|prod| Cloud[Cloud Path]

    subgraph Local["Local Execution"]
        Direct[direct=True]
        Direct --> InProcess[run_spark_step in-process]
        InProcess --> NoMain[No main.py]
        InProcess --> NoPack[No root.zip packaging]
    end

    subgraph Cloud["Cloud Execution"]
        DataprocSubmit[DataprocSparkRunner]
        DataprocSubmit --> BuildZip[Build root.zip]
        BuildZip --> Upload[Upload to GCS]
        Upload --> SparkSubmit["spark-submit main.py --py-files root.zip"]
        SparkSubmit --> MainPy[main.py on cluster]
    end

    MainPy --> MapInPandas[mapInPandas distributed inference]
```

---

## Level 4: Spark Inference Flow (mapInPandas)

How distributed prediction works on Dataproc.

```mermaid
flowchart TB
    subgraph Driver["Driver (main.py)"]
        LoadConfig[Load run_config.json]
        CreateSpark[Create SparkSession]
        ReadInput[spark.read.csv / parquet]
        GetPredictor[Import predictor class from config]
        MapInPandas[df.mapInPandas predict_fn]
        WriteOutput[Write Parquet output]
    end

    subgraph Workers["Workers (mapInPandas)"]
        P1[Partition 1]
        P2[Partition 2]
        P3[Partition N]
    end

    subgraph PartitionLogic["Per-partition logic"]
        LoadModel[predictor.load_model storage, path]
        IterBatches[For each batch in iterator]
        PredictChunk[predictor.predict_chunk batch]
        Yield[Yield DataFrame with predictions]
    end

    LoadConfig --> CreateSpark
    CreateSpark --> ReadInput
    ReadInput --> GetPredictor
    GetPredictor --> MapInPandas
    MapInPandas --> P1
    MapInPandas --> P2
    MapInPandas --> P3

    P1 --> LoadModel
    LoadModel --> IterBatches
    IterBatches --> PredictChunk
    PredictChunk --> Yield

    MapInPandas --> WriteOutput
```

---

## Level 5: Pipeline Data Flow

From config to step execution.

```mermaid
flowchart TB
    subgraph Config["Configuration"]
        DAG[DAG YAML train_infer.yaml]
        StepYAML[Step YAMLs train.yaml, inference.yaml]
        Env[env: dev | local_spark | prod]
    end

    subgraph Resolved["Resolved RunConfig"]
        StepConfig[StepConfig: name, type, module, class]
        EnvConfig[EnvConfig: runner, storage, etb, base_path]
    end

    subgraph Context["ExecutionContext"]
        Storage[Storage]
        ETB[ETB]
        Runner[Runner]
        RunConfig[RunConfig]
    end

    subgraph Execution["Execution"]
        Instantiate[Instantiate step class]
        Run[step.run context, **kwargs]
        Result[Result]
    end

    DAG --> StepYAML
    StepYAML --> Env
    Env --> StepConfig
    Env --> EnvConfig

    StepConfig --> RunConfig
    EnvConfig --> Context

    Context --> Instantiate
    Instantiate --> Run
    Run --> Result
```

---

## Level 6: Serving Modes (BasePredictor)

Same core logic, different invocation wrappers.

```mermaid
flowchart TB
    subgraph Core["BasePredictor Core"]
        LoadModel[load_model storage, path]
        PredictChunk[predict_chunk data]
    end

    subgraph BatchLocal["BatchLocal"]
        Run[run data]
        Run --> InProcess[In-process]
        InProcess --> Script[Script / CLI]
    end

    subgraph OnlineREST["OnlineREST"]
        PredictEndpoint["/predict"]
        PredictEndpoint --> HTTP[HTTP request-response]
        HTTP --> Service[Vertex AI / Cloud Run]
    end

    subgraph BatchSpark["BatchSpark"]
        MapInPandas[mapInPandas partition fn]
        MapInPandas --> MapSide[Map-side on workers]
        MapSide --> Dataproc[Dataproc cluster]
    end

    LoadModel --> Run
    PredictChunk --> Run

    LoadModel --> PredictEndpoint
    PredictChunk --> PredictEndpoint

    LoadModel --> MapInPandas
    PredictChunk --> MapInPandas
```

---

## Level 7: Artifact Hierarchy

Where artifacts live (Feature > Model > Version).

```mermaid
flowchart TB
    BasePath["base_path (injected by orchestrator)"]
    
    BasePath --> Feature["feature/"]
    Feature --> Model["model_name/"]
    Model --> Version["version/"]
    Version --> Artifacts["model.pkl, metrics.json, ..."]

    subgraph Example["Example"]
        F[simple/]
        M[simple_model/]
        V[dev/]
        A[model.pkl]
        F --> M
        M --> V
        V --> A
    end
```

---

## Level 8: Step Types & Execution Modes

```mermaid
flowchart LR
    subgraph TrainStep["TrainStep"]
        T_Local[Procedural Local]
        T_Batch[Batch Dataproc]
    end

    subgraph InferenceStep["InferenceStep"]
        I_Local[Procedural Local]
        I_Online[OnlineREST Vertex AI]
        I_Spark[SparkBatch Dataproc]
    end

    TrainStep --> T_Local
    TrainStep --> T_Batch

    InferenceStep --> I_Local
    InferenceStep --> I_Online
    InferenceStep --> I_Spark
```
