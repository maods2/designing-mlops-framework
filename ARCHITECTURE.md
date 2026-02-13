# MLOps Platform Architecture

Architecture diagrams for the MLOps framework, project model, orchestration, and CI/CD.

---

## 1. Framework Diagram

High-level structure of the MLOps framework library.

```mermaid
flowchart TB
    subgraph DataScientist ["Data Scientist Code"]
        Steps[Steps: PreprocessStep, TrainStep, InferenceStep, DataDriftStep, ModelMonitorStep]
        Custom[custom/ feature_engineering, evaluation, data_loader, drift, monitoring]
    end

    subgraph Framework ["mlops-framework Package"]
        subgraph Core [Core]
            RunContext[RunContext]
            ExecContext[ExecutionContext]
            StepTypes[PreprocessStep, TrainStep, InferenceStep, DataDriftStep, ModelMonitorStep]
        end
        subgraph Backends [Backends]
            LocalRunner[LocalRunner]
            VertexRunner[VertexRunner]
            DataprocRunner[DataprocRunner]
            LocalStorage[LocalStorage]
            GCSStorage[GCSStorage]
            NoOpTracker[NoOpTracker]
            LocalTracker[LocalTracker]
            VertexTracker[VertexTracker]
        end
        subgraph Compiler [Compiler]
            YAMLParser[YAML Parser]
            AirflowBuilder[Airflow DAG Builder]
        end
        CLI[CLI: mlops run, compile]
    end

    Steps --> ExecContext
    Steps --> Custom
    ExecContext --> RunContext
    LocalRunner --> LocalStorage
    LocalRunner --> NoOpTracker
    LocalRunner --> LocalTracker
    VertexRunner --> GCSStorage
    VertexRunner --> VertexTracker
    DataprocRunner --> GCSStorage
    CLI --> LocalRunner
    CLI --> Compiler
    YAMLParser --> AirflowBuilder
```

---

## 2. Project Model Diagram

Structure of a model project (e.g. part_failure_model) and data flow between steps.

```mermaid
flowchart LR
    subgraph Project [part_failure_model Project]
        subgraph Pipeline [pipeline/]
            PipelineYAML[pipeline.yaml]
            ConfigYAML[config.yaml]
        end
        subgraph Steps [steps/]
            Preprocess[PartFailurePreprocess]
            Train[PartFailureTrain]
            Inference[PartFailureInference]
            DataDrift[PartFailureDataDrift]
            ModelMonitor[PartFailureModelMonitor]
        end
        subgraph Custom [custom/]
            FE[feature_engineering]
            EV[evaluation]
            DL[data_loader]
            Drift[drift]
            Monitoring[monitoring]
        end
        Model[model.py PartFailureModel]
    end

    subgraph Artifacts [Artifacts]
        TrainData[train_data.pkl]
        ModelArtifact[model.pkl]
        Predictions[predictions.pkl]
    end

    PipelineYAML --> Preprocess
    PipelineYAML --> Train
    PipelineYAML --> Inference
    ConfigYAML --> Preprocess
    ConfigYAML --> Train
    ConfigYAML --> Inference
    Preprocess -->|build_features| FE
    Preprocess -->|load_raw_data| DL
    Preprocess -->|outputs| TrainData
    Train -->|loads| TrainData
    Train -->|compute_metrics| EV
    Train -->|uses| Model
    Train -->|outputs| ModelArtifact
    Inference -->|loads| ModelArtifact
    Inference -->|build_features| FE
    Inference -->|load_raw_data| DL
    Inference -->|outputs| Predictions
    Preprocess -->|train_data as reference| DataDrift
    Train -->|model| ModelMonitor
    DataDrift -->|compute_drift| Drift
    DataDrift -->|outputs| DriftReport[drift_report]
    ModelMonitor -->|compute_model_health| Monitoring
    ModelMonitor -->|outputs| MonitoringReport[monitoring_report]
```

---

## 2b. Monitoring Steps Data Flow

Data drift and model monitoring run alongside the main pipeline.

```mermaid
flowchart LR
    subgraph Existing [Existing Pipeline]
        P[preprocess] --> T[train]
        T --> I[inference]
    end
    subgraph Monitoring [Monitoring Steps]
        DD[data_drift]
        MM[model_monitor]
    end
    P -->|train_data as reference| DD
    T -->|model| MM
    DD -->|drift_report| Artifacts
    MM -->|monitoring_report| Artifacts
```

- **DataDriftStep**: Compares reference (training) vs current (production) distribution; outputs `drift_report` with PSI, max drift.
- **ModelMonitorStep**: Evaluates model on recent labeled data; outputs `monitoring_report` with accuracy, prediction stats.

---

## 3. Big Picture: Orchestration, Framework, Model Repo, Vertex AI, Dataproc

End-to-end view across repos, orchestration, and cloud services.

```mermaid
flowchart TB
    subgraph Repos [Repositories]
        FrameworkRepo[mlops-framework repo]
        ModelRepo[part_failure_model repo]
    end

    subgraph Orchestrator [Orchestration Layer]
        Airflow[Airflow DAGs]
        VertexPipelines[Vertex AI Pipelines]
        CLI[CLI mlops run]
    end

    subgraph Package [Package / Deploy]
        PyPI[PyPI / Artifact Registry]
        FrameworkPkg[mlops-framework wheel]
    end

    subgraph GCP [Google Cloud Platform]
        subgraph Vertex [Vertex AI]
            VertexTraining[Vertex AI Training Job]
            VertexExperiments[Vertex AI Experiments]
        end
        subgraph Dataproc [Dataproc]
            DataprocCluster[Dataproc Cluster]
            SparkJob[Spark / Batch Prediction Job]
        end
        GCS[Cloud Storage GCS]
    end

    subgraph Jobs [Jobs]
        PreprocessJob[Job: preprocess]
        TrainJob[Job: train]
        InferenceJob[Job: inference]
        DataDriftJob[Job: data_drift]
        ModelMonitorJob[Job: model_monitor]
    end

    FrameworkRepo --> FrameworkPkg
    ModelRepo -->|depends on| FrameworkPkg
    FrameworkPkg -->|published| PyPI

    CLI --> PreprocessJob
    CLI --> TrainJob
    CLI --> InferenceJob
    CLI --> DataDriftJob
    CLI --> ModelMonitorJob
    Airflow --> PreprocessJob
    Airflow --> TrainJob
    Airflow --> InferenceJob
    Airflow --> DataDriftJob
    Airflow --> ModelMonitorJob
    VertexPipelines --> TrainJob
    VertexPipelines --> InferenceJob

    PreprocessJob -->|LocalRunner or Vertex| GCS
    TrainJob -->|VertexRunner| VertexTraining
    TrainJob -->|VertexTracker| VertexExperiments
    TrainJob --> GCS
    InferenceJob -->|DataprocRunner| DataprocCluster
    DataprocCluster --> SparkJob
    InferenceJob --> GCS

    ModelRepo -->|pipeline YAML| Airflow
    ModelRepo -->|steps, model, custom| VertexTraining
    ModelRepo -->|steps, model, custom| DataprocCluster
```

---

## 4. Orchestrator to Jobs Flow

How the orchestrator launches jobs and how jobs map to backends.

```mermaid
flowchart TB
    subgraph Orchestrator [Orchestrators]
        CLI[CLI]
        Airflow[Airflow]
        Vertex[Vertex Pipelines]
    end

    subgraph Jobs [Jobs Step Executions]
        J1[Job: preprocess]
        J2[Job: train]
        J3[Job: inference]
        J4[Job: data_drift]
        J5[Job: model_monitor]
    end

    subgraph Backends [Backend Selection]
        Local[LocalRunner: local dev]
        VertexRunner[VertexRunner: GCS + Vertex Training]
        DataprocRunner[DataprocRunner: batch prediction]
    end

    subgraph Tracking [Experiment Tracking]
        NoOp[NoOpTracker: preprocess, inference]
        LocalTracker[LocalTracker: train, data_drift, model_monitor local]
        VertexTracker[VertexTracker: train, data_drift, model_monitor cloud]
    end

    CLI -->|run preprocess| J1
    CLI -->|run train| J2
    CLI -->|run inference| J3
    CLI -->|run data_drift| J4
    CLI -->|run model_monitor| J5
    Airflow --> J1
    Airflow --> J2
    Airflow --> J3
    Airflow --> J4
    Airflow --> J5
    Vertex --> J2
    Vertex --> J3

    J1 --> Local
    J2 --> Local
    J2 --> VertexRunner
    J3 --> Local
    J3 --> DataprocRunner
    J4 --> Local
    J5 --> Local

    J1 --> NoOp
    J2 --> LocalTracker
    J2 --> VertexTracker
    J3 --> NoOp
    J4 --> LocalTracker
    J4 --> VertexTracker
    J5 --> LocalTracker
    J5 --> VertexTracker
```

---

## 5. Vertex AI Training and Dataproc Prediction

Cloud execution paths for training and batch prediction.

```mermaid
flowchart TB
    subgraph Training [Training Path - Vertex AI]
        Airflow1[Airflow DAG]
        VertexPipeline[Vertex AI Pipelines]
        VertexTraining[Vertex AI Training Job]
        GCSArtifacts[GCS: artifacts/]
        VertexExp[Vertex AI Experiments]
    end

    subgraph Prediction [Prediction Path - Dataproc]
        Airflow2[Airflow DAG]
        DataprocSubmit[Dataproc Job Submit]
        DataprocCluster[Dataproc Cluster]
        SparkPred[Spark Batch Prediction]
        GCSInput[GCS: inference input]
        GCSOutput[GCS: predictions output]
    end

    Airflow1 -->|trigger| VertexPipeline
    VertexPipeline --> VertexTraining
    VertexTraining -->|save model| GCSArtifacts
    VertexTraining -->|log metrics| VertexExp

    Airflow2 -->|trigger| DataprocSubmit
    DataprocSubmit --> DataprocCluster
    DataprocCluster -->|load model| GCSArtifacts
    DataprocCluster -->|read| GCSInput
    SparkPred -->|write| GCSOutput
```

---

## 6. CI/CD Diagram

Continuous integration and deployment pipeline.

```mermaid
flowchart LR
    subgraph Dev [Development]
        DS[Data Scientist]
        Code[Code push]
        PR[Pull Request]
    end

    subgraph CI [CI Pipeline]
        Lint[Lint / Format]
        UnitTest[Unit Tests]
        Build[Build Package]
        Publish[Publish to Registry]
    end

    subgraph CD [CD Pipeline]
        DeployDAG[Deploy Airflow DAG]
        DeployVertex[Deploy Vertex Pipeline]
        ModelReg[Model Registry]
    end

    subgraph Triggers [Triggers]
        PushMain[Push to main]
        TagRelease[Tag release]
        Manual[Manual trigger]
    end

    DS --> Code
    Code --> PR
    PR -->|merge| PushMain
    PushMain --> Lint
    Lint --> UnitTest
    UnitTest --> Build
    Build --> Publish
    TagRelease --> Build
    Manual --> DeployDAG
    Manual --> DeployVertex

    Publish -->|framework version| DeployDAG
    Publish -->|framework version| DeployVertex
    DeployDAG -->|DAG file| AirflowEnv[Airflow Environment]
    DeployVertex -->|pipeline spec| VertexEnv[Vertex Pipelines]
    VertexTraining[Vertex Training] --> ModelReg
```

---

## 7. Component Summary

| Component | Purpose |
|-----------|---------|
| **mlops-framework** | Library: steps, runners, tracking, storage, compiler |
| **part_failure_model** | Model project: pipeline, steps, custom modules |
| **Airflow** | Orchestrator: schedule and run jobs locally or on GCP |
| **Vertex AI Training** | Cloud training jobs with experiment tracking |
| **Dataproc** | Batch prediction at scale (Spark) |
| **GCS** | Artifact and model storage |
| **CI/CD** | Build, test, publish framework; deploy DAGs and pipelines |
