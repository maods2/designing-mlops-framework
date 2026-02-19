Perfeito. Vou manter **exatamente o mesmo estilo estrutural que você usou**, mas com as melhorias conceituais aplicadas e responsabilidades mais bem separadas.

---

# Framework Components

* **Orchestrator**
* **ML Framework Lib**
* **Model code**
* **Model Registry** *(novo componente explícito)*

---

# Responsabilities:

### Orchestrator (Airflow – Not implemented here)

> Parse pipeline YAML and resolve environment configuration.

> Define and provide args for:

* Runner
* Storage
* Experiment Tracker
* Serving Mode (for inference only)
* StepType
* Model metadata (feature, model, version)

> Trigger the step execution.

Example resolution:

```
env=dev
runner=LocalRunner
storage=LocalFileSystem
tracker=LocalJson
serving_mode=InProcess
step=TrainStep
feature=fraud
model=xgboost
version=auto
```

The Orchestrator decides:

* **where** it runs
* **which infrastructure**
* **which configuration**

It does NOT implement execution logic.

---

### ML Framework Lib

> Provide common abstractions and computational primitives.

Responsibilities:

* Define base interfaces:

  * Runner
  * Storage
  * ExperimentTracker
  * ModelRegistry
  * Step
  * ServingWrapper
* Create and inject ExecutionContext
* Manage artifact persistence through Storage
* Manage model version resolution through ModelRegistry
* Provide base classes:

  * BaseTrainer
  * BasePredictor
* Ensure infrastructure is fully abstracted from model logic

The framework is the **platform abstraction layer**.

---

### Model Code

> Implement ML logic only.

Responsibilities:

* Implement:

  * `train()`
  * `predict_chunk()`
* Define feature schema
* Define artifact structure
* Use provided Storage abstraction to save/load artifacts
* Never depend on:

  * Dataproc
  * Vertex AI
  * GCS
  * REST frameworks
  * Airflow

Model code must be environment-agnostic.

---

### Model Registry (New Explicit Component)

> Responsible for model version resolution and lifecycle management.

Responsibilities:

* Resolve version:

  * latest
  * production
  * specific version
* Promote model versions
* Maintain metadata

Example hierarchy:

```
Feature
   └── Model
         └── Version
               └── Artifacts
```

Storage handles bytes.
Registry handles logical resolution.

---

# Primitives

## Runner (Where execution happens)

Defines execution environment.

* LocalRunner
* DataprocRunner
* VertexAIRunner

Responsibilities:

* Submit job
* Execute step
* Handle environment-specific runtime concerns

Runner does NOT define business logic.

---

## Storage (Artifact persistence)

* LocalFileSystem
* GCS

Responsibilities:

* save(path, object)
* load(path)
* manage artifact paths

Storage does NOT resolve model versions.

---

## Experiment Tracker

* None
* LocalJson
* MLflow
* Vertex AI

Used only in TrainStep.

Responsibilities:

* log_params
* log_metrics
* log_artifacts

---

## ModelRegistry

Resolves:

```
(feature, model, version) → artifact location
```

Handles logical versioning independent of physical storage.

---

## Step (What is executed)

* TrainStep
* InferenceStep

Responsibilities:

* Receive ExecutionContext
* Call Trainer or Predictor
* Remain agnostic to:

  * Runner
  * Serving mode
  * Infrastructure

Step defines the pipeline unit of work.

---

## ServingWrapper (How inference is invoked)

Separate abstraction from Step.

Serving modes:

* InProcess
* REST
* SparkBatch

Responsibilities:

* Wrap BasePredictor
* Expose prediction entrypoint
* Adapt invocation method

Core prediction logic must always live in:

```
BasePredictor:
    load_model()
    predict_chunk()
```

ServingWrapper only changes invocation pattern.

---

# Execution Model

### Training Flow

```
Orchestrator
    ↓
Runner
    ↓
TrainStep
    ↓
BaseTrainer
    ↓
Storage + Tracker + Registry
```

---

### Inference Flow

```
Orchestrator
    ↓
Runner
    ↓
InferenceStep
    ↓
BasePredictor
    ↓
ServingWrapper (optional)
    ↓
Storage + Registry
```

---

# Key Design Improvements Applied

* ExecutionMode removed from Step
* Runner separated from ServingMode
* ModelRegistry explicitly introduced
* ServingWrapper separated from InferenceStep
* Clear separation between logical versioning and storage backend
* Strict environment-agnostic model design

---

# Architectural Principles

* Local-first development
* Configuration-driven promotion to production
* Infrastructure fully abstracted
* Single prediction core shared across all serving modes
* Extensible primitives via interface implementation
* No modification of model code when changing environment

