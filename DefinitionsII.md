
---

# 🧠 Core Domain Types

```python
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional
```

---

# 1️⃣ Workload & Execution Definitions

```python
class WorkloadType(Enum):
    TRAINING = "training"
    INFERENCE = "inference"


class ExecutionNature(Enum):
    JOB = "job"
    SERVICE = "service"


class ExecutionTarget(Enum):
    LOCAL = "local"
    CLOUD = "cloud"
    EMULATED_CLOUD = "emulated_cloud"
```

---

# 2️⃣ Step Abstractions

```python
class Step(ABC):
    workload_type: WorkloadType
    execution_nature: ExecutionNature

    @abstractmethod
    def run(self, context: "ExecutionContext"):
        pass
```

---

## TrainStep

```python
class TrainStep(Step):
    workload_type = WorkloadType.TRAINING
    execution_nature = ExecutionNature.JOB

    def run(self, context: "ExecutionContext"):
        trainer = context.trainer
        trainer.train()
```

---

## InferenceStep

```python
class InferenceStep(Step):
    workload_type = WorkloadType.INFERENCE

    def __init__(self, execution_nature: ExecutionNature):
        self.execution_nature = execution_nature

    def run(self, context: "ExecutionContext"):
        predictor = context.predictor
        invocation = context.invocation_strategy
        invocation.invoke(predictor)
```

---

# 3️⃣ Execution Context

> Injected after resolution.
> Does NOT expose runners.

```python
class ExecutionContext:

    def __init__(
        self,
        storage: "Storage",
        artifact_store: "ArtifactStore",
        experiment_tracker: Optional["ExperimentTracker"],
        invocation_strategy: Optional["InvocationStrategy"],
        runtime_config: dict,
        environment_metadata: dict,
        trainer: Optional["BaseTrainer"],
        predictor: Optional["BasePredictor"],
    ):
        self.storage = storage
        self.artifact_store = artifact_store
        self.experiment_tracker = experiment_tracker
        self.invocation_strategy = invocation_strategy
        self.runtime_config = runtime_config
        self.environment_metadata = environment_metadata
        self.trainer = trainer
        self.predictor = predictor
```

---

# 4️⃣ Runners (Infrastructure Drivers)

Runners execute steps — steps never see runners.

```python
class JobRunner(ABC):

    @abstractmethod
    def execute(self, step: Step, context: ExecutionContext):
        pass
```

```python
class ServiceRunner(ABC):

    @abstractmethod
    def start(self, step: Step, context: ExecutionContext):
        pass
```

---

## Example Concrete Runners

```python
class LocalJobRunner(JobRunner):
    def execute(self, step: Step, context: ExecutionContext):
        step.run(context)
```

```python
class DataprocJobRunner(JobRunner):
    def execute(self, step: Step, context: ExecutionContext):
        # submit distributed job
        pass
```

```python
class EndpointServiceRunner(ServiceRunner):
    def start(self, step: Step, context: ExecutionContext):
        # start REST endpoint
        pass
```

---

# 5️⃣ Invocation Layer (Inference Only)

```python
class InvocationStrategy(ABC):

    @abstractmethod
    def invoke(self, predictor: "BasePredictor"):
        pass
```

---

## Concrete Strategies

```python
class InProcessInvocation(InvocationStrategy):
    def invoke(self, predictor: "BasePredictor"):
        predictor.predict_chunk(...)
```

```python
class RESTInvocation(InvocationStrategy):
    def invoke(self, predictor: "BasePredictor"):
        # expose HTTP interface
        pass
```

```python
class DistributedInvocation(InvocationStrategy):
    def invoke(self, predictor: "BasePredictor"):
        # distributed Spark invocation
        pass
```

---

# 6️⃣ ML Core Contracts

## Trainer

```python
class BaseTrainer(ABC):

    @abstractmethod
    def train(self):
        pass
```

---

## Predictor

```python
class BasePredictor(ABC):

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def predict_chunk(self, data):
        pass
```

---

# 7️⃣ Infrastructure Abstractions

## Storage

```python
class Storage(ABC):

    @abstractmethod
    def save(self, path: str, obj):
        pass

    @abstractmethod
    def load(self, path: str):
        pass
```

---

## Artifact Store

```python
class ArtifactStore(ABC):

    @abstractmethod
    def register_model(self, model_name: str, metadata: dict):
        pass

    @abstractmethod
    def resolve_model(self, model_name: str, version: str):
        pass
```

---

## Experiment Tracker

```python
class ExperimentTracker(ABC):

    @abstractmethod
    def log_metric(self, name: str, value: float):
        pass

    @abstractmethod
    def log_param(self, name: str, value):
        pass
```

---

# 8️⃣ Profile & Resolution Layer

## Profile

```python
class Profile:

    def __init__(
        self,
        execution_target: ExecutionTarget,
        job_runner: JobRunner,
        service_runner: Optional[ServiceRunner],
        storage: Storage,
        artifact_store: ArtifactStore,
        experiment_tracker: Optional[ExperimentTracker],
        default_invocation_strategy: Optional[InvocationStrategy],
    ):
        self.execution_target = execution_target
        self.job_runner = job_runner
        self.service_runner = service_runner
        self.storage = storage
        self.artifact_store = artifact_store
        self.experiment_tracker = experiment_tracker
        self.default_invocation_strategy = default_invocation_strategy
```

---

## Primitive Resolver

```python
class PrimitiveResolver:

    def resolve(self, step: Step, profile: Profile) -> tuple:
        """
        Returns:
            runner, context
        """

        invocation = None
        if step.workload_type == WorkloadType.INFERENCE:
            invocation = profile.default_invocation_strategy

        context = ExecutionContext(
            storage=profile.storage,
            artifact_store=profile.artifact_store,
            experiment_tracker=profile.experiment_tracker
                if step.workload_type == WorkloadType.TRAINING
                else None,
            invocation_strategy=invocation,
            runtime_config={},
            environment_metadata={},
            trainer=None,
            predictor=None,
        )

        if step.execution_nature == ExecutionNature.JOB:
            return profile.job_runner, context

        if step.execution_nature == ExecutionNature.SERVICE:
            return profile.service_runner, context
```

---

# 🔎 Final Relationship Overview

High-level dependency direction:

```
Step
  ↓
ExecutionContext
  ↓
Infrastructure Interfaces

Runner
  ↓
Step.run(context)

Profile
  ↓
PrimitiveResolver
  ↓
ExecutionContext
```

### 5. Profile Redesign (Environment-Oriented Only)
| Component / Profile           | `local`                      | `local-spark`                | `cloud-batch`                          | `cloud-online`           |
| :---------------------------- | :--------------------------- | :--------------------------- | :------------------------------------- | :----------------------- |
| **ExecutionTarget**           | `LOCAL`                      | `LOCAL`                      | `CLOUD`                                | `CLOUD`                  |
| **JobRunner**                 | `LocalJobRunner`             | `LocalSparkJobRunner`        | `DataprocJobRunner`                    | `CloudJobRunner`         |
| **ServiceRunner**             | `LocalServiceRunner`         | `LocalServiceRunner`         | `None`                                 | `EndpointServiceRunner`  |
| **Storage**                   | `LocalStorage`               | `LocalStorage`               | `CloudObjectStorage`                   | `CloudObjectStorage`     |
| **ArtifactStore**             | `LocalArtifactStore`         | `LocalArtifactStore`         | `CloudArtifactStore`                   | `CloudArtifactStore`     |
| **ExperimentTracker**         | `LocalJsonTracker`           | `LocalJsonTracker`           | `CloudTracker`                         | `CloudTracker`           |
| **DefaultInvocationStrategy** | `InProcessInvocation`        | `DistributedInvocation`      | `DistributedInvocation`                | `RESTInvocation`         |
| **Cloud Parity Mode**         | Native                       | Native                       | Native                                 | Native                   |
| **Primary Use Case**          | Single-machine dev & testing | Local distributed validation | Distributed training & batch inference | Online inference serving |

### Optional Extension (Cloud Emulation Profiles)
To support forced cloud behavior locally (as discussed in the design):

| Component / Profile           | `cloud-batch-emulated`                   | `cloud-online-emulated`                  |
| :---------------------------- | :--------------------------------------- | :--------------------------------------- |
| **ExecutionTarget**           | `EMULATED_CLOUD`                         | `EMULATED_CLOUD`                         |
| **JobRunner**                 | `LocalDataprocEmulator`                  | `LocalCloudJobEmulator`                  |
| **ServiceRunner**             | `None`                                   | `LocalEndpointEmulator`                  |
| **Storage**                   | `LocalStorage` (Cloud-compatible layout) | `LocalStorage` (Cloud-compatible layout) |
| **ArtifactStore**             | `LocalArtifactStore` (Cloud semantics)   | `LocalArtifactStore` (Cloud semantics)   |
| **ExperimentTracker**         | `LocalJsonTracker`                       | `LocalJsonTracker`                       |
| **DefaultInvocationStrategy** | `DistributedInvocation`                  | `RESTInvocation`                         |
| **Primary Use Case**          | Local simulation of cloud batch behavior | Local simulation of cloud online serving |



Below is the consolidated and formalized architecture definition incorporating all refinements discussed:

* Explicit separation of **WorkloadType**
* Explicit **ExecutionNature**
* Explicit **ExecutionTarget**
* Invocation decoupled and overrideable
* Runners removed from ExecutionContext
* Profiles purely environment-oriented
* Cloud ↔ Local parity supported


# 1. Architectural Overview

The platform separates concerns across four orthogonal dimensions:

1. **Workload Type** – What is being executed
2. **Execution Nature** – How it executes (lifecycle model)
3. **Execution Target** – Where it executes (environment class)
4. **Invocation Strategy** – How prediction is invoked

These dimensions are intentionally independent to ensure extensibility and cloud/local parity.

---

# 2. Core Conceptual Axes

## 2.1 WorkloadType

Defines the business domain of execution.

```
TrainingWorkload
InferenceWorkload
```

### TrainingWorkload

* Always produces artifacts
* Always procedural
* Always Job-based
* Requires ExperimentTracker
* Never uses InvocationStrategy

### InferenceWorkload

* Does not produce model artifacts
* May be Job-based (batch)
* May be Service-based (online)
* Requires InvocationStrategy

---

## 2.2 ExecutionNature

Defines lifecycle model.

```
JobExecution
ServiceExecution
```

### JobExecution

* Finite lifecycle
* Blocking or asynchronous
* Resource allocation required

### ServiceExecution

* Long-lived
* Request-response
* Scalable

---

## 2.3 ExecutionTarget

Defines infrastructure class.

```
LOCAL
CLOUD
EMULATED_CLOUD
```

### LOCAL

Native local infrastructure.

### CLOUD

Real cloud infrastructure.

### EMULATED_CLOUD

Local execution simulating cloud semantics for parity testing.

---

## 2.4 InvocationStrategy (Inference Only)

Defines how prediction logic is invoked.

```
InProcessInvocation
DistributedInvocation
RESTInvocation
```

Responsibilities:

* Transport adaptation
* Distribution adaptation
* Preserve BasePredictor contract

InvocationStrategy is:

* Required for InferenceWorkload
* Ignored for TrainingWorkload
* Defaulted by Profile
* Overrideable per Step

---

# 3. Core Architectural Components

---

## 3.1 Step

A Step declares:

* workload_type
* execution_nature

It does not resolve infrastructure.

```
Step
  ├── TrainStep
  └── InferenceStep
```

---

## 3.2 Runners (Infrastructure Drivers)

Runners execute steps.

```
JobRunner
ServiceRunner
```

They:

* Wrap execution
* Provide isolation
* Do not contain business logic

Steps never access runners.

---

## 3.3 ExecutionContext

Injected after resolution.

Contains:

* Storage
* ArtifactStore
* ExperimentTracker (training only)
* InvocationStrategy (inference only)
* RuntimeConfig
* EnvironmentMetadata
* Trainer or Predictor

Does NOT contain runners.

---

## 3.4 Infrastructure Abstractions

```
Storage
ArtifactStore
ExperimentTracker
```

### Storage

Physical persistence.

### ArtifactStore

Logical resolution and versioning.

### ExperimentTracker

Training metrics & metadata tracking.

---

# 4. Profile System

Profiles define infrastructure defaults only.

They specify:

* ExecutionTarget
* JobRunner
* ServiceRunner
* Storage
* ArtifactStore
* ExperimentTracker
* DefaultInvocationStrategy

Profiles do NOT encode:

* Training vs inference behavior
* Step intent

---

## Example Profiles

### local

* ExecutionTarget: LOCAL
* JobRunner: LocalJobRunner
* ServiceRunner: LocalServiceRunner
* Storage: LocalStorage
* ArtifactStore: LocalArtifactStore
* ExperimentTracker: LocalJsonTracker
* DefaultInvocationStrategy: InProcessInvocation

---

### cloud-batch

* ExecutionTarget: CLOUD
* JobRunner: DataprocJobRunner
* ServiceRunner: None
* Storage: CloudObjectStorage
* ArtifactStore: CloudArtifactStore
* ExperimentTracker: CloudTracker
* DefaultInvocationStrategy: DistributedInvocation

---

### cloud-online

* ExecutionTarget: CLOUD
* JobRunner: CloudJobRunner
* ServiceRunner: EndpointServiceRunner
* Storage: CloudObjectStorage
* ArtifactStore: CloudArtifactStore
* ExperimentTracker: CloudTracker
* DefaultInvocationStrategy: RESTInvocation

---

### cloud-batch-emulated

* ExecutionTarget: EMULATED_CLOUD
* JobRunner: LocalDataprocEmulator
* ServiceRunner: None
* Storage: LocalStorage (cloud-compatible layout)
* ArtifactStore: LocalArtifactStore (cloud semantics)
* ExperimentTracker: LocalJsonTracker
* DefaultInvocationStrategy: DistributedInvocation

---

# 5. Resolution Flow

Primitive resolution is deterministic.

```
Profile
    ↓
PrimitiveResolver
    ↓
ExecutionContext
    ↓
Runner.execute(step, context)
```

Resolution rules:

* Runner chosen by ExecutionNature
* InvocationStrategy:
  Step override OR Profile default
* ExperimentTracker injected only for TrainingWorkload
* EmulatedCloud swaps infrastructure adapters only

---

# 6. Execution Rules

---

## Training

* WorkloadType: TrainingWorkload
* ExecutionNature: JobExecution
* Uses JobRunner
* Requires ExperimentTracker
* No InvocationStrategy

---

## Inference (Batch)

* WorkloadType: InferenceWorkload
* ExecutionNature: JobExecution
* Uses JobRunner
* Uses InvocationStrategy (Distributed or InProcess)

---

## Inference (Online)

* WorkloadType: InferenceWorkload
* ExecutionNature: ServiceExecution
* Uses ServiceRunner
* Uses InvocationStrategy (REST)

---

# 7. Architectural Guarantees

This architecture guarantees:

* Explicit separation of training and inference domains
* Clear lifecycle distinction (Job vs Service)
* Environment isolation via ExecutionTarget
* Cloud ↔ Local parity via EMULATED_CLOUD
* No runner leakage into business logic
* Invocation mechanics isolated from prediction core
* Profile-driven infrastructure resolution
* Extensibility for new runners, storages, invocation strategies, or targets

---

# 8. Design Principles

* Orthogonal abstraction layers
* Infrastructure inversion
* Explicit workload semantics
* Environment-agnostic model code
* Local-first with cloud-scale symmetry
* Deterministic primitive resolution
* Composable and overrideable execution behavior

---

# Final Shape

The platform is no longer:

“A framework that runs ML code.”

It is now:

A structured execution platform with explicit workload semantics, lifecycle isolation, and environment symmetry.
