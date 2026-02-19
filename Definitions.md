### Framework Components
 - **Orchestrator**
 - **ML Framework Lib**
 - **Model code**

Responsabilities: 
 - Orchestrator (Airflow - Not implemented here)
    > Define and provide args for Runner, Storage, ETB types based on configuration defined for step and enviroment evoked. It will trigger the step.

    > env=dev, runner=LocalRunner, storage=LocalFileSystem, ETB=LocalJson, StepType=TrainStep, stepName=
 - ML Framework Lib
    > Provide common utility functions across models, such as cabability to save and load artifacts, and wrap up core code used to manage the computational primitives 
 - Model code
    > It will implement model code and logic, as well as, the process of saving and loading artifacts.

### Primitives
- Runner:
    - LocalRunner
    - Dataproc
    - VertexAI

- Storage:
    - LocalFileSystem
    - GCS

- Experiment Tracking Backend (ETB):
    - None
    - LocalJson
    - MlFlow
    - Vertex AI

- Step:
    - TrainStep
        - Execution mode:
            - Procedural (Local)
            - Batch (Dataproc)
    - InferenceStep
        - Execution mode:
            - Procedural (Local)
            - OnlineREST (Vertex AI)
            - SparkBatch (Dataproc)


- Feature > Model > Version > artifacts

### Serving Modes Summary

| Serving mode | Entry point | Invocation | Typical deployment |
|--------------|-------------|------------|---------------------|
| **BatchLocal** | `predictor.run()` | In-process | Script / CLI |
| **OnlineREST** | FastAPI `/predict` | Request–response | Vertex AI Endpoint, Cloud Run |
| **BatchSpark** | `mapInPandas` partition fn | Map-side | Dataproc |

All three use the same core logic (`BasePredictor.load_model()` + `predict_chunk()`); only the **invocation wrapper** differs:

```
                    BasePredictor (load_model, predict_chunk)
                                    │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
        ▼                          ▼                          ▼
   ProceduralLocal                 OnlineREST              BatchSpark
   ──────────                 ───────────             ──────────
   Entry: run()               Entry: /predict          Entry: mapInPandas
   Invoke: in-process         Invoke: HTTP             Invoke: map-side
   Deploy: script             Deploy: service          Deploy: cluster job
```

## 8. Data Flow: How It Fits Together

```
Pipeline Definition (YAML)
    │
    ├── Steps: [Train, Inference]  ← 
Orchestrator (Airflow DAG receives pipeline yaml config)    
    │
    └── RunContext / Config
            │
            ├── Runner: LocalRunner | VertexAI | dataproc
            ├── storage: LocalFileSystem | gcs 
            ├── Experiment Tracking Backend (ETB): None | LocalJson | Vertex AI | mlflow   (for TrainStep)
            └── serving:  Procedural (Local) | sparSparkBatch (Dataproc)
k | OnlineREST (Vertex AI)            (for InferenceStep)
                    │
                    ▼
                    │
                    ▼
            ExecutionContext(storage, tracker, ...)
                    │
                    ▼
            Step(context).run()
```
