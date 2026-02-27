# Framework analysis and improvement proposal

## 1) Current framework strengths

The current framework already gives a clean entry point for data scientists:

- YAML-driven workflows with model module entrypoints (`module`) and per-model optional configs.
- A clear `ExecutionContext` abstraction for artifact save/load and experiment logging.
- Profile-based infra switching (`local`, `local-spark`, `cloud-batch`, etc.) for storage/tracking/invocation strategy.
- Local-to-cloud packaging flow (`root.zip` + Spark entrypoint).

This makes experimentation simple and keeps model code lightweight.

## 2) Main gaps against your target operating model

Based on the existing schema and loader, there are a few gaps for multi-environment + multi-country operation:

1. **Flat config model**
   - Current workflow config is a single DAG file with model list.
   - No native multi-layer merge of defaults + environment + country + runtime overrides.

2. **Profile granularity**
   - Runtime `profile` currently selects infra adapters, but not a typed config contract per profile.
   - No concept of *required keys* by profile (e.g., `prod` requiring stricter settings).

3. **Job and orchestration coupling**
   - DAG file stores both workflow and job details directly.
   - No reusable job catalog with explicit dependencies that can be promoted across envs/countries.

4. **Cloud runtime config is underspecified**
   - `compute` is currently a coarse label (`xs/s/m/l`) without standardized mapping per provider.
   - Image names, service account, network, region, retry policy are not first-class.

5. **Validation and governance**
   - No explicit schema validation for required fields per pipeline type/profile.
   - Risk of late failure at runtime instead of early failure at config load time.

## 3) Proposed config architecture

Introduce a **4-layer config composition model**.

### Layer A: global defaults
Shared across all environments and geographies.

```yaml
# config/defaults.yaml
project:
  name: churn-platform
  artifact_root: gs://ml-artifacts

runtime_defaults:
  compute_class: s
  retries: 1
  timeout_minutes: 120

tracking:
  provider: vertexai
  experiment_prefix: churn
```

### Layer B: profile overlays (dev/qa/prod)
Defines required controls and cloud policy by environment.

```yaml
# config/profiles/prod.yaml
inherits: defaults
required:
  - cloud.region
  - cloud.service_account
  - cloud.vpc

cloud:
  region: europe-west1
  service_account: ml-prod@my-proj.iam.gserviceaccount.com
  vpc: projects/my-proj/global/networks/prod-vpc
  subnet: regions/europe-west1/subnetworks/prod-ml
  labels:
    environment: prod

runtime_defaults:
  retries: 3
  timeout_minutes: 240
```

### Layer C: country/domain overlays (CountryA/CountryB)
Local data policy + dataset naming + residency constraints.

```yaml
# config/domains/CountryA.yaml
required:
  - data.raw_dataset
  - data.prediction_output_dataset

data:
  residency_region: europe-west1
  raw_dataset: countryA_raw
  prediction_output_dataset: countryA_predictions
```

### Layer D: job definitions + pipeline DAG definitions
A reusable job catalog and DAG references.

```yaml
# config/jobs/train_model.yaml
job_name: train_model
entrypoint: example_model.train
type: training
artifacts:
  outputs: [model.pkl, metrics.json]
cloud:
  image: europe-west1-docker.pkg.dev/my-proj/ml/train:1.2.0
  compute:
    class: m
    vertex:
      machine_type: n1-standard-8
      accelerator: null
```

```yaml
# config/pipelines/train_countryA_prod.yaml
pipeline_name: train_countryA_prod
profile: prod
domain: CountryA
jobs:
  - id: prepare_features
    use: prepare_features
  - id: train_model
    use: train_model
    depends_on: [prepare_features]
  - id: evaluate_model
    use: evaluate_model
    depends_on: [train_model]
```

## 4) Resolution and precedence logic

At runtime, build one resolved config object:

1. load defaults
2. merge profile overlay
3. merge domain overlay
4. merge pipeline/job-level overrides
5. apply CLI/runtime overrides (highest precedence)

Conflict policy:
- deep merge for dicts
- replace for scalar/list unless explicitly marked mergeable
- blocked keys for lower layers (e.g., domain cannot override `cloud.service_account`)

## 5) Required-config contract by profile

Add explicit validation rules per profile:

- `dev`: permissive, local defaults allowed.
- `qa`: requires deterministic dataset pointers and image tag.
- `prod`: requires region, service account, VPC, KMS key, retry policy, observability tags, and approved image registry.

Recommended implementation:
- Define a Pydantic model for the resolved config.
- Add profile-specific validators (`if profile == "prod" then ...`).
- Fail fast during config load.

## 6) Job model and cloud runtime contract

For each job, make cloud runtime explicit and portable:

```yaml
cloud:
  platform: vertexai
  image: ...
  compute:
    class: m
    cpu: 8
    memory_gb: 32
    gpu: 0
  scheduling:
    max_retries: 3
    timeout_minutes: 180
  execution:
    service_account: ...
    region: ...
    network: ...
```

Then implement a translation layer:

- abstract job spec -> Vertex AI custom job spec
- abstract job spec -> Dataproc batch spec
- abstract job spec -> local docker run (optional)

This keeps user-facing YAML stable while infrastructure adapters evolve.

## 7) Automatic DAG orchestration

Use `depends_on` in pipeline config to auto-build a DAG graph:

- Parse jobs into nodes.
- Validate acyclic graph.
- Topologically sort for execution.
- Dispatch nodes to profile-specific invocation strategy.

Optional next step: compiler/exporters
- Export to Airflow DAG.
- Export to Vertex Pipelines/KFP.
- Keep local runner for development parity.

## 8) Artifact and model logging improvements

Strengthen reproducibility and governance:

- Standard metadata for every run:
  - git commit SHA
  - resolved config hash
  - data snapshot/table version
  - container image digest
- Artifact manifest saved beside model:
  - `artifacts_manifest.json` with URI, checksum, producer job
- Promote model registry record:
  - training metrics
  - evaluation gate results
  - source pipeline and environment

## 9) Suggested implementation plan

### Phase 1 (low risk)
- Add config composition loader and deep-merge utilities.
- Add resolved-config validation with profile-specific required keys.
- Keep current DAG schema backward compatible.

### Phase 2
- Introduce job catalog + pipeline DAG references (`use:`).
- Add explicit cloud runtime blocks and provider translators.

### Phase 3
- Add DAG compiler/exporters (Airflow/KFP) while preserving local runner.
- Add policy checks and promotion workflow (qa -> prod).

## 10) Why this helps data scientists

- They keep writing only model logic (`entrypoint`) and optional hyperparameters.
- Environment/country differences live in overlays, not duplicated DAG files.
- Deployment and orchestration are generated from declarative config.
- Failures are caught at config validation time instead of during cloud job execution.
