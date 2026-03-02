# Changelog

All notable changes to `mlplatform` are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [0.3.0] - 2026-03-02

### Changed
- **Pipeline restructure**: `example_model/pipeline/` now contains only DAG YAML files
  (`train.yaml`, `predict.yaml`). The former `dags/` subdirectory has been removed.
- **Step configs relocated**: `example_model/pipeline/steps/` content (train, inference envs)
  has moved to `mlplatform/mlplatform/config/` as framework-level generic templates.
  Model-specific details (module, class) are no longer embedded in the framework config.
- **New DAG format**: DAG YAML files now support a Databricks-like
  `resources.jobs.deployment` block for orchestration config alongside the existing
  framework values (`workflow_name`, `models`, `pipeline_type`, etc.).
- **`predict_chunk` renamed to `predict`**: `BasePredictor.predict_chunk()` is now
  `BasePredictor.predict()`. All internal call sites updated. **Breaking change** — update
  any predictor implementations to rename the method.

### Added
- **Config profiles**: DAG YAML files can declare `config: [global, dev]` to load and
  merge named config profile files from the `config/` directory at the project root.
  - `config/global.yaml` — baseline settings.
  - `config/local.yaml` — local development overrides.
  - `config/dev.yaml` — development environment overrides.
  - CLI: `mlplatform run --dag train.yaml --config global,local` overrides the YAML key.
  - Profiles are merged in order (later overrides earlier); DAG YAML values win last.
- **Config subpackage**: `pip install mlplatform[config]` installs only the config
  parsing layer (pyyaml + pydantic) without the full framework dependencies.
- **Schema module** (`mlplatform.schema`): `PredictionInputSchema` lets data scientists
  declare the expected input columns, dtypes, and required/optional flags.
  - `SchemaValidationError` raised on schema mismatch.
  - Import: `from mlplatform.schema import PredictionInputSchema`.
- **`config_profiles` field** on `WorkflowConfig`: records which config profiles were
  loaded for a given run.
- **`config_names` parameter** on `run_workflow`, `dev_predict`, `dev_context`:
  programmatic override of config profile selection.
- **`--config` CLI flag** on `mlplatform run`: comma-separated config profile names.
- **Spark `--input-path` / `--output-path` / `--project-root` CLI args**:
  `mlplatform/mlplatform/spark/main.py` now accepts these arguments to override config
  values and support local dev without building `root.zip`.
- **Local path bootstrap** in `spark/main.py`: when `--packages` is omitted, the module
  auto-adds `project_root` and `mlplatform/` to `sys.path` for local testing.
- **Tests**: pytest-based test suite in `tests/` with unit, integration, and e2e layers.
  - `tests/unit/` — config loader, schema, predictor interface, prediction schema.
  - `tests/integration/` — workflow runs, dev_predict, config merging.
  - `tests/e2e/` — full train-then-predict cycle.
- **CI**: GitHub Actions workflow (`.github/workflows/ci.yml`) running lint + tests on
  push/PR for Python 3.9, 3.10, and 3.11.
- **CD**: Disabled CD workflow scaffold (`.github/workflows/cd.yml`) for future JFrog
  publish and version tagging — uncomment when ready.

### Fixed
- `spark/main.py` `_run_spark_inference` now accepts `input_path` and `output_path`
  overrides, matching the expected CLI contract.

### Deprecated
- `predict_chunk` method name — replaced by `predict` with no alias kept.

---

## [0.2.0] - 2025-XX-XX

- Initial public release with pluggable Storage, ExperimentTracker, and InvocationStrategy.
- `BaseTrainer` / `BasePredictor` abstractions.
- Profiles: local, local-spark, cloud-train, cloud-online, cloud-batch.
- In-process, SparkBatch, and FastAPI invocation strategies.
- YAML-driven DAG config loading.
- CLI: `mlplatform run` and `mlplatform build-package`.
- PySpark batch prediction via `mapInPandas`.
- Spark config serializer and `build_root_zip` packager.
