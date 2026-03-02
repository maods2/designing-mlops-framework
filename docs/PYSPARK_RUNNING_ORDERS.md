# PySpark Running Orders

Implementation orders for PySpark batch prediction and training. Use this as the canonical reference when running or implementing PySpark jobs.

---

## 1. Entry Point: main.py

**`mlplatform/mlplatform/spark/main.py`** is the single entry point for PySpark jobs.

- Loads CLI arguments and calls the batch prediction (or training) runner.
- Passes `--config`, `--input-path`, `--output-path`, `--packages` to the framework.

### CLI Arguments

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `--config` | Yes | Path to run config JSON (local file or `gs://` URI) |
| `--input-path` | No | Override input path (file or GCS). If omitted, use value from config. |
| `--output-path` | No | Override output path (file or GCS). If omitted, use value from config. |
| `--packages` | No | Comma-separated paths to zip packages (e.g. `dist/root.zip`) |
| `--project-root` | No | Project root for local path bootstrap (only when `--packages` is not used) |

---

## 2. Path Resolution

**With `--packages` (cloud-like):**

- Add each zip to `sys.path`.
- `example_model` and `mlplatform` are resolved from inside the zip.
- Use for cloud runs and local tests that simulate cloud.

**Without `--packages` (local dev):**

- Add `project_root` and `project_root/mlplatform` to `sys.path`.
- Use when developing locally without building `root.zip`.

---

## 3. Cloud Run (Dataproc / VertexAI)

### Build root.zip

```bash
mlplatform build-package --model-package example_model --project-root . --output-dir dist
```

### root.zip Contents

```text
root.zip
  example_model/
    __init__.py
    train.py
    predict.py
    constants.py
    ...
  mlplatform/   # loaded inside train and predict code
    __init__.py
    core/
    config/
    invocation/
    ...
  config/       # optional, included when present in project
```

### Generate Config JSON

```python
from mlplatform.config import load_pipeline_config
from mlplatform.spark.config_serializer import write_workflow_config

pipeline = load_pipeline_config(
    "example_model/pipeline/predict.yaml",
    task_id="predict",
    config_names=["global", "predict-local"],
)
task_cfg = pipeline.tasks[0]
write_workflow_config(
    pipeline, task_cfg,
    "dist/spark_config.json",
    base_path="gs://bucket/artifacts",
    version="v1.0",
    profile="local-spark",
)
```

### Submit to Spark

```bash
spark-submit \
  mlplatform/mlplatform/spark/main.py \
  --py-files dist/root.zip \
  -- \
  --config gs://bucket/spark_config.json \
  --input-path gs://bucket/input.parquet \
  --output-path gs://bucket/output.parquet \
  --packages dist/root.zip
```

- `--py-files dist/root.zip` distributes the zip to workers.
- `--packages dist/root.zip` adds it to `sys.path` so imports work on driver and workers.

---

## 4. Local Run (No Zip)

For quick local testing without building `root.zip`:

```bash
python -m mlplatform.spark.main \
  --config dist/spark_config.json \
  --input-path example_model/data/sample_inference.csv \
  --output-path dist/predictions.parquet
```

- No `--packages`: project root and mlplatform are added to `sys.path`.
- Config JSON must exist (e.g. from `write_workflow_config`).

---

## 5. Local Run (With Zip)

To test the same layout as cloud:

```bash
# 1. Build root.zip
mlplatform build-package --model-package example_model --output-dir dist

# 2. Generate config (or use existing)
# ... write_workflow_config ...

# 3. Run with --packages
python -m mlplatform.spark.main \
  --config dist/spark_config.json \
  --input-path example_model/data/sample_inference.csv \
  --output-path dist/predictions.parquet \
  --packages dist/root.zip
```

---

## 6. Project Layout for Local Testing

```text
project_root/
  example_model/
    train.py
    predict.py
    constants.py
    data/
      sample_train.csv
      sample_inference.csv
  dist/
    root.zip
  mlplatform/
  config/
  spark_pred_config.json
  predictions.parquet
```

---

## 7. Orchestrator Integration

When an external orchestrator (e.g. Databricks, Airflow) runs the job:

1. Build `root.zip` (or use a pre-built artifact).
2. Generate config JSON with `write_workflow_config`.
3. Invoke `spark-submit` with `--py-files root.zip` and pass `--config`, `--input-path`, `--output-path`, `--packages`.

The orchestrator should pass `--packages` with the path to `root.zip` so imports resolve correctly.

---

## 8. Checklist

- [ ] Build `root.zip` with `mlplatform build-package`
- [ ] Generate config JSON with `write_workflow_config`
- [ ] Cloud: use `--py-files root.zip` and `--packages root.zip`
- [ ] Local dev: omit `--packages` for path bootstrap
- [ ] Local test with zip: use `--packages dist/root.zip` to simulate cloud-like
