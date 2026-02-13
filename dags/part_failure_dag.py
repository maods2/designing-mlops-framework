"""Auto-generated Airflow DAG from MLOps pipeline YAML."""

from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator


DEFAULT_RUN_CONTEXT = {"base_path": "./artifacts", "run_id": "run_default", "tracking_enabled": False, "tracking_backend": "noop"}

def run_step(step_id: str, class_path: str, run_context: dict = None, project_root: str = None, env: str = "dev", **context):
    import importlib
    import os
    import yaml
    from pathlib import Path

    # run_context from op_kwargs, DAG params, or default
    rc_params = (context.get("params") or {}).get("run_context")
    run_context = run_context or rc_params or DEFAULT_RUN_CONTEXT
    project_root = Path(project_root or os.getcwd())
    if str(project_root) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(project_root))

    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    step_class = getattr(module, class_name)

    # Merge step config from pipeline/config.yaml
    config_path = project_root / "pipeline" / "config.yaml"
    full_config = {}
    if config_path.exists():
        with open(config_path, "r") as f:
            full_config = yaml.safe_load(f) or {}
    steps_cfg = full_config.get("steps", {})
    env_cfg = full_config.get("environments", {}).get(env, {}).get(step_id, {})
    step_base = steps_cfg.get(step_id, {}).copy()
    for k, v in env_cfg.items():
        step_base[k] = v
    config = step_base

    from mlops_framework.core.run_context import RunContext
    rc = RunContext.model_validate(run_context) if run_context else None

    if rc and rc.base_path.startswith("gs://"):
        from urllib.parse import urlparse
        parsed = urlparse(rc.base_path)
        bucket_name = parsed.netloc
        from mlops_framework.backends.execution.vertex_runner import VertexRunner
        runner = VertexRunner(bucket_name=bucket_name, experiment_name=rc.experiment_name or "default", run_id=rc.run_id)
        runner.run_context = rc
    else:
        from mlops_framework.backends.execution.local_runner import LocalRunner
        runner = LocalRunner(
            artifacts_path=str(project_root / "artifacts"),
            runs_path=str(project_root / "runs"),
            tracking=rc.tracking_enabled if rc else False,
            tracking_backend=rc.tracking_backend if rc else "noop",
            run_context=rc,
        )
    runner.run_step(step_class, config=config)


with DAG(
    dag_id="part_failure_pipeline",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "generated"],
    params={"run_context": DEFAULT_RUN_CONTEXT},
) as dag:
    step_preprocess = PythonOperator(
        task_id="step_preprocess",
        python_callable=run_step,
        op_kwargs={"step_id": "preprocess", "class_path": "steps.preprocess.PartFailurePreprocess", "run_context": DEFAULT_RUN_CONTEXT},
    )

    step_train = PythonOperator(
        task_id="step_train",
        python_callable=run_step,
        op_kwargs={"step_id": "train", "class_path": "steps.train.PartFailureTrain", "run_context": DEFAULT_RUN_CONTEXT},
    )

    step_inference = PythonOperator(
        task_id="step_inference",
        python_callable=run_step,
        op_kwargs={"step_id": "inference", "class_path": "steps.inference.PartFailureInference", "run_context": DEFAULT_RUN_CONTEXT},
    )

    step_preprocess >> step_train

    step_train >> step_inference
