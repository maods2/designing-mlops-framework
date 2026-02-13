"""Build Airflow DAG from PipelineDef."""

from datetime import datetime

from mlops_framework.compiler.yaml_parser import PipelineDef, StepDef


def build_airflow_dag(pipeline_def: PipelineDef) -> str:
    """
    Generate Airflow DAG Python code from PipelineDef.

    Each step becomes a PythonOperator that runs the step class. RunContext
    is passed via op_kwargs from DAG params; use VertexRunner when base_path
    is gs://, else LocalRunner.

    Example DAG params (RunContext as dict):
        default_args={"run_context": {
            "base_path": "gs://my-bucket/artifacts",
            "run_id": "{{ ds_nodash }}",
            "model_name": "churn_v1",
            "tracking_enabled": True,
        }}
    """
    dag_id = pipeline_def.name.replace("-", "_")
    steps = pipeline_def.steps

    run_step_body = '''
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
'''

    lines = [
        '"""Auto-generated Airflow DAG from MLOps pipeline YAML."""',
        "",
        "from datetime import datetime",
        "from airflow import DAG",
        "from airflow.operators.python import PythonOperator",
        "",
        run_step_body,
        "",
        "with DAG(",
        f'    dag_id="{dag_id}",',
        '    start_date=datetime(2024, 1, 1),',
        "    catchup=False,",
        '    tags=["mlops", "generated"],',
        "    params={\"run_context\": DEFAULT_RUN_CONTEXT},",
        ") as dag:",
    ]

    task_ids = []
    for s in steps:
        task_id = f"step_{s.id}"
        task_ids.append(task_id)
        lines.append(f'    {task_id} = PythonOperator(')
        lines.append(f'        task_id="{task_id}",')
        lines.append(f'        python_callable=run_step,')
        lines.append(f'        op_kwargs={{"step_id": "{s.id}", "class_path": "{s.class_path}", "run_context": DEFAULT_RUN_CONTEXT}},')
        lines.append("    )")
        lines.append("")

    # Add dependencies (inside with block)
    for s in steps:
        task_id = f"step_{s.id}"
        for dep in s.depends_on:
            dep_task = f"step_{dep}"
            if dep_task in task_ids:
                lines.append(f"    {dep_task} >> {task_id}")
                lines.append("")

    return "\n".join(lines)
