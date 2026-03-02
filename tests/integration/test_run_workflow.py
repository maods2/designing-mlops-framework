"""Integration tests for run_workflow (training and prediction)."""

from __future__ import annotations

import pytest

from mlplatform.runner import run_workflow, _build_context
from mlplatform.config.loader import load_workflow_config


class TestRunTrainingWorkflow:
    def test_training_workflow_runs(self, legacy_train_dag_path, sample_train_data, artifacts_dir):
        """Training workflow completes and model artifact is saved."""
        workflow = load_workflow_config(legacy_train_dag_path)
        model_cfg = workflow.models[0]
        ctx = _build_context(workflow, model_cfg, "local", "test_v1", str(artifacts_dir))
        ctx.optional_configs["train_data"] = sample_train_data

        from example_model.train import MyTrainer
        trainer = MyTrainer()
        trainer.context = ctx
        trainer.train()

        model_path = (
            artifacts_dir
            / workflow.feature_name
            / model_cfg.model_name
            / "test_v1"
            / "model.pkl"
        )
        assert model_path.exists(), f"Model artifact not found at {model_path}"

    def test_run_workflow_returns_ok(self, legacy_train_dag_path, sample_train_data, artifacts_dir, monkeypatch):
        """run_workflow returns 'ok' for each model on success."""
        workflow = load_workflow_config(legacy_train_dag_path)
        model_cfg = workflow.models[0]

        original_build = _build_context.__wrapped__ if hasattr(_build_context, "__wrapped__") else None

        import mlplatform.runner as runner_mod

        orig_build = runner_mod._build_context

        def patched_build(wf, mc, profile, version, base_path, commit_hash=None):
            ctx = orig_build(wf, mc, profile, version, str(artifacts_dir), commit_hash)
            ctx.optional_configs["train_data"] = sample_train_data
            return ctx

        monkeypatch.setattr(runner_mod, "_build_context", patched_build)

        results = run_workflow(legacy_train_dag_path, base_path=str(artifacts_dir))
        assert all(v == "ok" for v in results.values()), f"Unexpected results: {results}"


class TestNewDagTrainingWorkflow:
    def test_new_dag_format_loads(self, train_dag_path):
        """New DAG format with resources.jobs.deployment block loads correctly."""
        cfg = load_workflow_config(train_dag_path)
        assert cfg.pipeline_type == "training"
        assert cfg.workflow_name == "example_workflow_sequential"
        assert len(cfg.models) >= 1

    def test_new_dag_with_config_profiles(self, train_dag_path, repo_root):
        """New DAG loads config profiles declared in config: key."""
        cfg = load_workflow_config(train_dag_path)
        assert len(cfg.config_profiles) > 0
