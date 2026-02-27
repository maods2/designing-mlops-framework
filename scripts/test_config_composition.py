#!/usr/bin/env python3
"""Tests for composed config loading and DAG ordering."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_composition_dev_country():
    from mlplatform.config.loader import load_workflow_config

    workflow = load_workflow_config(
        ROOT / "template_orchestrated_training_dag.yaml",
        config_profile="dev",
        domain="CountryA",
    )
    assert workflow.workflow_name == "eds_orchestrated_training"
    assert len(workflow.models) == 2
    assert workflow.models[1].depends_on == ["prepare_features"]
    assert workflow.models[1].image == "us-docker.pkg.dev/my-project/ml/train:latest"
    print("PASS test_composition_dev_country")


def test_prod_required_keys_fail_fast():
    from mlplatform.config.loader import load_workflow_config

    try:
        load_workflow_config(ROOT / "template_orchestrated_training_dag.yaml", config_profile="prod", domain="CountryA")
    except ValueError as exc:
        assert "cloud.service_account" in str(exc)
        print("PASS test_prod_required_keys_fail_fast")
        return
    raise AssertionError("Expected required-key validation failure for prod profile")


def test_topological_order():
    from mlplatform.config.loader import load_workflow_config
    from mlplatform.config.dag import topological_order

    workflow = load_workflow_config(ROOT / "template_orchestrated_training_dag.yaml", config_profile="dev", domain="CountryA")
    ordered = topological_order(workflow.models)
    assert [m.model_name for m in ordered] == ["prepare_features", "train_model"]
    print("PASS test_topological_order")


if __name__ == "__main__":
    test_composition_dev_country()
    test_prod_required_keys_fail_fast()
    test_topological_order()
