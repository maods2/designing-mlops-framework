"""Shared pytest fixtures for mlplatform framework tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure mlplatform package and project root are importable
_mlplatform_pkg = Path(__file__).resolve().parent.parent
_repo_root = _mlplatform_pkg.parent
for _p in [str(_repo_root), str(_mlplatform_pkg)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return _repo_root


@pytest.fixture(scope="session")
def legacy_train_dag_path(repo_root: Path) -> Path:
    return repo_root / "template_training_dag.yaml"


@pytest.fixture(scope="session")
def legacy_predict_dag_path(repo_root: Path) -> Path:
    return repo_root / "template_prediction_dag.yaml"


@pytest.fixture(scope="session")
def train_dag_path(repo_root: Path) -> Path:
    return repo_root / "example_model" / "pipeline" / "train.yaml"


@pytest.fixture(scope="session")
def predict_dag_path(repo_root: Path) -> Path:
    return repo_root / "example_model" / "pipeline" / "predict.yaml"
