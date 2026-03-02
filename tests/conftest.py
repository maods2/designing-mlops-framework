"""Shared pytest fixtures for mlplatform tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import pandas as pd
from sklearn.datasets import make_classification

# Ensure the monorepo root and mlplatform are importable
_repo_root = Path(__file__).resolve().parent.parent
for _p in [str(_repo_root), str(_repo_root / "mlplatform")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return _repo_root


@pytest.fixture(scope="session")
def train_dag_path(repo_root: Path) -> Path:
    return repo_root / "example_model" / "pipeline" / "train.yaml"


@pytest.fixture(scope="session")
def predict_dag_path(repo_root: Path) -> Path:
    return repo_root / "example_model" / "pipeline" / "predict.yaml"


@pytest.fixture(scope="session")
def legacy_train_dag_path(repo_root: Path) -> Path:
    return repo_root / "template_training_dag.yaml"


@pytest.fixture(scope="session")
def legacy_predict_dag_path(repo_root: Path) -> Path:
    return repo_root / "template_prediction_dag.yaml"


@pytest.fixture
def sample_train_data() -> dict:
    """Small synthetic training dataset (50 rows, 5 features)."""
    X, y = make_classification(n_samples=50, n_features=5, random_state=42)
    return {
        "X": pd.DataFrame(X, columns=["f0", "f1", "f2", "f3", "f4"]),
        "y": pd.Series(y),
    }


@pytest.fixture
def sample_inference_df() -> pd.DataFrame:
    """Small synthetic inference DataFrame (10 rows, 5 features)."""
    X, _ = make_classification(n_samples=10, n_features=5, random_state=99)
    return pd.DataFrame(X, columns=["f0", "f1", "f2", "f3", "f4"])


@pytest.fixture
def artifacts_dir(tmp_path: Path) -> Path:
    """Temporary directory for test artifacts."""
    d = tmp_path / "artifacts"
    d.mkdir()
    return d
