"""Pytest fixtures for part-failure model tests."""

import sys
from pathlib import Path

import pytest

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def project_root():
    """Project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def synthetic_train_data():
    """Synthetic train_data dict matching preprocess output."""
    from custom.data_loader import create_synthetic_training_data

    X, y = create_synthetic_training_data(n_samples=200, n_features=10, seed=42)
    # Split like preprocess (80/20)
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
    }


@pytest.fixture
def trained_model(synthetic_train_data):
    """Trained PartFailureModel using synthetic data."""
    from model import PartFailureModel

    train_data = synthetic_train_data
    model = PartFailureModel(n_estimators=10, max_depth=5, random_state=42)
    model.fit(train_data["X_train"], train_data["y_train"])
    return model


@pytest.fixture
def tmp_artifacts(tmp_path):
    """Temporary directory for artifacts (used as artifacts_path)."""
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    return artifacts_dir


@pytest.fixture
def exec_context(tmp_artifacts, synthetic_train_data, trained_model):
    """ExecutionContext with storage pre-seeded with train_data and model."""
    from mlops_framework.backends.storage.local import LocalStorage
    from mlops_framework.backends.tracking.noop import NoOpTracker
    from mlops_framework.core.context import ExecutionContext

    storage = LocalStorage(base_path=str(tmp_artifacts))
    storage.save("train_data", synthetic_train_data)
    storage.save("model", trained_model)

    context = ExecutionContext(
        storage=storage,
        tracker=NoOpTracker(),
        logger=lambda _: None,
        config={
            "reference_artifact": "train_data",
            "current_data_path": "data/production_sample.csv",
            "validation_path": "data/validation_latest.csv",
        },
        run_context=None,
    )
    return context
