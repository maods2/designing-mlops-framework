"""Unit tests for mlplatform.artifacts."""

import tempfile
from pathlib import Path

import pytest

from mlplatform.artifacts import Artifact, create_artifacts
from mlplatform.artifacts.registry import ArtifactRegistry


class TestCreateArtifacts:
    def test_standalone_local(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = create_artifacts(
                backend="local",
                base_path=tmp,
                feature_name="demo",
                model_name="model",
                version="v1",
            )
            assert isinstance(artifacts, ArtifactRegistry)
            path = artifacts.resolve_path("model.pkl")
            assert path == "demo/model/v1/model.pkl"
            artifacts.save("model.pkl", {"x": 1})
            loaded = artifacts.load("model.pkl")
            assert loaded == {"x": 1}

    def test_standalone_with_feature_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = create_artifacts(
                backend="local",
                base_path=tmp,
                feature="churn",
                feature_name=None,
                model_name="model",
                version="v1",
            )
            path = artifacts.resolve_path("x.pkl")
            assert "churn" in path

    def test_requires_feature_or_feature_name(self):
        with pytest.raises(ValueError, match="feature_name|feature"):
            create_artifacts(
                backend="local",
                base_path="/tmp",
                model_name="m",
                version="v1",
            )

    def test_gcs_requires_bucket_or_base_bucket(self):
        with pytest.raises(ValueError, match="bucket|base_bucket"):
            create_artifacts(
                backend="gcs",
                feature_name="f",
                model_name="m",
                version="v1",
            )


class TestArtifactAlias:
    def test_artifact_local(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Artifact(
                model_name="m",
                feature="f",
                version="v1",
                base_path=tmp,
                backend="local",
            )
            assert isinstance(artifact, ArtifactRegistry)
            artifact.save("x.pkl", {"a": 1})
            assert artifact.load("x.pkl") == {"a": 1}

    def test_artifact_project_id_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Artifact(
                model_name="m",
                feature="f",
                project_id="proj-1",
                base_path=tmp,
                backend="local",
            )
            artifact.save("x.pkl", 1)
            assert artifact.load("x.pkl") == 1


class TestArtifactRegistry:
    def test_save_load_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = create_artifacts(
                backend="local",
                base_path=tmp,
                feature_name="f",
                model_name="m",
                version="v1",
            )
            artifacts.save("metrics.json", {"accuracy": 0.95})
            m = artifacts.load("metrics.json")
            assert m["accuracy"] == 0.95

    def test_save_load_metadata(self):
        from datetime import datetime, timezone

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = create_artifacts(
                backend="local",
                base_path=tmp,
                feature_name="f",
                model_name="m",
                version="v1",
            )
            metadata = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "extra": "value",
            }
            artifacts.save("metadata.json", metadata)
            m = artifacts.load("metadata.json")
            assert m["extra"] == "value"
            assert "timestamp" in m

    def test_unified_save_figure(self):
        """save() with .png path and Figure stores as PNG bytes."""
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = create_artifacts(
                backend="local",
                base_path=tmp,
                feature_name="f",
                model_name="m",
                version="v1",
            )
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ax.plot([1, 2, 3])
            artifacts.save("plot.png", fig)
            plt.close(fig)
            raw = artifacts.load("plot.png")
            assert isinstance(raw, bytes)
            assert raw[:8] == b"\x89PNG\r\n\x1a\n"

    def test_unified_save_json(self):
        """save() with .json path and dict stores as JSON; load() returns dict."""
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = create_artifacts(
                backend="local",
                base_path=tmp,
                feature_name="f",
                model_name="m",
                version="v1",
            )
            artifacts.save("metrics.json", {"acc": 0.95})
            m = artifacts.load("metrics.json")
            assert m["acc"] == 0.95

    def test_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = create_artifacts(
                backend="local",
                base_path=tmp,
                feature_name="f",
                model_name="m",
                version="v1",
            )
            assert artifacts.exists("x.pkl") is False
            artifacts.save("x.pkl", 1)
            assert artifacts.exists("x.pkl") is True

    def test_resolve_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = create_artifacts(
                backend="local",
                base_path=tmp,
                feature_name="demo",
                model_name="model",
                version="v1",
            )
            assert artifacts.resolve_path("model.pkl") == "demo/model/v1/model.pkl"


class TestNoneStorage:
    """When backend is not set, storage is inactive: save no-ops, load raises."""

    def test_save_noop_warns(self, caplog):
        import logging

        caplog.set_level(logging.WARNING)
        artifacts = create_artifacts(
            feature_name="f",
            model_name="m",
            version="v1",
        )
        artifacts.save("x.pkl", {"a": 1})
        assert any("Storage backend not activated" in r.message for r in caplog.records)

    def test_load_raises_warns(self, caplog):
        import logging

        caplog.set_level(logging.WARNING)
        artifacts = create_artifacts(
            feature_name="f",
            model_name="m",
            version="v1",
        )
        with pytest.raises(FileNotFoundError, match="not activated"):
            artifacts.load("x.pkl")

    def test_exists_returns_false(self):
        artifacts = create_artifacts(
            feature_name="f",
            model_name="m",
            version="v1",
        )
        assert artifacts.exists("x.pkl") is False
