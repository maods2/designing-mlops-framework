"""Unit tests for mlplatform.storage."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from mlplatform.storage.local import LocalFileSystem


class TestLocalFileSystem:
    def test_save_load_joblib(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalFileSystem(tmp)
            store.save("a/b/model.pkl", {"x": 1})
            loaded = store.load("a/b/model.pkl")
            assert loaded == {"x": 1}

    def test_save_bytes_load_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalFileSystem(tmp)
            store.save_bytes("data.json", b'{"a": 1}')
            loaded = store.load("data.json")
            assert loaded == b'{"a": 1}'

    def test_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalFileSystem(tmp)
            assert store.exists("x") is False
            store.save("x", 1)
            assert store.exists("x") is True

    def test_list_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalFileSystem(tmp)
            store.save("a/1.pkl", 1)
            store.save("a/2.pkl", 2)
            store.save("b/3.pkl", 3)
            listed = store.list_artifacts("")
            assert len(listed) == 3
            assert "a/1.pkl" in listed
            assert "a/2.pkl" in listed
            assert "b/3.pkl" in listed

    def test_list_artifacts_with_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalFileSystem(tmp)
            store.save("a/1.pkl", 1)
            store.save("a/2.pkl", 2)
            store.save("b/3.pkl", 3)
            listed = store.list_artifacts("a")
            assert len(listed) == 2
            assert "a/1.pkl" in listed
            assert "a/2.pkl" in listed

    def test_delete(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalFileSystem(tmp)
            store.save("x", 1)
            assert store.exists("x") is True
            store.delete("x")
            assert store.exists("x") is False

    def test_creates_base_path_if_not_exists(self):
        """LocalFileSystem creates base_path directory on init if it does not exist."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "nested" / "artifacts"
            assert not base.exists()
            store = LocalFileSystem(str(base))
            assert base.exists()
            assert base.is_dir()
