"""Shared pytest fixtures for mlplatform framework tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure mlplatform package and project root are importable
_repo_root = Path(__file__).resolve().parent.parent
_mlplatform_src = _repo_root / "mlplatform"
for _p in [str(_repo_root), str(_mlplatform_src)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return _repo_root
