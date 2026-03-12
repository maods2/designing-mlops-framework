"""Minimal setup.py for editable install compatibility.

Provides name/version when pip falls back to setup.py develop.
Full config (dependencies, extras) comes from pyproject.toml.
"""
import re
from pathlib import Path

from setuptools import setup

_version_file = Path(__file__).resolve().parent / "mlplatform" / "_version.py"
_match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', _version_file.read_text())
_version = _match.group(1) if _match else "0.0.0"

setup(name="mlplatform", version=_version)
