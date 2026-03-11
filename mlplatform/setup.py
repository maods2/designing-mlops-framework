"""Minimal setup.py for PEP 660 editable install support.

When pip falls back to setup.py develop, explicit name/version ensure
correct metadata display (avoids UNKNOWN 0.0.0).
"""
import re
from pathlib import Path

from setuptools import setup

_version_file = Path(__file__).resolve().parent / "mlplatform" / "_version.py"
_match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', _version_file.read_text())
_version = _match.group(1) if _match else "0.0.0"

setup(name="mlplatform", version=_version)
