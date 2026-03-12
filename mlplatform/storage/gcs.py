"""Google Cloud Storage backend for artifact persistence."""

from __future__ import annotations

import io
import os
from typing import Any

import joblib

from mlplatform.storage.base import Storage

_RAW_BYTES_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".json", ".html", ".htm")


def _content_type_for_path(path: str) -> str:
    ext = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".json": "application/json",
        ".html": "text/html",
        ".htm": "text/html",
    }.get(ext, "application/octet-stream")


def _is_raw_bytes_path(path: str) -> bool:
    ext = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
    return ext in _RAW_BYTES_EXTS


class GCSStorage(Storage):
    """Google Cloud Storage backend.

    ``base_path`` should be a ``gs://bucket/prefix`` URI. Artifacts are
    stored as blobs under ``{base_path}/{path}`` using joblib serialization.

    ``project`` is the GCP project ID.  Resolution order:

    1. Explicit ``project`` argument.
    2. ``GOOGLE_CLOUD_PROJECT`` environment variable.
    3. ADC / metadata server (automatic on GCP — Vertex AI, Cloud Run, GCE).

    Local setup (one-time):

    .. code-block:: bash

        gcloud auth application-default login
        export GOOGLE_CLOUD_PROJECT=my-dev-project  # or add to .env
    """

    def __init__(self, base_path: str, project: str | None = None) -> None:
        from google.cloud import storage as gcs

        resolved_project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.base_path = base_path.rstrip("/")
        self._client = gcs.Client(project=resolved_project)
        bucket_name, self._prefix = self._parse_gs_uri(self.base_path)
        self._bucket = self._client.bucket(bucket_name)

    @staticmethod
    def _parse_gs_uri(uri: str) -> tuple[str, str]:
        """Split ``gs://bucket/prefix`` into ``(bucket, prefix)``."""
        if not uri.startswith("gs://"):
            raise ValueError(f"GCS base_path must start with gs://, got: {uri}")
        without_scheme = uri[5:]
        parts = without_scheme.split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        return bucket, prefix

    def _blob_path(self, path: str) -> str:
        if self._prefix:
            return f"{self._prefix}/{path}"
        return path

    def save(self, path: str, obj: Any) -> None:
        buf = io.BytesIO()
        joblib.dump(obj, buf)
        buf.seek(0)
        blob = self._bucket.blob(self._blob_path(path))
        blob.upload_from_file(buf, content_type="application/octet-stream")

    def save_bytes(self, path: str, data: bytes) -> None:
        """Upload raw bytes with content-type inferred from path."""
        blob = self._bucket.blob(self._blob_path(path))
        content_type = _content_type_for_path(path)
        blob.upload_from_string(data, content_type=content_type)

    def load(self, path: str) -> Any:
        blob = self._bucket.blob(self._blob_path(path))
        buf = io.BytesIO()
        blob.download_to_file(buf)
        buf.seek(0)
        data = buf.read()
        # Raw bytes (PNG, JSON, HTML) stored via save_bytes — return as-is
        if _is_raw_bytes_path(path):
            return data
        buf.seek(0)
        return joblib.load(buf)

    def exists(self, path: str) -> bool:
        return self._bucket.blob(self._blob_path(path)).exists()
