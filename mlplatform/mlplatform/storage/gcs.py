"""Google Cloud Storage backend for artifact persistence."""

from __future__ import annotations

import io
import os
from typing import Any

import joblib

from mlplatform.storage.base import Storage


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

    def load(self, path: str) -> Any:
        blob = self._bucket.blob(self._blob_path(path))
        buf = io.BytesIO()
        blob.download_to_file(buf)
        buf.seek(0)
        return joblib.load(buf)

    def exists(self, path: str) -> bool:
        return self._bucket.blob(self._blob_path(path)).exists()

    def list_artifacts(self, prefix: str = "") -> list[str]:
        full_prefix = self._blob_path(prefix) if prefix else self._prefix
        blobs = self._client.list_blobs(self._bucket, prefix=full_prefix)
        base = f"{self._prefix}/" if self._prefix else ""
        return [
            blob.name[len(base):] if blob.name.startswith(base) else blob.name
            for blob in blobs
        ]

    def delete(self, path: str) -> None:
        self._bucket.blob(self._blob_path(path)).delete()
