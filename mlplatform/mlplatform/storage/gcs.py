"""Google Cloud Storage backend for artifact persistence."""

from __future__ import annotations

import io
import tempfile
from typing import Any

import joblib

from mlplatform.storage.base import Storage


class GCSStorage(Storage):
    """Google Cloud Storage backend.

    ``base_path`` should be a ``gs://bucket/prefix`` URI. Artifacts are
    stored as blobs under ``{base_path}/{path}`` using joblib serialization.
    """

    def __init__(self, base_path: str) -> None:
        from google.cloud import storage as gcs

        self.base_path = base_path.rstrip("/")
        self._client = gcs.Client()
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

    def list(self, prefix: str = "") -> list[str]:
        """List immediate children under base_path/prefix (simulates directory listing)."""
        list_prefix = self._blob_path(prefix)
        if list_prefix and not list_prefix.endswith("/"):
            list_prefix = list_prefix + "/"
        elif not list_prefix:
            list_prefix = ""
        iterator = self._bucket.list_blobs(prefix=list_prefix, delimiter="/")
        names: list[str] = []
        # Iterate to trigger API call; prefixes are populated on the iterator
        for blob in iterator:
            rel = blob.name[len(list_prefix) :] if list_prefix else blob.name
            if "/" not in rel and rel:
                names.append(rel)
        # Subdirectories (common prefixes)
        for prefix_path in getattr(iterator, "prefixes", ()):
            name = prefix_path.rstrip("/").split("/")[-1]
            if name:
                names.append(name)
        return names
