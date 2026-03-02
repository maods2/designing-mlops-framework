"""VersionResolver - resolves artifact version (latest or specific) for prediction."""

from __future__ import annotations

from typing import Callable

from mlplatform.storage.base import Storage


def resolve_version(
    storage_factory: Callable[[str], Storage],
    artifact_base_path: str,
    feature_name: str,
    model_name: str,
    model_version: str,
) -> str:
    """Resolve model_version to a concrete version string for artifact loading.

    Args:
        storage_factory: Factory to create Storage with a given base path.
        artifact_base_path: Path under which version dirs live (e.g. ./artifacts/artifacts/example).
        feature_name: Feature name for the model.
        model_name: Model name identifier.
        model_version: "latest" or a specific version string (e.g. "2025-01-15-10-30").

    Returns:
        Concrete version string to use for artifact paths.

    Raises:
        FileNotFoundError: If model_version is "latest" but no matching versions exist.
    """
    if model_version and model_version.lower() != "latest":
        return model_version

    # Create storage at artifact_base_path to list version directories
    storage = storage_factory(artifact_base_path)
    prefix = f"{feature_name}_{model_name}_train_"
    children = storage.list("")
    matching = [c for c in children if c.startswith(prefix)]
    if not matching:
        raise FileNotFoundError(
            f"No artifact versions found for {feature_name}/{model_name} "
            f"(expected dirs matching '{prefix}*' under {artifact_base_path})"
        )
    # Extract version from dir name: {feature}_{model}_train_{version}
    versions = []
    for name in matching:
        if name.startswith(prefix):
            ver = name[len(prefix) :]
            if ver:
                versions.append((ver, name))
    if not versions:
        raise FileNotFoundError(
            f"No valid artifact versions found for {feature_name}/{model_name}"
        )
    # Sort by version string (lexicographic; timestamps like 2025-01-15-10-30 sort correctly)
    versions.sort(key=lambda x: x[0], reverse=True)
    return versions[0][0]
