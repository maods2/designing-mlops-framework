"""DAG utilities for dependency ordering."""

from __future__ import annotations

from mlplatform.config.schema import ModelConfig


def topological_order(models: list[ModelConfig]) -> list[ModelConfig]:
    """Return models in dependency order based on ``depends_on``."""
    if not models:
        return []

    by_name = {m.model_name: m for m in models}
    indegree = {m.model_name: 0 for m in models}
    edges: dict[str, list[str]] = {m.model_name: [] for m in models}

    for model in models:
        for dep in model.depends_on:
            if dep not in by_name:
                raise ValueError(f"Unknown dependency '{dep}' referenced by '{model.model_name}'")
            edges[dep].append(model.model_name)
            indegree[model.model_name] += 1

    queue = [name for name, deg in indegree.items() if deg == 0]
    ordered_names: list[str] = []
    while queue:
        current = queue.pop(0)
        ordered_names.append(current)
        for nxt in edges[current]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)

    if len(ordered_names) != len(models):
        raise ValueError("Workflow dependency cycle detected in models.depends_on")

    return [by_name[name] for name in ordered_names]
