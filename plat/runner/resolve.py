"""Dynamic class resolution for trainer/predictor modules.

Supports two formats:

* **Explicit** — ``"example_model.train:MyTrainer"`` (module path + class name
  separated by a colon).  This is the recommended form because it is
  unambiguous and faster.

* **Scan** (legacy) — ``"example_model.train"`` (module path only).  The
  resolver imports the module and returns the first subclass of *base_class*
  found via ``dir()``.  This is kept for backward compatibility.
"""

from __future__ import annotations

import importlib


def resolve_class(module_path: str, base_class: type) -> type:
    """Import a module and resolve the target class.

    Args:
        module_path: Either ``"pkg.module:ClassName"`` (explicit) or
            ``"pkg.module"`` (scan for first *base_class* subclass).
        base_class: The ABC that the resolved class must inherit from.

    Returns:
        The resolved class.

    Raises:
        ImportError: If the module cannot be imported or no suitable
            subclass is found.
    """
    if ":" in module_path:
        mod_path, class_name = module_path.rsplit(":", 1)
        mod = importlib.import_module(mod_path)
        cls = getattr(mod, class_name, None)
        if cls is None:
            raise ImportError(f"Class '{class_name}' not found in {mod_path}")
        if not (isinstance(cls, type) and issubclass(cls, base_class)):
            raise ImportError(
                f"'{class_name}' in {mod_path} is not a subclass of {base_class.__name__}"
            )
        return cls

    # Legacy scan: find first subclass in module
    mod = importlib.import_module(module_path)
    for attr_name in dir(mod):
        attr = getattr(mod, attr_name)
        if isinstance(attr, type) and issubclass(attr, base_class) and attr is not base_class:
            return attr
    raise ImportError(f"No {base_class.__name__} subclass found in {module_path}")
