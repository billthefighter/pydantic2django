"""typed2django compatibility namespace.

This package provides a forward-compatible import path that mirrors the
existing implementation in `pydantic2django`. Any import such as
`typed2django.core.base_generator` will be dynamically resolved to
`pydantic2django.core.base_generator`.

Distribution name: django-typed2django
Import package name: typed2django
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys


class _AliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    PREFIX = "typed2django"
    TARGET_PREFIX = "pydantic2django"

    def find_spec(self, fullname: str, path=None, target=None):  # type: ignore[override]
        if not fullname.startswith(self.PREFIX):
            return None
        target_name = fullname.replace(self.PREFIX, self.TARGET_PREFIX, 1)
        spec = importlib.util.find_spec(target_name)
        if spec is None:
            return None
        # Return a spec that uses this loader to finish the aliasing
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):  # type: ignore[override]
        # Use default module creation semantics
        return None

    def exec_module(self, module):  # type: ignore[override]
        # Redirect the module to the pydantic2django counterpart
        target_name = module.__spec__.name.replace(self.PREFIX, self.TARGET_PREFIX, 1)  # type: ignore[attr-defined]
        target = importlib.import_module(target_name)
        sys.modules[module.__name__] = target


# Install the aliasing finder/loader at import time
if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())

# Re-export public API of the original top-level package for convenience
try:
    from pydantic2django import *  # noqa: F401,F403
except Exception:  # pragma: no cover - defensive
    pass
