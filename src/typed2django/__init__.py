"""
typed2django

Compatibility alias package that re-exports the public API from pydantic2django.
This allows `import typed2django` while the internal package remains `pydantic2django`.
"""

from pydantic2django import *  # noqa: F401,F403

try:  # Keep __all__ consistent if defined upstream
    from pydantic2django import __all__ as _ALL  # type: ignore

    __all__ = _ALL  # type: ignore[assignment]
except Exception:  # pragma: no cover - fallback if upstream does not define __all__
    # Expose nothing explicitly; star import above already populates globals
    pass
