"""Lightweight xmlschema package initializer.

Exports objects lazily to avoid importing Django-dependent code unless needed.

Also exposes a minimal in-memory registry for dynamically generated Django
model classes so that utilities like the XML ingestor can resolve model
classes even when the target app is not installed in Django's app registry.
This is especially useful for unsaved instance workflows in tests.
"""

from typing import TYPE_CHECKING

__all__ = [
    "XmlSchemaDjangoModelGenerator",
    "XmlInstanceIngestor",
    "register_generated_model",
    "get_generated_model",
]

# --- Lightweight generated models registry ---
# Structure: { app_label: { ModelName: ModelClass } }
_GENERATED_MODELS_REGISTRY: dict[str, dict[str, type]] = {}


def register_generated_model(app_label: str, model_class: type) -> None:
    """Register a dynamically created Django model class for later lookup.

    This does not touch Django's app registry; it merely stores a reference
    so that tools can resolve classes by (app_label, model_name) when needed.
    """
    try:
        name = getattr(model_class, "__name__", None) or str(model_class)
    except Exception:
        name = str(model_class)
    bucket = _GENERATED_MODELS_REGISTRY.setdefault(app_label, {})
    bucket[name] = model_class


def get_generated_model(app_label: str, model_name: str):
    """Retrieve a previously registered model class, if available."""
    return _GENERATED_MODELS_REGISTRY.get(app_label, {}).get(model_name)


if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from .generator import XmlSchemaDjangoModelGenerator as XmlSchemaDjangoModelGenerator
    from .ingestor import XmlInstanceIngestor as XmlInstanceIngestor


def __getattr__(name: str):
    if name == "XmlSchemaDjangoModelGenerator":
        # Import lazily to avoid importing Django models at package import time
        from .generator import XmlSchemaDjangoModelGenerator  # type: ignore

        return XmlSchemaDjangoModelGenerator
    if name == "XmlInstanceIngestor":
        from .ingestor import XmlInstanceIngestor  # type: ignore

        return XmlInstanceIngestor
    raise AttributeError(f"module 'pydantic2django.xmlschema' has no attribute {name!r}")
