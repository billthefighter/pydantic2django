"""Lightweight xmlschema package initializer.

Exports objects lazily to avoid importing Django-dependent code unless needed.
"""

__all__ = ["XmlSchemaDjangoModelGenerator"]


def __getattr__(name: str):
    if name == "XmlSchemaDjangoModelGenerator":
        # Import lazily to avoid importing Django models at package import time
        from .generator import XmlSchemaDjangoModelGenerator  # type: ignore

        return XmlSchemaDjangoModelGenerator
    raise AttributeError(f"module 'pydantic2django.xmlschema' has no attribute {name!r}")
