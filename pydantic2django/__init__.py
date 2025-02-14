"""
Pydantic2Django - Convert Pydantic models to Django models dynamically.
"""

__version__ = "0.1.0"

from .core import make_django_model

__all__ = ["make_django_model"]
