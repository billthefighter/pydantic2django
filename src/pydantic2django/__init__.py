"""
Pydantic2Django - Generate Django models from Pydantic models.

This package provides utilities for generating Django models from Pydantic models
and converting between them.
"""

__version__ = "1.0.7"

# Don't import modules that might cause circular imports
# We'll import them directly in the files that need them

# Safe, lightweight exports only (avoid importing Django at package import time)
from .core.imports import ImportHandler
from .core.typing import configure_core_typing_logging  # Corrected function name

__all__ = [
    "ImportHandler",
    "configure_core_typing_logging",
]
