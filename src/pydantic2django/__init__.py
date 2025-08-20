"""
Pydantic2Django - Generate Django models from Pydantic models.

This package provides utilities for generating Django models from Pydantic models
and converting between them.
"""

__version__ = "1.0.0"

# Don't import modules that might cause circular imports
# We'll import them directly in the files that need them

# Restore admin imports with correct path
from .core.imports import ImportHandler
from .core.typing import configure_core_typing_logging  # Corrected function name
from .django.admin import (
    DynamicModelAdmin,
    register_model_admin,
    register_model_admins,
)

# Corrected import paths based on current structure
from .django.models import Pydantic2DjangoBaseClass

__all__ = [
    # Type-safe model creation
    "Pydantic2DjangoBaseClass",
    # "DjangoModelFactory", # Commented out
    # Registry and dependency management - Commented out
    # "find_missing_models",
    # "topological_sort",
    # Model discovery - Commented out
    # "ModelDiscovery",
    # Admin interface - Restored with correct path
    "DynamicModelAdmin",
    "register_model_admin",
    "register_model_admins",
    # Import handling
    "ImportHandler",
    # Logging
    "configure_core_typing_logging",  # Corrected function name
]
