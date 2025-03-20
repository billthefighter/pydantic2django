"""
Pydantic2Django - Generate Django models from Pydantic models.

This package provides utilities for generating Django models from Pydantic models
and converting between them.
"""

__version__ = "0.1.0"

# Don't import modules that might cause circular imports
# We'll import them directly in the files that need them

from .admin import (
    DynamicModelAdmin,
    register_model_admin,
    register_model_admins,
)
from .base_django_model import Pydantic2DjangoBaseClass
from .discovery import (
    ModelDiscovery,
    find_missing_models,
    topological_sort,
)
from .factory import DjangoModelFactory
from .field_type_mapping import (
    TypeMapper,
    TypeMappingDefinition,
)
from .type_handler import configure_type_handler_logging

__all__ = [
    # Type-safe model creation
    "Pydantic2DjangoBaseClass",
    "DjangoModelFactory",
    # Registry and dependency management
    "find_missing_models",
    "topological_sort",
    # Model discovery
    "ModelDiscovery",
    # Admin interface
    "DynamicModelAdmin",
    "register_model_admin",
    "register_model_admins",
    # Type mapping
    "TypeMapper",
    "TypeMappingDefinition",
    # Logging
    "configure_type_handler_logging",
]
