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
from .core import clear_model_registry
from .discovery import (
    ModelDiscovery,
    register_django_model,
    topological_sort,
    validate_model_references,
)
from .factory import DjangoModelFactory
from .field_type_mapping import (
    TYPE_MAPPINGS,
    TypeMapper,
    TypeMappingDefinition,
)

__all__ = [
    # Type-safe model creation
    "Pydantic2DjangoBaseClass",
    "DjangoModelFactory",
    # Core model conversion
    "clear_model_registry",
    # Registry and dependency management
    "validate_model_references",
    "topological_sort",
    "register_django_model",
    # Model discovery
    "ModelDiscovery",
    # Admin interface
    "DynamicModelAdmin",
    "register_model_admin",
    "register_model_admins",
    # Type mapping
    "TypeMapper",
    "TypeMappingDefinition",
    "TYPE_MAPPINGS",
]
