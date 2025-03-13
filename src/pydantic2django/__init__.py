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
from .core import clear_model_registry, make_django_model
from .discovery import (
    ModelDiscovery,
    discover_models,
    get_discovered_models,
    get_django_model,
    get_django_models,
    get_model_dependencies_recursive,
    normalize_model_name,
    register_django_model,
    setup_dynamic_models,
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
    "make_django_model",
    "clear_model_registry",
    # Registry and dependency management
    "normalize_model_name",
    "get_model_dependencies_recursive",
    "validate_model_references",
    "topological_sort",
    "register_django_model",
    # Model discovery
    "ModelDiscovery",
    "discover_models",
    "setup_dynamic_models",
    "get_discovered_models",
    "get_django_models",
    "get_django_model",
    # Admin interface
    "DynamicModelAdmin",
    "register_model_admin",
    "register_model_admins",
    # Type mapping
    "TypeMapper",
    "TypeMappingDefinition",
    "TYPE_MAPPINGS",
]
