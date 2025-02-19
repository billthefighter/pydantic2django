"""
pydantic2django - Convert Pydantic models to Django models
"""

__version__ = "0.1.0"

from .core import make_django_model, clear_model_registry
from .registry import (
    ModelRegistryManager,
    normalize_model_name,
    get_model_dependencies_recursive,
    validate_model_references,
    topological_sort,
    register_django_model,
)
from .discovery import (
    discover_models,
    setup_dynamic_models,
    get_discovered_models,
    get_django_models,
)
from .admin import (
    DynamicModelAdmin,
    register_model_admin,
    register_model_admins,
)

__all__ = [
    # Core model conversion
    "make_django_model",
    "clear_model_registry",
    
    # Registry and dependency management
    "ModelRegistryManager",
    "normalize_model_name",
    "get_model_dependencies_recursive",
    "validate_model_references",
    "topological_sort",
    "register_django_model",
    
    # Model discovery
    "discover_models",
    "setup_dynamic_models",
    "get_discovered_models",
    "get_django_models",
    
    # Admin interface
    "DynamicModelAdmin",
    "register_model_admin",
    "register_model_admins",
]
