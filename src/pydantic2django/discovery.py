"""
High-level interface for model discovery and registration.

This module provides a simplified interface for discovering and registering
Pydantic models as Django models.
"""
import logging
from typing import Optional, cast

from django.db import models
from pydantic import BaseModel

from .factory import DjangoModelFactory
from .registry import ModelRegistryManager
from .types import DjangoBaseModel, T

logger = logging.getLogger(__name__)

# Global registry instance
_registry: Optional[ModelRegistryManager] = None


def get_registry() -> ModelRegistryManager:
    """Get or create the global registry instance."""
    global _registry
    if _registry is None:
        _registry = ModelRegistryManager()
    return _registry


def discover_models(package_names: list[str], app_label: str = "django_llm") -> None:
    """
    Discover and analyze Pydantic models from specified packages.

    Args:
        package_names: List of package names to search for models
        app_label: The Django app label to use for model registration
    """
    registry = get_registry()

    logger.info(f"Discovering models from packages {package_names} for {app_label}...")

    # Discover models from all packages
    for package_name in package_names:
        registry.discover_models(package_name, app_label=app_label)

    logger.info(f"Discovered {len(registry.discovered_models)} models")


def setup_dynamic_models(app_label: str = "django_llm", skip_admin: bool = False) -> dict[str, type[models.Model]]:
    """
    Set up dynamic models from discovered Pydantic models.

    This function should be called during migration operations, not during app initialization.
    It creates Django models from discovered Pydantic models.

    Args:
        app_label: The Django app label to use for model registration
        skip_admin: Whether to skip registering models with the Django admin interface

    Returns:
        Dict mapping model names to Django model classes
    """
    registry = get_registry()
    return registry.setup_models(app_label, skip_admin)


def get_discovered_models() -> dict[str, type[BaseModel]]:
    """Get all discovered Pydantic models."""
    return get_registry().discovered_models


def get_django_models(app_label: str = "django_llm") -> dict[str, type[models.Model]]:
    """Get all registered Django models for the given app label."""
    return get_registry().django_models


def get_django_model(
    pydantic_model: type[T] | type[type[T]], app_label: str = "django_llm"
) -> type[DjangoBaseModel[T]]:
    """
    Get a Django model with proper type hints for a given Pydantic model.

    Args:
        pydantic_model: The Pydantic model class or a callable that returns the model class
        app_label: The Django app label to use for model registration

    Example:
        from your_package.models import UserPydantic

        UserDjango = discovery.get_django_model(UserPydantic)
        user = UserDjango(name="John")
        user.get_display_name()  # IDE completion works!
    """
    registry = get_registry()

    # Handle both direct type and callable returning type
    if callable(pydantic_model) and not isinstance(pydantic_model, type):
        # If it's a fixture function, call it to get the actual model
        actual_model = cast(type[T], pydantic_model())
    else:
        # If it's already a type, use it directly
        actual_model = cast(type[T], pydantic_model)

    model_name = actual_model.__name__

    # Set up models for the specified app_label if not already done
    if not registry.django_models:
        registry.setup_models(app_label=app_label)

    # Look for the model in the registry
    for model in registry.django_models.values():
        if getattr(model, "_pydantic_model", None) == actual_model:
            return cast(type[DjangoBaseModel[T]], model)

    # If not found, try to create it
    django_model, _ = DjangoModelFactory[T].create_model(actual_model, app_label=app_label)

    return django_model
