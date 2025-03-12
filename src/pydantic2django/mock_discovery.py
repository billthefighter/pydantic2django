"""
Mock implementation of the discovery module for examples and tests.

This module provides simplified mock implementations of the discovery functions
to allow examples and tests to run without requiring the full implementation.
"""
from collections.abc import Callable
from typing import Optional

from django.db import models
from pydantic import BaseModel

# Storage for discovered models
_discovered_models: dict[str, type[BaseModel]] = {}
_django_models: dict[str, type[models.Model]] = {}


def discover_models(
    package_names: list[str],
    app_label: str = "django_app",
    filter_function: Optional[Callable[[str, type[BaseModel]], bool]] = None,
) -> None:
    """
    Mock implementation of discover_models.

    In a real implementation, this would scan the packages for Pydantic models.
    For our mock, we'll just use the models that have been registered manually.

    Args:
        package_names: List of package names to search for models
        app_label: The Django app label to use for model registration
        filter_function: Optional function to filter discovered models
    """
    # In a real implementation, this would discover models from the packages
    # For our mock, we'll just use the models that have been registered manually
    pass


def register_model(name: str, model: type[BaseModel]) -> None:
    """
    Register a Pydantic model for discovery.

    Args:
        name: The name of the model
        model: The Pydantic model class
    """
    _discovered_models[name] = model


def register_django_model(name: str, model: type[models.Model]) -> None:
    """
    Register a Django model.

    Args:
        name: The name of the model
        model: The Django model class
    """
    _django_models[name] = model


def setup_dynamic_models() -> None:
    """
    Mock implementation of setup_dynamic_models.

    In a real implementation, this would create Django models from the discovered Pydantic models.
    For our mock, we'll just use the Django models that have been registered manually.
    """
    # In a real implementation, this would create Django models
    # For our mock, we'll just use the Django models that have been registered manually
    pass


def get_discovered_models() -> dict[str, type[BaseModel]]:
    """
    Get all discovered Pydantic models.

    Returns:
        Dict of discovered Pydantic models
    """
    return _discovered_models


def get_django_models() -> dict[str, type[models.Model]]:
    """
    Get all registered Django models.

    Returns:
        Dict of registered Django models
    """
    return _django_models


def clear() -> None:
    """Clear all registered models."""
    _discovered_models.clear()
    _django_models.clear()
