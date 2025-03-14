"""
Mock implementation of the discovery module for examples and tests.

This module provides simplified mock implementations of the discovery functions
to allow examples and tests to run without requiring the full implementation.
"""
from collections.abc import Callable
from typing import Optional

from django.db import models
from pydantic import BaseModel
from pydantic2django.discovery import ModelDiscovery

# Storage for discovered models
_discovered_models: dict[str, type[BaseModel]] = {}
_django_models: dict[str, type[models.Model]] = {}
_model_has_context: dict[str, bool] = {}


class MockDiscovery(ModelDiscovery):
    """Mock implementation of ModelDiscovery for testing."""

    def __init__(self):
        """Initialize a new MockDiscovery instance."""
        super().__init__()
        self.discovered_models = _discovered_models
        self.django_models = _django_models

    def discover_models(
        self,
        package_names: list[str],
        app_label: str = "django_app",
        filter_function: Optional[Callable[[type[BaseModel]], bool]] = None,
    ) -> None:
        """Mock implementation of discover_models."""
        self.discovered_models = _discovered_models
        self.app_label = app_label

    def setup_dynamic_models(self, app_label: str = "django_app") -> dict[str, type[models.Model]]:
        """Mock implementation of setup_dynamic_models."""
        self.app_label = app_label
        self.django_models = _django_models
        return self.django_models

    def get_model_has_context(self) -> dict[str, bool]:
        """Get the dictionary of models with context."""
        return _model_has_context


def register_model(name: str, model: type[BaseModel], has_context: bool = False) -> None:
    """
    Register a Pydantic model for discovery.

    Args:
        name: The name of the model
        model: The Pydantic model class
        has_context: Whether the model has context fields
    """
    _discovered_models[name] = model
    _model_has_context[name] = has_context


def register_django_model(name: str, model: type[models.Model]) -> None:
    """
    Register a Django model.

    Args:
        name: The name of the model
        model: The Django model class
    """
    _django_models[name] = model


def get_discovered_models() -> dict[str, type[BaseModel]]:
    """
    Get all discovered Pydantic models.

    Returns:
        Dict of discovered Pydantic models
    """
    return _discovered_models


def get_django_models() -> dict[str, type[models.Model]]:
    """Get all registered Django models."""
    # Create Django models from discovered Pydantic models
    for name, pydantic_model in _discovered_models.items():
        if name not in _django_models:
            # Create a simple Django model with a name field
            class DynamicModel(models.Model):
                class Meta:
                    app_label = "test_app"

            DynamicModel.__name__ = name
            _django_models[name] = DynamicModel

    return _django_models


def get_model_has_context() -> dict[str, bool]:
    """Get the dictionary of models with context."""
    return _model_has_context


def clear() -> None:
    """Clear all registered models."""
    _discovered_models.clear()
    _django_models.clear()
    _model_has_context.clear()
