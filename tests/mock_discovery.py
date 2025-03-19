"""
Mock implementation of the discovery module for examples and tests.

This module provides simplified mock implementations of the discovery functions
to allow examples and tests to run without requiring the full implementation.
"""
from collections.abc import Callable
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("mock_discovery")

from django.db import models
from pydantic import BaseModel
from pydantic2django.discovery import ModelDiscovery
from pydantic2django.relationships import RelationshipConversionAccessor, RelationshipMapper

# Storage for discovered models
_discovered_models: dict[str, type[BaseModel]] = {}
_django_models: dict[str, type[models.Model]] = {}
_model_has_context: dict[str, bool] = {}

# Initialize relationships
_relationships = RelationshipConversionAccessor()


class MockDiscovery(ModelDiscovery):
    """Mock implementation of ModelDiscovery for testing."""

    def __init__(self):
        """Initialize a new MockDiscovery instance."""
        logger.debug("Initializing MockDiscovery")
        super().__init__()
        self.discovered_models = _discovered_models
        self.django_models = _django_models
        self.filtered_models = {}  # Will hold the filtered models
        self.relationship_accessor = _relationships  # Use the global relationship accessor

        # Add our pydantic models to the relationship accessor directly
        # Instead of using add_pydantic_model which might have different signature
        for name, model in self.discovered_models.items():
            logger.debug(f"Adding Pydantic model {name} to relationship accessor")
            relationship = RelationshipMapper(pydantic_model=model, django_model=None, context=None)
            self.relationship_accessor.available_relationships.append(relationship)

        logger.debug(f"MockDiscovery initialized with {len(_discovered_models)} discovered models")

    def discover_models(
        self,
        package_names: list[str],
        app_label: str = "django_app",
        filter_function: Optional[Callable[[type[BaseModel]], bool]] = None,
    ) -> None:
        """Mock implementation of discover_models."""
        logger.debug(f"discover_models called with packages: {package_names}, app_label: {app_label}")
        self.discovered_models = _discovered_models
        self.app_label = app_label

        # Populate filtered_models with fully qualified names
        self.filtered_models = {}
        for name, model in self.discovered_models.items():
            qualified_name = f"{app_label}.{name}"
            self.filtered_models[qualified_name] = model

        logger.debug(
            f"discover_models found {len(self.discovered_models)} models: {list(self.discovered_models.keys())}"
        )
        logger.debug(f"filtered_models contains: {list(self.filtered_models.keys())}")

    def get_registration_order(self) -> list[str]:
        """Get the registration order for models.

        Returns:
            List of fully qualified model names (app_label.model_name)
        """
        logger.debug("get_registration_order called")
        # Simply return the keys from discovered_models in alphabetical order
        model_names = list(self.discovered_models.keys())
        model_names.sort()  # Sort alphabetically

        # Add app_label prefix
        result = [f"{self.app_label}.{name}" for name in model_names]
        logger.debug(f"Registration order: {result}")
        return result

    def get_models_in_registration_order(self) -> list[type[BaseModel]]:
        """Get models in registration order.

        Returns:
            List of Pydantic model classes in registration order
        """
        logger.debug("get_models_in_registration_order called")
        registration_order = self.get_registration_order()
        models = [self.filtered_models[name] for name in registration_order]
        logger.debug(f"Returning {len(models)} models in registration order")
        return models

    def setup_dynamic_models(self, app_label: str = "django_app") -> dict[str, type[models.Model]]:
        """Mock implementation of setup_dynamic_models."""
        logger.debug(f"setup_dynamic_models called with app_label: {app_label}")
        self.app_label = app_label

        # Create Django models for each discovered Pydantic model
        for name, pydantic_model in self.discovered_models.items():
            if name not in _django_models:
                logger.debug(f"Creating Django model for {name}")

                # Create the Django model class name
                # Use standard Django naming convention - no 'Model' suffix
                model_class_name = f"Django{name}"
                model_attrs = {
                    "Meta": type("Meta", (), {"app_label": self.app_label}),
                    "__module__": f"{self.app_label}.models.{name.lower()}",
                }

                DynamicModel = type(model_class_name, (models.Model,), model_attrs)
                _django_models[name] = DynamicModel
                logger.debug(f"Created model class {model_class_name}")

                # Map the relationship
                self.map_model_relationship(pydantic_model, DynamicModel)
            else:
                logger.debug(f"Django model for {name} already exists")
                # Ensure relationship is mapped for existing models
                if name in _django_models:
                    self.map_model_relationship(pydantic_model, _django_models[name])

        self.django_models = _django_models
        logger.debug(
            f"setup_dynamic_models created {len(self.django_models)} models: {list(self.django_models.keys())}"
        )
        return self.django_models

    def map_model_relationship(self, pydantic_model: type[BaseModel], django_model: type[models.Model]) -> None:
        """Map the relationship between a Pydantic model and a Django model."""
        logger.debug(f"Mapping relationship: {pydantic_model.__name__} <-> {django_model.__name__}")
        self.relationship_accessor.map_relationship(pydantic_model, django_model)

    def get_relationship_accessor(self) -> RelationshipConversionAccessor:
        """Get the RelationshipConversionAccessor."""
        return self.relationship_accessor

    def get_model_has_context(self) -> dict[str, bool]:
        """Get the dictionary of models with context."""
        logger.debug(f"get_model_has_context returning {len(_model_has_context)} items")
        return _model_has_context


def register_model(name: str, model: type[BaseModel], has_context: bool = False) -> None:
    """
    Register a Pydantic model for discovery.

    Args:
        name: The name of the model
        model: The Pydantic model class
        has_context: Whether the model has context fields
    """
    logger.debug(f"Registering model {name}, has_context={has_context}")
    _discovered_models[name] = model
    _model_has_context[name] = has_context


def register_django_model(name: str, model: type[models.Model]) -> None:
    """
    Register a Django model.

    Args:
        name: The name of the model
        model: The Django model class
    """
    logger.debug(f"Registering Django model {name}")
    _django_models[name] = model


def map_relationship(pydantic_model: type[BaseModel], django_model: type[models.Model]) -> None:
    """
    Map a relationship between a Pydantic model and a Django model.

    Args:
        pydantic_model: The Pydantic model class
        django_model: The Django model class
    """
    logger.debug(f"Mapping relationship: {pydantic_model.__name__} <-> {django_model.__name__}")
    _relationships.map_relationship(pydantic_model, django_model)


def get_discovered_models() -> dict[str, type[BaseModel]]:
    """
    Get all discovered Pydantic models.

    Returns:
        Dict of discovered Pydantic models
    """
    logger.debug(f"get_discovered_models returning {len(_discovered_models)} models: {list(_discovered_models.keys())}")
    return _discovered_models


def get_django_models() -> dict[str, type[models.Model]]:
    """Get all registered Django models."""
    logger.debug("get_django_models called")

    # Create Django models from discovered Pydantic models
    for name, pydantic_model in _discovered_models.items():
        if name not in _django_models:
            logger.debug(f"Creating Django model for {name}")

            # Create a unique model class for each model
            model_class_name = f"Django{name}"
            model_attrs = {
                "Meta": type("Meta", (), {"app_label": "django_llm"}),
                "__module__": f"django_llm.models.{name.lower()}",
            }

            DynamicModel = type(model_class_name, (models.Model,), model_attrs)
            _django_models[name] = DynamicModel
            logger.debug(f"Created model class {model_class_name}")

            # Map the relationship
            map_relationship(pydantic_model, DynamicModel)
        else:
            logger.debug(f"Django model for {name} already exists")

    logger.debug(f"get_django_models returning {len(_django_models)} models: {list(_django_models.keys())}")
    return _django_models


def get_model_has_context() -> dict[str, bool]:
    """Get the dictionary of models with context."""
    logger.debug(f"get_model_has_context returning {len(_model_has_context)} items")
    return _model_has_context


def get_relationship_accessor() -> RelationshipConversionAccessor:
    """Get the RelationshipConversionAccessor."""
    return _relationships


def clear() -> None:
    """Clear all registered models and relationships."""
    logger.debug("Clearing all registered models and relationships")
    _discovered_models.clear()
    _django_models.clear()
    _model_has_context.clear()
    _relationships.available_relationships.clear()
