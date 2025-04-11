"""
Mock implementation of the discovery module for examples and tests.

This module provides simplified mock implementations of the discovery functions
to allow examples and tests to run without requiring the full implementation.
"""
from collections.abc import Callable
from typing import Optional, Union, get_origin, get_args, List, Type, Any, TypeVar
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("mock_discovery")

from django.db import models
from pydantic import BaseModel
from dataclasses import is_dataclass, dataclass  # Import dataclass decorator too

# Define ModelType simply as a TypeVar without a complex bound for now
ModelType = TypeVar("ModelType")

# Corrected imports based on src structure
from pydantic2django.core.discovery import BaseDiscovery  # Corrected path
from pydantic2django.core.relationships import RelationshipConversionAccessor, RelationshipMapper  # Corrected path
from pydantic2django.core.context import ModelContext  # Corrected context import path

# --- Global State --- #
# Renamed for clarity
_registered_pydantic_models: dict[str, type[BaseModel]] = {}
_registered_dataclasses: dict[str, Type] = {}  # Using Type which is Any, maybe refine?
_registered_django_models: dict[str, type[models.Model]] = {}
_model_has_context: dict[str, bool] = {}
_model_contexts: dict[str, ModelContext] = {}
_relationships = RelationshipConversionAccessor()
_field_overrides: dict[str, dict[str, dict[str, str]]] = {}

# --- Helper Functions for Managing Global State --- #


def clear() -> None:
    """Clear all stored models, relationships, contexts, and overrides."""
    global _registered_pydantic_models, _registered_dataclasses, _registered_django_models
    global _model_has_context, _relationships, _field_overrides, _model_contexts
    logger.debug("Clearing all mock discovery state")
    _registered_pydantic_models = {}
    _registered_dataclasses = {}
    _registered_django_models = {}
    _model_has_context = {}
    _field_overrides = {}
    _model_contexts = {}
    _relationships = RelationshipConversionAccessor()


def register_model(name: str, model: Type, has_context: bool = False) -> None:
    """
    Register a Pydantic model or Dataclass for discovery.

    Args:
        name: The name of the model.
        model: The Pydantic model or Dataclass class.
        has_context: Whether the model has context fields.
    """
    logger.debug(
        f"Registering model '{name}', type: {'Dataclass' if is_dataclass(model) else 'Pydantic'}, has_context={has_context}"
    )
    if is_dataclass(model):
        _registered_dataclasses[name] = model
    elif isinstance(model, type) and issubclass(model, BaseModel):
        _registered_pydantic_models[name] = model
    else:
        logger.warning(f"Attempted to register unsupported model type for '{name}': {type(model)}")
        return  # Don't register unknown types

    _model_has_context[name] = has_context
    # Automatically add the model to the relationship accessor upon registration
    relationship = RelationshipMapper(
        pydantic_model=model if isinstance(model, type) and issubclass(model, BaseModel) else None,
        dataclass_model=model if is_dataclass(model) else None,
        django_model=None,
        context=None,
    )
    _relationships.available_relationships.append(relationship)


def register_django_model(name: str, model: type[models.Model]) -> None:
    """
    Register a Django model, usually a mock or predefined one.

    Args:
        name: The logical name associated with the Django model (often matching a Pydantic/Dataclass name).
        model: The Django model class.
    """
    logger.debug(f"Registering Django model '{name}' ({model.__name__})")
    _registered_django_models[name] = model


def map_relationship(model1: Type, model2: Union[type[models.Model], Type]) -> None:
    """
    Explicitly map a relationship between a Pydantic/Dataclass model and its Django counterpart,
    or between two Pydantic/Dataclass models if needed for the relationship accessor.

    Args:
        model1: The Pydantic model or Dataclass class.
        model2: The corresponding Django model class or another Pydantic/Dataclass.
    """
    logger.debug(
        f"Mapping relationship: {getattr(model1, '__name__', str(model1))} <-> {getattr(model2, '__name__', str(model2))}"
    )
    # Let the relationship accessor handle the details
    if isinstance(model2, type) and issubclass(model2, models.Model):
        _relationships.map_relationship(model1, model2)
    else:
        # Handle mapping between two non-Django models if RelationshipAccessor supports it
        # For now, assume we're mapping to a Django model or use register_model for discovery
        logger.warning(
            f"map_relationship currently primarily supports mapping to Django models. Mapping {model1.__name__} <-> {model2.__name__}"
        )
        # You might need to extend RelationshipAccessor or RelationshipMapper logic
        # if direct Pydantic <-> Pydantic mapping is needed beyond simple discovery.


def set_field_override(model_name: str, field_name: str, field_type: str, target_model_name: str) -> None:
    """
    Set a field override for a model during Django model generation.

    Args:
        model_name: The name of the model (Pydantic/Dataclass).
        field_name: The name of the field in the model.
        field_type: The desired Django field type (e.g., "ForeignKey", "OneToOneField").
        target_model_name: The name of the target model for the relationship.
    """
    logger.debug(
        f"Setting field override for {model_name}.{field_name}: Type={field_type}, Target='{target_model_name}'"
    )
    if model_name not in _field_overrides:
        _field_overrides[model_name] = {}
    _field_overrides[model_name][field_name] = {"field_type": field_type, "target_model": target_model_name}


def register_context(name: str, context: ModelContext) -> None:
    """
    Register a model context for a model.

    Args:
        name: The name of the model (Pydantic/Dataclass).
        context: The ModelContext instance.
    """
    logger.debug(f"Registering context for model: {name}")
    _model_contexts[name] = context


# --- Global State Getters --- #


def get_registered_models() -> dict[str, Type]:
    """Get all registered Pydantic models and Dataclasses."""
    # Combine Pydantic and Dataclass dictionaries
    all_models = {**_registered_pydantic_models, **_registered_dataclasses}
    logger.debug(f"get_registered_models returning {len(all_models)} models: {list(all_models.keys())}")
    return all_models


def get_registered_django_models() -> dict[str, type[models.Model]]:
    """Get all registered Django models (mocks or predefined)."""
    logger.debug(
        f"get_registered_django_models returning {len(_registered_django_models)} models: {list(_registered_django_models.keys())}"
    )
    return _registered_django_models


def get_model_has_context() -> dict[str, bool]:
    """Get the dictionary indicating which models have context."""
    logger.debug(f"get_model_has_context returning {len(_model_has_context)} items")
    return _model_has_context


def get_relationship_accessor() -> RelationshipConversionAccessor:
    """Get the singleton RelationshipConversionAccessor instance."""
    return _relationships


def get_field_overrides() -> dict[str, dict[str, dict[str, str]]]:
    """Get the field overrides dictionary."""
    return _field_overrides


def get_model_contexts() -> dict[str, ModelContext]:
    """Get all registered model contexts."""
    return _model_contexts


def has_field_override(model_name: str, field_name: str) -> bool:
    """Check if a field override exists for a specific field."""
    return model_name in _field_overrides and field_name in _field_overrides[model_name]


def get_field_override(model_name: str, field_name: str) -> Optional[dict[str, str]]:
    """Get the field override details for a specific field."""
    return _field_overrides.get(model_name, {}).get(field_name)


# --- Mock Discovery Class --- #


class MockDiscovery(BaseDiscovery[Type]):  # Use Type instead of ModelType
    """Mock implementation of BaseDiscovery for testing."""

    def __init__(
        self,
        model_type_to_discover: str = "all",  # 'pydantic', 'dataclass', 'all'
        initial_models: Optional[dict[str, Type]] = None,
    ):
        """
        Initialize a new MockDiscovery instance.

        Args:
            model_type_to_discover: Specifies which type of models ('pydantic', 'dataclass', 'all')
                                   this instance should pretend to discover.
            initial_models: Optionally pre-populate with specific models for this instance
                           (rarely needed, usually use global registration).
        """
        logger.debug(f"Initializing MockDiscovery (type: {model_type_to_discover})")
        super().__init__()  # Initialize base class
        self.model_type_to_discover = model_type_to_discover

        # Instance state - primarily for filtering/behavior, not data storage
        # self.discovered_models remains the main dict from BaseDiscovery
        # self.filtered_models is also from BaseDiscovery
        # self.dependencies is also from BaseDiscovery

        # If initial_models are provided, use them instead of global ones for this instance.
        # This deviates from using only global state but allows instance-specific test scenarios.
        self._instance_models = initial_models

    def _get_source_models(self) -> dict[str, Type]:
        """Returns the models this instance should 'discover' from."""
        if self._instance_models is not None:
            logger.debug("Using instance-specific initial models.")
            return self._instance_models

        logger.debug("Using globally registered models.")
        if self.model_type_to_discover == "pydantic":
            return _registered_pydantic_models
        elif self.model_type_to_discover == "dataclass":
            return _registered_dataclasses
        else:  # 'all'
            return {**_registered_pydantic_models, **_registered_dataclasses}

    # --- Overriding BaseDiscovery abstract methods --- #

    def _is_target_model(self, obj: Any) -> bool:
        """Check if an object is the type of model this instance targets."""
        is_pydantic = isinstance(obj, type) and issubclass(obj, BaseModel)
        is_dc = is_dataclass(obj) and isinstance(obj, type)

        if self.model_type_to_discover == "pydantic":
            return is_pydantic
        elif self.model_type_to_discover == "dataclass":
            return is_dc
        else:  # 'all'
            return is_pydantic or is_dc

    def _default_eligibility_filter(self, model: Type) -> bool:
        """Mock eligibility: Assume all discovered models are eligible."""
        # Add specific mock filters if needed (e.g., exclude models starting with '_')
        return True

    def analyze_dependencies(self) -> None:
        """Mock dependency analysis - does nothing for now."""
        # In a real scenario, this would analyze fields of self.filtered_models
        logger.debug("Mock analyze_dependencies called.")
        self.dependencies = {}

    def discover_models(
        self,
        packages: list[str],  # Renamed from package_names to match base class
        app_label: str = "django_app",
        user_filters: Optional[
            Union[Callable[[Type], bool], list[Callable[[Type], bool]]]
        ] = None,  # Renamed & updated type
        # Removed unused mock-specific parameters like _source_module_override
    ) -> None:
        """Mock discovery: Populates filtered_models based on registered models and filters."""
        logger.info("Mock discover_models called.")
        source_models = self._get_source_models()
        self.all_models = source_models.copy()  # Store all found before filtering
        self.filtered_models = {}

        # Normalize user_filters
        filters_to_apply = []
        if user_filters:
            if isinstance(user_filters, list):
                filters_to_apply.extend(user_filters)
            else:
                filters_to_apply.append(user_filters)

        for name, model in source_models.items():
            is_eligible = self._default_eligibility_filter(model)
            if is_eligible:
                for user_filter in filters_to_apply:
                    try:
                        # Adapt filter signature if needed, BaseDiscovery expects Callable[[Type[TModel]], bool]
                        if not user_filter(model):
                            is_eligible = False
                            break
                    except Exception as e:
                        logger.error(f"Error applying user filter to {name}: {e}")
                        is_eligible = False
                        break

            if is_eligible:
                self.filtered_models[name] = model

        logger.info(f"Mock discovery finished. Filtered models: {list(self.filtered_models.keys())}")
        # Call analyze_dependencies after filtering
        self.analyze_dependencies()

    def get_models_in_registration_order(self) -> list[Type]:
        """Mock topological sort - just returns filtered models in arbitrary order."""
        logger.debug("Mock get_models_in_registration_order called.")
        # Add proper topological sort if dependencies are analyzed
        return list(self.filtered_models.values())

    # --- Mock-specific methods (kept if still relevant) --- #
    def get_relationship_accessor(self) -> RelationshipConversionAccessor:
        """Get the global RelationshipConversionAccessor."""
        return get_relationship_accessor()

    def get_field_overrides(self) -> dict:
        """Get the global field overrides dictionary."""
        return get_field_overrides()

    def get_model_contexts(self) -> dict[str, ModelContext]:
        """Get the global model contexts dictionary."""
        return get_model_contexts()

    def get_model_has_context(self) -> dict[str, bool]:
        """Get the global dictionary indicating which models have context."""
        return get_model_has_context()


# --- Remove redundant/complex internal relationship logic --- #
# ( _setup_nested_model_relationships and _map_model_relationship removed )
