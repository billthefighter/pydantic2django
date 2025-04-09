"""
Mock implementation of the discovery module for examples and tests.

This module provides simplified mock implementations of the discovery functions
to allow examples and tests to run without requiring the full implementation.
"""
from collections.abc import Callable
from typing import Optional, Union, get_origin, get_args, List, Type, Any
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("mock_discovery")

from django.db import models
from pydantic import BaseModel
from dataclasses import is_dataclass  # Import is_dataclass

# Corrected imports based on src structure
from pydantic2django.discovery.core import BaseDiscovery, ModelType  # Import BaseDiscovery and ModelType
from pydantic2django.relationships import RelationshipConversionAccessor, RelationshipMapper
from pydantic2django.context import ModelContext  # Corrected context import path

# --- Global State --- #
# Renamed for clarity
_registered_pydantic_models: dict[str, type[BaseModel]] = {}
_registered_dataclasses: dict[str, Type] = {}  # Added for dataclasses
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


def register_model(name: str, model: Type[ModelType], has_context: bool = False) -> None:
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
    elif issubclass(model, BaseModel):
        _registered_pydantic_models[name] = model
    else:
        logger.warning(f"Attempted to register unsupported model type for '{name}': {type(model)}")
        return  # Don't register unknown types

    _model_has_context[name] = has_context
    # Automatically add the model to the relationship accessor upon registration
    # This assumes we want a Django counterpart eventually
    relationship = RelationshipMapper(
        pydantic_model=model if issubclass(model, BaseModel) else None,
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


def map_relationship(model1: Type[ModelType], model2: Union[type[models.Model], Type[ModelType]]) -> None:
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


def get_registered_models() -> dict[str, Type[ModelType]]:
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


class MockDiscovery(BaseDiscovery[ModelType]):  # Inherit from BaseDiscovery
    """Mock implementation of BaseDiscovery for testing."""

    def __init__(
        self,
        model_type_to_discover: str = "all",  # 'pydantic', 'dataclass', 'all'
        initial_models: Optional[dict[str, Type[ModelType]]] = None,
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

    def _get_source_models(self) -> dict[str, Type[ModelType]]:
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
        is_pydantic = isinstance(obj, type) and issubclass(obj, BaseModel) and obj is not BaseModel
        is_dc = isinstance(obj, type) and is_dataclass(obj)

        if self.model_type_to_discover == "pydantic":
            return is_pydantic
        elif self.model_type_to_discover == "dataclass":
            return is_dc
        else:  # 'all'
            return is_pydantic or is_dc

    def _default_eligibility_filter(self, model: Type[ModelType]) -> bool:
        """Mock eligibility filter. For testing, assume all registered models are eligible."""
        # Keep it simple for the mock, specific filters aren't usually tested here.
        return True

    def analyze_dependencies(self) -> None:
        """Mock dependency analysis. Assume simple or no dependencies for tests."""
        logger.info("Mock analyze_dependencies: Performing simplified analysis.")
        # In a mock, we often don't need complex dependency analysis.
        # We can assume an order or let tests dictate it if necessary.
        # Populate self.dependencies based on filtered_models, assuming no interdependencies for simplicity.
        self.dependencies = {model: set() for model in self.filtered_models.values()}
        logger.debug(
            f"Mock dependencies created: { {k.__name__: {dep.__name__ for dep in v} for k,v in self.dependencies.items()} }"
        )

    # get_models_in_registration_order is inherited from BaseDiscovery and uses self.dependencies

    # --- Mock-specific implementations or overrides --- #

    def discover_models(
        self,
        package_names: list[str],  # Mock doesn't actually use package_names
        app_label: str = "django_app",
        filter_function: Optional[Callable[[Type[ModelType]], bool]] = None,
    ) -> None:
        """
        Mock implementation: Populates discovered_models from the registered global state
        or instance state, applying filters.
        """
        logger.debug(f"Mock discover_models called (packages ignored), app_label: {app_label}")
        self.app_label = app_label
        source_models = self._get_source_models()

        # Use the discovery logic from BaseDiscovery
        super().discover_models(
            package_names=[],  # Pass empty list as we use registered models
            app_label=app_label,
            filter_function=filter_function,
            # Provide the source lookup directly to the base method
            _source_module_override=None,  # Not needed
            _initial_discovery_dict_override=source_models,
        )

        logger.debug(f"Mock discover_models finished. Discovered: {list(self.discovered_models.keys())}")
        logger.debug(f"Filtered models: {list(self.filtered_models.keys())}")

    def setup_dynamic_models(self, app_label: Optional[str] = None) -> dict[str, type[models.Model]]:
        """
        Mock implementation: Creates mock Django model classes for discovered/filtered models.
        Uses globally registered Django models if available, otherwise creates new mocks.
        Ensures relationships are mapped using the global accessor.
        """
        effective_app_label = app_label or self.app_label or "django_app"
        logger.debug(f"Mock setup_dynamic_models called with app_label: {effective_app_label}")

        created_django_models: dict[str, type[models.Model]] = {}
        models_to_process = self.get_models_in_registration_order()  # Use ordered models

        for model in models_to_process:
            model_name = model.__name__
            # Check if a Django model is already globally registered for this name
            if model_name in _registered_django_models:
                django_model = _registered_django_models[model_name]
                logger.debug(f"Using pre-registered Django model for '{model_name}': {django_model.__name__}")
            elif model_name in created_django_models:
                django_model = created_django_models[model_name]
                logger.debug(f"Using already created mock Django model for '{model_name}': {django_model.__name__}")
            else:
                # Create a new mock Django model
                logger.debug(f"Creating mock Django model for {model_name}")
                # Use a consistent naming scheme, maybe just the model name if unique
                django_model_name = f"{model_name}"  # Simple name matching
                model_attrs = {
                    "Meta": type("Meta", (), {"app_label": effective_app_label}),
                    "__module__": f"{effective_app_label}.models",  # Mock module path
                    # Add a field to avoid empty model issues in some Django versions
                    "mock_id": models.AutoField(primary_key=True),
                }
                try:
                    django_model = type(django_model_name, (models.Model,), model_attrs)
                except TypeError as e:
                    logger.error(f"Failed to create mock Django model '{django_model_name}': {e}")
                    logger.error(f"Attributes attempted: {model_attrs}")
                    continue  # Skip this model

                logger.debug(f"Created mock Django model class {django_model_name}")
                # Store locally created mock for this run
                created_django_models[model_name] = django_model
                # Optionally, register it globally? Might cause issues if not cleared.
                # register_django_model(model_name, django_model)

            # Ensure the relationship is mapped using the global accessor
            _relationships.map_relationship(model, django_model)

        # Return the models created/used in *this* call
        logger.debug(
            f"setup_dynamic_models returning {len(created_django_models)} models created/mapped in this run: { {k: v.__name__ for k,v in created_django_models.items()} }"
        )
        # Note: This returns only newly created models in this run.
        # To get ALL potentially relevant Django models, use get_registered_django_models()
        return created_django_models

    # --- Methods to access global state (convenience) --- #
    # These could be removed if direct use of global getters is preferred

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
