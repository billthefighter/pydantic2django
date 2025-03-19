"""
High-level interface for model discovery and registration.

This module provides a simplified interface for discovering and registering
Pydantic models as Django models.
"""
import importlib
import inspect
import logging
from abc import ABC
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Optional, get_args, get_origin

from django.db import models
from pydantic import BaseModel

from .types import is_pydantic_model

logger = logging.getLogger(__name__)


def normalize_model_reference(model_ref: str | type[models.Model]) -> str:
    """
    Normalize a model reference to a consistent format.

    Args:
        model_ref: Either a string reference ('app.Model' or 'Model') or a model class

    Returns:
        Normalized model name in the format 'ModelName'
    """
    if isinstance(model_ref, str):
        # Extract just the model name if it includes app_label
        model_name = model_ref.split(".")[-1]
    else:
        # Get the name from the class
        model_name = model_ref.__name__

    return model_name


def find_missing_models(models: dict[str, type], dependencies: dict[str, set[str]]) -> list[str]:
    """
    Validate that all model references exist.

    Args:
        models: Dictionary of model names to model classes
        dependencies: Dictionary of model names to their dependencies

    Returns:
        List of error messages for non-existent model references
    """
    missing_models = []
    normalized_models = {normalize_model_reference(name): cls for name, cls in models.items()}

    # Check each model's dependencies
    for model_name, deps in dependencies.items():
        for dep in deps:
            # Skip self-references
            if dep == model_name or dep == "self":
                continue

            # Check if the dependency exists in normalized models
            normalized_dep = normalize_model_reference(dep)
            if normalized_dep not in normalized_models:
                logger.warning(f"Model {model_name} references non-existent model {dep}")
                missing_models.append(dep)

    return missing_models


def topological_sort(dependencies: dict[str, set[str]]) -> list[str]:
    """
    Sort models topologically based on their dependencies.

    Args:
        dependencies: Dict mapping model names to their dependencies

    Returns:
        List of model names in dependency order
    """
    # Track visited and sorted nodes
    visited = set()
    temp_visited = set()
    sorted_nodes = []

    def visit(node: str):
        if node in temp_visited:
            # Cyclic dependency detected - break the cycle
            logger.warning(f"Cyclic dependency detected for {node}")
            return
        if node in visited:
            return

        temp_visited.add(node)

        # Visit dependencies
        for dep in dependencies.get(node, set()):
            if dep != node and dep != "self":  # Skip self-references
                visit(dep)

        temp_visited.remove(node)
        visited.add(node)
        sorted_nodes.append(node)

    # Visit all nodes
    for node in dependencies:
        if node not in visited:
            visit(node)

    return sorted_nodes


def exclude_models(model_names: list[str]) -> Callable[[str, type[BaseModel]], bool]:
    """
    Create a filter function that excludes models with specific names.

    Args:
        model_names: List of model names to exclude

    Returns:
        A filter function that returns False for models in the exclusion list
    """
    return lambda model_name, _: model_name not in model_names


def include_models(model_names: list[str]) -> Callable[[str, type[BaseModel]], bool]:
    """
    Create a filter function that includes only models with specific names.

    Args:
        model_names: List of model names to include

    Returns:
        A filter function that returns True only for models in the inclusion list
    """
    return lambda model_name, _: model_name in model_names


def has_field(field_name: str) -> Callable[[str, type[BaseModel]], bool]:
    """
    Create a filter function that includes only models with a specific field.

    Args:
        field_name: The name of the field that models must have

    Returns:
        A filter function that returns True only for models with the specified field
    """
    return lambda _, model_class: field_name in model_class.model_fields


def always_include(_model_name: str, _model_class: type[BaseModel]) -> bool:
    """
    A stub filter function that always returns True.

    This can be used as a starting point for creating custom filter functions
    for the discover_models function.

    Args:
        _model_name: The normalized name of the model
        _model_class: The Pydantic model class

    Returns:
        True, always including the model
    """
    return True


class ModelDiscovery:
    """
    Manages the discovery, registration, and access to Pydantic models and Django models.

    This class encapsulates the functionality for discovering Pydantic models,
    converting them to Django models, and providing access to both.

    Example:
        # Create a discovery instance
        discovery = ModelDiscovery()

        # Discover models from packages
        discovery.discover_models(["your_package.models"], app_label="your_app")

        # Set up Django models
        django_models = discovery.setup_dynamic_models(app_label="your_app")
    """

    def __init__(self):
        """Initialize a new ModelDiscovery instance with its own registry."""
        self.discovered_models: dict[str, type[BaseModel]] = {}
        self.filtered_models: dict[str, type[BaseModel]] = {}
        self.normalized_models: dict[str, type[BaseModel]] = {}
        self.dependencies: dict[str, set[str]] = {}
        self.django_models: dict[str, type[models.Model]] = {}
        self._registration_order: Optional[list[str]] = None
        self.app_label: Optional[str] = None

    def discover_models(
        self,
        package_names: list[str],
        app_label: str = "django_llm",
        filter_function: Optional[Callable[[type[BaseModel]], bool]] = None,
    ) -> None:
        """
        Discover and analyze Pydantic models from specified packages.

        This is a convenience function that uses the singleton ModelDiscovery instance.

        Args:
            package_names: List of package names to search for models
            app_label: The Django app label to use for model registration
            filter_function: Optional function to filter discovered models.
                             Takes model_name and model_class as arguments and returns
                             True if the model should be included, False otherwise.
                             Use the provided helper functions like `always_include`,
                             `include_models`, `exclude_models`, or `has_field`.

        Example:
            ```python
            # Using the helper functions
            from pydantic2django.discovery import has_field, exclude_models, include_models

            # Only include models with a specific field
            discover_models(['my_package'], filter_function=has_field('email'))

            # Exclude specific models
            discover_models(['my_package'], filter_function=exclude_models(['InternalConfig', 'PrivateData']))

            # Include only specific models
            discover_models(['my_package'], filter_function=include_models(['User', 'Product']))

            # Or with a lambda for custom logic
            discover_models(['my_package'], filter_function=lambda name, _: not name.startswith('Internal'))

            # Using the provided stub function for more complex filtering
            from pydantic2django.discovery import always_include

            def custom_filter(model_name: str, model_class: type[BaseModel]) -> bool:
                # Add your custom filtering logic here
                if model_name in ['PrivateModel', 'InternalConfig']:
                    return False
                return always_include(model_name, model_class)

            discover_models(['my_package'], filter_function=custom_filter)
            ```
        """
        logger.info(f"Discovering models from packages {package_names} for {app_label}...")

        # Discover models from all packages
        for package_name in package_names:
            discovered_models = self._discover_models_from_package(package_name, app_label=app_label)

            # Apply filter function if provided
            if filter_function:
                filtered_out = []

                for model_name, model_cls in discovered_models.items():
                    if filter_function(model_cls):
                        self.filtered_models[model_name] = model_cls
                    else:
                        filtered_out.append(model_name)

                if filtered_out:
                    logger.info(f"Filtered out {len(filtered_out)} models: {', '.join(filtered_out)}")
            else:
                self.filtered_models.update(discovered_models)

        logger.info(f"Discovered {len(self.discovered_models)} models")
        logger.info(f"Filtered discovered models to {len(self.filtered_models)} models")

    def _should_convert_to_django_model(self, model_class: type[BaseModel]) -> bool:
        """
        Determine if a Pydantic model should be converted to a Django model.

        Args:
            model_class: The Pydantic model class to check

        Returns:
            True if the model should be converted, False otherwise
        """
        # Skip models that directly inherit from ABC
        if ABC in model_class.__bases__:
            logger.info(f"Skipping {model_class.__name__} because it directly inherits from ABC")
            return False

        # Skip models that are marked as abstract
        if getattr(model_class, "__abstract__", False):
            logger.info(f"Skipping {model_class.__name__} because it is marked as abstract")
            return False

        return True

    def _discover_models_from_package(
        self, package_name: str, app_label: str = "django_llm"
    ) -> dict[str, type[BaseModel]]:
        """
        Discover Pydantic models in a package.

        Args:
            package_name: The package to search for models
            app_label: The Django app label to use for model registration

        Returns:
            Dict of discovered models
        """
        discovered_models = {}

        try:
            # Get the package
            package = importlib.import_module(package_name)

            # Handle modules without __file__ attribute (e.g., dynamically created modules in tests)
            if not hasattr(package, "__file__") or not package.__file__:
                # For modules without __file__, directly inspect the module's attributes
                logger.info(f"Module {package_name} has no __file__ attribute, inspecting directly")
                for name, obj in inspect.getmembers(package):
                    logger.info(f"Inspecting {name}: {obj}")
                    if is_pydantic_model(obj) and self._should_convert_to_django_model(obj):
                        logger.info(f"Found Pydantic model: {name}")
                        model_name = normalize_model_reference(name)
                        discovered_models[model_name] = obj
                    else:
                        if inspect.isclass(obj):
                            logger.info(f"Not a Pydantic model: {name}, bases: {obj.__bases__}")

                # Update registry state before returning
                self.discovered_models.update(discovered_models)

                # Update normalized models (but defer dependency analysis)
                for model_name, model_cls in discovered_models.items():
                    self.normalized_models[model_name] = model_cls
                    self.dependencies[model_name] = set()  # Initialize empty dependencies

                return discovered_models

            package_path = Path(package.__file__).parent

            # Walk through all Python files in the package
            for py_file in package_path.rglob("*.py"):
                if py_file.name.startswith("_"):
                    continue

                # Convert file path to module path
                relative_path = py_file.relative_to(package_path)
                module_path = f"{package_name}.{'.'.join(relative_path.with_suffix('').parts)}"

                try:
                    # Import the module
                    module = importlib.import_module(module_path)

                    # Find all Pydantic models in the module
                    for name, obj in inspect.getmembers(module):
                        if is_pydantic_model(obj) and self._should_convert_to_django_model(obj):
                            model_name = normalize_model_reference(name)
                            discovered_models[model_name] = obj

                except Exception as e:
                    logger.error(f"Error importing {module_path}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error discovering models in {package_name}: {e}")

        # Update registry state
        self.discovered_models.update(discovered_models)

        # Update normalized models (but defer dependency analysis)
        for model_name, model_cls in discovered_models.items():
            self.normalized_models[model_name] = model_cls
            self.dependencies[model_name] = set()  # Initialize empty dependencies

        return discovered_models

    def analyze_dependencies(self, app_label: str) -> None:
        """
        Analyze dependencies between models.

        Args:
            app_label: The Django app label to use for model registration
        """
        logger.info("Analyzing model dependencies...")
        self.app_label = app_label  # Store app_label for use in get_registration_order

        # Clear existing dependencies
        self.dependencies.clear()

        # First analyze dependencies from Pydantic models
        for model_name, pydantic_model in self.filtered_models.items():
            self.dependencies[model_name] = set()
            for _, field in pydantic_model.model_fields.items():
                annotation = field.annotation
                if annotation is not None:
                    # Handle direct model references
                    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
                        dep_name = normalize_model_reference(annotation.__name__)
                        self.dependencies[model_name].add(dep_name)
                    # Handle generic types (List, Dict, etc.)
                    elif hasattr(annotation, "__origin__"):
                        origin = get_origin(annotation)
                        args = get_args(annotation)
                        if origin in (list, dict):
                            for arg in args:
                                if inspect.isclass(arg) and issubclass(arg, BaseModel):
                                    dep_name = normalize_model_reference(arg.__name__)
                                    self.dependencies[model_name].add(dep_name)

    def validate_dependencies(self) -> list[str]:
        """
        Validate that all model dependencies exist.

        Returns:
            List of error messages for missing dependencies
        """

        return find_missing_models(self.normalized_models, self.dependencies)

    def get_registration_order(self) -> list[str]:
        """
        Get the order in which models should be registered based on their dependencies.

        Returns:
            List of model names in dependency order
        """
        if self._registration_order is None:
            errors = self.validate_dependencies()
            if errors:
                logger.warning(f"The following models are referenced but not available: {errors}")
                logger.warning("These will be appended to context. and removed from the dependency graph.")
                # for error in errors:
                #    del self.dependencies[error]

            # Create a copy of dependencies that only includes models that are in filtered_models
            filtered_dependencies = {}
            for model_name in self.dependencies:
                # Only include dependencies for models that exist in filtered_models
                if model_name in self.filtered_models:
                    filtered_dependencies[model_name] = set()
                    # Only include dependencies that exist in filtered_models
                    for dep in self.dependencies[model_name]:
                        if dep in self.filtered_models:
                            filtered_dependencies[model_name].add(dep)

            # Get the raw order from topological sort using the filtered dependencies
            raw_order = topological_sort(filtered_dependencies)
            # Store the unqualified model names order
            self._registration_order = raw_order
            # Return with app label for logging purposes
            registration_models = "\n".join([f"  - {model}" for model in raw_order])
            logger.info(f"Registration order: \n{registration_models}")
        return self._registration_order

    def get_qualified_registration_order(self) -> list[str]:
        """
        Get the fully qualified registration order (with app label).

        Returns:
            List of model names with app label prefix in dependency order
        """
        return [f"{self.app_label}.{model_name}" for model_name in self.get_registration_order()]

    def get_models_in_registration_order(self) -> list[type[BaseModel]]:
        """Get all discovered Pydantic models in registration order."""
        try:
            return [self.filtered_models[model_name] for model_name in self.get_registration_order()]
        except KeyError as e:
            logger.error(f"Model {e} not found in filtered models")
            logger.error(f"Available models: {self.filtered_models.keys()}")
            raise

    def get_app_dependencies(self) -> dict[str, set[str]]:
        """
        Get a mapping of app dependencies based on model relationships.
        This can be used by Django apps to determine initialization order.

        Returns:
            Dict mapping app labels to their dependent app labels
        """
        app_deps = defaultdict(set)

        for model_name, deps in self.dependencies.items():
            app_label = model_name.split(".")[0]
            for dep in deps:
                dep_app = dep.split(".")[0]
                if dep_app != app_label:
                    app_deps[app_label].add(dep_app)

        return dict(app_deps)

    def get_model_dependencies_recursive(
        self, model: type[BaseModel], app_label: str, visited: set[str] | None = None
    ) -> set[str]:
        """
        Get all dependencies for a model recursively.

        Args:
            model: The Pydantic model to analyze
            app_label: The app label to use for model references
            visited: Set of already visited models to prevent infinite recursion

        Returns:
            Set of model names that this model depends on
        """
        deps = set()
        if visited is None:
            visited = set()

        model_name = normalize_model_reference(model.__name__)
        if model_name in visited:
            return deps

        visited.add(model_name)

        # For Pydantic models, analyze field dependencies
        if is_pydantic_model(model):
            for _, field in model.model_fields.items():
                annotation = field.annotation
                if annotation is not None:
                    # Handle direct model references
                    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
                        dep_name = normalize_model_reference(annotation.__name__)
                        deps.add(dep_name)
                    # Handle generic types (List, Dict, etc.)
                    elif hasattr(annotation, "__origin__"):
                        origin = get_origin(annotation)
                        args = get_args(annotation)
                        if origin in (list, dict):
                            for arg in args:
                                if inspect.isclass(arg) and issubclass(arg, BaseModel):
                                    dep_name = normalize_model_reference(arg.__name__)
                                    deps.add(dep_name)

        return deps

    def get_discovered_models(self) -> dict[str, type[BaseModel]]:
        """Get all discovered Pydantic models."""
        return self.discovered_models

    def get_django_models(self, app_label: str = "django_llm") -> dict[str, type[models.Model]]:
        """Get all registered Django models for the given app label."""
        return self.django_models

    def clear(self) -> None:
        """Clear the registry, removing all discovered and registered models."""
        self.discovered_models.clear()
        self.normalized_models.clear()
        self.dependencies.clear()
        self.django_models.clear()
        self._registration_order = None
