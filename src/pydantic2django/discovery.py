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
from typing import Any, Optional, cast, get_args, get_origin

from django.apps import apps
from django.db import models
from pydantic import BaseModel

from .base_django_model import Pydantic2DjangoBaseClass
from .core import make_django_model
from .factory import DjangoModelFactory
from .field_utils import is_pydantic_model
from .types import T
from .utils import normalize_model_name

logger = logging.getLogger(__name__)


def get_model_dependencies_recursive(model: type[models.Model], app_label: str) -> set[str]:
    """
    Get all dependencies for a model recursively.

    Args:
        model: The Django model to analyze
        app_label: The app label to use for model registration

    Returns:
        Set of model names that this model depends on
    """
    deps = set()

    for field in model._meta.get_fields():
        if hasattr(field, "remote_field") and field.remote_field:
            target = field.remote_field.model
            if isinstance(target, str):
                if "." not in target:
                    if not target.startswith("Django"):
                        target = f"Django{target}"
                    target = f"{app_label}.{target}"
                deps.add(target)
            elif inspect.isclass(target):
                target_name = target.__name__
                if not target_name.startswith("Django"):
                    target_name = f"Django{target_name}"
                deps.add(f"{app_label}.{target_name}")

        # Check through relationships for ManyToManyField
        if isinstance(field, models.ManyToManyField):
            remote_field = field.remote_field
            if isinstance(remote_field, models.ManyToManyRel):
                through = remote_field.through
                if through and through is not models.ManyToManyRel:
                    if isinstance(through, str):
                        if "." not in through:
                            if not through.startswith("Django"):
                                through = f"Django{through}"
                            through = f"{app_label}.{through}"
                        deps.add(through)
                    elif inspect.isclass(through):
                        through_name = through.__name__
                        if not through_name.startswith("Django"):
                            through_name = f"Django{through_name}"
                        deps.add(f"{app_label}.{through_name}")

    return deps


def validate_model_references(models: dict[str, type], dependencies: dict[str, set[str]]) -> list[str]:
    """
    Validate that all model references exist.

    Args:
        models: Dict mapping normalized model names to their classes
        dependencies: Dict mapping model names to their dependencies

    Returns:
        List of error messages for missing models
    """
    errors = []
    for model_name, deps in dependencies.items():
        for dep in deps:
            if dep not in models and dep != "self":
                errors.append(f"Model '{model_name}' references non-existent model '{dep}'")
    return errors


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


def register_django_model(model: type[models.Model], app_label: str) -> None:
    """
    Register a Django model with the app registry.

    Args:
        model: The Django model to register
        app_label: The app label to register under
    """
    model_name = model._meta.model_name
    if not model_name:
        return

    try:
        # Check if model is already registered
        existing = apps.get_registered_model(app_label, model_name)
        if existing is not model:
            # Unregister existing model if it's different
            apps.all_models[app_label].pop(model_name, None)
            # Register the new model
            apps.register_model(app_label, model)
    except LookupError:
        # Model not registered yet
        apps.register_model(app_label, model)


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

        # Get a specific Django model with proper type hints
        UserDjango = discovery.get_django_model(UserPydantic)
        user = UserDjango(name="John")
    """

    def __init__(self):
        """Initialize a new ModelDiscovery instance with its own registry."""
        self.discovered_models: dict[str, type[BaseModel]] = {}
        self.normalized_models: dict[str, type[BaseModel]] = {}
        self.dependencies: dict[str, set[str]] = {}
        self.django_models: dict[str, type[models.Model]] = {}
        self._registration_order: Optional[list[str]] = None

    def _should_convert_to_django_model(self, model_class: type[BaseModel]) -> bool:
        """
        Determine if a Pydantic model should be converted to a Django model.

        Args:
            model_class: The Pydantic model class to check

        Returns:
            True if the model should be converted, False otherwise
        """
        # Skip models that inherit from ABC
        if ABC in model_class.__mro__:
            logger.info(f"Skipping {model_class.__name__} because it inherits from ABC")
            return False

        return True

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
                filtered_models = {}
                filtered_out = []

                for model_name, model_cls in discovered_models.items():
                    if filter_function(model_cls):
                        filtered_models[model_name] = model_cls
                    else:
                        filtered_out.append(model_name)

                        # Remove from registry if it was added
                        if model_name in self.discovered_models:
                            del self.discovered_models[model_name]
                        if model_name in self.normalized_models:
                            del self.normalized_models[model_name]
                        if model_name in self.dependencies:
                            del self.dependencies[model_name]

                if filtered_out:
                    logger.info(f"Filtered out {len(filtered_out)} models: {', '.join(filtered_out)}")

        logger.info(f"Discovered {len(self.discovered_models)} models")

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
                        model_name = name  # Don't normalize the name for test modules
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
                            model_name = normalize_model_name(name)
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

    def setup_dynamic_models(
        self, app_label: str = "django_llm", skip_admin: bool = False
    ) -> dict[str, type[models.Model]]:
        """
        Set up dynamic models from discovered Pydantic models.

        This method should be called during migration operations, not during app initialization.
        It creates Django models from discovered Pydantic models.

        Args:
            app_label: The Django app label to use for model registration
            skip_admin: Whether to skip registering models with the Django admin interface

        Returns:
            Dict mapping model names to Django model classes
        """
        return self.setup_models(app_label, skip_admin)

    def setup_models(self, app_label: str = "django_llm", skip_admin: bool = False) -> dict[str, type[models.Model]]:
        """
        Set up all discovered models with proper relationships.

        Args:
            app_label: The Django app label to use for model registration
            skip_admin: Whether to skip registering models with the Django admin interface

        Returns:
            Dict mapping model names to Django model classes
        """
        logger.info(f"Setting up models for app {app_label}")
        logger.info("Discovered models:")
        for name, model in self.discovered_models.items():
            logger.info(f"  - {name}: {model}")

        # Create Django models for each discovered model
        for model_name, pydantic_model in self.discovered_models.items():
            logger.info(f"Creating Django model for {model_name}")
            try:
                django_model, _ = DjangoModelFactory[Any].create_model(pydantic_model, app_label=app_label)
                self.django_models[model_name] = django_model
                logger.info(f"Successfully created Django model: {django_model}")

                # Register the model with Django's app registry
                register_django_model(django_model, app_label)
                logger.info(f"Registered model with Django app registry: {django_model._meta.model_name}")
            except Exception as e:
                logger.error(f"Error creating Django model for {model_name}: {e}")

        logger.info("Final Django models:")
        for name, model in self.django_models.items():
            logger.info(f"  - {name}: {model}")

        return self.django_models

    def analyze_dependencies(self, app_label: str) -> None:
        """
        Analyze dependencies between models after they've been registered.
        Should be called after all models are registered with Django.

        Args:
            app_label: The Django app label the models are registered under
        """
        for model_name, _model_cls in self.normalized_models.items():
            try:
                # Get the registered Django model
                django_model = apps.get_model(app_label, model_name)
                deps = get_model_dependencies_recursive(django_model, app_label)
                self.dependencies[model_name] = deps
                logger.info(f"Dependencies for {model_name}: {deps}")
            except Exception as e:
                logger.warning(f"Could not analyze dependencies for {model_name}: {e}")
                continue

    def validate_dependencies(self) -> list[str]:
        """
        Validate that all model dependencies exist.

        Returns:
            List of error messages for missing dependencies
        """
        return validate_model_references(self.normalized_models, self.dependencies)

    def get_registration_order(self) -> list[str]:
        """
        Get the order in which models should be registered based on their dependencies.

        Returns:
            List of model names in dependency order
        """
        if self._registration_order is None:
            errors = self.validate_dependencies()
            if errors:
                raise ValueError(f"Cannot determine registration order due to dependency errors: {errors}")
            self._registration_order = topological_sort(self.dependencies)
        return self._registration_order

    def register_models(self, app_label: str = "django_llm") -> dict[str, type[models.Model]]:
        """
        Register all discovered models in dependency order.

        Args:
            app_label: The Django app label to use for registration

        Returns:
            Dict mapping model names to registered Django model classes
        """
        # Get registration order
        ordered_models = self.get_registration_order()

        # Register models in order
        for full_name in ordered_models:
            try:
                _, model_name = full_name.split(".")
                if model_name not in self.normalized_models:
                    continue

                pydantic_model = self.normalized_models[model_name]

                # Create and register Django model
                django_model, _ = make_django_model(
                    pydantic_model,
                    app_label=app_label,
                    db_table=f"{app_label}_{model_name.lower()}",
                    check_migrations=False,
                )
                self.django_models[model_name] = django_model
                register_django_model(django_model, app_label)

            except Exception as e:
                logger.error(f"Error registering model {full_name}: {e}")
                continue

        # Analyze dependencies after all models are registered
        self.analyze_dependencies(app_label)

        return self.django_models

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

    def get_model_dependencies(self, model: type[BaseModel]) -> set[str]:
        """
        Get dependencies for a single Pydantic model.

        Args:
            model: The Pydantic model to analyze

        Returns:
            Set of model names that this model depends on
        """
        deps = set()
        for field in model.model_fields.values():
            annotation = field.annotation
            if annotation is not None:
                # Check for List/Dict types containing Pydantic models
                if hasattr(annotation, "__origin__"):
                    origin = get_origin(annotation)
                    args = get_args(annotation)
                    if origin in (list, dict):
                        for arg in args:
                            if inspect.isclass(arg) and issubclass(arg, BaseModel):
                                deps.add(arg.__name__)
                # Check for direct Pydantic model references
                elif inspect.isclass(annotation) and issubclass(annotation, BaseModel):
                    deps.add(annotation.__name__)
        return deps

    def get_discovered_models(self) -> dict[str, type[BaseModel]]:
        """Get all discovered Pydantic models."""
        return self.discovered_models

    def get_django_models(self, app_label: str = "django_llm") -> dict[str, type[models.Model]]:
        """Get all registered Django models for the given app label."""
        return self.django_models

    def get_django_model(
        self, pydantic_model: type[T] | type[type[T]], app_label: str = "django_llm"
    ) -> type[Pydantic2DjangoBaseClass[T]]:
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
        # Handle both direct type and callable returning type
        if callable(pydantic_model) and not isinstance(pydantic_model, type):
            # If it's a fixture function, call it to get the actual model
            actual_model = cast(type[T], pydantic_model())
        else:
            # If it's already a type, use it directly
            actual_model = cast(type[T], pydantic_model)

        # Set up models for the specified app_label if not already done
        if not self.django_models:
            self.setup_models(app_label=app_label)

        # Get the fully qualified name of the Pydantic model
        module_name = actual_model.__module__
        class_name = actual_model.__name__
        fully_qualified_name = f"{module_name}.{class_name}"

        # Look for the model in the registry by checking the object_type field
        for model in self.django_models.values():
            if getattr(model, "object_type", None) == fully_qualified_name:
                return cast(type[Pydantic2DjangoBaseClass[T]], model)

        # If not found, try to create it
        django_model, _ = DjangoModelFactory[T].create_model(actual_model, app_label=app_label)

        return django_model

    def clear(self) -> None:
        """Clear the registry, removing all discovered and registered models."""
        self.discovered_models.clear()
        self.normalized_models.clear()
        self.dependencies.clear()
        self.django_models.clear()
        self._registration_order = None


# Create a singleton instance for convenience
discovery = ModelDiscovery()


# Convenience functions that use the singleton instance
def discover_models(
    package_names: list[str],
    app_label: str = "django_llm",
    filter_function: Optional[Callable[[str, type[BaseModel]], bool]] = None,
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
    discovery.discover_models(package_names, app_label, filter_function)


def setup_dynamic_models(app_label: str = "django_llm", skip_admin: bool = False) -> dict[str, type[models.Model]]:
    """
    Set up dynamic models from discovered Pydantic models.

    This is a convenience function that uses the singleton ModelDiscovery instance.

    Args:
        app_label: The Django app label to use for model registration
        skip_admin: Whether to skip registering models with the Django admin interface

    Returns:
        Dict mapping model names to Django model classes
    """
    return discovery.setup_dynamic_models(app_label, skip_admin)


def get_discovered_models() -> dict[str, type[BaseModel]]:
    """
    Get all discovered Pydantic models.

    This is a convenience function that uses the singleton ModelDiscovery instance.

    Returns:
        Dict of discovered Pydantic models
    """
    return discovery.get_discovered_models()


def get_django_models(app_label: str = "django_llm") -> dict[str, type[models.Model]]:
    """
    Get all registered Django models for the given app label.

    This is a convenience function that uses the singleton ModelDiscovery instance.

    Args:
        app_label: The Django app label to use for model registration

    Returns:
        Dict of registered Django models
    """
    return discovery.get_django_models(app_label)


def get_django_model(pydantic_model: type[T], app_label: str = "django_llm") -> type[Pydantic2DjangoBaseClass[T]]:
    """
    Get a Django model with proper type hints for a given Pydantic model.

    This is a convenience function that uses the singleton ModelDiscovery instance.

    Args:
        pydantic_model: The Pydantic model class
        app_label: The Django app label to use for model registration

    Returns:
        Django model class with proper type hints
    """
    return discovery.get_django_model(pydantic_model, app_label)
