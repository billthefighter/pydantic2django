"""
Model registry management for pydantic2django.

This module provides utilities for discovering, tracking dependencies,
and managing registration of Pydantic models as Django models.
"""
import importlib
import inspect
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, get_args, get_origin

from django.apps import apps
from django.db import models
from pydantic import BaseModel

from .core import make_django_model
from .factory import DjangoModelFactory

logger = logging.getLogger(__name__)


def normalize_model_name(name: str) -> str:
    """
    Normalize a model name by removing generic type parameters and ensuring proper Django model naming.

    Args:
        name: The model name to normalize

    Returns:
        Normalized model name
    """
    # Remove generic type parameters
    name = re.sub(r"\[.*?\]", "", name)

    # Ensure Django prefix
    if not name.startswith("Django"):
        name = f"Django{name}"

    return name


def get_model_dependencies_recursive(model: type[models.Model], app_label: str) -> set[str]:
    """Get all dependencies for a model recursively."""
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
            print(f"Warning: Cyclic dependency detected for {node}")
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


def is_pydantic_model(obj: type) -> bool:
    """Check if an object is a Pydantic model class."""
    return inspect.isclass(obj) and issubclass(obj, BaseModel) and obj != BaseModel


class ModelRegistryManager:
    """
    Manages the discovery, dependency tracking, and registration order of Pydantic models
    for conversion to Django models.
    """

    def __init__(self):
        self.discovered_models: dict[str, type[BaseModel]] = {}
        self.normalized_models: dict[str, type[BaseModel]] = {}
        self.dependencies: dict[str, set[str]] = {}
        self.django_models: dict[str, type[models.Model]] = {}
        self._registration_order: Optional[list[str]] = None

    def discover_models(self, package_name: str, app_label: str = "django_llm") -> dict[str, type[BaseModel]]:
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
            if not package.__file__:
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
                        if is_pydantic_model(obj):
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
                django_model = apps.get_model(app_label, model_name.replace("Django", ""))
                deps = get_model_dependencies_recursive(django_model, app_label)
                self.dependencies[model_name] = deps
            except Exception as e:
                print(f"Warning: Could not analyze dependencies for {model_name}: {e}")
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
                print(f"Error registering model {full_name}: {e}")
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

    def clear(self):
        """Clear all discovered and registered models."""
        self.discovered_models.clear()
        self.normalized_models.clear()
        self.dependencies.clear()
        self.django_models.clear()
        self._registration_order = None
