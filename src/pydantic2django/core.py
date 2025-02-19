"""
Core functionality for converting Pydantic models to Django models.
"""
from typing import Any, Dict, Optional, Set, TypeVar, cast, List, Type
import inspect
import re

from django.apps import apps
from django.db import models
from pydantic import BaseModel

from .fields import get_django_field
from .methods import create_django_model_with_methods
from .migrations import check_model_migrations

T = TypeVar("T", bound=BaseModel)

_converted_models: Dict[str, type[models.Model]] = {}
_model_dependencies: Dict[str, Set[str]] = {}


def normalize_model_name(name: str) -> str:
    """
    Normalize a model name by removing generic type parameters and ensuring proper Django model naming.
    
    Args:
        name: The model name to normalize
        
    Returns:
        Normalized model name
    """
    # Remove generic type parameters
    name = re.sub(r'\[.*?\]', '', name)
    
    # Ensure Django prefix
    if not name.startswith('Django'):
        name = f'Django{name}'
        
    return name


def get_model_dependencies_recursive(model: Type[models.Model], app_label: str) -> Set[str]:
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


def validate_model_references(models: Dict[str, type], dependencies: Dict[str, Set[str]]) -> List[str]:
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
            if dep not in models and dep != 'self':
                errors.append(f"Model '{model_name}' references non-existent model '{dep}'")
    return errors


def topological_sort(dependencies: Dict[str, Set[str]]) -> List[str]:
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
            if dep != node and dep != 'self':  # Skip self-references
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


def make_django_model(
    pydantic_model: type[T],
    base_django_model: Optional[type[models.Model]] = None,
    check_migrations: bool = True,
    skip_relationships: bool = False,
    existing_model: Optional[type[models.Model]] = None,
    **options: Any,
) -> tuple[type[models.Model], Optional[dict[str, models.Field]]]:
    """
    Convert a Pydantic model to a Django model, with optional base Django model inheritance.

    Args:
        pydantic_model: The Pydantic model class to convert
        base_django_model: Optional base Django model to inherit from
        check_migrations: Whether to check for needed migrations
        skip_relationships: Whether to skip relationship fields (useful during initial model creation)
        existing_model: Optional existing model to update with new fields
        **options: Additional options for customizing the conversion

    Returns:
        A tuple of (django_model, field_updates) where:
        - django_model is the Django model class that corresponds to the Pydantic model
        - field_updates is a dict of fields that need to be added to an existing model, or None
    """
    # Check if model was already converted and we're not updating an existing model
    model_key = f"{pydantic_model.__module__}.{pydantic_model.__name__}"
    if model_key in _converted_models and not existing_model:
        return _converted_models[model_key], None

    # Initialize dependencies for this model
    _model_dependencies[model_key] = set()

    # Get all fields from the Pydantic model
    pydantic_fields = pydantic_model.model_fields

    # Create Django model fields
    django_fields = {}
    relationship_fields = {}

    for field_name, field_info in pydantic_fields.items():
        try:
            # Skip id field if we're updating an existing model
            if field_name == 'id' and existing_model:
                continue

            # Create the Django field
            django_field = get_django_field(field_name, field_info, skip_relationships=skip_relationships)

            # Handle relationship fields differently based on skip_relationships
            if isinstance(django_field, (models.ForeignKey, models.ManyToManyField, models.OneToOneField)):
                if skip_relationships:
                    # Store relationship fields for later
                    relationship_fields[field_name] = django_field
                    continue
                else:
                    # Track dependencies for relationship fields
                    related_model = django_field.remote_field.model
                    if isinstance(related_model, str):
                        _model_dependencies[model_key].add(related_model)

            django_fields[field_name] = django_field

        except ValueError as e:
            # Log warning about skipped field
            import warnings
            warnings.warn(f"Skipping field {field_name}: {str(e)}", stacklevel=2)
            continue

    # If we're updating an existing model, return only the relationship fields
    if existing_model:
        return existing_model, relationship_fields

    # Check for field collisions if a base Django model is provided
    if base_django_model:
        base_fields = base_django_model._meta.get_fields()
        base_field_names = {field.name for field in base_fields}

        # Check for collisions
        collision_fields = set(django_fields.keys()) & base_field_names
        if collision_fields:
            raise ValueError(f"Field collision detected with base model. Conflicting fields: {collision_fields}")

    # Determine base classes
    base_classes = [base_django_model] if base_django_model else [models.Model]

    # Set up Meta options
    if "app_label" not in options:
        raise ValueError("app_label must be provided in options")
    meta_app_label = options["app_label"]
    meta_db_table = options.get("db_table", f"{meta_app_label}_{pydantic_model.__name__.lower()}")

    class Meta:
        app_label = meta_app_label
        db_table = meta_db_table
        verbose_name = (getattr(pydantic_model, "__doc__", "") or "").strip() or pydantic_model.__name__
        verbose_name_plural = f"{verbose_name}s"

    # Create the model attributes
    attrs = {
        "__module__": pydantic_model.__module__,
        "Meta": Meta,
        **django_fields
    }

    # Create the Django model
    model_name = f"Django{pydantic_model.__name__}"
    django_model = type(model_name, tuple(base_classes), attrs)

    # Only store the model if we're not skipping relationships
    # This ensures we don't store incomplete models during the first pass
    if not skip_relationships:
        _converted_models[model_key] = django_model

    return django_model, relationship_fields if skip_relationships else None


def get_model_dependencies() -> Dict[str, Set[str]]:
    """Get the current model dependency graph."""
    return _model_dependencies.copy()


def clear_model_registry() -> None:
    """Clear the model registry and dependencies."""
    _converted_models.clear()
    _model_dependencies.clear()
