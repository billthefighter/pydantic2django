"""
Core functionality for converting Pydantic models to Django models.

This module provides the core functionality for converting individual Pydantic models
to Django models, handling field conversion and model creation.
"""
from typing import Any, Dict, Optional, Type, TypeVar, cast, Tuple
import inspect

from django.db import models
from pydantic import BaseModel

from .fields import get_django_field
from .methods import create_django_model_with_methods

T = TypeVar("T", bound=BaseModel)

# Cache for converted models to prevent duplicate conversions
_converted_models: Dict[str, type[models.Model]] = {}


def make_django_model(
    pydantic_model: type[T],
    base_django_model: Optional[type[models.Model]] = None,
    check_migrations: bool = True,
    skip_relationships: bool = False,
    existing_model: Optional[type[models.Model]] = None,
    **options: Any,
) -> Tuple[type[models.Model], Optional[Dict[str, models.Field]]]:
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


def clear_model_registry() -> None:
    """Clear the model conversion cache."""
    _converted_models.clear()
