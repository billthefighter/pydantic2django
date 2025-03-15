"""
Core functionality for converting Pydantic models to Django models.

This module provides the core functionality for converting individual Pydantic models
to Django models, handling field conversion and model creation.
"""
import logging
from typing import Any, Optional, TypeVar, cast

from django.db import models
from pydantic import BaseModel

from .context_storage import ModelContext, create_context_for_model
from .field_converter import convert_field

T = TypeVar("T", bound=BaseModel)

# Cache for converted models to prevent duplicate conversions
_converted_models: dict[str, type[models.Model]] = {}

logger = logging.getLogger(__name__)


def clear_model_registry() -> None:
    """Clear the internal model registry cache.

    This is useful for testing and when you need to force model regeneration.
    """
    _converted_models.clear()


def make_django_model(
    pydantic_model: type[T],
    base_django_model: Optional[type[models.Model]] = None,
    check_migrations: bool = True,
    skip_relationships: bool = False,
    existing_model: Optional[type[models.Model]] = None,
    class_name_prefix: str = "Django",
    strict: bool = False,
    **options: Any,
) -> tuple[type[models.Model], Optional[dict[str, models.Field]], Optional[ModelContext]]:
    """
    Convert a Pydantic model to a Django model, with optional base Django model inheritance.

    Args:
        pydantic_model: The Pydantic model class to convert
        base_django_model: Optional base Django model to inherit from
        check_migrations: Whether to check for needed migrations
        skip_relationships: Whether to skip relationship fields (useful during initial model creation)
        existing_model: Optional existing model to update with new fields
        class_name_prefix: Prefix to use for the generated Django model class name
        strict: If True, raise an error on field collisions; if False, keep base model fields
        **options: Additional options for customizing the conversion

    Returns:
        A tuple of (django_model, field_updates, model_context) where:
        - django_model is the Django model class that corresponds to the Pydantic model
        - field_updates is a dict of fields that need to be added to an existing model, or None
        - model_context is the ModelContext object containing context information, or None if not needed

    Raises:
        ValueError: If app_label is not provided in options or if field type cannot be mapped
    """
    if "app_label" not in options:
        raise ValueError("app_label must be provided in options")

    logger.debug(f"Converting Pydantic model {pydantic_model.__name__}")
    if base_django_model:
        logger.debug(f"Using base Django model {base_django_model.__name__}")

    # Check if model was already converted and we're not updating an existing model
    model_key = f"{pydantic_model.__module__}.{pydantic_model.__name__}"
    if model_key in _converted_models and not existing_model:
        logger.debug(f"Returning cached model for {model_key}")
        return _converted_models[model_key], None, None

    # Get all fields from the Pydantic model
    pydantic_fields = pydantic_model.model_fields
    logger.debug(f"Processing {len(pydantic_fields)} fields from Pydantic model")

    # Create Django model fields
    django_fields = {}
    relationship_fields = {}
    invalid_fields = []

    for field_name, field_info in pydantic_fields.items():
        try:
            # Skip id field if we're updating an existing model
            if field_name == "id" and existing_model:
                continue

            # Create the Django field
            django_field = convert_field(
                field_name,
                field_info,
                skip_relationships=skip_relationships,
                app_label=options["app_label"],
            )

            # Handle relationship fields differently based on skip_relationships
            if isinstance(
                django_field,
                (models.ForeignKey, models.ManyToManyField, models.OneToOneField),
            ):
                if skip_relationships:
                    # Store relationship fields for later
                    relationship_fields[field_name] = django_field
                    continue

            django_fields[field_name] = django_field

        except (ValueError, TypeError) as e:
            logger.warning(f"Skipping field {field_name}: {str(e)}")
            invalid_fields.append((field_name, str(e)))
            continue

    # If we have invalid fields, raise a ValueError
    if invalid_fields and not skip_relationships and not options.get("ignore_errors", False):
        error_msg = "Failed to convert the following fields:\n"
        for field_name, error in invalid_fields:
            error_msg += f"  - {field_name}: {error}\n"
        raise ValueError(error_msg)

    # If we're updating an existing model, return only the relationship fields
    if existing_model:
        logger.debug(f"Returning relationship fields for existing model {existing_model.__name__}")
        return existing_model, relationship_fields, None

    # Check for field collisions if a base Django model is provided
    if base_django_model:
        # Use hasattr to safely check for _meta
        if hasattr(base_django_model, "_meta"):
            base_fields = base_django_model._meta.get_fields()
            base_field_names = {field.name for field in base_fields}
            logger.debug(f"Checking field collisions with base model {base_django_model.__name__}")

            # Check for collisions
            collision_fields = set(django_fields.keys()) & base_field_names
            if collision_fields:
                if strict:
                    # In strict mode, raise an error with helpful message
                    error_msg = (
                        f"Field collision detected with base model: {collision_fields}. "
                        f"Options: 1) Change the base model fields, 2) Rename the Pydantic fields, "
                        f"or 3) Set strict=False to keep base model fields and discard Pydantic fields."
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                else:
                    # In non-strict mode, keep base model fields and discard Pydantic fields
                    logger.warning(
                        f"Field collision detected with base model: {collision_fields}. "
                        f"Keeping base model fields and discarding Pydantic fields."
                    )
                    for field_name in collision_fields:
                        django_fields.pop(field_name, None)

    # Determine base classes
    base_classes = [base_django_model] if base_django_model else [models.Model]
    logger.debug(f"Using base classes: {[cls.__name__ for cls in base_classes]}")

    # Set up Meta options
    meta_app_label = options["app_label"]
    meta_db_table = options.get("db_table", f"{meta_app_label}_{pydantic_model.__name__.lower()}")

    # Create Meta class
    meta_attrs = {
        "app_label": meta_app_label,
        "db_table": meta_db_table,
        "abstract": False,  # Ensure model is not abstract
        "managed": True,  # Ensure model is managed by Django
    }

    # Add verbose names if available
    doc = (getattr(pydantic_model, "__doc__", "") or "").strip()
    meta_attrs["verbose_name"] = doc or pydantic_model.__name__
    meta_attrs["verbose_name_plural"] = f"{meta_attrs['verbose_name']}s"

    # If inheriting from an abstract model, we still need to set app_label
    # to avoid Django's error about missing app_label
    if (
        base_django_model
        and hasattr(base_django_model, "_meta")
        and getattr(base_django_model._meta, "abstract", False)
    ):
        # Keep app_label even for abstract base models
        logger.debug("Keeping app_label for model with abstract base")

    # Create Meta class
    if base_django_model and hasattr(base_django_model, "_meta"):
        # Inherit from base model's Meta class if it exists
        base_meta = getattr(base_django_model._meta, "original_attrs", {})
        meta_attrs.update(base_meta)
        # Ensure model is not abstract even if base model is
        meta_attrs["abstract"] = False
        meta_attrs["managed"] = True
        # Always ensure app_label is set
        meta_attrs["app_label"] = meta_app_label
        Meta = type("Meta", (object,), meta_attrs)
        logger.debug(f"Created Meta class inheriting from {base_django_model.__name__}")
    else:
        Meta = type("Meta", (), meta_attrs)
        logger.debug("Created new Meta class")

    # Create the model attributes
    attrs = {
        "__module__": pydantic_model.__module__,  # Use the Pydantic model's module
        "Meta": Meta,
        **django_fields,
    }

    # Create the Django model
    model_name = f"{class_name_prefix}{pydantic_model.__name__}"

    # Use the correct base class
    bases = tuple(base_classes)

    # Create the model class
    model = type(model_name, bases, attrs)
    django_model = cast(type[models.Model], model)

    # Create context object if needed
    model_context = None
    if any(getattr(field, "is_relationship", False) for field in django_fields.values()):
        model_context = create_context_for_model(django_model, pydantic_model)

    logger.debug(f"Created Django model {model_name}")

    # Register the model with Django if it has a Meta class with app_label
    if hasattr(django_model, "_meta") and hasattr(django_model._meta, "app_label"):
        from django.apps import apps

        app_label = django_model._meta.app_label

        try:
            apps.get_registered_model(app_label, model_name)
        except LookupError:
            apps.register_model(app_label, django_model)
            logger.debug(f"Registered model {model_name} with app {app_label}")

    # Cache the model if not updating an existing one
    if not existing_model:
        _converted_models[model_key] = django_model
        logger.debug(f"Cached model {model_key}")

    return django_model, relationship_fields, model_context
