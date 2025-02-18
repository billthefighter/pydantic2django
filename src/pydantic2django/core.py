"""
Core functionality for converting Pydantic models to Django models.
"""
from typing import Any, Optional, TypeVar

from django.db import models
from pydantic import BaseModel

from .fields import get_django_field
from .methods import create_django_model_with_methods
from .migrations import check_model_migrations

T = TypeVar("T", bound=BaseModel)


def make_django_model(
    pydantic_model: type[T],
    base_django_model: Optional[type[models.Model]] = None,
    check_migrations: bool = True,
    **options: Any,
) -> tuple[type[models.Model], Optional[list[str]]]:
    """
    Convert a Pydantic model to a Django model, with optional base Django model inheritance.

    Args:
        pydantic_model: The Pydantic model class to convert
        base_django_model: Optional base Django model to inherit from
        check_migrations: Whether to check for needed migrations
        **options: Additional options for customizing the conversion

    Returns:
        A tuple of (django_model, migration_operations) where:
        - django_model is the Django model class that corresponds to the Pydantic model
        - migration_operations is a list of needed migration operations or None if checking is disabled

    Raises:
        ValueError: If there are field collisions between Pydantic and base Django model
    """
    # Get all fields from the Pydantic model
    pydantic_fields = pydantic_model.model_fields

    # Create Django model fields
    django_fields = {}
    for field_name, field_info in pydantic_fields.items():
        try:
            django_field = get_django_field(field_name, field_info)
            django_fields[field_name] = django_field
        except ValueError as e:
            # Log warning about skipped field
            import warnings

            warnings.warn(f"Skipping field {field_name}: {str(e)}", stacklevel=2)
            continue

    # Check for field collisions if a base Django model is provided
    if base_django_model:
        base_fields = base_django_model._meta.get_fields()
        base_field_names = {field.name for field in base_fields}

        # Check for collisions
        collision_fields = set(django_fields.keys()) & base_field_names
        if collision_fields:
            raise ValueError(f"Field collision detected with base model. " f"Conflicting fields: {collision_fields}")

    # Determine base classes
    base_classes = [base_django_model] if base_django_model else [models.Model]

    # Set up Meta options
    class Meta:
        app_label = options.get("app_label", "django_pydantic")
        db_table = options.get("db_table", pydantic_model.__name__.lower())
        verbose_name = (getattr(pydantic_model, "__doc__", "") or "").strip() or pydantic_model.__name__
        verbose_name_plural = f"{verbose_name}s"

    # Create the model attributes
    attrs = {"__module__": pydantic_model.__module__, "Meta": Meta, **django_fields}

    # Create the Django model with copied methods
    django_model = create_django_model_with_methods(
        pydantic_model.__name__, pydantic_model, attrs, base_classes=base_classes
    )

    # Check for needed migrations if enabled
    operations = None
    if check_migrations:
        _, operations = check_model_migrations(django_model)

    return django_model, operations
