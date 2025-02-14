"""
Core functionality for converting Pydantic models to Django models.
"""
from typing import Any, TypeVar

from django.db import models
from pydantic import BaseModel

from .fields import get_django_field
from .methods import create_django_model_with_methods

T = TypeVar("T", bound=BaseModel)


def make_django_model(pydantic_model: type[T], **options: Any) -> type[models.Model]:
    """
    Convert a Pydantic model to a Django model.

    Args:
        pydantic_model: The Pydantic model class to convert
        **options: Additional options for customizing the conversion

    Returns:
        A Django model class that corresponds to the Pydantic model

    Example:
        >>> class User(BaseModel):
        ...     name: str
        ...     age: int
        ...
        ...     @property
        ...     def is_adult(self) -> bool:
        ...         return self.age >= 18
        ...
        ...     def get_display_name(self) -> str:
        ...         return f"{self.name} ({self.age})"
        ...
        >>> DjangoUser = make_django_model(User)
        >>> user = DjangoUser(name="John", age=25)
        >>> user.is_adult
        True
        >>> user.get_display_name()
        'John (25)'
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

    # Set up Meta options
    class Meta:
        app_label = options.get("app_label", "django_pydantic")
        db_table = options.get("db_table", pydantic_model.__name__.lower())
        verbose_name = getattr(pydantic_model, "__doc__", "").strip() or pydantic_model.__name__
        verbose_name_plural = f"{verbose_name}s"

    # Create the model attributes
    attrs = {"__module__": pydantic_model.__module__, "Meta": Meta, **django_fields}

    # Create and return the Django model with copied methods
    return create_django_model_with_methods(pydantic_model.__name__, pydantic_model, attrs)
