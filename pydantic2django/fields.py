"""
Field mapping between Pydantic and Django models.
"""
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any, Union, cast, get_args, get_origin
from uuid import UUID

from django.db import models
from pydantic import BaseModel, EmailStr
from pydantic.fields import FieldInfo

# Mapping of Python/Pydantic types to Django field classes
FIELD_TYPE_MAPPING: dict[type[Any], type[models.Field]] = {
    str: models.CharField,
    int: models.IntegerField,
    float: models.FloatField,
    bool: models.BooleanField,
    datetime: models.DateTimeField,
    date: models.DateField,
    time: models.TimeField,
    timedelta: models.DurationField,
    Decimal: models.DecimalField,
    UUID: models.UUIDField,
    EmailStr: models.EmailField,
    bytes: models.BinaryField,
}

# Default max_lengths for certain field types
DEFAULT_MAX_LENGTHS = {
    models.CharField: 255,
    models.EmailField: 254,  # RFC 5321
}


def get_field_kwargs(field_info: FieldInfo, field_type: type[Any]) -> dict[str, Any]:
    """
    Extract Django field kwargs from Pydantic field info.

    Args:
        field_info: Pydantic field information
        field_type: The resolved field type (after handling Optional/Union)

    Returns:
        Dictionary of kwargs for Django field initialization
    """
    kwargs: dict[str, Any] = {
        "null": not field_info.is_required(),
        "blank": not field_info.is_required(),
    }

    # Get field constraints from metadata
    metadata = field_info.json_schema_extra or {}
    if isinstance(metadata, dict):
        if "title" in metadata:
            kwargs["verbose_name"] = metadata["title"]
        if "description" in metadata:
            kwargs["help_text"] = metadata["description"]
        if "max_length" in metadata:
            kwargs["max_length"] = metadata["max_length"]
        if "on_delete" in metadata:
            kwargs["on_delete"] = metadata["on_delete"]
        if "related_name" in metadata:
            kwargs["related_name"] = metadata["related_name"]
        if "through" in metadata:
            kwargs["through"] = metadata["through"]
        if "through_fields" in metadata:
            kwargs["through_fields"] = metadata["through_fields"]
        if "symmetrical" in metadata:
            kwargs["symmetrical"] = metadata["symmetrical"]

    # Handle specific field types
    if field_type == str and "max_length" not in kwargs:
        kwargs["max_length"] = DEFAULT_MAX_LENGTHS[models.CharField]

    elif field_type == Decimal:
        # Get decimal constraints from metadata or use defaults
        kwargs["max_digits"] = metadata.get("max_digits", 19)
        kwargs["decimal_places"] = metadata.get("decimal_places", 4)

    return kwargs


def is_pydantic_model(type_: Any) -> bool:
    """Check if a type is a Pydantic model."""
    try:
        return isinstance(type_, type) and issubclass(type_, BaseModel)
    except TypeError:
        return False


def resolve_field_type(field_type: Any) -> tuple[type[Any], bool]:
    """
    Resolve the actual type from Optional/Union/List/Set types.

    Args:
        field_type: The type to resolve

    Returns:
        Tuple of (resolved_type, is_collection)
    """
    is_collection = False
    origin = get_origin(field_type)

    if origin is Union:
        args = get_args(field_type)
        if len(args) == 2 and type(None) in args:
            field_type = next(arg for arg in args if arg is not type(None))

    if origin in (list, list, set, set):
        args = get_args(field_type)
        if len(args) == 1:
            field_type = args[0]
            is_collection = True

    return cast(type[Any], field_type), is_collection


def get_relationship_field(field_name: str, field_info: FieldInfo, field_type: type[BaseModel]) -> models.Field:
    """
    Create a relationship field based on the field type and metadata.

    Args:
        field_name: Name of the field
        field_info: Pydantic field information
        field_type: The field type (should be a Pydantic model)

    Returns:
        Django relationship field
    """
    kwargs = get_field_kwargs(field_info, field_type)
    metadata = field_info.json_schema_extra or {}

    # Default on_delete to CASCADE if not specified
    kwargs.setdefault("on_delete", models.CASCADE)

    # Handle self-referential relationships
    to_model = "self" if metadata.get("self", False) else field_type.__name__

    # Check if it's a collection type
    _, is_collection = resolve_field_type(field_info.annotation)
    if is_collection:
        # It's a many-to-many relationship
        return models.ManyToManyField(
            to_model,
            through=kwargs.pop("through", None),
            through_fields=kwargs.pop("through_fields", None),
            symmetrical=kwargs.pop("symmetrical", None) if to_model == "self" else None,
            **kwargs,
        )

    # Check for one-to-one relationship hint in metadata
    if metadata.get("one_to_one", False):
        return models.OneToOneField(to_model, **kwargs)

    # Default to ForeignKey (many-to-one) relationship
    return models.ForeignKey(to_model, **kwargs)


def get_django_field(field_name: str, field_info: FieldInfo) -> models.Field:
    """
    Convert a Pydantic field to a Django model field.

    Args:
        field_name: Name of the field
        field_info: Pydantic field information

    Returns:
        Django model field instance

    Raises:
        ValueError: If field type cannot be mapped to a Django field
    """
    field_type, _ = resolve_field_type(field_info.annotation)

    # Handle relationship fields
    if is_pydantic_model(field_type):
        return get_relationship_field(field_name, field_info, cast(type[BaseModel], field_type))

    # Handle regular fields
    if field_type in FIELD_TYPE_MAPPING:
        field_class = FIELD_TYPE_MAPPING[field_type]
        field_kwargs = get_field_kwargs(field_info, field_type)
        return field_class(**field_kwargs)

    raise ValueError(f"Cannot map Pydantic field type {field_type} to Django field")
