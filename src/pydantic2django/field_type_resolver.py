"""
Field type resolution for Django model fields.

This module provides centralized field type detection and resolution logic
to ensure consistency across the codebase.
"""
import inspect
from enum import Enum
from typing import Any, Optional, Union, get_args, get_origin

from django.db import models
from pydantic import BaseModel

from .serialization import is_serializable


def is_serializable_type(field_type: Any) -> bool:
    """
    Check if a type is serializable (can be stored in the database).

    Args:
        field_type: The type to check

    Returns:
        True if the type is serializable, False otherwise
    """
    # Handle Optional types
    origin = get_origin(field_type)
    args = get_args(field_type)

    if origin is Union and type(None) in args:
        # For Optional types, check the inner type
        inner_type = next(arg for arg in args if arg is not type(None))
        return is_serializable_type(inner_type)

    # Basic Python types that are always serializable
    basic_types = (str, int, float, bool, dict, list, set)
    if field_type in basic_types:
        return True

    # Handle collection types
    if origin in (list, dict, set):
        # For collections, check if all type arguments are serializable
        return all(is_serializable_type(arg) for arg in args)

    # Handle Pydantic models (they can be serialized to JSON)
    if inspect.isclass(field_type) and issubclass(field_type, BaseModel):
        return True

    # Handle Enums (they can be serialized)
    if inspect.isclass(field_type) and issubclass(field_type, Enum):
        return True

    # For class types, check if they have a serialization method
    if inspect.isclass(field_type):
        # Create a dummy instance to test serialization
        try:
            instance = object.__new__(field_type)
            return is_serializable(instance)
        except Exception:
            # If we can't create an instance, assume it's not serializable
            return False

    # If none of the above conditions are met, it's not serializable
    return False


def is_pydantic_model(obj: Any) -> bool:
    """
    Check if an object is a Pydantic model class.

    Args:
        obj: The object to check

    Returns:
        True if the object is a Pydantic model class, False otherwise
    """
    try:
        return issubclass(obj, BaseModel)
    except TypeError:
        return False


class FieldTypeResolver:
    """
    Centralized field type resolution for Django model fields.
    This class handles all field type detection logic to ensure consistency
    across the codebase.
    """

    @staticmethod
    def resolve_field_type(
        field_type: Any,
    ) -> tuple[type[models.Field], dict[str, Any]]:
        """
        Resolve a field type to its corresponding Django field type and attributes.

        Args:
            field_type: The type to resolve (can be a Pydantic field type, Python type, etc.)

        Returns:
            Tuple of (Django field class, field attributes)
        """
        # Handle Optional types first
        origin = get_origin(field_type)
        args = get_args(field_type)
        field_kwargs = {}

        if origin is Union and type(None) in args:
            # Get the non-None type
            field_type = next(arg for arg in args if arg is not type(None))
            field_kwargs["null"] = True
            field_kwargs["blank"] = True
            # Update origin and args for the actual type
            origin = get_origin(field_type)
            args = get_args(field_type)

        # Check if it's an abstract base class
        if inspect.isabstract(field_type):
            field_kwargs["is_relationship"] = True
            return models.TextField, field_kwargs

        # Check if it's a non-serializable type
        if not is_serializable_type(field_type):
            field_kwargs["is_relationship"] = True
            return models.TextField, field_kwargs

        # Check if it's a Pydantic model (direct relationship)
        if is_pydantic_model(field_type):
            field_kwargs["on_delete"] = models.CASCADE
            return models.ForeignKey, field_kwargs

        # Check if it's a list of Pydantic models (many-to-many relationship)
        if origin is list and args and is_pydantic_model(args[0]):
            return models.ManyToManyField, field_kwargs

        # Handle basic Python types
        if field_type is str:
            field_kwargs["max_length"] = 255
            return models.CharField, field_kwargs
        elif field_type is int:
            return models.IntegerField, field_kwargs
        elif field_type is float:
            return models.FloatField, field_kwargs
        elif field_type is bool:
            return models.BooleanField, field_kwargs
        elif field_type is dict or origin is dict:
            return models.JSONField, field_kwargs
        elif field_type is list or origin is list:
            return models.JSONField, field_kwargs

        # For any other serializable type, use JSONField
        if is_serializable_type(field_type):
            return models.JSONField, field_kwargs

        # Default to TextField for unknown types
        return models.TextField, field_kwargs

    @staticmethod
    def is_relationship_field(field_type: Any) -> bool:
        """
        Check if a field type represents a relationship.

        Args:
            field_type: The type to check

        Returns:
            True if the field type represents a relationship, False otherwise
        """
        # Handle Optional types
        origin = get_origin(field_type)
        args = get_args(field_type)
        if origin is Union and type(None) in args:
            field_type = next(arg for arg in args if arg is not type(None))
            origin = get_origin(field_type)
            args = get_args(field_type)

        # Direct Pydantic model reference
        if is_pydantic_model(field_type):
            return True

        # List of Pydantic models
        if origin is list and args and is_pydantic_model(args[0]):
            return True

        # Abstract base class or non-serializable type
        if inspect.isabstract(field_type) or not is_serializable_type(field_type):
            return True

        return False

    @staticmethod
    def get_relationship_type(field_type: Any) -> Optional[type[models.Field]]:
        """
        Get the specific type of relationship field.

        Args:
            field_type: The type to check

        Returns:
            The Django field class for the relationship, or None if not a relationship
        """
        # Handle Optional types
        origin = get_origin(field_type)
        args = get_args(field_type)
        if origin is Union and type(None) in args:
            field_type = next(arg for arg in args if arg is not type(None))
            origin = get_origin(field_type)
            args = get_args(field_type)

        # Abstract base class or non-serializable type
        if inspect.isabstract(field_type) or not is_serializable_type(field_type):
            return models.TextField

        # Direct Pydantic model reference
        if is_pydantic_model(field_type):
            return models.ForeignKey

        # List of Pydantic models
        if origin is list and args and is_pydantic_model(args[0]):
            return models.ManyToManyField

        return None
