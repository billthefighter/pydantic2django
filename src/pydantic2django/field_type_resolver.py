"""
Field type resolution for Django model fields.

This module provides centralized field type detection and resolution logic
to ensure consistency across the codebase.
"""
from typing import Any, Optional, Union, get_args, get_origin

from django.db import models

from pydantic2django.field_utils import RelationshipFieldHandler
from pydantic2django.types import is_serializable_type


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
        # Delegate relationship handling to RelationshipFieldHandler
        field_class, kwargs = RelationshipFieldHandler.detect_field_type(field_type)
        if field_class:
            return field_class, kwargs

        # Handle Optional types
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
        # Delegate to RelationshipFieldHandler
        return RelationshipFieldHandler.is_relationship_field(field_type)

    @staticmethod
    def get_relationship_type(field_type: Any) -> Optional[type[models.Field]]:
        """
        Get the specific type of relationship field.

        Args:
            field_type: The type to check

        Returns:
            The Django field class for the relationship, or None if not a relationship
        """
        # Delegate to RelationshipFieldHandler
        return RelationshipFieldHandler.get_relationship_type(field_type)
