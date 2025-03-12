"""
Shared utilities for field handling across pydantic2django modules.

This module contains common utilities, constants, and type definitions used by
field_type_mapping.py, fields.py, and static_django_model_generator.py to reduce
code duplication and improve maintainability.
"""
import logging
import re
from collections.abc import Callable
from enum import Enum
from typing import (
    Any,
    Optional,
    Union,
    cast,
    get_args,
    get_origin,
)

from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from pydantic import BaseModel
from pydantic.fields import FieldInfo

# Configure logging
logger = logging.getLogger("pydantic2django.field_utils")


def is_pydantic_model(obj: Any) -> bool:
    """
    Check if an object is a Pydantic model class.

    Args:
        obj: The object to check

    Returns:
        True if the object is a Pydantic model class, False otherwise
    """
    if not inspect_is_class_or_type(obj):
        return False

    # Check if it's a subclass of BaseModel
    try:
        return issubclass(obj, BaseModel)
    except TypeError:
        return False


def inspect_is_class_or_type(obj: Any) -> bool:
    """
    Check if an object is a class or type.

    Args:
        obj: The object to check

    Returns:
        True if the object is a class or type, False otherwise
    """
    import inspect

    return inspect.isclass(obj)


def sanitize_related_name(name: str, model_name: str = "", field_name: str = "") -> str:
    """
    Sanitize a related name for Django models.

    Args:
        name: The name to sanitize
        model_name: Optional model name for context
        field_name: Optional field name for context

    Returns:
        A sanitized related name
    """
    # Replace spaces and special characters with underscores
    sanitized = re.sub(r"[^\w]", "_", name)

    # Ensure it starts with a letter or underscore
    if sanitized and not sanitized[0].isalpha() and sanitized[0] != "_":
        sanitized = f"_{sanitized}"

    # If it's empty, generate a name based on model and field
    if not sanitized and model_name and field_name:
        sanitized = f"{model_name.lower()}_{field_name.lower()}_set"
    elif not sanitized and model_name:
        sanitized = f"{model_name.lower()}_set"
    elif not sanitized and field_name:
        sanitized = f"{field_name.lower()}_set"
    elif not sanitized:
        sanitized = "related_items"

    return sanitized


class FieldAttributeHandler:
    """
    Handles extraction and processing of field attributes from Pydantic field info.
    """

    @staticmethod
    def handle_field_attributes(
        field_info: FieldInfo,
        extra: Optional[Union[dict[str, Any], Callable[[FieldInfo], dict[str, Any]]]] = None,
    ) -> dict[str, Any]:
        """
        Extract and process field attributes from Pydantic field info.

        Args:
            field_info: The Pydantic field info
            extra: Optional extra attributes or callable to get extra attributes

        Returns:
            A dictionary of field attributes
        """
        kwargs = {}

        # Handle null/blank based on whether the field is optional
        is_optional = not field_info.is_required
        kwargs["null"] = is_optional
        kwargs["blank"] = is_optional

        # Handle default value
        if field_info.default is not None and field_info.default != Ellipsis:
            kwargs["default"] = field_info.default

        # Handle description as help_text
        if field_info.description:
            kwargs["help_text"] = field_info.description

        # Handle title as verbose_name
        if field_info.title:
            kwargs["verbose_name"] = field_info.title

        # Handle validators from field constraints
        # In Pydantic v2, constraints are stored in the field's metadata
        metadata = field_info.metadata
        if isinstance(metadata, dict):
            gt = metadata.get("gt")
            if gt is not None:
                kwargs.setdefault("validators", []).append(MinValueValidator(limit_value=gt))

            ge = metadata.get("ge")
            if ge is not None:
                kwargs.setdefault("validators", []).append(MinValueValidator(limit_value=ge))

            lt = metadata.get("lt")
            if lt is not None:
                kwargs.setdefault("validators", []).append(MaxValueValidator(limit_value=lt))

            le = metadata.get("le")
            if le is not None:
                kwargs.setdefault("validators", []).append(MaxValueValidator(limit_value=le))

        # Process extra attributes
        if extra:
            if callable(extra):
                extra_kwargs = extra(field_info)
                kwargs.update(extra_kwargs)
            else:
                kwargs.update(extra)

        return kwargs


# Define a type for relationship fields to help with type checking
class RelationshipField(models.Field):
    """Base class for relationship fields to help with type checking."""

    to: Any
    related_name: Optional[str]


class ForeignKeyField(RelationshipField):
    """Type for ForeignKey fields to help with type checking."""

    on_delete: Any


class ManyToManyField(RelationshipField):
    """Type for ManyToManyField fields to help with type checking."""

    through: Any


class RelationshipFieldHandler:
    """
    Handles creation and processing of relationship fields.
    """

    @staticmethod
    def get_related_model_name(field: models.Field) -> Optional[str]:
        """
        Get the related model name from a relationship field.

        Args:
            field: The relationship field

        Returns:
            The related model name or None if it couldn't be determined
        """
        related_model_name = None

        # Try different ways to get the related model
        if hasattr(field, "related_model") and field.related_model is not None:
            if isinstance(field.related_model, str):
                related_model_name = field.related_model
            else:
                try:
                    related_model_name = field.related_model.__name__
                except (AttributeError, TypeError):
                    pass

        # Try remote_field.model if related_model didn't work
        if not related_model_name and hasattr(field, "remote_field") and field.remote_field is not None:
            remote_field = field.remote_field
            if hasattr(remote_field, "model") and remote_field.model is not None:
                if isinstance(remote_field.model, str):
                    related_model_name = remote_field.model
                else:
                    try:
                        related_model_name = remote_field.model.__name__
                    except (AttributeError, TypeError):
                        pass

        # Try to_field as a last resort
        if not related_model_name:
            # Cast the field to the appropriate type for type checking
            if isinstance(field, models.ForeignKey) or isinstance(field, models.OneToOneField):
                rel_field = cast(ForeignKeyField, field)
            elif isinstance(field, models.ManyToManyField):
                rel_field = cast(ManyToManyField, field)
            else:
                rel_field = cast(RelationshipField, field)

            # Check if the field has a 'to' attribute
            if hasattr(rel_field, "to") and rel_field.to is not None:
                if isinstance(rel_field.to, str):
                    related_model_name = rel_field.to
                else:
                    try:
                        related_model_name = rel_field.to.__name__
                    except (AttributeError, TypeError):
                        pass

        return related_model_name

    @staticmethod
    def create_relationship_field(
        field_name: str,
        field_info: FieldInfo,
        field_type: Any,
        app_label: str,
        model_name: Optional[str] = None,
    ) -> Optional[models.Field]:
        """
        Create a relationship field based on the field type.

        Args:
            field_name: The name of the field
            field_info: The Pydantic field info
            field_type: The field type
            app_label: The Django app label
            model_name: Optional model name for context

        Returns:
            A Django relationship field or None if the field type is not supported
        """
        # Get field attributes
        kwargs = FieldAttributeHandler.handle_field_attributes(field_info)

        # Get the related model type
        origin = get_origin(field_type)
        args = get_args(field_type)

        # Handle List[Model] as ManyToManyField
        if origin is list and args and is_pydantic_model(args[0]):
            related_model = args[0]
            related_name = sanitize_related_name(
                getattr(field_info, "related_name", ""),
                model_name or "",
                field_name,
            )

            return models.ManyToManyField(
                to=f"{app_label}.{related_model.__name__}",
                related_name=related_name,
                **kwargs,
            )

        # Handle direct model reference as ForeignKey
        if is_pydantic_model(field_type):
            related_name = sanitize_related_name(
                getattr(field_info, "related_name", ""),
                model_name or "",
                field_name,
            )

            return models.ForeignKey(
                to=f"{app_label}.{field_type.__name__}",
                on_delete=models.CASCADE,  # Default to CASCADE
                related_name=related_name,
                **kwargs,
            )

        # Handle Optional[Model] as ForeignKey with null=True
        if origin is Union and type(None) in args:
            # Find the model type in the Union
            model_type = next((arg for arg in args if is_pydantic_model(arg)), None)
            if model_type:
                related_name = sanitize_related_name(
                    getattr(field_info, "related_name", ""),
                    model_name or "",
                    field_name,
                )

                # Ensure null and blank are True for optional relationships
                kwargs["null"] = True
                kwargs["blank"] = True

                return models.ForeignKey(
                    to=f"{app_label}.{model_type.__name__}",
                    on_delete=models.SET_NULL,  # Use SET_NULL for optional relationships
                    related_name=related_name,
                    **kwargs,
                )

        # Not a relationship field
        return None


def get_default_max_length(field_name: str, field_type: type[models.Field]) -> Optional[int]:
    """
    Get the default max_length for a field based on its name and type.

    Args:
        field_name: The name of the field
        field_type: The Django field type

    Returns:
        The default max_length or None if not applicable
    """
    # Default max_length for CharField
    if field_type == models.CharField:
        # Special cases based on field name
        if "email" in field_name.lower():
            return 254  # Standard max length for email fields
        elif "password" in field_name.lower():
            return 128  # Common length for password fields
        elif "phone" in field_name.lower():
            return 20  # Reasonable length for phone numbers
        elif "url" in field_name.lower() or "link" in field_name.lower():
            return 200  # Reasonable length for URLs
        elif "name" in field_name.lower() or "title" in field_name.lower():
            return 100  # Reasonable length for names/titles
        elif "description" in field_name.lower() or "summary" in field_name.lower():
            return 500  # Longer text for descriptions
        elif "code" in field_name.lower() or "id" in field_name.lower():
            return 50  # Reasonable length for codes/IDs
        else:
            return 255  # Default max length

    # No max_length for other field types
    return None


def handle_enum_field(field_type: type[Enum], kwargs: dict[str, Any]) -> models.Field:
    """
    Create a Django field for an Enum type.

    Args:
        field_type: The Enum type
        kwargs: Additional field attributes

    Returns:
        A Django field for the Enum
    """
    # Get all enum values
    enum_values = [item.value for item in field_type]

    # Determine the type of the enum values
    if all(isinstance(val, int) for val in enum_values):
        # Integer enum
        return models.IntegerField(
            choices=[(item.value, item.name) for item in field_type],
            **kwargs,
        )
    elif all(isinstance(val, str) for val in enum_values):
        # String enum
        max_length = max(len(val) for val in enum_values)
        return models.CharField(
            max_length=max_length,
            choices=[(item.value, item.name) for item in field_type],
            **kwargs,
        )
    else:
        # Mixed type enum - use TextField with choices
        return models.TextField(
            choices=[(str(item.value), item.name) for item in field_type],
            **kwargs,
        )


def get_model_fields(model_class: type[BaseModel]) -> dict[str, FieldInfo]:
    """
    Get the fields from a Pydantic model.

    Args:
        model_class: The Pydantic model class

    Returns:
        A dictionary mapping field names to field info
    """
    return model_class.model_fields
