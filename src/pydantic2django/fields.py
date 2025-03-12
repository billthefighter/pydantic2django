"""
Field mapping between Pydantic and Django models.
"""
import logging
from typing import Any, Optional, Union, cast, get_args, get_origin

from django.db import models
from pydantic import BaseModel
from pydantic.fields import FieldInfo

# Import shared utilities
from pydantic2django.field_utils import (
    RelationshipFieldHandler,
    get_default_max_length,
    is_pydantic_model,
    sanitize_related_name,
)

# Configure logging
logger = logging.getLogger(__name__)


def get_field_attributes(field_info: FieldInfo, extra: Any = None) -> dict[str, Any]:
    """
    Extract field attributes from a Pydantic field.

    Args:
        field_info: The Pydantic field info
        extra: Optional extra attributes or callable to get extra attributes

    Returns:
        Dictionary of field attributes
    """
    kwargs = {}

    # Handle null/blank based on whether the field is optional
    is_optional = field_info.is_required is False
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

    # Process extra attributes
    if extra:
        if callable(extra):
            extra_kwargs = extra(field_info)
            kwargs.update(extra_kwargs)
        else:
            kwargs.update(extra)

    return kwargs


def handle_id_field(field_name: str, field_info: FieldInfo) -> Optional[models.Field]:
    """
    Handle potential ID field naming conflicts with Django's automatic primary key.

    Args:
        field_name: The original field name
        field_info: The Pydantic field info

    Returns:
        A Django field instance configured as a primary key
    """
    # Check if this is an ID field (case insensitive)
    if field_name.lower() == "id":
        field_type = field_info.annotation

        # Determine the field type based on the annotation
        if field_type is int:
            field_class = models.AutoField
        elif field_type is str:
            field_class = models.CharField
        else:
            # Default to AutoField for other types
            field_class = models.AutoField

        # Create field kwargs
        field_kwargs = {
            "primary_key": True,
            "verbose_name": f"Custom {field_name} (used as primary key)",
        }

        # Add max_length for CharField
        if field_class is models.CharField:
            field_kwargs["max_length"] = 255

        # Create and return the field
        return field_class(**field_kwargs)

    return None


class FieldConverter:
    """
    Converts Pydantic fields to Django model fields.
    """

    def __init__(self, app_label: str = "django_llm"):
        """
        Initialize the field converter.

        Args:
            app_label: The Django app label to use for model registration
        """
        self.app_label = app_label

    def _resolve_field_type(self, field_type: Any) -> tuple[type[models.Field], bool]:
        """
        Resolve the Django field type for a given Pydantic field type.

        Args:
            field_type: The Pydantic field type

        Returns:
            Tuple of (Django field class, is_relationship)
        """
        # Handle Optional types
        origin = get_origin(field_type)
        args = get_args(field_type)

        if origin is Union and type(None) in args:
            # This is an Optional type, get the actual type
            field_type = next(arg for arg in args if arg is not type(None))

        # Check for List/Dict types
        if origin is list:
            # Check if it's a list of Pydantic models
            if args and is_pydantic_model(args[0]):
                return models.ManyToManyField, True
            return models.JSONField, False
        elif origin is dict:
            return models.JSONField, False

        # Handle basic types
        if field_type is str:
            return models.CharField, False
        elif field_type is int:
            return models.IntegerField, False
        elif field_type is float:
            return models.FloatField, False
        elif field_type is bool:
            return models.BooleanField, False
        elif field_type is dict:
            return models.JSONField, False
        elif field_type is list:
            return models.JSONField, False

        # Handle relationship fields (Pydantic models)
        if is_pydantic_model(field_type):
            return models.ForeignKey, True

        # Default to TextField for unknown types
        return models.TextField, False

    def convert_field(
        self,
        field_name: str,
        field_info: FieldInfo,
        skip_relationships: bool = False,
    ) -> Optional[models.Field]:
        """
        Convert a Pydantic field to a Django model field.

        Args:
            field_name: The name of the field
            field_info: The Pydantic field info
            skip_relationships: Whether to skip relationship fields

        Returns:
            A Django model field or None if the field should be skipped
        """
        # Handle special case for id field
        if field_name == "id":
            id_field = handle_id_field(field_name, field_info)
            if id_field is not None:
                return id_field

        # Get field type from annotation
        field_type = field_info.annotation

        # Resolve field type
        django_field_type, is_relationship = self._resolve_field_type(field_type)

        # Skip relationship fields if requested
        if skip_relationships and is_relationship:
            return None

        # Handle relationship fields
        if is_relationship:
            result = RelationshipFieldHandler.create_relationship_field(
                field_name, field_info, field_type, self.app_label
            )
            return result or models.JSONField(null=True, blank=True)

        # Get field attributes
        field_attrs = get_field_attributes(field_info)

        # Handle specific field types
        if django_field_type is models.CharField and "max_length" not in field_attrs:
            field_attrs["max_length"] = get_default_max_length(field_name, django_field_type)

        # Create the field
        try:
            return django_field_type(**field_attrs)
        except Exception as e:
            logger.error(f"Error creating field {field_name}: {str(e)}")
            return models.TextField(null=True, blank=True)


def convert_field(
    field_name: str,
    field_info: FieldInfo,
    skip_relationships: bool = False,
    app_label: str = "django_llm",
    model_name: Optional[str] = None,
) -> Optional[models.Field]:
    """
    Convert a Pydantic field to a Django field.
    This is the main entry point for field conversion.

    Args:
        field_name: The name of the field
        field_info: The Pydantic field info
        skip_relationships: Whether to skip relationship fields
        app_label: The app label to use for model registration
        model_name: The name of the model to reference (for relationships)

    Returns:
        A Django field instance or None if the field should be skipped
    """
    # Use the FieldConverter for all field conversion
    converter = FieldConverter(app_label)

    # Get field type from annotation
    field_type = field_info.annotation

    # Handle special case for direct model relationships when model_name is provided
    if model_name is not None:
        # Check for list of models (many-to-many)
        origin = get_origin(field_type)
        args = get_args(field_type)

        # Handle list of models (many-to-many)
        if origin is list and args:
            first_arg = args[0]
            if first_arg is not None and is_pydantic_model(first_arg):
                # At this point, we know first_arg is a Pydantic model class
                model_class = cast(type[BaseModel], first_arg)
                if skip_relationships:
                    return None

                # Get the related model name
                related_model_name = model_class.__name__
                if not related_model_name.startswith("Django"):
                    related_model_name = f"Django{related_model_name}"

                # Create a many-to-many field
                related_name = sanitize_related_name(
                    field_name,
                    model_name=model_name if model_name else "",
                    field_name=field_name,
                )
                return models.ManyToManyField(
                    f"{app_label}.{related_model_name}",
                    related_name=related_name,
                    blank=True,
                )

        # Handle direct model relationship
        if is_pydantic_model(field_type):
            # At this point, we know field_type is a Pydantic model class
            model_class = cast(type[BaseModel], field_type)
            if skip_relationships:
                return None

            # Get the related model name
            related_model_name = model_class.__name__
            if not related_model_name.startswith("Django"):
                related_model_name = f"Django{related_model_name}"

            # Create a foreign key
            related_name = sanitize_related_name(
                field_name,
                model_name=model_name if model_name else "",
                field_name=field_name,
            )
            return models.ForeignKey(
                f"{app_label}.{related_model_name}",
                on_delete=models.CASCADE,
                related_name=related_name,
                null=True,
                blank=True,
            )

    # Use the converter for standard field conversion
    try:
        return converter.convert_field(field_name, field_info, skip_relationships)
    except Exception as e:
        logger.error(f"Error converting field {field_name}: {str(e)}")
        return models.TextField(null=True, blank=True)
