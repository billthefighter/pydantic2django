"""
Field mapping between Pydantic and Django models.
"""
import logging
from typing import Any, Optional, Union, get_args, get_origin

from django.db import models
from pydantic.fields import FieldInfo

# Import shared utilities
from pydantic2django.field_utils import (
    RelationshipFieldHandler,
    get_default_max_length,
    process_default_value,
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

    # Handle default value using the standardized function
    default_value = process_default_value(field_info.default)
    if default_value is not models.NOT_PROVIDED:
        kwargs["default"] = default_value

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
        # First check if it's a relationship field
        relationship_field = RelationshipFieldHandler.detect_relationship_type(field_type)
        if relationship_field:
            return relationship_field, True

        # Handle Optional types
        origin = get_origin(field_type)
        args = get_args(field_type)

        if origin is Union and type(None) in args:
            # This is an Optional type, get the actual type
            field_type = next(arg for arg in args if arg is not type(None))
            # Recursively resolve the inner type
            inner_field, is_rel = self._resolve_field_type(field_type)
            return inner_field, is_rel

        # Handle basic types
        if field_type is str:
            return models.CharField, False
        elif field_type is int:
            return models.IntegerField, False
        elif field_type is float:
            return models.FloatField, False
        elif field_type is bool:
            return models.BooleanField, False
        elif field_type is dict or origin is dict:
            return models.JSONField, False
        elif field_type is list or origin is list:
            return models.JSONField, False

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
    # Get field type from annotation
    field_type = field_info.annotation

    # First try to handle it as a relationship field
    if not skip_relationships:
        relationship_field = RelationshipFieldHandler.create_relationship_field(
            field_name=field_name,
            field_info=field_info,
            field_type=field_type,
            app_label=app_label,
            model_name=model_name,
        )
        if relationship_field:
            return relationship_field

    # If it's not a relationship or we're skipping relationships,
    # use the FieldConverter for standard field conversion
    try:
        converter = FieldConverter(app_label)
        return converter.convert_field(field_name, field_info, skip_relationships)
    except Exception as e:
        logger.error(f"Error converting field {field_name}: {str(e)}")
        return models.TextField(null=True, blank=True)
