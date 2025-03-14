"""
Field mapping between Pydantic and Django models.
"""
import logging
from typing import Optional

from django.db import models
from pydantic.fields import FieldInfo

from pydantic2django.field_type_resolver import FieldTypeResolver

# Import shared utilities
from pydantic2django.field_utils import (
    FieldAttributeHandler,
    RelationshipFieldHandler,
)

# Configure logging
logger = logging.getLogger(__name__)


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

    def convert_field(self, field_name: str, field_info: FieldInfo) -> models.Field:
        """
        Convert a Pydantic field to a Django model field.

        Args:
            field_name: The name of the field
            field_info: The Pydantic field info

        Returns:
            A Django model field instance
        """
        # Handle potential ID field naming conflicts
        id_field = handle_id_field(field_name, field_info)
        if id_field:
            return id_field

        # Get the field type and attributes from the resolver
        field_type = field_info.annotation
        field_class, field_kwargs = FieldTypeResolver.resolve_field_type(field_type)

        # Create field kwargs
        field_kwargs.update(FieldAttributeHandler.get_field_kwargs(field_info))

        # Create the field instance
        return field_class(**field_kwargs)


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
    # Handle potential ID field naming conflicts
    id_field = handle_id_field(field_name, field_info)
    if id_field:
        return id_field

    # Get field type from annotation
    field_type = field_info.annotation

    # Get the field type and attributes from the resolver
    field_class, field_kwargs = FieldTypeResolver.resolve_field_type(field_type)

    # Check if it's a relationship field
    if FieldTypeResolver.is_relationship_field(field_type):
        if skip_relationships:
            return None
        return RelationshipFieldHandler.create_relationship_field(
            field_name, field_info, field_type, app_label, model_name
        )

    # For non-relationship fields, merge field attributes
    kwargs = FieldAttributeHandler.get_field_kwargs(field_info)
    kwargs.update(field_kwargs)

    # Create and return the field
    return field_class(**kwargs)
