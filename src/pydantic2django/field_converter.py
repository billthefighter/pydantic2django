"""
Field mapping between Pydantic and Django models.
"""
import logging
from typing import Optional, Union, get_args, get_origin

from django.db import models
from pydantic.fields import FieldInfo

from pydantic2django.field_type_mapping import TypeMapper
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

    Raises:
        ValueError: If the field type cannot be mapped to a Django field
    """
    # Handle potential ID field naming conflicts
    id_field = handle_id_field(field_name, field_info)
    if id_field:
        return id_field

    # Get field type from annotation
    field_type = field_info.annotation
    is_optional = False

    # Handle Optional types
    origin = get_origin(field_type)
    if origin is Union:
        args = get_args(field_type)
        if len(args) == 2 and type(None) in args:
            # This is an Optional type
            field_type = next(arg for arg in args if arg is not type(None))
            is_optional = True

    # Get the field mapping from TypeMapper
    mapping = TypeMapper.get_mapping_for_type(field_type)
    if not mapping:
        raise ValueError(f"Could not map field type {field_type} to a Django field")

    # If it's a relationship field and we're skipping relationships, return None
    if mapping.is_relationship and skip_relationships:
        return None

    # Get field attributes from both TypeMapper and FieldAttributeHandler
    kwargs = TypeMapper.get_field_attributes(field_type)

    # Add field attributes from field_info
    field_attrs = FieldAttributeHandler.handle_field_attributes(field_info)
    kwargs.update(field_attrs)

    # For Optional types, set null and blank to True
    if is_optional:
        kwargs["null"] = True
        kwargs["blank"] = True

    # For relationship fields, use RelationshipFieldHandler
    if mapping.is_relationship:
        return RelationshipFieldHandler.create_field(
            field_name=field_name,
            field_info=field_info,
            field_type=field_type,
            app_label=app_label,
            model_name=model_name,
        )

    # Create and return the field
    return mapping.django_field(**kwargs)
