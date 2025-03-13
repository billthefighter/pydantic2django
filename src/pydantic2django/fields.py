"""
Field mapping between Pydantic and Django models.
"""
import logging
from typing import Any, Optional, Union, get_args, get_origin

from django.db import models
from pydantic.fields import FieldInfo

# Import shared utilities
from pydantic2django.field_utils import (
    FieldAttributeHandler,
    RelationshipFieldHandler,
    is_serializable_type,
    process_default_value,
)

from .field_type_mapping import get_django_field_type, get_field_kwargs

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

        # Check if the type is serializable
        if is_serializable_type(field_type):
            return models.JSONField, False
        else:
            # For non-serializable types, use a special field that indicates context is required
            return (
                models.TextField,
                True,
            )  # Using True to indicate this is a context field

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

        # Get the field type and whether it's a relationship
        field_type = field_info.annotation
        field_class, is_context = self._resolve_field_type(field_type)

        # Create field kwargs
        field_kwargs = FieldAttributeHandler.get_field_kwargs(field_info)

        # Create the field instance
        field = field_class(**field_kwargs)

        # For non-serializable fields, set the is_relationship attribute
        # We use is_relationship to indicate this is a context field that needs
        # to be provided when converting back to Pydantic
        if is_context:
            field.is_relationship = True

        return field


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
        return converter.convert_field(field_name, field_info)
    except Exception as e:
        logger.error(f"Error converting field {field_name}: {str(e)}")
        return models.TextField(null=True, blank=True)


def _resolve_field_type(
    field_type: type[Any],
    field_info: Optional[dict[str, Any]] = None,
) -> type[models.Field]:
    """
    Resolve a Python type to a Django field type.

    Args:
        field_type: The Python type to resolve
        field_info: Optional field information from Pydantic

    Returns:
        The resolved Django field type
    """
    # Handle Optional types
    origin = get_origin(field_type)
    if origin is Union:
        args = get_args(field_type)
        # Check if it's an Optional type (Union with NoneType)
        if len(args) == 2 and type(None) in args:
            # Get the non-None type
            field_type = next(arg for arg in args if arg is not type(None))
            if field_info is None:
                field_info = {}
            field_info["nullable"] = True

    return get_django_field_type(field_type, field_info)


def create_django_field(
    field_name: str,
    field_type: type[Any],
    field_info: Optional[FieldInfo] = None,
) -> models.Field:
    """
    Create a Django model field from a Pydantic field.

    Args:
        field_name: The name of the field
        field_type: The Python type of the field
        field_info: Optional Pydantic field information

    Returns:
        A Django model field
    """
    # Convert FieldInfo to dict if present
    field_info_dict: dict[str, Any] = {}
    if field_info:
        # Extract relevant attributes from FieldInfo
        field_info_dict = {
            "nullable": getattr(field_info, "default", None) is None,
            "allow_blank": getattr(field_info, "allow_blank", False),
            "max_length": getattr(field_info, "max_length", None),
            "relationship_type": getattr(field_info, "relationship_type", None),
            "related_name": getattr(field_info, "related_name", None),
            "through": getattr(field_info, "through", None),
        }

    # Get the Django field type
    django_field_type = _resolve_field_type(field_type, field_info_dict)

    # Get field kwargs
    kwargs = get_field_kwargs(field_type, field_info_dict)

    # Create and return the field
    return django_field_type(**kwargs)
