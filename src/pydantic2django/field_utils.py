"""
Shared utilities for field handling across pydantic2django modules.

This module contains common utilities, constants, and type definitions used by
field_type_mapping.py, fields.py, and static_django_model_generator.py to reduce
code duplication and improve maintainability.
"""
import inspect
import logging
import re
from typing import (
    Any,
    Optional,
    TypeAlias,
    Union,
    get_args,
    get_origin,
)

from django.db import models
from django.utils.functional import Promise
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from pydantic2django.types import is_pydantic_model, is_serializable_type

# Type aliases for relationship field types
RelationshipFieldType: TypeAlias = Optional[
    type[models.ForeignKey] | type[models.ManyToManyField] | type[models.OneToOneField]
]
RelationshipFieldKwargs: TypeAlias = dict[str, Any]
RelationshipFieldDetectionResult: TypeAlias = tuple[RelationshipFieldType, RelationshipFieldKwargs]

# Configure logging
logger = logging.getLogger("pydantic2django.field_utils")


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


class FieldSerializer:
    """
    Handles extraction and processing of field attributes into string form
    from Django model fields.
    """

    @staticmethod
    def serialize_field_attributes(field: models.Field) -> list[str]:
        """
        Serialize Django model field attributes to a list of parameter strings.

        Args:
            field: The Django model field

        Returns:
            List of parameter strings in the format "param=value"
        """
        params = []

        # Common field parameters
        if hasattr(field, "verbose_name") and field.verbose_name:
            params.append(f"verbose_name='{sanitize_string(field.verbose_name)}'")

        if hasattr(field, "help_text") and field.help_text:
            params.append(f"help_text='{sanitize_string(field.help_text)}'")

        if hasattr(field, "null") and field.null:
            params.append(f"null={field.null}")

        if hasattr(field, "blank") and field.blank:
            params.append(f"blank={field.blank}")

        # Handle choices for fields like EnumFields
        if hasattr(field, "choices") and field.choices:
            choices_repr = str(field.choices)
            params.append(f"choices={choices_repr}")

        # Only include default if it was explicitly set (None)
        if hasattr(field, "default") and field.default is not None:
            # Check if this is a Django-provided default or one we explicitly set
            if not isinstance(field, (models.AutoField | models.BigAutoField)):
                # if it's a string, we need to make sure it can be safely converted to a string in our template
                if isinstance(field.default, str):
                    params.append(f"default='{sanitize_string(field.default)}'")
                # if it's not a string, we pass the default value as is
                else:
                    params.append(f"default={field.default}")

        # Field-specific parameters
        if isinstance(field, models.CharField) and hasattr(field, "max_length"):
            params.append(f"max_length={field.max_length}")

        if isinstance(field, models.DecimalField):
            if hasattr(field, "max_digits"):
                params.append(f"max_digits={field.max_digits}")
            if hasattr(field, "decimal_places"):
                params.append(f"decimal_places={field.decimal_places}")

        # TODO: Add support for relationship fields

        return params

    @staticmethod
    def serialize_field(field: models.Field) -> str:
        """
        Serialize a Django model field to its string representation.

        Args:
            field: The Django model field

        Returns:
            String representation of the field
        """
        field_type = type(field).__name__
        params = FieldSerializer.serialize_field_attributes(field)

        # Handle relationship fields
        if isinstance(field, models.ForeignKey):
            # Get the related model name safely
            related_model_name = get_related_model_name(field)
            if related_model_name:
                # Ensure the model name has the app label if needed
                if "." not in related_model_name:
                    # Get app label from the field's model if available
                    app_label = None
                    if hasattr(field, "model") and hasattr(field.model, "_meta"):
                        app_label = field.model._meta.app_label
                    # Use the related field's app label as a fallback
                    elif hasattr(field.remote_field, "model") and hasattr(field.remote_field.model, "_meta"):
                        app_label = field.remote_field.model._meta.app_label

                    if app_label:
                        related_model_name = f"{app_label}.{related_model_name}"

                params.append(f"to='{related_model_name}'")
            else:
                params.append("to='self'")  # Default to self-reference if we can't determine

            # Get the on_delete behavior safely
            try:
                on_delete = getattr(field, "on_delete", None)
                if on_delete and hasattr(on_delete, "__name__"):
                    params.append(f"on_delete=models.{on_delete.__name__}")
                else:
                    params.append("on_delete=models.CASCADE")  # Default to CASCADE
            except Exception:
                params.append("on_delete=models.CASCADE")  # Default to CASCADE

            # Add related_name if present - Access it via remote_field
            related_name = getattr(field.remote_field, "related_name", None)
            if related_name:
                params.append(f"related_name='{sanitize_related_name(related_name)}'")

        if isinstance(field, models.ManyToManyField):
            # Get the related model name safely
            related_model_name = get_related_model_name(field)
            if related_model_name:
                # Ensure the model name has the app label if needed
                if "." not in related_model_name:
                    # Get app label from the field's model if available
                    app_label = None
                    if hasattr(field, "model") and hasattr(field.model, "_meta"):
                        app_label = field.model._meta.app_label
                    # Use the related field's app label as a fallback
                    elif hasattr(field.remote_field, "model") and hasattr(field.remote_field.model, "_meta"):
                        app_label = field.remote_field.model._meta.app_label

                    if app_label:
                        related_model_name = f"{app_label}.{related_model_name}"

                params.append(f"to='{related_model_name}'")
            else:
                raise ValueError(f"Related model not found for {field}")

            # Add related_name if present - this should be sanitized since it's a Python identifier
            related_name = getattr(field, "related_name", None)
            if related_name:
                params.append(f"related_name='{sanitize_related_name(related_name)}'")

            # Add through model if present - use direct class reference
            through = getattr(field, "through", None)
            if through:
                # Skip auto-generated through models
                # Django's auto-created through models are instances of ManyToManyRel
                # or have the auto_created flag set to True
                if not (isinstance(through, models.ManyToManyRel) or getattr(through, "auto_created", False)):
                    # If through is a string, use it directly
                    # Otherwise, try to get the class name or fall back to string representation
                    through_name = (
                        through
                        if isinstance(through, str)
                        else (through.__name__ if hasattr(through, "__name__") else str(through))
                    )
                    if "." in through_name:
                        # Check for and fix duplicate app labels
                        parts = through_name.split(".")
                        if len(parts) >= 2 and parts[0] == parts[1]:
                            # Remove duplicate app label
                            through_name = f"{parts[0]}.{'.'.join(parts[2:])}"
                        params.append(f"through='{through_name}'")
                    else:
                        # Direct class reference for same-app models
                        params.append(f"through={through_name}")

        # Handle context fields (non-serializable fields)
        if isinstance(field, models.TextField) and getattr(field, "is_relationship", False):
            params.append("is_relationship=True")

        # Join parameters and return definition
        final_def = f"models.{field_type}({', '.join(params)})"
        logger.debug(f"SERIALIZE_FIELD: Final definition for {field.name}: {final_def}")
        return final_def


def get_model_fields(model_class: type[BaseModel]) -> dict[str, FieldInfo]:
    """
    Get the fields from a Pydantic model.

    Args:
        model_class: The Pydantic model class

    Returns:
        A dictionary mapping field names to field info
    """
    return model_class.model_fields


def sanitize_string(value: Union[str, Promise, Any]) -> str:
    """
    Sanitize a string for safe inclusion in generated code.

    Escapes special characters like quotes and newlines to prevent syntax errors
    in the generated code.

    Args:
        value: The string to sanitize. Can be a string, Django's Promise object, or any other type

    Returns:
        A sanitized string safe for code generation
    """
    # Convert value to string first
    str_value = str(value)

    # Replace backslashes first to avoid double-escaping
    str_value = str_value.replace("\\", "\\\\")

    # Escape single quotes since we're using them for string literals
    str_value = str_value.replace("'", "\\'")

    # Replace newlines with escaped newlines
    str_value = str_value.replace("\n", "\\n")

    # Replace tabs with escaped tabs
    str_value = str_value.replace("\t", "\\t")

    # Replace carriage returns
    str_value = str_value.replace("\r", "\\r")

    return str_value


def get_relationship_metadata(field_type: Any) -> dict[str, Any]:
    """
    Extract relationship-specific metadata.

    Args:
        field_type: The field type to analyze

    Returns:
        Dictionary of relationship metadata
    """
    metadata = {}

    # Handle Optional types
    origin = get_origin(field_type)
    args = get_args(field_type)
    if origin is Union and type(None) in args:
        field_type = next(arg for arg in args if arg is not type(None))
        metadata["optional"] = True
        origin = get_origin(field_type)
        args = get_args(field_type)

    # Determine relationship type
    if origin is list and args and is_pydantic_model(args[0]):
        metadata["type"] = "many_to_many"
        metadata["model"] = args[0]
    elif is_pydantic_model(field_type):
        metadata["type"] = "foreign_key"
        metadata["model"] = field_type
    elif inspect.isabstract(field_type) or not is_serializable_type(field_type):
        metadata["type"] = "foreign_key"
        metadata["model"] = field_type

    return metadata


def get_related_model_name(field: models.Field) -> Optional[str]:
    """
    Get the name of the related model from a relationship field.

    Args:
        field: The Django model field

    Returns:
        The name of the related model or None if not found
    """
    try:
        if isinstance(
            field,
            (models.ForeignKey | models.ManyToManyField | models.OneToOneField),
        ):
            remote_field = field.remote_field
            if remote_field and remote_field.model:
                if isinstance(remote_field.model, str):
                    return remote_field.model
                elif hasattr(remote_field.model, "__name__"):
                    return remote_field.model.__name__
    except Exception:
        pass
    return None


def is_pydantic_model_field_optional(field_type: Any) -> bool:
    """
    Check if a Pydantic model field is optional.
    """
    return get_origin(field_type) is Union and type(None) in get_args(field_type)


def balanced(s: str) -> bool:
    """
    Check if a string has balanced brackets.
    """
    pairs = {"{": "}", "(": ")", "[": "]"}
    stack = []
    for c in s:
        if c in "{[(":
            stack.append(c)
        elif stack and c == pairs[stack[-1]]:
            stack.pop()
        else:
            return False
    return len(stack) == 0
