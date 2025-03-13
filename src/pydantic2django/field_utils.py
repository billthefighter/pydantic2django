"""
Shared utilities for field handling across pydantic2django modules.

This module contains common utilities, constants, and type definitions used by
field_type_mapping.py, fields.py, and static_django_model_generator.py to reduce
code duplication and improve maintainability.
"""
import inspect
import logging
import re
from collections.abc import Callable
from enum import Enum
from typing import (
    Any,
    Optional,
    Union,
    get_args,
    get_origin,
)

from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.utils.functional import Promise
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from .serialization import is_serializable

# Configure logging
logger = logging.getLogger("pydantic2django.field_utils")


def is_valid_default_value(value: Any) -> bool:
    """
    Check if a value is a valid default value for a Django field.

    Args:
        value: The value to check

    Returns:
        True if the value is valid as a default, False otherwise
    """
    return value is not None and value != Ellipsis and value != PydanticUndefined and value != models.NOT_PROVIDED


def process_default_value(value: Any) -> Any:
    """
    Process a default value for use in a Django field.

    Args:
        value: The default value to process

    Returns:
        The processed default value, or models.NOT_PROVIDED if the value is invalid
    """
    if not is_valid_default_value(value):
        return models.NOT_PROVIDED
    return value


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

        # Handle default value using the standardized function
        # Only include default if it was explicitly set (not models.NOT_PROVIDED)
        if hasattr(field, "default") and field.default != models.NOT_PROVIDED:
            # Check if this is a Django-provided default or one we explicitly set
            if not isinstance(field, (models.AutoField | models.BigAutoField)):
                default_value = process_default_value(field.default)
                if default_value is not models.NOT_PROVIDED:
                    if isinstance(default_value, str):
                        params.append(f"default='{sanitize_string(default_value)}'")
                    else:
                        params.append(f"default={default_value}")

        # Field-specific parameters
        if isinstance(field, models.CharField) and hasattr(field, "max_length"):
            params.append(f"max_length={field.max_length}")

        if isinstance(field, models.DecimalField):
            if hasattr(field, "max_digits"):
                params.append(f"max_digits={field.max_digits}")
            if hasattr(field, "decimal_places"):
                params.append(f"decimal_places={field.decimal_places}")

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
        params = FieldAttributeHandler.serialize_field_attributes(field)

        # Handle relationship fields
        if isinstance(field, models.ForeignKey):
            # Get the related model name safely
            related_model_name = RelationshipFieldHandler.get_related_model_name(field)
            if related_model_name:
                # Use direct class reference for type checking
                if "." in related_model_name:
                    # If it's a cross-app reference, keep it as a string
                    params.append(f"to='{related_model_name}'")
                else:
                    # Direct class reference for same-app models
                    params.append(f"to={related_model_name}")
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

        if isinstance(field, models.ManyToManyField):
            # Get the related model name safely
            related_model_name = RelationshipFieldHandler.get_related_model_name(field)
            if related_model_name:
                # Use direct class reference for type checking
                if "." in related_model_name:
                    # If it's a cross-app reference, keep it as a string
                    params.append(f"to='{related_model_name}'")
                else:
                    # Direct class reference for same-app models
                    params.append(f"to={related_model_name}")
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
                if not (isinstance(through, str) and through.endswith("_edges") or through.endswith("_nodes")):
                    through_name = through.__name__ if hasattr(through, "__name__") else str(through)
                    if "." in through_name:
                        # If it's a cross-app reference, keep it as a string
                        params.append(f"through='{through_name}'")
                    else:
                        # Direct class reference for same-app models
                        params.append(f"through={through_name}")

        # Handle context fields (non-serializable fields)
        if isinstance(field, models.TextField) and getattr(field, "is_relationship", False):
            params.append("is_relationship=True")

        # Join parameters and return definition
        return f"models.{field_type}({', '.join(params)})"

    @staticmethod
    def get_field_kwargs(field_info: FieldInfo) -> dict[str, Any]:
        """
        Get field kwargs from Pydantic field info.

        Args:
            field_info: The Pydantic field info

        Returns:
            Dict of field kwargs
        """
        kwargs = {}

        # Handle null/blank based on whether the field is required
        kwargs["null"] = not field_info.is_required()
        kwargs["blank"] = not field_info.is_required()

        # Add description as help_text if available
        if field_info.description:
            kwargs["help_text"] = field_info.description

        # Add default if available and valid
        if is_valid_default_value(field_info.default):
            kwargs["default"] = process_default_value(field_info.default)

        # Add max_length for CharField if available in field metadata
        if hasattr(field_info, "metadata") and isinstance(field_info.metadata, dict):
            max_length = field_info.metadata.get("max_length")
            if max_length is not None:
                kwargs["max_length"] = max_length
        elif isinstance(field_info.annotation, type) and field_info.annotation == str:
            kwargs["max_length"] = 255  # Default max_length for strings

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
    """Handles relationship fields between Django models."""

    @staticmethod
    def get_related_model_name(field: models.Field) -> Optional[str]:
        """
        Get the name of the related model from a relationship field.

        Args:
            field: The Django model field

        Returns:
            The name of the related model or None if not found
        """
        try:
            if isinstance(field, (models.ForeignKey | models.ManyToManyField)):
                remote_field = field.remote_field
                if remote_field and remote_field.model:
                    if isinstance(remote_field.model, str):
                        return remote_field.model
                    elif hasattr(remote_field.model, "__name__"):
                        return remote_field.model.__name__
        except Exception:
            pass
        return None

    @staticmethod
    def is_relationship_field(field: models.Field) -> bool:
        """
        Check if a field is a relationship field.

        Args:
            field: The Django model field

        Returns:
            True if the field is a relationship field, False otherwise
        """
        # Check for standard relationship fields
        if isinstance(field, (models.ForeignKey | models.ManyToManyField)):
            return True

        # Check for context fields marked as relationships
        if isinstance(field, models.TextField) and getattr(field, "is_relationship", False):
            return True

        return False

    @staticmethod
    def get_relationship_type(field: models.Field) -> Optional[str]:
        """
        Get the type of relationship for a field.

        Args:
            field: The Django model field

        Returns:
            The type of relationship or None if not a relationship field
        """
        if isinstance(field, models.ForeignKey):
            return "ForeignKey"
        elif isinstance(field, models.ManyToManyField):
            return "ManyToManyField"
        elif isinstance(field, models.TextField) and getattr(field, "is_relationship", False):
            return "ContextField"
        return None

    @staticmethod
    def detect_relationship_type(field_type: Any) -> Optional[type[models.Field]]:
        """
        Detect if a field type represents a relationship and return the appropriate Django field type.

        Args:
            field_type: The field type to check

        Returns:
            Django field type if it's a relationship, None otherwise
        """
        origin = get_origin(field_type)
        args = get_args(field_type)

        # Handle Optional[...]
        if origin is Union and type(None) in args:
            # Get the non-None type
            inner_type = next(arg for arg in args if arg is not type(None))
            # Only recurse if the inner type might be a relationship
            if is_pydantic_model(inner_type) or (
                get_origin(inner_type) in (list, dict) and any(is_pydantic_model(arg) for arg in get_args(inner_type))
            ):
                return RelationshipFieldHandler.detect_relationship_type(inner_type)
            return None

        # Handle List[Model]
        if origin is list and args and is_pydantic_model(args[0]):
            return models.ManyToManyField

        # Handle Dict[str, Model] or Dict[Any, Model]
        if origin is dict and len(args) == 2 and is_pydantic_model(args[1]):
            return models.ManyToManyField

        # Handle direct Model reference
        if is_pydantic_model(field_type):
            return models.ForeignKey

        return None

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
            A Django relationship field or None if the field type is not a relationship
        """
        # First detect if this is a relationship field
        field_class = RelationshipFieldHandler.detect_relationship_type(field_type)
        if not field_class:
            return None

        # Get field attributes
        kwargs = FieldAttributeHandler.handle_field_attributes(field_info)

        # Get the origin and args for type analysis
        origin = get_origin(field_type)
        args = get_args(field_type)

        # Handle Optional types
        if origin is Union and type(None) in args:
            # For Optional fields, ensure they're nullable
            kwargs["null"] = True
            kwargs["blank"] = True
            # Get the actual type (not None)
            field_type = next(arg for arg in args if arg is not type(None))
            # Update origin and args for the actual type
            origin = get_origin(field_type)
            args = get_args(field_type)

        # Get the model class based on the field type
        if origin is list and args:
            model_class = args[0]
        elif origin is dict and len(args) == 2:
            model_class = args[1]
        else:
            model_class = field_type

        # Get the related name
        related_name = sanitize_related_name(
            getattr(field_info, "related_name", ""),
            model_name or "",
            field_name,
        )

        # Create the appropriate field type
        if field_class is models.ManyToManyField:
            return models.ManyToManyField(
                to=f"{app_label}.{RelationshipFieldHandler.get_related_model_name(model_class)}",
                related_name=related_name,
                **kwargs,
            )
        elif field_class is models.ForeignKey:
            # For ForeignKey, we need on_delete
            kwargs["on_delete"] = models.CASCADE  # Default to CASCADE
            return models.ForeignKey(
                to=f"{app_label}.{RelationshipFieldHandler.get_related_model_name(model_class)}",
                related_name=related_name,
                **kwargs,
            )

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
