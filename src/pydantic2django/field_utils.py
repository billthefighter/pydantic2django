"""
Shared utilities for field handling across pydantic2django modules.

This module contains common utilities, constants, and type definitions used by
field_type_mapping.py, fields.py, and static_django_model_generator.py to reduce
code duplication and improve maintainability.
"""
import inspect
import logging
import re
from dataclasses import dataclass, field
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


class FieldAttributeHandler:
    """
    Handles extraction and processing of field attributes into string form
    from Pydantic field info.
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


@dataclass
class RelationshipFieldHandler:
    """Handles relationship fields between Django models."""

    available_relationships: dict[str, type[BaseModel]] = field(default_factory=dict)

    @staticmethod
    def detect_field_type(
        field_type: Any,
    ) -> RelationshipFieldDetectionResult:
        """
        Single entry point for relationship type detection.

        Args:
            field_type: The field type to check

        Returns:
            Tuple of (Django field class, field attributes) or (None, {}) if not a relationship
        """
        # Handle Optional types
        origin = get_origin(field_type)
        args = get_args(field_type)
        field_kwargs = {}

        if origin is Union and type(None) in args:
            field_type = next(arg for arg in args if arg is not type(None))
            field_kwargs["null"] = True
            field_kwargs["blank"] = True
            origin = get_origin(field_type)
            args = get_args(field_type)

        # Detect pydantic models
        if is_pydantic_model(field_type):
            # If field type is in available relationships, it should be converted into a relationship
            if field_type in RelationshipFieldHandler.available_relationships.values():
                # Direct Pydantic model reference (ForeignKey)
                if is_pydantic_model(field_type):
                    field_kwargs["on_delete"] = models.CASCADE
                    return models.ForeignKey, field_kwargs

                # List of Pydantic models (ManyToManyField)
                if origin is list and args and is_pydantic_model(args[0]):
                    return models.ManyToManyField, field_kwargs

                # Abstract base class or non-serializable type (ForeignKey)
                if inspect.isabstract(field_type) or not is_serializable_type(field_type):
                    field_kwargs["on_delete"] = models.CASCADE
                    return models.ForeignKey, field_kwargs
            else:
                # Field needs to be  added to context, return none
                pass

        return None, {"context_field": True}

    @staticmethod
    def create_field(
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
        # Get the field type and attributes
        field_class, field_kwargs = RelationshipFieldHandler.detect_field_type(field_type)
        if not field_class:
            return None

        # Get field attributes
        kwargs = FieldAttributeHandler.handle_field_attributes(field_info)
        kwargs.update(field_kwargs)

        # Get the model class based on the field type
        origin = get_origin(field_type)
        args = get_args(field_type)
        if origin is list and args:
            model_class = args[0]
        elif origin is dict and len(args) == 2:
            model_class = args[1]
        else:
            model_class = field_type

        # Get the model name, handling both string and class references
        if isinstance(model_class, str):
            target_model_name = model_class
        elif inspect.isclass(model_class):
            target_model_name = model_class.__name__
            if not target_model_name.startswith("Django"):
                target_model_name = f"Django{target_model_name}"
        else:
            logger.warning(f"Invalid model class type for field {field_name}: {type(model_class)}")
            return None

        # Get the related name
        related_name = sanitize_related_name(
            getattr(field_info, "related_name", ""),
            model_name or "",
            field_name,
        )

        # Create the appropriate field type
        if field_class is models.ManyToManyField:
            return models.ManyToManyField(
                to=f"{app_label}.{target_model_name}",
                related_name=related_name,
                **kwargs,
            )
        else:  # ForeignKey or OneToOneField
            # Ensure field_class is a valid relationship field type that accepts 'to' and 'related_name'
            if field_class in (models.ForeignKey, models.OneToOneField):
                return field_class(
                    to=f"{app_label}.{target_model_name}",
                    related_name=related_name,
                    **kwargs,
                )
            else:
                logger.warning(f"Unexpected field class {field_class} for relationship field {field_name}")
                return None

    @staticmethod
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

    @staticmethod
    def is_relationship_field(field_type: Any) -> bool:
        """
        Check if a field type represents a relationship.

        Args:
            field_type: The type to check

        Returns:
            True if the field type represents a relationship, False otherwise
        """
        field_class, _ = RelationshipFieldHandler.detect_field_type(field_type)
        return field_class is not None

    @staticmethod
    def get_relationship_type(field_type: Any) -> RelationshipFieldType:
        """
        Get the specific type of relationship field.

        Args:
            field_type: The type to check

        Returns:
            The Django field class for the relationship, or None if not a relationship
        """
        field_class, _ = RelationshipFieldHandler.detect_field_type(field_type)
        return field_class

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
