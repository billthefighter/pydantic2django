"""
Field mapping between Pydantic and Django models.
"""
import re
from collections.abc import Callable
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar, Union, cast, get_args, get_origin
from uuid import UUID

from django.core.validators import MaxValueValidator, MinValueValidator, RegexValidator
from django.db import models
from pydantic import BaseModel, EmailStr
from pydantic.config import JsonDict
from pydantic.fields import FieldInfo

# Mapping of Python/Pydantic types to Django field classes
FIELD_TYPE_MAPPING: dict[type[Any], type[models.Field]] = {
    str: models.CharField,
    int: models.IntegerField,
    float: models.FloatField,
    bool: models.BooleanField,
    datetime: models.DateTimeField,
    date: models.DateField,
    time: models.TimeField,
    timedelta: models.DurationField,
    Decimal: models.DecimalField,
    UUID: models.UUIDField,
    EmailStr: models.EmailField,
    bytes: models.BinaryField,
    dict: models.JSONField,
    list: models.JSONField,
    set: models.JSONField,
    Any: models.JSONField,  # Map Any to JSONField
    Path: models.CharField,  # Map Path to CharField
    Union: models.JSONField,  # Map Union to JSONField
    Enum: models.CharField,  # Map Enum to CharField
    type: models.CharField,  # Map Type to CharField
    Protocol: models.JSONField,  # Map Protocol to JSONField
    Callable: models.JSONField,  # Map Callable to JSONField
}

# Default max_lengths for certain field types
DEFAULT_MAX_LENGTHS = {
    models.CharField: 255,
    models.EmailField: 254,  # RFC 5321
    "error": 1000,  # For error messages
    "description": 1000,  # For descriptions
    "api_key": 512,  # For API keys
    "git_commit": 40,  # For git commit hashes
    "response_schema": 2000,  # For response schemas
    "parent_task_id": 255,  # For task IDs
    "name": 255,  # For names
    "family": 255,  # For family names
    "format": 100,  # For format strings
    "content_type": 100,  # For content types
    "path": 1000,  # For file paths
    "file_path": 1000,  # For file paths
    "file_name": 255,  # For file names
    "function_name": 255,  # For function names
    "model": 255,  # For model names
    "provider": 255,  # For provider names
    "type": 100,  # For type names
    "status": 50,  # For status values
    "author": 255,  # For author names
    "change_type": 50,  # For change types
    "number": 50,  # For version numbers
    "min_api_version": 50,  # For API versions
    "recommended_replacement": 255,  # For model replacement names
    "default_agent_type": 100,  # For agent types
    "reminder_template": 1000,  # For reminder templates
    "capabilities_detector": 255,  # For capability detector names
    "api_base": 1000,  # For API base URLs
    "session_id": 255,  # For session IDs
    "storage_path": 1000,  # For storage paths
    "base_path": 1000,  # For base paths
    "system_prompt": 4000,  # For system prompts
    "user_prompt": 4000,  # For user prompts
    "notes": 1000,  # For notes fields
    "tax_code": 50,  # For tax codes
    "google_api_key": 512,  # For error messages
}


def safe_str_tuple(value: Any) -> tuple[str, str]:
    """Convert a key-value pair to a string tuple safely."""
    if isinstance(value, list | tuple) and len(value) == 2:
        return str(value[0]), str(value[1])
    return str(value), str(value)


def get_field_kwargs(
    field_name: str, field_info: Any, metadata: Union[dict[str, Any], JsonDict, Callable[..., Any], None]
) -> dict[str, Any]:
    """Get kwargs for creating a Django field from a Pydantic field."""
    kwargs = {}

    # Convert metadata to dict if it's not None and not a callable
    meta_dict = dict(metadata) if metadata is not None and not callable(metadata) else {}

    # Handle nullability
    is_primary_key = meta_dict.get("primary_key", False) or field_name == "id"
    kwargs["null"] = False if is_primary_key else meta_dict.get("nullable", True)
    kwargs["blank"] = False if is_primary_key else meta_dict.get("nullable", True)

    # Handle primary key
    if is_primary_key:
        kwargs["primary_key"] = True

    # Handle relationship fields
    if meta_dict.get("is_relation", False):
        kwargs["on_delete"] = models.CASCADE  # Only add on_delete for relationship fields
        if meta_dict.get("related_name"):
            kwargs["related_name"] = meta_dict["related_name"]
        if meta_dict.get("through"):
            kwargs["through"] = meta_dict["through"]
        if meta_dict.get("through_fields"):
            kwargs["through_fields"] = meta_dict["through_fields"]
        if meta_dict.get("symmetrical") is not None:
            kwargs["symmetrical"] = meta_dict["symmetrical"]

    # Handle indexes
    if meta_dict.get("db_index"):
        kwargs["db_index"] = True
    if meta_dict.get("unique"):
        kwargs["unique"] = True

    # Handle choices
    if meta_dict.get("choices"):
        try:
            choices = []
            for choice in meta_dict["choices"]:
                if isinstance(choice, list | tuple) and len(choice) == 2:
                    choices.append((str(choice[0]), str(choice[1])))
            if choices:
                kwargs["choices"] = choices
        except (TypeError, IndexError):
            pass  # Skip invalid choices

    # Handle RangeConfig fields
    if hasattr(field_info, "annotation") and str(field_info.annotation).endswith("RangeConfig"):
        # Convert RangeConfig to JSONField
        kwargs["default"] = dict
        return kwargs

    # Handle validation and max_length for string fields
    if meta_dict.get("max_length"):
        kwargs["max_length"] = meta_dict["max_length"]
    elif (
        field_info.annotation == str
        or (hasattr(field_info, "annotation") and str(field_info.annotation).endswith("str"))
        or (hasattr(field_info, "annotation") and "Path" in str(field_info.annotation))
    ):
        # Use field-specific max_length if available, otherwise use default
        kwargs["max_length"] = DEFAULT_MAX_LENGTHS.get(
            field_name,
            DEFAULT_MAX_LENGTHS.get(
                "path"
                if "path" in field_name.lower()
                else "storage_path"
                if "storage" in field_name.lower()
                else models.CharField,
                DEFAULT_MAX_LENGTHS[models.CharField],
            ),
        )

    if meta_dict.get("min_value") is not None:
        kwargs["validators"] = kwargs.get("validators", []) + [MinValueValidator(meta_dict["min_value"])]
    if meta_dict.get("max_value") is not None:
        kwargs["validators"] = kwargs.get("validators", []) + [MaxValueValidator(meta_dict["max_value"])]
    if meta_dict.get("regex"):
        kwargs["validators"] = kwargs.get("validators", []) + [RegexValidator(meta_dict["regex"])]

    return kwargs


def is_pydantic_model(type_: Any) -> bool:
    """Check if a type is a Pydantic model."""
    try:
        return isinstance(type_, type) and issubclass(type_, BaseModel)
    except TypeError:
        return False


def resolve_field_type(field_type: Any) -> tuple[type[Any], bool]:
    """
    Resolve the actual type from Optional/Union/List/Set types.

    Args:
        field_type: The type to resolve

    Returns:
        Tuple of (resolved_type, is_collection)
    """
    is_collection = False
    origin = get_origin(field_type)

    # Handle generic types
    if hasattr(field_type, "__origin__"):
        # For generic types like List[T], Dict[K, V], etc.
        origin = field_type.__origin__
        args = field_type.__args__

        # Handle TypeVar and generic parameters
        if any(hasattr(arg, "__bound__") or str(arg).startswith("~") for arg in args):
            # For generic type parameters, use Any
            return Any, False

        # Handle List/Set with generic type parameter
        if origin in (list, list, set, set):
            if len(args) == 1:
                field_type = args[0]
                is_collection = True
        elif origin in (dict, dict):
            return dict, False
        # Handle other generic types (including Generic base classes)
        elif origin is Generic or hasattr(origin, "__parameters__"):
            return Any, False

    # Handle Union types
    elif origin is Union:
        args = get_args(field_type)
        if len(args) == 2 and type(None) in args:
            # Handle Optional types
            field_type = next(arg for arg in args if arg is not type(None))
            origin = get_origin(field_type)
        else:
            # For complex Union types, use JSONField
            return Any, False

    # Handle basic collection types
    elif origin in (list, list, set, set):
        args = get_args(field_type)
        if len(args) == 1:
            field_type = args[0]
            is_collection = True

    elif origin in (dict, dict):
        return dict, False

    # Handle Protocol types
    if hasattr(field_type, "__protocol__") or (origin and hasattr(origin, "__protocol__")):
        return Protocol, False

    # Handle Callable types
    if origin is Callable or (callable(field_type) and not isinstance(field_type, type)):
        return Callable, False

    # Handle TypeVar
    if isinstance(field_type, TypeVar):
        return Any, False

    return cast(type[Any], field_type), is_collection


def sanitize_related_name(name: str, model_name: str = "", field_name: str = "") -> str:
    """
    Convert a string into a valid Python identifier for use as a related_name.
    Ensures uniqueness by incorporating model and field names.

    Args:
        name: The string to convert
        model_name: Name of the model containing the field
        field_name: Name of the field

    Returns:
        A valid Python identifier
    """
    # Start with model name and field name to ensure uniqueness
    prefix = ""
    if model_name:
        prefix += f"{model_name.lower()}_"
    if field_name:
        prefix += f"{field_name.lower()}_"

    # Handle empty or None name
    if not name:
        name = "related"

    # Replace invalid characters with underscores
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)

    # Remove consecutive underscores
    name = re.sub(r"_+", "_", name)

    # Ensure it starts with a letter or underscore
    if not name or (not name[0].isalpha() and name[0] != "_"):
        name = f"_{name}"

    # Combine prefix and name
    result = f"{prefix}{name}".lower()

    # Remove any leading or trailing underscores
    result = result.strip("_")

    # If result is empty after all processing, use a default
    if not result:
        result = f"{prefix}related"

    # Ensure the name doesn't exceed Django's field name length limit (63 characters)
    if len(result) > 63:
        # If too long, use a hash of the full name to ensure uniqueness
        import hashlib

        hash_suffix = hashlib.md5(result.encode()).hexdigest()[:8]
        result = f"{result[:54]}_{hash_suffix}"

    return result


def get_relationship_field(field_name: str, field_info: FieldInfo, field_type: type[BaseModel]) -> models.Field:
    """
    Create a relationship field based on the field type and metadata.

    Args:
        field_name: Name of the field
        field_info: Pydantic field information
        field_type: The field type (should be a Pydantic model)

    Returns:
        Django relationship field
    """
    kwargs = get_field_kwargs(field_name, field_info, field_info.json_schema_extra or {})
    metadata = field_info.json_schema_extra or {}

    # Handle self-referential relationships
    if metadata.get("self", False):
        to_model = "self"
        model_name = "self"
    else:
        # Convert model name to Django model name
        model_name = field_type.__name__
        if not model_name.startswith("Django"):
            model_name = f"Django{model_name}"
        to_model = f"django_llm.{model_name}"

    # Generate a unique related_name if not explicitly provided
    if not kwargs.get("related_name"):
        base_name = "%(class)s"  # Use %(class)s for model inheritance support

        # Add type parameters for generic types
        type_params = []
        if hasattr(field_type, "__origin__"):
            args = getattr(field_type, "__args__", [])
            for arg in args:
                try:
                    if hasattr(arg, "__name__"):
                        type_params.append(arg.__name__)
                    elif hasattr(arg, "_name"):
                        type_params.append(arg._name)
                    else:
                        # Handle special cases like TypeVar, Any, etc.
                        arg_str = str(arg).replace("typing.", "")
                        # Remove angle brackets and their contents
                        arg_str = re.sub(r"\[.*?\]", "", arg_str)
                        # Remove any remaining special characters
                        arg_str = re.sub(r"[^a-zA-Z0-9_]", "_", arg_str)
                        type_params.append(arg_str)
                except (AttributeError, TypeError):
                    continue

        # Create a unique suffix based on field type and parameters
        suffix = "_".join(filter(None, type_params)) if type_params else ""

        # Generate the related name
        related_name = sanitize_related_name(name=suffix, model_name=base_name, field_name=field_name)

        kwargs["related_name"] = related_name

    # Check if it's a collection type
    _, is_collection = resolve_field_type(field_info.annotation)
    if is_collection:
        # It's a many-to-many relationship
        return models.ManyToManyField(
            to_model,
            through=kwargs.pop("through", None),
            through_fields=kwargs.pop("through_fields", None),
            symmetrical=kwargs.pop("symmetrical", None) if to_model == "self" else None,
            **kwargs,
        )

    # Check for one-to-one relationship hint in metadata
    if metadata.get("one_to_one", False):
        return models.OneToOneField(to_model, on_delete=models.CASCADE, **kwargs)

    # Default to ForeignKey (many-to-one) relationship
    return models.ForeignKey(to_model, on_delete=models.CASCADE, **kwargs)


def get_field_type(field_info: Any) -> tuple[type[models.Field], bool]:
    """
    Get the Django field type for a Pydantic field.

    Args:
        field_info: Pydantic field information

    Returns:
        Tuple of (field_type, is_collection)
    """
    field_type, is_collection = resolve_field_type(field_info.annotation)

    # Handle special cases
    if str(field_type).startswith("typing.Any"):
        return models.JSONField, False
    elif str(field_type).startswith("typing.Union"):
        return models.JSONField, False
    elif str(field_type).startswith("typing.Dict"):
        return models.JSONField, False
    elif str(field_type).startswith("typing.List"):
        return models.JSONField, False
    elif str(field_type).startswith("typing.Set"):
        return models.JSONField, False
    elif str(field_type).startswith("typing.Type"):
        return models.CharField, False
    elif str(field_type).startswith("typing.TypeVar"):
        return models.JSONField, False
    elif str(field_type).startswith("typing.Generic"):
        return models.JSONField, False
    elif str(field_type).endswith("Path"):
        return models.CharField, False
    elif str(field_type).endswith("Config"):
        return models.JSONField, False
    elif str(field_type).endswith("Registry"):
        return models.JSONField, False
    elif str(field_type).endswith("Manager"):
        return models.JSONField, False
    elif str(field_type).endswith("Storage"):
        return models.JSONField, False
    elif str(field_type).endswith("Loader"):
        return models.JSONField, False
    elif str(field_type).endswith("Response"):
        return models.JSONField, False
    elif str(field_type).endswith("Metrics"):
        return models.JSONField, False
    elif str(field_type).endswith("Usage"):
        return models.JSONField, False
    elif str(field_type).endswith("Format"):
        return models.JSONField, False
    elif str(field_type).endswith("Metadata"):
        return models.JSONField, False
    elif str(field_type).endswith("Info"):
        return models.JSONField, False
    elif str(field_type).endswith("Attachment"):
        return models.JSONField, False
    elif "Enum" in str(field_type):
        return models.CharField, False

    # Use standard mapping
    if field_type in FIELD_TYPE_MAPPING:
        return FIELD_TYPE_MAPPING[field_type], is_collection

    # Default to JSONField for complex types
    return models.JSONField, False


def get_django_field(field_name: str, field_info: FieldInfo, skip_relationships: bool = False) -> models.Field:
    """
    Create a Django field from a Pydantic field.

    Args:
        field_name: Name of the field
        field_info: Pydantic field information
        skip_relationships: Whether to skip relationship fields

    Returns:
        Django field instance

    Raises:
        ValueError: If field type cannot be mapped
    """
    field_type, is_collection = resolve_field_type(field_info.annotation)

    # Handle special types first
    if field_type in (Protocol, Callable):
        return models.JSONField(null=True, blank=True, help_text=f"Serialized {field_type.__name__} object")

    # Handle Pydantic models (relationships)
    if is_pydantic_model(field_type):
        if skip_relationships:
            # Return a placeholder field for relationships when skipping
            return models.JSONField(
                null=True, blank=True, help_text=f"Placeholder for {field_type.__name__} relationship"
            )
        return get_relationship_field(field_name, field_info, field_type)

    # Get the Django field class
    django_field_class = FIELD_TYPE_MAPPING.get(field_type)
    if not django_field_class:
        # If no direct mapping exists, fall back to JSONField
        django_field_class = models.JSONField

    # Get field kwargs
    kwargs = get_field_kwargs(field_name, field_info, field_info.json_schema_extra)

    # Handle collection types
    if is_collection and not issubclass(django_field_class, models.JSONField):
        return models.JSONField(**kwargs)

    # Create and return the field
    return django_field_class(**kwargs)
