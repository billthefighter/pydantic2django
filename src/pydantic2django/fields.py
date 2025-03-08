"""
Field mapping between Pydantic and Django models.
"""
import logging
import re
from collections.abc import Callable
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar, Union, get_args, get_origin
from uuid import UUID

from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from pydantic import BaseModel, EmailStr
from pydantic.config import JsonDict
from pydantic.fields import FieldInfo
from pydantic.networks import IPvAnyAddress
from pydantic.types import Json

logger = logging.getLogger(__name__)

# Type-based mappings
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
    Any: models.JSONField,
    Path: models.FilePathField,
    Union: models.JSONField,
    Enum: models.CharField,
    type: models.CharField,
    Protocol: models.JSONField,
    Callable: models.JSONField,
    IPvAnyAddress: models.GenericIPAddressField,
    Json: models.JSONField,
}


class TypeSuffix(Enum):
    """Common type suffixes and their corresponding Django field types."""

    ANY = "typing.Any"
    DICT = "typing.Dict"
    LIST = "typing.List"
    SET = "typing.Set"
    UNION = "typing.Union"
    TYPE = "typing.Type"
    TYPE_VAR = "typing.TypeVar"
    GENERIC = "typing.Generic"
    PATH = "Path"
    CONFIG = "Config"
    REGISTRY = "Registry"
    MANAGER = "Manager"
    STORAGE = "Storage"
    LOADER = "Loader"
    RESPONSE = "Response"
    METRICS = "Metrics"
    USAGE = "Usage"
    FORMAT = "Format"
    METADATA = "Metadata"
    INFO = "Info"
    ATTACHMENT = "Attachment"

    def get_field_class(self) -> type[models.Field]:
        """Get the Django field class for this type suffix."""
        return STRING_TYPE_MAPPING[self.value]


# String-based type mappings
STRING_TYPE_MAPPING: dict[str, type[models.Field]] = {
    TypeSuffix.ANY.value: models.JSONField,
    TypeSuffix.DICT.value: models.JSONField,
    TypeSuffix.LIST.value: models.JSONField,
    TypeSuffix.SET.value: models.JSONField,
    TypeSuffix.UNION.value: models.JSONField,
    TypeSuffix.TYPE.value: models.CharField,
    TypeSuffix.TYPE_VAR.value: models.JSONField,
    TypeSuffix.GENERIC.value: models.JSONField,
    TypeSuffix.PATH.value: models.CharField,
    TypeSuffix.CONFIG.value: models.JSONField,
    TypeSuffix.REGISTRY.value: models.JSONField,
    TypeSuffix.MANAGER.value: models.JSONField,
    TypeSuffix.STORAGE.value: models.JSONField,
    TypeSuffix.LOADER.value: models.JSONField,
    TypeSuffix.RESPONSE.value: models.JSONField,
    TypeSuffix.METRICS.value: models.JSONField,
    TypeSuffix.USAGE.value: models.JSONField,
    TypeSuffix.FORMAT.value: models.JSONField,
    TypeSuffix.METADATA.value: models.JSONField,
    TypeSuffix.INFO.value: models.JSONField,
    TypeSuffix.ATTACHMENT.value: models.JSONField,
}


class FieldPattern(Enum):
    """Field patterns and their default max lengths."""

    ERROR = "error"
    DESCRIPTION = "description"
    API_KEY = "api_key"
    GIT_COMMIT = "git_commit"
    RESPONSE_SCHEMA = "response_schema"
    PARENT_TASK_ID = "parent_task_id"
    NAME = "name"
    FAMILY = "family"
    FORMAT = "format"
    CONTENT_TYPE = "content_type"
    PATH = "path"
    FILE_PATH = "file_path"
    FILE_NAME = "file_name"
    FUNCTION_NAME = "function_name"
    MODEL = "model"
    PROVIDER = "provider"
    TYPE = "type"
    STATUS = "status"
    AUTHOR = "author"
    CHANGE_TYPE = "change_type"
    NUMBER = "number"
    MIN_API_VERSION = "min_api_version"
    RECOMMENDED_REPLACEMENT = "recommended_replacement"
    DEFAULT_AGENT_TYPE = "default_agent_type"
    REMINDER_TEMPLATE = "reminder_template"
    CAPABILITIES_DETECTOR = "capabilities_detector"
    API_BASE = "api_base"
    SESSION_ID = "session_id"
    STORAGE_PATH = "storage_path"
    BASE_PATH = "base_path"
    SYSTEM_PROMPT = "system_prompt"
    USER_PROMPT = "user_prompt"
    NOTES = "notes"
    TAX_CODE = "tax_code"
    GOOGLE_API_KEY = "google_api_key"

    def get_max_length(self) -> int:
        """Get the default max length for this field pattern."""
        return FIELD_PATTERN_LENGTHS[self]


# Field pattern max lengths
FIELD_PATTERN_LENGTHS: dict[FieldPattern, int] = {
    FieldPattern.ERROR: 1000,
    FieldPattern.DESCRIPTION: 1000,
    FieldPattern.API_KEY: 512,
    FieldPattern.GIT_COMMIT: 40,
    FieldPattern.RESPONSE_SCHEMA: 2000,
    FieldPattern.PARENT_TASK_ID: 255,
    FieldPattern.NAME: 255,
    FieldPattern.FAMILY: 255,
    FieldPattern.FORMAT: 100,
    FieldPattern.CONTENT_TYPE: 100,
    FieldPattern.PATH: 1000,
    FieldPattern.FILE_PATH: 1000,
    FieldPattern.FILE_NAME: 255,
    FieldPattern.FUNCTION_NAME: 255,
    FieldPattern.MODEL: 255,
    FieldPattern.PROVIDER: 255,
    FieldPattern.TYPE: 100,
    FieldPattern.STATUS: 50,
    FieldPattern.AUTHOR: 255,
    FieldPattern.CHANGE_TYPE: 50,
    FieldPattern.NUMBER: 50,
    FieldPattern.MIN_API_VERSION: 50,
    FieldPattern.RECOMMENDED_REPLACEMENT: 255,
    FieldPattern.DEFAULT_AGENT_TYPE: 100,
    FieldPattern.REMINDER_TEMPLATE: 1000,
    FieldPattern.CAPABILITIES_DETECTOR: 255,
    FieldPattern.API_BASE: 1000,
    FieldPattern.SESSION_ID: 255,
    FieldPattern.STORAGE_PATH: 1000,
    FieldPattern.BASE_PATH: 1000,
    FieldPattern.SYSTEM_PROMPT: 4000,
    FieldPattern.USER_PROMPT: 4000,
    FieldPattern.NOTES: 1000,
    FieldPattern.TAX_CODE: 50,
    FieldPattern.GOOGLE_API_KEY: 512,
}

# Default max lengths for Django field types
FIELD_TYPE_MAX_LENGTHS: dict[type[models.Field], int] = {
    models.CharField: 255,
    models.EmailField: 254,  # RFC 5321
}


def safe_str_tuple(value: Any) -> tuple[str, str]:
    """Convert a key-value pair to a string tuple safely."""
    if isinstance(value, list | tuple) and len(value) == 2:
        return str(value[0]), str(value[1])
    return str(value), str(value)


def _handle_basic_attributes(field_info: FieldInfo) -> dict[str, Any]:
    """Handle basic field attributes like title and description."""
    kwargs: dict[str, Any] = {}
    if hasattr(field_info, "title") and field_info.title:
        kwargs["verbose_name"] = field_info.title
    if hasattr(field_info, "description") and field_info.description:
        kwargs["help_text"] = field_info.description
    return kwargs


def _handle_string_constraints(constraints: dict[str, Any], kwargs: dict[str, Any]) -> None:
    """Handle string-related constraints."""
    if "max_length" in constraints:
        kwargs["max_length"] = constraints["max_length"]
    if "min_length" in constraints:
        kwargs["min_length"] = constraints["min_length"]


def _handle_numeric_constraints(constraints: dict[str, Any], kwargs: dict[str, Any]) -> None:
    """Handle numeric constraints like gt, lt, ge, le."""
    if "gt" in constraints:
        kwargs["validators"] = kwargs.get("validators", []) + [MinValueValidator(constraints["gt"])]
    if "lt" in constraints:
        kwargs["validators"] = kwargs.get("validators", []) + [MaxValueValidator(constraints["lt"])]
    if "ge" in constraints:
        kwargs["validators"] = kwargs.get("validators", []) + [MinValueValidator(constraints["ge"])]
    if "le" in constraints:
        kwargs["validators"] = kwargs.get("validators", []) + [MaxValueValidator(constraints["le"])]


def _handle_field_attributes(extra_dict: dict[str, Any], kwargs: dict[str, Any]) -> None:
    """Handle Django-specific field attributes."""
    field_attrs = ["verbose_name", "help_text", "unique", "db_index"]
    for attr in field_attrs:
        if attr in extra_dict:
            kwargs[attr] = extra_dict[attr]


def _handle_decimal_fields(field_info: FieldInfo, extra_dict: dict[str, Any], kwargs: dict[str, Any]) -> None:
    """Handle decimal field specific attributes."""
    if field_info.annotation == Decimal:
        kwargs["max_digits"] = extra_dict.get("max_digits", 10)
        kwargs["decimal_places"] = extra_dict.get("decimal_places", 2)


def _handle_default_max_length(field_info: FieldInfo, kwargs: dict[str, Any]) -> None:
    """Handle default max_length for string fields."""
    if "max_length" not in kwargs:
        base_type = field_info.annotation
        if get_origin(base_type) is Union:
            args = get_args(base_type)
            base_type = next((t for t in args if t is not type(None)), str)
        if base_type == str:
            kwargs["max_length"] = 255


def get_field_kwargs(
    field_name: str,
    field_info: FieldInfo,
    extra: Union[JsonDict, dict[str, Any], Callable[..., Any], None],
) -> dict[str, Any]:
    """Get kwargs for creating a Django field."""
    kwargs = {}

    # Convert extra to dict, handling callable case
    extra_dict = {} if callable(extra) else (dict(extra) if extra is not None else {})

    # Get constraints from field_info metadata
    metadata = getattr(field_info, "metadata", [])

    # Process constraints from metadata
    for constraint in metadata:
        constraint_type = type(constraint).__name__
        if constraint_type == "MaxLen":
            kwargs["max_length"] = constraint.max_length
        elif constraint_type == "_PydanticGeneralMetadata":
            if hasattr(constraint, "max_digits"):
                kwargs["max_digits"] = constraint.max_digits
            if hasattr(constraint, "decimal_places"):
                kwargs["decimal_places"] = constraint.decimal_places
        elif constraint_type == "Gt":
            kwargs["validators"] = kwargs.get("validators", []) + [MinValueValidator(constraint.gt)]
        elif constraint_type == "Lt":
            kwargs["validators"] = kwargs.get("validators", []) + [MaxValueValidator(constraint.lt)]

    # Process constraints from extra_dict
    if extra_dict:
        if "max_length" in extra_dict:
            kwargs["max_length"] = extra_dict["max_length"]
        if "max_digits" in extra_dict:
            kwargs["max_digits"] = extra_dict["max_digits"]
        if "decimal_places" in extra_dict:
            kwargs["decimal_places"] = extra_dict["decimal_places"]
        if "gt" in extra_dict:
            kwargs["validators"] = kwargs.get("validators", []) + [MinValueValidator(extra_dict["gt"])]
        if "lt" in extra_dict:
            kwargs["validators"] = kwargs.get("validators", []) + [MaxValueValidator(extra_dict["lt"])]

    # Handle field metadata
    if hasattr(field_info, "title") and field_info.title:
        kwargs["verbose_name"] = field_info.title
    if hasattr(field_info, "description") and field_info.description:
        kwargs["help_text"] = field_info.description

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

    # Handle Optional types (Union[T, None])
    if origin is Union:
        args = get_args(field_type)
        if len(args) == 2 and type(None) in args:
            field_type = next(arg for arg in args if arg is not type(None))
            origin = get_origin(field_type)
        else:
            return Any, False

    # Handle generic types
    if hasattr(field_type, "__origin__"):
        origin = field_type.__origin__
        args = field_type.__args__

        # Handle TypeVar and generic parameters
        if any(hasattr(arg, "__bound__") or str(arg).startswith("~") for arg in args):
            return Any, False

        # Handle List/Set with generic type parameter
        if origin in (list, list, set, set):
            if len(args) == 1:
                field_type = args[0]
                is_collection = True
        elif origin in (dict, dict):
            return dict, False
        elif origin is Generic or hasattr(origin, "__parameters__"):
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

    # If field_type is None at this point, return Any
    if field_type is None:
        return Any, False

    return field_type, is_collection


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

    # Preserve leading underscore if it exists
    has_leading_underscore = name.startswith("_")

    # Ensure it starts with a letter or underscore if it doesn't already
    if not name[0].isalpha() and not has_leading_underscore:
        name = f"_{name}"

    # Combine prefix and name
    result = f"{prefix}{name}".lower()

    # Remove any trailing underscores
    result = result.rstrip("_")

    # If result is empty after all processing, use a default
    if not result:
        result = f"{prefix}related"

    # Ensure the name doesn't exceed Django's field name length limit (63 characters)
    if len(result) > 63:
        # Use a consistent suffix for truncated names
        result = f"{result[:54]}_{'a' * 8}"

    return result


def get_relationship_field(field_name: str, field_info: FieldInfo, field_type: type[BaseModel]) -> models.Field:
    """Create a relationship field based on the field type and metadata."""
    kwargs = get_field_kwargs(field_name, field_info, field_info.json_schema_extra or {})
    metadata = field_info.json_schema_extra or {}

    # Convert model name to Django model name
    model_name = field_type.__name__
    if not model_name.startswith("Django"):
        model_name = f"Django{model_name}"

    # Use the model name as a string to avoid circular dependencies
    # Django expects "app_label.ModelName" format
    to_model = f"testapp.{model_name}"

    # Handle one-to-one relationships
    if metadata.get("one_to_one", False):
        kwargs.pop("one_to_one", None)  # Remove one_to_one from kwargs
        return models.OneToOneField(to_model, on_delete=models.CASCADE, **kwargs)

    # Handle many-to-many relationships
    if isinstance(field_info.annotation, list) or get_origin(field_info.annotation) in (
        list,
        set,
    ):
        return models.ManyToManyField(to_model, **kwargs)

    # Default to ForeignKey
    return models.ForeignKey(to_model, on_delete=models.CASCADE, **kwargs)


class TypeResolver:
    """Handles type resolution and field type mapping."""

    def __init__(self):
        self._field_type_mapping = FIELD_TYPE_MAPPING.copy()

    def resolve_type(self, field_type: Any) -> tuple[type[models.Field], bool]:
        """Resolve a Python/Pydantic type to a Django field type."""
        origin_type = get_origin(field_type)
        is_collection = False

        # Handle Optional/Union types
        if origin_type is Union:
            args = get_args(field_type)
            if len(args) == 2 and type(None) in args:
                # For Optional types, use the non-None type
                field_type = next(arg for arg in args if arg is not type(None))
                return self.resolve_type(field_type)
            return models.JSONField, False

        # Handle collection types
        if origin_type in (list, set, dict):
            is_collection = True
            return models.JSONField, is_collection

        # Handle basic types
        if field_type in self._field_type_mapping:
            return self._field_type_mapping[field_type], is_collection

        # Try string representation for typing types and suffixes
        field_type_str = str(field_type)
        if field_type_str in STRING_TYPE_MAPPING:
            return STRING_TYPE_MAPPING[field_type_str], is_collection

        # Check for suffix matches using TypeSuffix enum
        for suffix in TypeSuffix:
            if field_type_str.endswith(suffix.value):
                return suffix.get_field_class(), is_collection

        # Handle enums specially
        if "Enum" in field_type_str:
            return models.CharField, is_collection

        # Handle class types (non-Pydantic)
        if isinstance(field_type, type):
            # Check if the class has a to_json/to_dict method
            if hasattr(field_type, "to_json") or hasattr(field_type, "to_dict"):
                return models.JSONField, is_collection
            # Check if the class has a __str__ method that's not the default object.__str__
            elif field_type.__str__ is not object.__str__:
                return models.TextField, is_collection
            else:
                # Default to JSONField for class instances
                return models.JSONField, is_collection

        # Default to JSONField for unknown types
        logger.debug(f"Using default JSONField for unknown type: {field_type}")
        return models.JSONField, is_collection

    def register_field_type(self, python_type: type, django_field: type[models.Field]) -> None:
        """Register a new field type mapping."""
        self._field_type_mapping[python_type] = django_field


class FieldTypeManager:
    """Manages field type mappings and their configurations."""

    def __init__(self):
        self._type_resolver = TypeResolver()
        self._field_type_max_lengths = FIELD_TYPE_MAX_LENGTHS.copy()
        self._pattern_lengths = FIELD_PATTERN_LENGTHS.copy()

    def get_field_type(self, field_type: Any) -> type[models.Field]:
        """Get the Django field type for a given Python/Pydantic type."""
        field_class, _ = self._type_resolver.resolve_type(field_type)
        return field_class

    def get_default_max_length(self, field_name: str, field_type: type[models.Field]) -> int:
        """Get the default max_length for a field type."""
        if field_type == models.CharField:
            # Check for specific field name patterns
            field_name_lower = field_name.lower()
            for pattern in FieldPattern:
                if pattern.value in field_name_lower:
                    return self._pattern_lengths[pattern]
            # If no pattern matched, return default CharField length
            return self._field_type_max_lengths.get(models.CharField, 255)
        elif field_type == models.EmailField:
            return self._field_type_max_lengths.get(models.EmailField, 254)
        return 255  # Default fallback

    def register_field_type(self, python_type: type, django_field: type[models.Field]) -> None:
        """Register a new field type mapping."""
        self._type_resolver.register_field_type(python_type, django_field)

    def register_max_length(self, pattern: FieldPattern, length: int) -> None:
        """Register a new max_length for a field pattern."""
        self._pattern_lengths[pattern] = length


def _handle_enum_field(field_type: type[Enum], kwargs: dict[str, Any]) -> models.Field:
    """
    Handle enum field by creating a CharField with choices.

    Args:
        field_type: The enum class
        kwargs: Additional field arguments

    Returns:
        A CharField with choices set from the enum
    """
    # Create choices from enum members
    choices = [(member.value, member.name) for member in field_type]

    # Set max_length based on the longest value if not already set
    if "max_length" not in kwargs:
        max_length = max(len(str(choice[0])) for choice in choices)
        kwargs["max_length"] = max(max_length, 1)  # Ensure at least length 1

    # Add choices to kwargs
    kwargs["choices"] = choices

    return models.CharField(**kwargs)


def handle_id_field(field_name: str, field_info: FieldInfo) -> tuple[str, dict[str, Any]]:
    """
    Handle potential ID field naming conflicts with Django's automatic primary key.

    Args:
        field_name: The original field name
        field_info: The Pydantic field info

    Returns:
        Tuple of (new_field_name, field_kwargs)
    """
    field_kwargs = {}
    new_field_name = field_name

    # Check if this is an ID field (case insensitive)
    if field_name.lower() == "id":
        # Rename the field to custom_id or similar
        new_field_name = "custom_id"
        # Add db_column to maintain the original column name in database
        field_kwargs["db_column"] = field_name
        # Add a helpful comment in verbose_name
        field_kwargs[
            "verbose_name"
        ] = f"Custom {field_name} (renamed from '{field_name}' to avoid conflict with Django's primary key)"

    return new_field_name, field_kwargs


class FieldConverter:
    """Converts Pydantic fields to Django model fields."""

    def __init__(self):
        self._type_manager = FieldTypeManager()

    def convert_field(self, field_name: str, field_info: FieldInfo) -> models.Field:
        """Convert a Pydantic field to a Django field."""
        # Handle potential ID field conflicts
        field_name, id_field_kwargs = handle_id_field(field_name, field_info)

        field_type = field_info.annotation
        origin_type = get_origin(field_type)

        # Get field kwargs and merge with any ID field kwargs
        kwargs = get_field_kwargs(field_name, field_info, field_info.json_schema_extra)
        kwargs.update(id_field_kwargs)

        # Handle nullable fields
        if origin_type is Union:
            args = get_args(field_type)
            if len(args) == 2 and type(None) in args:
                kwargs["null"] = True
                kwargs["blank"] = True
                field_type = next(arg for arg in args if arg is not type(None))

        # Handle collection relationships
        if origin_type in (list, set):
            args = get_args(field_type)
            if args and is_pydantic_model(args[0]):
                return self._handle_relationship_field(field_name, args[0], field_info)

        # Handle direct relationships
        if field_type is not None and is_pydantic_model(field_type):
            return self._handle_relationship_field(field_name, field_type, field_info)

        # Handle enums
        if isinstance(field_type, type) and issubclass(field_type, Enum):
            return _handle_enum_field(field_type, kwargs)

        # Get Django field class for other types
        django_field_class = self._type_manager.get_field_type(field_type)

        # Handle max_length for CharField if not already set
        if issubclass(django_field_class, models.CharField) and "max_length" not in kwargs:
            kwargs["max_length"] = self._type_manager.get_default_max_length(field_name, django_field_class)

        # Create and return the Django field
        return django_field_class(**kwargs)

    def _handle_relationship_field(
        self, field_name: str, field_type: type[BaseModel], field_info: FieldInfo
    ) -> models.Field:
        """Handle relationship fields (OneToOne, ForeignKey, ManyToMany)."""
        extra = field_info.json_schema_extra or {}
        kwargs = get_field_kwargs(field_name, field_info, extra)

        # Convert model name to Django model name
        model_name = field_type.__name__
        if not model_name.startswith("Django"):
            model_name = f"Django{model_name}"

        # Use the model name as a string to avoid circular dependencies
        # Django expects "app_label.ModelName" format
        to_model = f"testapp.{model_name}"

        # Handle OneToOne relationships
        if extra.get("one_to_one", False):
            return models.OneToOneField(to_model, on_delete=models.CASCADE, **kwargs)

        # Handle ManyToMany relationships
        origin_type = get_origin(field_info.annotation)
        if origin_type in (list, set):
            args = get_args(field_info.annotation)
            if args and is_pydantic_model(args[0]):
                inner_model_name = args[0].__name__
                if not inner_model_name.startswith("Django"):
                    inner_model_name = f"Django{inner_model_name}"
                to_model = f"testapp.{inner_model_name}"
                return models.ManyToManyField(to_model, **kwargs)
            return models.JSONField(**kwargs)
        elif isinstance(field_info.annotation, (list, set)):
            return models.JSONField(**kwargs)

        # Default to ForeignKey
        return models.ForeignKey(to_model, on_delete=models.CASCADE, **kwargs)


# Update the get_django_field function to use the converter
def get_django_field(field_name: str, field_info: FieldInfo, skip_relationships: bool = False) -> models.Field:
    """Convert a Pydantic field to a Django field."""
    converter = FieldConverter()
    return converter.convert_field(field_name, field_info)
