"""
Field mapping between Pydantic and Django models.
"""
import logging
import re
from collections.abc import Callable
from decimal import Decimal
from enum import Enum
from typing import (
    Any,
    Generic,
    Optional,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from pydantic import BaseModel
from pydantic.config import JsonDict
from pydantic.fields import FieldInfo

from .field_type_mapping import (
    TypeMapper,
)

logger = logging.getLogger(__name__)


def is_pydantic_model(type_: Any) -> bool:
    """
    Check if a type is a Pydantic model or collection of Pydantic models.

    Args:
        type_: The type to check

    Returns:
        True if the type is a Pydantic model or collection containing Pydantic models
    """
    try:
        # Direct Pydantic model check
        if isinstance(type_, type) and issubclass(type_, BaseModel):
            return True

        # Check for collections (List, Dict, Set, etc.)
        origin = get_origin(type_)
        if origin in (list, set, dict):
            args = get_args(type_)
            return any(is_pydantic_model(arg) for arg in args)

        return False
    except TypeError:
        return False


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
    # Handle empty or None name
    if not name:
        name = "related"

    # Convert camelCase and PascalCase to snake_case
    # Insert underscore between lowercase and uppercase letters
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()

    # Replace invalid characters with underscores
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)

    # Remove consecutive underscores
    name = re.sub(r"_+", "_", name)

    # Start with model name and field name to ensure uniqueness
    prefix = ""
    if model_name and field_name:
        expected_prefix = f"{model_name.lower()}_{field_name.lower()}"
        # Only add prefix if the name doesn't already contain it
        if expected_prefix != name and not name.startswith(f"{expected_prefix}_"):
            prefix = f"{expected_prefix}_"
    elif model_name:
        prefix += f"{model_name.lower()}_"
    elif field_name:
        prefix += f"{field_name.lower()}_"

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


class FieldAttributeHandler:
    """Handles field attribute processing and constraints."""

    @staticmethod
    def handle_field_attributes(
        field_info: FieldInfo,
        extra: Union[JsonDict, dict[str, Any], Callable[..., Any], None] = None,
    ) -> dict[str, Any]:
        """
        Process field attributes and constraints to generate Django field kwargs.

        Args:
            field_info: The Pydantic field info
            extra: Additional field attributes from json_schema_extra

        Returns:
            Dictionary of kwargs for Django field creation
        """
        kwargs = {}
        # Convert extra to dict, handling callable case
        extra_dict = {} if extra is None or callable(extra) else dict(extra)

        # Check if field type is Decimal for proper constraint handling
        is_decimal_field = field_info.annotation == Decimal

        # Basic attributes
        if hasattr(field_info, "title") and field_info.title:
            kwargs["verbose_name"] = field_info.title
        if hasattr(field_info, "description") and field_info.description:
            kwargs["help_text"] = field_info.description

        # Process constraints from metadata
        metadata = getattr(field_info, "metadata", [])
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
                # Convert to Decimal if field is Decimal type
                gt_value = Decimal(str(constraint.gt)) if is_decimal_field else constraint.gt
                kwargs["validators"] = kwargs.get("validators", []) + [MinValueValidator(gt_value)]
            elif constraint_type == "Lt":
                # Convert to Decimal if field is Decimal type
                lt_value = Decimal(str(constraint.lt)) if is_decimal_field else constraint.lt
                kwargs["validators"] = kwargs.get("validators", []) + [MaxValueValidator(lt_value)]

        # Process constraints from extra_dict
        if extra_dict:
            # String constraints
            if "max_length" in extra_dict:
                kwargs["max_length"] = extra_dict["max_length"]
            if "min_length" in extra_dict:
                kwargs["min_length"] = extra_dict["min_length"]

            # Numeric constraints
            if "max_digits" in extra_dict:
                kwargs["max_digits"] = extra_dict["max_digits"]
            if "decimal_places" in extra_dict:
                kwargs["decimal_places"] = extra_dict["decimal_places"]
            if "gt" in extra_dict:
                # Convert to Decimal if field is Decimal type
                gt_value = Decimal(str(extra_dict["gt"])) if is_decimal_field else extra_dict["gt"]
                kwargs["validators"] = kwargs.get("validators", []) + [MinValueValidator(gt_value)]
            if "lt" in extra_dict:
                # Convert to Decimal if field is Decimal type
                lt_value = Decimal(str(extra_dict["lt"])) if is_decimal_field else extra_dict["lt"]
                kwargs["validators"] = kwargs.get("validators", []) + [MaxValueValidator(lt_value)]
            if "ge" in extra_dict:
                # Convert to Decimal if field is Decimal type
                ge_value = Decimal(str(extra_dict["ge"])) if is_decimal_field else extra_dict["ge"]
                kwargs["validators"] = kwargs.get("validators", []) + [MinValueValidator(ge_value)]
            if "le" in extra_dict:
                # Convert to Decimal if field is Decimal type
                le_value = Decimal(str(extra_dict["le"])) if is_decimal_field else extra_dict["le"]
                kwargs["validators"] = kwargs.get("validators", []) + [MaxValueValidator(le_value)]

            # Django-specific field attributes
            field_attrs = ["verbose_name", "help_text", "unique", "db_index"]
            for attr in field_attrs:
                if attr in extra_dict:
                    kwargs[attr] = extra_dict[attr]

        # Handle decimal fields
        if field_info.annotation == Decimal and "max_digits" not in kwargs:
            kwargs["max_digits"] = extra_dict.get("max_digits", 10)
            kwargs["decimal_places"] = extra_dict.get("decimal_places", 2)

        # Handle default max_length for string fields
        if "max_length" not in kwargs:
            base_type = field_info.annotation
            if get_origin(base_type) is Union:
                args = get_args(base_type)
                base_type = next((t for t in args if t is not type(None)), str)
            if base_type == str:
                kwargs["max_length"] = 255

        return kwargs


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
        # Set as primary key
        field_kwargs["primary_key"] = True
        # Add a helpful comment in verbose_name
        field_kwargs["verbose_name"] = f"Custom {field_name} (used as primary key)"

    return new_field_name, field_kwargs


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

    # Create and return the CharField
    return models.CharField(**kwargs)


class FieldConverter:
    """Converts Pydantic fields to Django model fields."""

    def __init__(self, app_label: str = "django_llm"):
        """
        Initialize the field converter.

        Args:
            app_label: The Django app label to use for model references
        """
        self.app_label = app_label
        self._attribute_handler = FieldAttributeHandler()
        # Direct reference to TypeMapper for type resolution
        self._type_mapper = TypeMapper

    def _resolve_field_type(self, field_type: Any) -> tuple[type[models.Field], bool]:
        """
        Resolve a Python/Pydantic type to a Django field type.

        Args:
            field_type: The type to resolve

        Returns:
            Tuple of (django_field_class, is_collection)
        """
        is_collection = False
        origin_type = get_origin(field_type)

        # Handle Optional types (Union[T, None])
        if origin_type is Union or str(origin_type) == "types.UnionType":
            args = get_args(field_type)
            if len(args) == 2 and type(None) in args:
                # For Optional types, use the non-None type
                non_none_type = next(arg for arg in args if arg is not type(None))
                field_class, is_collection = self._resolve_field_type(non_none_type)
                return field_class, is_collection
            return models.JSONField, False

        # Handle collection types
        if origin_type in (list, set):
            is_collection = True
            args = get_args(field_type)
            if args and len(args) == 1:
                # Check if the collection contains Pydantic models
                if is_pydantic_model(args[0]):
                    # This will be handled by relationship field logic
                    return models.ManyToManyField, True
                # Otherwise, use JSONField for collections
                return models.JSONField, True
        elif origin_type is dict:
            return models.JSONField, False

        # Handle generic types with __origin__ attribute
        if hasattr(field_type, "__origin__"):
            origin = field_type.__origin__
            args = field_type.__args__

            # Handle TypeVar and generic parameters
            if any(hasattr(arg, "__bound__") or str(arg).startswith("~") for arg in args):
                return models.JSONField, False

            # Handle List/Set with generic type parameter
            if origin in (list, set):
                if len(args) == 1:
                    if is_pydantic_model(args[0]):
                        return models.ManyToManyField, True
                    field_type = args[0]
                    is_collection = True
            elif origin in (dict,):
                return models.JSONField, False
            elif origin is Generic or hasattr(origin, "__parameters__"):
                return models.JSONField, False

        # Handle Protocol types
        if hasattr(field_type, "__protocol__") or (origin_type and hasattr(origin_type, "__protocol__")):
            return models.JSONField, False

        # Handle Callable types
        if origin_type is Callable or (callable(field_type) and not isinstance(field_type, type)):
            return models.JSONField, False

        # Handle TypeVar
        if isinstance(field_type, TypeVar):
            return models.JSONField, False

        # Handle Pydantic models (direct relationship)
        if is_pydantic_model(field_type):
            return models.ForeignKey, False

        # Handle Enum types
        if isinstance(field_type, type) and issubclass(field_type, Enum):
            return models.CharField, False

        # Use TypeMapper to find the field type
        django_field = self._type_mapper.python_to_django_field(field_type)
        return django_field, is_collection

    def _get_default_max_length(self, field_name: str, field_type: type[models.Field]) -> int:
        """
        Get the default max_length for a field.

        Args:
            field_name: The name of the field
            field_type: The Django field type

        Returns:
            The default max_length value
        """
        return self._type_mapper.get_max_length(field_name, field_type) or 255

    def _create_relationship_field(
        self,
        field_name: str,
        field_info: FieldInfo,
        field_type: Any,
    ) -> models.Field:
        """
        Create a relationship field based on the field type and metadata.

        Args:
            field_name: The name of the field
            field_info: The Pydantic field info
            field_type: The Pydantic model type for the relationship

        Returns:
            A Django relationship field (ForeignKey, ManyToManyField, or OneToOneField)
        """
        # Get field kwargs
        kwargs = self._attribute_handler.handle_field_attributes(field_info, field_info.json_schema_extra)
        metadata = field_info.json_schema_extra or {}

        # Convert model name to Django model name
        from .utils import normalize_model_name

        model_name = normalize_model_name(field_type.__name__)

        # Use the model name as a string to avoid circular dependencies
        # Django expects "app_label.ModelName" format
        to_model = f"{self.app_label}.{model_name}"

        # Handle one-to-one relationships
        if metadata.get("one_to_one", False):
            kwargs.pop("one_to_one", None)  # Remove one_to_one from kwargs
            return models.OneToOneField(to_model, on_delete=models.CASCADE, **kwargs)

        # Handle many-to-many relationships
        origin_type = get_origin(field_info.annotation)
        if origin_type in (list, set) or isinstance(field_info.annotation, (list, set)):
            return models.ManyToManyField(to_model, **kwargs)

        # Default to ForeignKey
        return models.ForeignKey(to_model, on_delete=models.CASCADE, **kwargs)

    def convert_field(
        self,
        field_name: str,
        field_info: FieldInfo,
        skip_relationships: bool = False,
    ) -> models.Field:
        """
        Convert a Pydantic field to a Django field.

        Args:
            field_name: The name of the field
            field_info: The Pydantic field info
            skip_relationships: Whether to skip relationship fields

        Returns:
            A Django model field
        """
        # Handle potential ID field conflicts
        field_name, id_field_kwargs = handle_id_field(field_name, field_info)

        # Get the field type and whether it's a collection
        field_type = field_info.annotation
        django_field_class, is_collection = self._resolve_field_type(field_type)

        # Get field kwargs and merge with any ID field kwargs
        kwargs = self._attribute_handler.handle_field_attributes(field_info, field_info.json_schema_extra)
        kwargs.update(id_field_kwargs)

        # Handle nullable fields
        origin_type = get_origin(field_type)
        if origin_type is Union:
            args = get_args(field_type)
            if len(args) == 2 and type(None) in args:
                kwargs["null"] = True
                kwargs["blank"] = True
                field_type = next(arg for arg in args if arg is not type(None))
                django_field_class, is_collection = self._resolve_field_type(field_type)

        # Handle optional fields with default=None
        if field_info.default is None and not field_info.is_required():
            kwargs["null"] = True
            kwargs["blank"] = True

        # Handle relationship fields
        if django_field_class in (
            models.ForeignKey,
            models.ManyToManyField,
            models.OneToOneField,
        ):
            if skip_relationships:
                # If skipping relationships, return a JSONField instead
                return models.JSONField(**kwargs)

            # Get the related model type
            if origin_type in (list, set):
                args = get_args(field_type)
                if args and is_pydantic_model(args[0]):
                    return self._create_relationship_field(field_name, field_info, args[0])
            elif is_pydantic_model(field_type):
                return self._create_relationship_field(field_name, field_info, field_type)

        # Handle enums
        if isinstance(field_type, type) and issubclass(field_type, Enum):
            return _handle_enum_field(field_type, kwargs)

        # Handle max_length for CharField if not already set
        if issubclass(django_field_class, models.CharField) and "max_length" not in kwargs:
            kwargs["max_length"] = self._get_default_max_length(field_name, django_field_class)

        # Create and return the Django field
        return django_field_class(**kwargs)


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

    # Handle special case for direct model relationships when model_name is provided
    if model_name is not None and is_pydantic_model(field_info.annotation):
        # Use the model name as a string to avoid circular dependencies
        # Django expects "app_label.ModelName" format
        to_model = f"{app_label}.{model_name}"

        # Handle one-to-one relationships
        if field_info.annotation is BaseModel:
            if skip_relationships:
                return None
            return models.OneToOneField(
                to_model,
                on_delete=models.CASCADE,
                related_name=f"{field_name}_set",
            )

        # Handle lists of models (many-to-many)
        if get_origin(field_info.annotation) is list and issubclass(get_args(field_info.annotation)[0], BaseModel):
            if skip_relationships:
                return None
            return models.ManyToManyField(
                to_model,
                related_name=f"{field_name}_set",
            )

    # Use the converter for standard field conversion
    return converter.convert_field(field_name, field_info, skip_relationships)
