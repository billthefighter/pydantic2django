"""
Field mapping between Pydantic and Django models.
"""
import logging
import re
import inspect
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
    List,
    Dict,
)
from abc import ABC

from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from pydantic import BaseModel, Field

# Determine Pydantic version
import pydantic
PYDANTIC_V2 = pydantic.__version__.startswith("2.")

if PYDANTIC_V2:
    # Pydantic v2 imports
    from pydantic.fields import FieldInfo as _FieldInfo
    # In Pydantic v2, we need to use model_fields instead of __fields__
    def get_model_fields(model_class):
        return model_class.model_fields
else:
    # Pydantic v1 imports
    from pydantic import FieldInfo as _FieldInfo
    # In Pydantic v1, we use __fields__
    def get_model_fields(model_class):
        return model_class.__fields__

# Use a consistent FieldInfo type for our code
FieldInfo = _FieldInfo

# Add compatibility layer for is_optional method
if not hasattr(FieldInfo, "is_optional"):
    def is_optional(self) -> bool:
        """Check if a field is optional (can be None)."""
        if PYDANTIC_V2:
            # In v2, check if None is allowed in the field
            return self.is_required is False
        else:
            # In v1, check if the field allows None
            return self.allow_none
    
    # Monkey patch the method
    setattr(FieldInfo, "is_optional", is_optional)

logger = logging.getLogger(__name__)


def is_pydantic_model(obj: Any) -> bool:
    """
    Check if an object is a Pydantic model class.

    Args:
        obj: The object to check

    Returns:
        True if the object is a Pydantic model class, False otherwise
    """
    if not inspect.isclass(obj):
        return False
    
    # Check if it's a Pydantic model
    is_pydantic = issubclass(obj, BaseModel)
    
    # Skip abstract base classes (inheriting from ABC)
    if is_pydantic and ABC in obj.__mro__:
        return False
    
    # In Pydantic v2, we need to check for model_fields attribute
    if is_pydantic and PYDANTIC_V2:
        return hasattr(obj, "model_fields")
    
    # In Pydantic v1, we check for __fields__ attribute
    if is_pydantic and not PYDANTIC_V2:
        return hasattr(obj, "__fields__")
    
    return is_pydantic


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
    """
    Handles extracting attributes from Pydantic field info for Django fields.
    """

    @staticmethod
    def handle_field_attributes(
        field_info: FieldInfo,
        extra: Union[Dict[str, Any], Callable[..., Any], None] = None,
    ) -> dict[str, Any]:
        """
        Extract attributes from Pydantic field info for Django fields.

        Args:
            field_info: The Pydantic field info
            extra: Extra attributes to include

        Returns:
            Dict of field attributes
        """
        attrs = {}

        # Handle null/blank
        try:
            # Try to use is_optional method if available
            null = field_info.is_optional()
        except (AttributeError, TypeError):
            # Fall back to checking if the field is required
            if PYDANTIC_V2:
                null = not field_info.is_required
            else:
                null = field_info.allow_none
                
        attrs["null"] = null
        attrs["blank"] = null  # Usually blank follows null

        # Handle default value
        if PYDANTIC_V2:
            has_default = field_info.default is not None and field_info.default != ...
        else:
            has_default = field_info.default is not None and field_info.default != Ellipsis
            
        if has_default:
            default_value = field_info.default
            if not callable(default_value) and default_value is not None:
                attrs["default"] = default_value

        # Handle description/help_text
        if hasattr(field_info, "description") and field_info.description:
            attrs["help_text"] = field_info.description

        # Add extra attributes
        if extra:
            if callable(extra):
                extra_attrs = extra(field_info)
            else:
                extra_attrs = extra
            attrs.update(extra_attrs)

        return attrs


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
        # Handle Optional types
        origin = get_origin(field_type)
        args = get_args(field_type)

        if origin is Union and type(None) in args:
            # This is an Optional type, get the actual type
            for arg in args:
                if arg is not type(None):
                    field_type = arg
                    break

        # Check for List/Dict types
        if origin in (list, List):
            return models.JSONField, False
        elif origin in (dict, Dict):
            return models.JSONField, False

        # Handle basic types
        if field_type is str:
            return models.CharField, False
        elif field_type is int:
            return models.IntegerField, False
        elif field_type is float:
            return models.FloatField, False
        elif field_type is bool:
            return models.BooleanField, False
        elif field_type is dict or field_type is Dict:
            return models.JSONField, False
        elif field_type is list or field_type is List:
            return models.JSONField, False

        # Handle relationship fields (Pydantic models)
        if inspect.isclass(field_type) and issubclass(field_type, BaseModel):
            return models.ForeignKey, True

        # Default to TextField for unknown types
        return models.TextField, False

    def _get_default_max_length(self, field_name: str, field_type: type[models.Field]) -> int:
        """
        Get the default max_length for a field based on its name and type.

        Args:
            field_name: The name of the field
            field_type: The Django field type

        Returns:
            The default max_length value
        """
        if field_type is models.CharField:
            if "name" in field_name or "title" in field_name:
                return 255
            elif "description" in field_name:
                return 1000
            elif "id" in field_name or field_name.endswith("_id"):
                return 100
            else:
                return 255
        return 255

    def _create_relationship_field(
        self,
        field_name: str,
        field_info: FieldInfo,
        field_type: Any,
    ) -> models.Field:
        """
        Create a relationship field (ForeignKey, ManyToManyField, etc.).

        Args:
            field_name: The name of the field
            field_info: The Pydantic field info
            field_type: The Pydantic field type

        Returns:
            A Django relationship field
        """
        # Handle Optional types
        origin = get_origin(field_type)
        args = get_args(field_type)

        if origin is Union and type(None) in args:
            # This is an Optional type, get the actual type
            for arg in args:
                if arg is not type(None):
                    field_type = arg
                    break

        # Handle List[Model] as ManyToManyField
        if origin in (list, List) and args:
            item_type = args[0]
            if inspect.isclass(item_type) and issubclass(item_type, BaseModel):
                # This is a List[Model], create a ManyToManyField
                model_name = item_type.__name__
                if not model_name.startswith("Django"):
                    model_name = f"Django{model_name}"

                # Get null/blank from field_info
                try:
                    # Try to use is_optional method if available
                    null = field_info.is_optional()
                except (AttributeError, TypeError):
                    # Fall back to checking if the field is required
                    if PYDANTIC_V2:
                        null = not field_info.is_required
                    else:
                        null = field_info.allow_none
                
                blank = null  # Usually blank follows null

                # Create the ManyToManyField
                related_name = sanitize_related_name(field_name)
                return models.ManyToManyField(
                    f"{self.app_label}.{model_name}",
                    related_name=related_name,
                    blank=blank,
                )

        # Handle direct Model reference as ForeignKey
        if inspect.isclass(field_type) and issubclass(field_type, BaseModel):
            model_name = field_type.__name__
            if not model_name.startswith("Django"):
                model_name = f"Django{model_name}"
            
            # Get null/blank from field_info
            try:
                # Try to use is_optional method if available
                null = field_info.is_optional()
            except (AttributeError, TypeError):
                # Fall back to checking if the field is required
                if PYDANTIC_V2:
                    null = not field_info.is_required
                else:
                    null = field_info.allow_none
                
            blank = null  # Usually blank follows null
            
            # Create the ForeignKey
            related_name = sanitize_related_name(field_name)
            return models.ForeignKey(
                f"{self.app_label}.{model_name}",
                on_delete=models.CASCADE,
                related_name=related_name,
                null=null,
                blank=blank,
            )

        # Default to JSONField for unknown relationship types
        return models.JSONField(null=True, blank=True)

    def convert_field(
        self,
        field_name: str,
        field_info: FieldInfo,
        skip_relationships: bool = False,
    ) -> models.Field:
        """
        Convert a Pydantic field to a Django model field.

        Args:
            field_name: The name of the field
            field_info: The Pydantic field info
            skip_relationships: Whether to skip relationship fields

        Returns:
            A Django model field
        """
        # Get field type
        if PYDANTIC_V2:
            field_type = field_info.annotation
        else:
            field_type = field_info.type_
            
        # Handle special case for id field
        if field_name == "id":
            return handle_id_field(field_name, field_info)[1]

        # Resolve field type
        django_field_type, is_relationship = self._resolve_field_type(field_type)

        # Skip relationship fields if requested
        if skip_relationships and is_relationship:
            return None

        # Handle relationship fields
        if is_relationship:
            return self._create_relationship_field(field_name, field_info, field_type)

        # Get field attributes
        field_attrs = FieldAttributeHandler.handle_field_attributes(field_info)

        # Handle specific field types
        if django_field_type is models.CharField:
            # Add max_length if not provided
            if "max_length" not in field_attrs:
                field_attrs["max_length"] = self._get_default_max_length(field_name, django_field_type)

        # Create the field
        return django_field_type(**field_attrs)


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
