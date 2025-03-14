import datetime
from collections.abc import Callable
from datetime import date, time, timedelta
from decimal import Decimal
from enum import Enum
from logging import getLogger
from pathlib import Path
from typing import (
    Any,
    Optional,
    Protocol,
    Union,
    get_args,
    get_origin,
)
from uuid import UUID

from django.db import models

# EmailStr and IPvAnyAddress are likely from pydantic
from pydantic import BaseModel, EmailStr, IPvAnyAddress, Json

from pydantic2django.field_utils import RelationshipFieldHandler

logger = getLogger(__name__)


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


class TypeMappingDefinition(BaseModel):
    """
    Definition of a mapping between a Python/Pydantic type and a Django field type.

    This class represents a single mapping between a Python type and a Django field type,
    with optional additional attributes like max_length.
    """

    python_type: type
    django_field: type[models.Field]
    max_length: Optional[int] = None

    # Class methods for creating common field types
    @classmethod
    def char_field(cls, python_type: type, max_length: int = 255) -> "TypeMappingDefinition":
        """Create a CharField mapping with the specified max_length."""
        return cls(
            python_type=python_type,
            django_field=models.CharField,
            max_length=max_length,
        )

    @classmethod
    def text_field(cls, python_type: type) -> "TypeMappingDefinition":
        """Create a TextField mapping."""
        return cls(python_type=python_type, django_field=models.TextField)

    @classmethod
    def json_field(cls, python_type: type) -> "TypeMappingDefinition":
        """Create a JSONField mapping."""
        return cls(python_type=python_type, django_field=models.JSONField)

    @classmethod
    def email_field(cls, python_type: type = EmailStr, max_length: int = 254) -> "TypeMappingDefinition":
        """Create an EmailField mapping with the specified max_length."""
        return cls(
            python_type=python_type,
            django_field=models.EmailField,
            max_length=max_length,
        )

    def matches_type(self, python_type: Any) -> bool:
        """Check if this definition matches the given Python type."""
        # Direct type equality check
        if self.python_type == python_type:
            return True

        # Handle complex types like List[str], Dict[str, Any], etc.
        origin = get_origin(python_type)
        if origin is not None:
            # For Optional[T] which is Union[T, None]
            if origin is Union:
                args = get_args(python_type)
                # If it's Optional[T] (Union[T, None])
                if type(None) in args and len(args) == 2:
                    # Get the non-None type
                    non_none_type = next(arg for arg in args if arg is not type(None))
                    # For Optional[str], match with str
                    if non_none_type == self.python_type:
                        return True
                # For other Union types, match with the json_field mapping
                elif self.django_field == models.JSONField and self.python_type in (
                    dict,
                    list,
                    set,
                ):
                    return True
            # For collection types, match with their base types
            elif origin in (list, dict, set) and origin == self.python_type:
                return True

        return False

    def matches_type_str(self, type_str: str) -> bool:
        """Check if this definition matches the string representation of a type."""
        return str(self.python_type) == type_str or type_str.endswith(str(self.python_type))


# Define all type mappings as TypeMappingDefinition instances
TYPE_MAPPINGS: list[TypeMappingDefinition] = [
    # Basic Python types
    TypeMappingDefinition(python_type=str, django_field=models.TextField),
    TypeMappingDefinition(python_type=int, django_field=models.IntegerField),
    TypeMappingDefinition(python_type=float, django_field=models.FloatField),
    TypeMappingDefinition(python_type=bool, django_field=models.BooleanField),
    TypeMappingDefinition(python_type=datetime.datetime, django_field=models.DateTimeField),
    TypeMappingDefinition(python_type=date, django_field=models.DateField),
    TypeMappingDefinition(python_type=time, django_field=models.TimeField),
    TypeMappingDefinition(python_type=timedelta, django_field=models.DurationField),
    TypeMappingDefinition(python_type=Decimal, django_field=models.DecimalField),
    TypeMappingDefinition(python_type=UUID, django_field=models.UUIDField),
    TypeMappingDefinition(python_type=EmailStr, django_field=models.EmailField, max_length=254),
    TypeMappingDefinition(python_type=bytes, django_field=models.BinaryField),
    # Collection types
    TypeMappingDefinition.json_field(dict),
    TypeMappingDefinition.json_field(list),
    TypeMappingDefinition.json_field(set),
    # Special types
    TypeMappingDefinition(python_type=Path, django_field=models.FilePathField),
    TypeMappingDefinition.char_field(Enum),
    TypeMappingDefinition.char_field(type),
    TypeMappingDefinition(python_type=IPvAnyAddress, django_field=models.GenericIPAddressField),
    TypeMappingDefinition.json_field(Json),
]


class TypeMapper:
    """
    Handles bidirectional mapping between Python/Pydantic types and Django field types.

    This class provides static methods for converting between Python types and Django field types,
    as well as determining appropriate field attributes like max_length.
    """

    class UnsupportedTypeError(Exception):
        """Exception raised when a type cannot be mapped to a Django field."""

        pass

    @classmethod
    def get_mapping_for_type(cls, python_type: Any) -> Optional[TypeMappingDefinition]:
        """
        Get the mapping definition for a specific Python type.

        Args:
            python_type: The Python type to find a mapping for

        Returns:
            TypeMappingDefinition if found, None otherwise
        """
        # Handle special typing forms that aren't concrete types
        if python_type is Any or python_type is Union or python_type is Protocol or python_type is Callable:
            # Default to JSONField for these special forms
            return TypeMappingDefinition(python_type=dict, django_field=models.JSONField)

        # Delegate relationship handling to RelationshipFieldHandler
        field_class, kwargs = RelationshipFieldHandler.detect_field_type(python_type)
        if field_class:
            return TypeMappingDefinition(
                python_type=python_type,
                django_field=field_class,
            )

        for mapping in TYPE_MAPPINGS:
            if mapping.matches_type(python_type):
                return mapping
        return None

    @classmethod
    def get_mapping_for_type_str(cls, type_str: str) -> Optional[TypeMappingDefinition]:
        """
        Get the mapping definition for a type string.

        Args:
            type_str: String representation of a type

        Returns:
            TypeMappingDefinition if found, None otherwise
        """
        # Only check regular type mappings
        for mapping in TYPE_MAPPINGS:
            if mapping.matches_type_str(type_str):
                return mapping
        return None

    @classmethod
    def filter_by_django_field(cls, django_field: type[models.Field]) -> list[TypeMappingDefinition]:
        """
        Filter mappings by Django field type.

        Args:
            django_field: The Django field type to filter by

        Returns:
            List of TypeMappingDefinition instances with the specified Django field
        """
        return [mapping for mapping in TYPE_MAPPINGS if mapping.django_field == django_field]

    @classmethod
    def is_type_supported(cls, python_type: Any) -> bool:
        """
        Check if a Python type is supported by any mapping.

        Args:
            python_type: The Python type to check

        Returns:
            True if the type is supported, False otherwise
        """
        return cls.get_mapping_for_type(python_type) is not None

    @classmethod
    def is_type_str_supported(cls, type_str: str) -> bool:
        """
        Check if a type string is supported by any mapping.

        Args:
            type_str: String representation of a type

        Returns:
            True if the type string is supported, False otherwise
        """
        return cls.get_mapping_for_type_str(type_str) is not None

    @classmethod
    def get_all_mappings(cls) -> list[TypeMappingDefinition]:
        """
        Get all available type mappings.

        Returns:
            List of TYPE_MAPPINGS
        """
        return TYPE_MAPPINGS

    @classmethod
    def get_django_field_for_type(cls, python_type: Any, strict: bool = True) -> type[models.Field]:
        """
        Get the Django field type for a Python type.

        Args:
            python_type: The Python type to find a Django field for
            strict: If True, raises UnsupportedTypeError when type is not supported

        Returns:
            Django field type if found

        Raises:
            UnsupportedTypeError: If strict=True and the type is not supported
        """
        mapping = cls.get_mapping_for_type(python_type)
        if mapping:
            return mapping.django_field

        if strict:
            raise cls.UnsupportedTypeError(f"No Django field mapping found for Python type: {python_type}")

        # Default to JSONField as a fallback if not strict
        logger.warning(f"No mapping found for {python_type}, defaulting to JSONField")
        return models.JSONField

    @classmethod
    def get_django_field_for_type_str(cls, type_str: str, strict: bool = True) -> type[models.Field]:
        """
        Get the Django field type for a type string.

        Args:
            type_str: String representation of a type
            strict: If True, raises UnsupportedTypeError when type is not supported

        Returns:
            Django field type if found

        Raises:
            UnsupportedTypeError: If strict=True and the type is not supported
        """
        mapping = cls.get_mapping_for_type_str(type_str)
        if mapping:
            return mapping.django_field

        if strict:
            raise cls.UnsupportedTypeError(f"No Django field mapping found for type string: {type_str}")

        # Default to JSONField as a fallback if not strict
        logger.warning(f"No mapping found for {type_str}, defaulting to JSONField")
        return models.JSONField

    @classmethod
    def get_field_attributes(cls, python_type: Any) -> dict[str, Any]:
        """
        Get the field attributes (like max_length) for a Python type.

        Args:
            python_type: The Python type to get attributes for

        Returns:
            Dictionary of field attributes
        """
        mapping = cls.get_mapping_for_type(python_type)
        if not mapping:
            return {}

        attributes = {}
        if mapping.max_length is not None:
            attributes["max_length"] = mapping.max_length

        return attributes

    @classmethod
    def register_type_mapping(cls, mapping: TypeMappingDefinition) -> None:
        """
        Register a new type mapping.

        Args:
            mapping: The TypeMappingDefinition to register
        """
        # Check if mapping already exists
        for i, existing in enumerate(TYPE_MAPPINGS):
            if existing.python_type == mapping.python_type:
                # Replace existing mapping
                TYPE_MAPPINGS[i] = mapping
                return

        # Add new mapping
        TYPE_MAPPINGS.append(mapping)

    @classmethod
    def python_to_django_field(cls, python_type: Any) -> type[models.Field]:
        """
        Get the Django field type for a Python type.
        Alias for get_django_field_for_type with strict=False.

        Args:
            python_type: The Python type to find a Django field for

        Returns:
            Django field type, defaulting to JSONField if not found
        """
        return cls.get_django_field_for_type(python_type, strict=False)

    @classmethod
    def get_max_length(cls, field_name: str, field_type: type[models.Field]) -> Optional[int]:
        """
        Get the default max_length for a field based on field name and type.

        Args:
            field_name: The name of the field
            field_type: The Django field type

        Returns:
            The default max_length or None if not applicable
        """
        # Use the shared utility function
        return get_default_max_length(field_name, field_type)
