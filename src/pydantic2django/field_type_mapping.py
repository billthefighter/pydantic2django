import datetime
from collections.abc import Callable
from dataclasses import dataclass
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

logger = getLogger(__name__)

# Type alias for python types that can be either a direct type or a collection type
PythonType = Union[type, list[type], dict[str, type]]


@dataclass
class TypeMappingDefinition:
    """
    Definition of a mapping between a Python/Pydantic type and a Django field type.

    This class represents a single mapping between a Python type and a Django field type,
    with optional additional attributes like max_length and relationship info.
    """

    python_type: PythonType
    django_field: type[models.Field]
    max_length: Optional[int] = None
    is_relationship: bool = False
    on_delete: Optional[Any] = None  # For ForeignKey relationships
    field_kwargs: dict[str, Any] = {}

    def __post_init__(self):
        """
        Post-initialization hook to set default values for max_length and field_kwargs.
        """
        if self.max_length is None and self.django_field == models.CharField:
            if isinstance(self.python_type, type) and hasattr(self.python_type, "__name__"):
                self.max_length = get_default_max_length(self.python_type.__name__, self.django_field)
            else:
                # Just assume 255 is fine for any CharField
                self.max_length = 255

    # Class methods for creating common field types

    @property
    def relationship_type(self) -> Optional[str]:
        """
        Get the relationship type for this mapping.
        """
        if self.django_field == models.ForeignKey:
            return "foreign_key"
        elif self.django_field == models.ManyToManyField:
            return "many_to_many"
        else:
            return None

    @classmethod
    def char_field(cls, python_type: PythonType, max_length: int = 255) -> "TypeMappingDefinition":
        """Create a CharField mapping with the specified max_length."""
        return cls(
            python_type=python_type,
            django_field=models.CharField,
            max_length=max_length,
        )

    @classmethod
    def text_field(cls, python_type: PythonType) -> "TypeMappingDefinition":
        """Create a TextField mapping."""
        return cls(python_type=python_type, django_field=models.TextField)

    @classmethod
    def json_field(cls, python_type: PythonType) -> "TypeMappingDefinition":
        """Create a JSONField mapping."""
        return cls(python_type=python_type, django_field=models.JSONField)

    @classmethod
    def email_field(cls, python_type: PythonType = EmailStr, max_length: int = 254) -> "TypeMappingDefinition":
        """Create an EmailField mapping with the specified max_length."""
        return cls(
            python_type=python_type,
            django_field=models.EmailField,
            max_length=max_length,
        )

    @classmethod
    def foreign_key(cls, python_type: PythonType) -> "TypeMappingDefinition":
        """Create a ForeignKey mapping."""
        return cls(
            python_type=python_type,
            django_field=models.ForeignKey,
            is_relationship=True,
            on_delete=models.CASCADE,
        )

    @classmethod
    def many_to_many(cls, python_type: PythonType) -> "TypeMappingDefinition":
        """Create a ManyToManyField mapping."""
        return cls(
            python_type=python_type,
            django_field=models.ManyToManyField,
            is_relationship=True,
        )

    @classmethod
    def enum_field(cls, python_type: type[Enum]) -> "TypeMappingDefinition":
        """Create an EnumField mapping."""
        enum_values = [item.value for item in python_type]

        # Determine the type of the enum values
        if all(isinstance(val, int) for val in enum_values):
            # Integer enum
            return cls(
                python_type=python_type,
                django_field=models.IntegerField,
                field_kwargs={"choices": [(item.value, item.name) for item in enum_values]},
            )
        elif all(isinstance(val, (str, int)) for val in enum_values):
            # String enum
            max_length = max(len(str(val)) for val in enum_values if isinstance(val, str))
            return cls(
                python_type=python_type,
                django_field=models.CharField,
                max_length=max_length,
                field_kwargs={"choices": [(item.value, item.name) for item in enum_values]},
            )
        elif all(isinstance(val, (str, int, float)) for val in enum_values):
            # Mixed type enum - use TextField with choices
            return cls(
                python_type=python_type,
                django_field=models.TextField,
                field_kwargs={"choices": [(str(item.value), item.name) for item in enum_values]},
            )
        # TODO: Add support for other enum types
        else:
            raise ValueError(f"Unsupported enum values: {enum_values}")

    def get_django_field(self, kwargs: Optional[dict[str, Any]] = None) -> models.Field:
        """
        Get the Django field type with the given kwargs.
        If this field has additional kwargs, they will be merged with the kwargs passed in.
        This is the preferred way to access the field type.
        Args:
            kwargs: Additional kwargs for the field

        Returns:
            The Django field type
        """
        if kwargs is None:
            kwargs = {}
        # Merge the kwargs with the field_kwargs
        kwargs.update(self.field_kwargs)
        # If this is a CharField, set the max_length
        if self.django_field == models.CharField and "max_length" not in kwargs and self.max_length is not None:
            kwargs["max_length"] = self.max_length
        return self.django_field(**kwargs)

    def matches_type(self, python_type: Any) -> bool:
        """Check if this definition matches the given Python type."""
        # Direct type equality check
        if self.python_type == python_type:
            return True

        # Handle complex types like List[str], Dict[str, Any], etc.
        origin = get_origin(python_type)
        args = get_args(python_type)
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
            elif origin in (list, dict, set):
                args = get_args(python_type)
                # For relationship fields, check if the inner type matches
                if self.is_relationship and args:
                    if self.relationship_type == "many_to_many":
                        # Get the origin and args of our python_type for comparison
                        our_origin = get_origin(self.python_type)
                        our_args = get_args(self.python_type)

                        if origin is list and our_origin is list:
                            # List[Model] case - check if inner types match
                            inner_type = args[0]
                            our_inner_type = our_args[0]
                            # Check if inner_type is a subclass of BaseModel
                            return (
                                isinstance(inner_type, type)
                                and isinstance(our_inner_type, type)
                                and issubclass(inner_type, BaseModel)
                                and self.django_field == models.ManyToManyField
                            )
                        elif origin is dict and our_origin is dict:
                            # Dict[str, Model] case - check if key and value types match
                            key_type, value_type = args
                            our_key_type, our_value_type = our_args
                            return (
                                key_type == our_key_type
                                and isinstance(value_type, type)
                                and isinstance(our_value_type, type)
                                and issubclass(value_type, BaseModel)
                                and self.django_field == models.ManyToManyField
                            )
                    else:
                        # For other relationship types
                        inner_type = args[0]
                        our_type = get_origin(self.python_type) or self.python_type
                        return (
                            isinstance(inner_type, type)
                            and isinstance(our_type, type)
                            and issubclass(inner_type, our_type)
                        )
                # For regular collection types
                return origin == get_origin(self.python_type)

        # For non-collection relationship types (e.g., ForeignKey)
        if self.is_relationship and not origin:
            our_type = get_origin(self.python_type) or self.python_type
            return isinstance(python_type, type) and isinstance(our_type, type) and issubclass(python_type, BaseModel)

        # For basic types, check if the python_type is a subclass of self.python_type
        if isinstance(self.python_type, type) and isinstance(python_type, type):
            return issubclass(python_type, self.python_type)

        return False


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


class TypeMapper:
    """
    Handles bidirectional mapping between Python/Pydantic types and Django field types.

    This class provides static methods for converting between Python types and Django field types,
    as well as determining appropriate field attributes like max_length.
    """

    # Define all type mappings as TypeMappingDefinition instances
    TYPE_MAPPINGS: list[TypeMappingDefinition] = [
        # Basic Python types
        TypeMappingDefinition(
            python_type=bool, django_field=models.BooleanField
        ),  # Move bool before int since bool is a subclass of int
        TypeMappingDefinition(python_type=str, django_field=models.TextField),
        TypeMappingDefinition(python_type=int, django_field=models.IntegerField),
        TypeMappingDefinition(python_type=float, django_field=models.FloatField),
        TypeMappingDefinition(python_type=datetime.datetime, django_field=models.DateTimeField),
        TypeMappingDefinition(python_type=date, django_field=models.DateField),
        TypeMappingDefinition(python_type=time, django_field=models.TimeField),
        TypeMappingDefinition(python_type=timedelta, django_field=models.DurationField),
        TypeMappingDefinition(python_type=Decimal, django_field=models.DecimalField),
        TypeMappingDefinition(python_type=UUID, django_field=models.UUIDField),
        TypeMappingDefinition(python_type=EmailStr, django_field=models.EmailField, max_length=254),
        TypeMappingDefinition(python_type=bytes, django_field=models.BinaryField),
        # Enum type
        TypeMappingDefinition.enum_field(Enum),
        # Collection types
        TypeMappingDefinition.json_field(dict),
        TypeMappingDefinition.json_field(list),
        TypeMappingDefinition.json_field(set),
        # Special types
        TypeMappingDefinition(python_type=Path, django_field=models.FilePathField),
        TypeMappingDefinition.char_field(type),
        TypeMappingDefinition(python_type=IPvAnyAddress, django_field=models.GenericIPAddressField),
        TypeMappingDefinition.json_field(Json),
        TypeMappingDefinition.enum_field(Enum),
        # Relationship base types - these serve as templates for dynamic relationships
        TypeMappingDefinition.foreign_key(BaseModel),
        # List-based many-to-many relationships
        TypeMappingDefinition.many_to_many(list[BaseModel]),
        # Dict-based many-to-many relationships (for named relationships)
        TypeMappingDefinition(
            python_type=dict[str, BaseModel],
            django_field=models.ManyToManyField,
            is_relationship=True,
        ),
    ]

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

        # Handle Optional types first
        origin = get_origin(python_type)
        args = get_args(python_type)

        if origin is Union and type(None) in args:
            # Get the non-None type
            python_type = next(arg for arg in args if arg is not type(None))

        # Handle list[BaseModel] and dict[str, BaseModel] for many-to-many relationships
        if origin in (list, dict) and args:
            inner_type = args[-1]  # Last type argument is the value type
            if isinstance(inner_type, type) and issubclass(inner_type, BaseModel):
                # Find the many-to-many mapping
                for mapping in cls.TYPE_MAPPINGS:
                    if mapping.django_field == models.ManyToManyField:
                        our_origin = get_origin(mapping.python_type)
                        if our_origin == origin:
                            return TypeMappingDefinition(
                                python_type=python_type,
                                django_field=models.ManyToManyField,
                                is_relationship=True,
                            )

        # Look for an existing type mapping
        for mapping in cls.TYPE_MAPPINGS:
            if mapping.matches_type(python_type):
                # For relationship types, create a new mapping with is_relationship=True
                if mapping.django_field in (models.ForeignKey, models.ManyToManyField):
                    return TypeMappingDefinition(
                        python_type=python_type,
                        django_field=mapping.django_field,
                        is_relationship=True,
                        on_delete=mapping.on_delete,
                    )
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
        return [mapping for mapping in cls.TYPE_MAPPINGS if mapping.django_field == django_field]

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
    def get_all_mappings(cls) -> list[TypeMappingDefinition]:
        """
        Get all available type mappings.

        Returns:
            List of self.TYPE_MAPPINGS
        """
        return cls.TYPE_MAPPINGS

    @classmethod
    def get_field_attributes(cls, python_type: Any) -> dict[str, Any]:
        """
        Get the field attributes (like max_length) for a Python type.

        Args:
            python_type: The Python type to get attributes for

        Returns:
            Dictionary of field attributes
        """
        field_kwargs = {}

        # Handle Optional types first
        origin = get_origin(python_type)
        args = get_args(python_type)

        if origin is Union and type(None) in args:
            # Get the non-None type
            python_type = next(arg for arg in args if arg is not type(None))
            field_kwargs["null"] = True
            field_kwargs["blank"] = True

        # Look for an existing type mapping
        mapping = cls.get_mapping_for_type(python_type)
        if mapping:
            if mapping.max_length is not None:
                field_kwargs["max_length"] = mapping.max_length
            if mapping.is_relationship and mapping.on_delete is not None:
                field_kwargs["on_delete"] = mapping.on_delete

        return field_kwargs

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
