"""
Type definitions for pydantic2django.
"""
import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, TypeVar, Union, get_args, get_origin

from django.db import models
from pydantic import BaseModel

# Type variable for BaseModel subclasses
T = TypeVar("T", bound=BaseModel)

# Type alias for Django model fields
DjangoField = Union[models.Field, type[models.Field]]

# Type alias for python types that can be either a direct type or a collection type
PythonType = Union[type, list[type], dict[str, type]]


def is_serializable_type(field_type: Any) -> bool:
    """
    Check if a type is serializable (can be stored in the database).

    A type is considered serializable if:
    1. It's a basic Python type (str, int, float, bool, dict, list, set, NoneType)
    2. It's a collection (list, dict, set) of serializable types
    3. It's a Pydantic model
    4. It's an Enum
    5. It has __get_pydantic_core_schema__ defined
    6. It has a serialization method (to_json, to_dict, etc.)

    Args:
        field_type: The type to check

    Returns:
        True if the type is serializable, False otherwise
    """
    # Handle typing.Any specially - it's not serializable
    if field_type is Any:
        return False

    # Handle NoneType (type(None)) specially - it is serializable
    if field_type is type(None):
        return True

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

    # Check if the type has __get_pydantic_core_schema__ (can be serialized)
    if hasattr(field_type, "__get_pydantic_core_schema__"):
        return True

    # Handle Pydantic models (they can be serialized to JSON)
    try:
        if inspect.isclass(field_type) and issubclass(field_type, BaseModel):
            return True
    except TypeError:
        # field_type might not be a class, which is fine
        pass

    # Handle Enums (they can be serialized)
    try:
        if inspect.isclass(field_type) and issubclass(field_type, Enum):
            return True
    except TypeError:
        # field_type might not be a class, which is fine
        pass

    # For class types, check if they have a serialization method
    if inspect.isclass(field_type):
        # Create a dummy instance to test serialization
        try:
            instance = object.__new__(field_type)
            return hasattr(instance, "to_json") or hasattr(instance, "to_dict") or hasattr(instance, "__json__")
        except Exception:
            # If we can't create an instance, assume it's not serializable
            return False

    # If none of the above conditions are met, it's not serializable
    return False


def is_pydantic_model(obj: Any) -> bool:
    """
    Check if an object is a Pydantic model class.

    Args:
        obj: The object to check

    Returns:
        True if the object is a Pydantic model class, False otherwise
    """
    try:
        return issubclass(obj, BaseModel)
    except TypeError:
        return False


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
    field_kwargs: dict[str, Any] = field(default_factory=dict)

    # Post-init logic that depended on get_default_max_length
    # will be handled in the django.mapping module where that function lives.

    # Property and class methods related to Django field creation
    # (relationship_type, char_field, text_field, etc.) are moved to django.mapping.

    def matches_type(self, python_type: Any) -> bool:
        """Check if this definition matches the given Python type."""
        actual_type = python_type
        origin = get_origin(python_type)
        args = get_args(python_type)

        # Handle Optional[T]
        if origin is Union and type(None) in args and len(args) == 2:
            actual_type = next((arg for arg in args if arg is not type(None)), None)
            if actual_type is None:  # Handle Optional[NoneType]
                return False

        # 1. Direct type equality check (most common case for simple types)
        if self.python_type == actual_type:
            return True

        # 2. Special case for JSONField matching generic collections if not matched directly
        # Check if this definition is for JSONField and the actual_type is a collection origin
        if self.django_field == models.JSONField:
            actual_origin = get_origin(actual_type)  # Use origin from actual_type if unwrapped
            if actual_origin in (list, dict, set, tuple):
                # Check if this mapping's python_type matches the collection origin
                if self.python_type == actual_origin:
                    return True

        # 3. Subclass check (use sparingly, mainly for things like EmailStr matching str)
        # Ensure both are actual classes before checking issubclass
        # Avoid comparing basic types like issubclass(int, str)
        basic_types = (str, int, float, bool, list, dict, set, tuple, bytes)
        if (
            inspect.isclass(self.python_type)
            and inspect.isclass(actual_type)
            and self.python_type not in basic_types
            and actual_type not in basic_types
        ):
            try:
                return issubclass(actual_type, self.python_type)
            except TypeError:
                return False  # issubclass fails if args aren't classes

        # If no match after all checks
        return False
