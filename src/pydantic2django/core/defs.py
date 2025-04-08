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
                # For other Union types, match with the json_field mapping if appropriate
                # (This check is simplified here, more specific checks happen in TypeMapper)
                elif self.django_field == models.JSONField and self.python_type in (
                    dict,
                    list,
                    set,
                ):
                    return True
            # For collection types, match with their base types or JSONField
            elif origin in (list, dict, set):
                # Match JSONField for collections
                if self.django_field == models.JSONField:
                    if (
                        (origin is set and self.python_type is set)
                        or (origin is dict and self.python_type is dict)
                        or (origin is list and self.python_type is list)
                    ):
                        return True

                args = get_args(python_type)
                # For relationship fields, check if the inner type matches (basic check)
                if self.is_relationship and args:
                    # More detailed relationship matching logic belongs in TypeMapper
                    # Check if inner type is BaseModel (simplified check)
                    inner_type = args[0]
                    return isinstance(inner_type, type) and issubclass(inner_type, BaseModel)

                # Simplified check for regular collections (more specific checks in TypeMapper)
                # return origin == get_origin(self.python_type)

        # For non-collection relationship types (e.g., ForeignKey)
        # Basic check: Is the python_type a Pydantic BaseModel?
        if self.is_relationship and not origin:
            return isinstance(python_type, type) and issubclass(python_type, BaseModel)

        # For basic types, check if the python_type is a subclass of self.python_type
        # Ensure both are actual types before calling issubclass
        if isinstance(self.python_type, type) and isinstance(python_type, type):
            try:
                return issubclass(python_type, self.python_type)
            except TypeError:  # Handle cases where issubclass raises TypeError (e.g., non-class types)
                return False

        # Fallback if no match found
        return False
