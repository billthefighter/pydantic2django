"""
Type definitions for pydantic2django.
"""
import inspect
from enum import Enum
from typing import Any, TypeVar, Union, get_args, get_origin

from django.db import models
from pydantic import BaseModel

# Type variable for BaseModel subclasses
T = TypeVar("T", bound=BaseModel)

# Type alias for Django model fields
DjangoField = Union[models.Field, type[models.Field]]


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
