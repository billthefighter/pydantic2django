from collections.abc import Callable
from enum import Enum
from typing import Any, Optional, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class SerializationMethod(Enum):
    """Enumeration of available serialization methods."""

    MODEL_DUMP = "model_dump"  # Pydantic's model_dump
    TO_JSON = "to_json"  # Custom to_json method
    TO_DICT = "to_dict"  # Custom to_dict method
    STR = "str"  # __str__ method
    DICT = "dict"  # __dict__ attribute
    NONE = "none"  # No serialization method found


def get_serialization_method(
    obj: Any,
) -> tuple[SerializationMethod, Optional[Callable[[], Any]]]:
    """
    Get the appropriate serialization method for an object.

    This function checks for various serialization methods in order of preference:
    1. model_dump (Pydantic)
    2. to_json
    3. to_dict
    4. __str__ (if overridden)
    5. __dict__

    Args:
        obj: The object to check for serialization methods

    Returns:
        A tuple of (SerializationMethod, Optional[Callable]). The callable is None if no method is found.
    """
    # Check for Pydantic model_dump
    if isinstance(obj, BaseModel):
        return SerializationMethod.MODEL_DUMP, obj.model_dump

    # Check for to_json method
    if hasattr(obj, "to_json"):
        return SerializationMethod.TO_JSON, obj.to_json

    # Check for to_dict method
    if hasattr(obj, "to_dict"):
        return SerializationMethod.TO_DICT, obj.to_dict

    # Check for overridden __str__ method
    if hasattr(obj, "__str__") and obj.__class__.__str__ is not object.__str__:
        return SerializationMethod.STR, obj.__str__

    # Check for __dict__ attribute
    if hasattr(obj, "__dict__"):

        def dict_serializer():
            return {
                "__class__": obj.__class__.__name__,
                "__module__": obj.__class__.__module__,
                "data": obj.__dict__,
            }

        return SerializationMethod.DICT, dict_serializer

    return SerializationMethod.NONE, None


def serialize_value(value: Any) -> Any:
    """
    Serialize a value using the most appropriate method.

    Args:
        value: The value to serialize

    Returns:
        The serialized value
    """
    method, serializer = get_serialization_method(value)

    if method == SerializationMethod.NONE:
        # If no serialization method is found, return the value as is
        return value

    if serializer is None:
        return value

    try:
        return serializer()
    except Exception:
        # If serialization fails, return the string representation
        return str(value)


def is_serializable(obj: Any) -> bool:
    """
    Check if an object is serializable.

    Args:
        obj: The object to check

    Returns:
        True if the object has a valid serialization method, False otherwise
    """
    method, _ = get_serialization_method(obj)
    return method != SerializationMethod.NONE
