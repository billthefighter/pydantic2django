import inspect
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo


def get_model_fields(model_class: type[BaseModel]) -> dict[str, FieldInfo]:
    """
    Get the fields from a Pydantic model, handling potential errors.

    Args:
        model_class: The Pydantic model class

    Returns:
        A dictionary of field names to FieldInfo objects.
    """
    if not inspect.isclass(model_class) or not issubclass(model_class, BaseModel):
        # logger.warning(f"Attempted to get fields from non-Pydantic model: {model_class}")
        return {}
    try:
        return model_class.model_fields
    except Exception:
        # logger.error(f"Error retrieving fields for Pydantic model {model_class.__name__}: {e}")
        return {}


def is_pydantic_model_field_optional(field_type: Any) -> bool:
    """Check if a Pydantic field type annotation represents an Optional field."""
    origin = get_origin(field_type)
    args = get_args(field_type)
    # Check for Union[T, NoneType]
    return origin is Union and type(None) in args
