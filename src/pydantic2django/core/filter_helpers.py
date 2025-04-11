from collections.abc import Callable

from pydantic import BaseModel


def exclude_models(model_names: list[str]) -> Callable[[str, type[BaseModel]], bool]:
    """
    Create a filter function that excludes models with specific names.

    Args:
        model_names: List of model names to exclude

    Returns:
        A filter function that returns False for models in the exclusion list
    """
    return lambda model_name, _: model_name not in model_names


def include_models(model_names: list[str]) -> Callable[[str, type[BaseModel]], bool]:
    """
    Create a filter function that includes only models with specific names.

    Args:
        model_names: List of model names to include

    Returns:
        A filter function that returns True only for models in the inclusion list
    """
    return lambda model_name, _: model_name in model_names


def has_field(field_name: str) -> Callable[[str, type[BaseModel]], bool]:
    """
    Create a filter function that includes only models with a specific field.

    Args:
        field_name: The name of the field that models must have

    Returns:
        A filter function that returns True only for models with the specified field
    """
    return lambda _, model_class: field_name in model_class.model_fields


def always_include(_model_name: str, _model_class: type[BaseModel]) -> bool:
    """
    A stub filter function that always returns True.

    This can be used as a starting point for creating custom filter functions
    for the discover_models function.

    Args:
        _model_name: The normalized name of the model
        _model_class: The Pydantic model class

    Returns:
        True, always including the model
    """
    return True
