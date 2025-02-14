"""
Support for copying methods and properties from Pydantic to Django models.
"""
from inspect import getmembers, isdatadescriptor, isroutine
from typing import Any, cast

from django.db import models
from pydantic import BaseModel

IGNORED_NAMES = {
    # Pydantic internal methods
    "model_config",
    "model_fields",
    "model_fields_set",
    "model_extra",
    "model_computed_fields",
    "model_post_init",
    "model_validate",
    "model_validate_json",
    "model_dump",
    "model_dump_json",
    "model_copy",
    # Django internal methods
    "clean",
    "clean_fields",
    "full_clean",
    "validate_unique",
    "save",
    "delete",
    "refresh_from_db",
    # Python special methods
    "__dict__",
    "__class__",
    "__module__",
    "__weakref__",
    "__annotations__",
    "__doc__",
    "__slots__",
    "__init__",
}


def is_property(obj: Any) -> bool:
    """Check if an object is a property or similar descriptor."""
    return isdatadescriptor(obj) or isinstance(obj, property)


def is_classmethod(obj: Any) -> bool:
    """Check if an object is a classmethod."""
    return isinstance(obj, classmethod)


def is_staticmethod(obj: Any) -> bool:
    """Check if an object is a staticmethod."""
    return isinstance(obj, staticmethod)


def copy_method(method: Any) -> Any:
    """
    Copy a method while preserving its type (regular, class, or static method).

    Args:
        method: The method to copy

    Returns:
        The copied method with the same type
    """
    if is_classmethod(method):
        return classmethod(method.__get__(None, object).__func__)
    elif is_staticmethod(method):
        return staticmethod(method.__get__(None, object))
    return method


def get_methods_and_properties(model: type[BaseModel]) -> dict[str, Any]:
    """
    Extract methods and properties from a Pydantic model.

    Args:
        model: The Pydantic model class

    Returns:
        Dictionary of attribute names and their values
    """
    attrs: dict[str, Any] = {}

    # Get all members that aren't in the ignore list
    for name, member in getmembers(model):
        if name.startswith("_") or name in IGNORED_NAMES:
            continue

        # Handle properties
        if is_property(member):
            attrs[name] = property(
                member.fget if hasattr(member, "fget") else None,
                member.fset if hasattr(member, "fset") else None,
                member.fdel if hasattr(member, "fdel") else None,
                member.__doc__,
            )
            continue

        # Handle methods
        if isroutine(member):
            attrs[name] = copy_method(member)
            continue

        # Handle class variables and other descriptors
        if not name.startswith("__"):
            attrs[name] = member

    return attrs


def create_django_model_with_methods(
    name: str, pydantic_model: type[BaseModel], django_attrs: dict[str, Any]
) -> type[models.Model]:
    """
    Create a Django model class with methods and properties from a Pydantic model.

    Args:
        name: Name for the new model class
        pydantic_model: The source Pydantic model
        django_attrs: Base attributes for the Django model (fields, Meta, etc.)

    Returns:
        A new Django model class with copied methods and properties
    """
    # Get methods and properties from Pydantic model
    copied_attrs = get_methods_and_properties(pydantic_model)

    # Combine with Django attributes, letting Django attrs take precedence
    attrs = {**copied_attrs, **django_attrs}

    # Create the model class
    model = type(name, (models.Model,), attrs)
    return cast(type[models.Model], model)
