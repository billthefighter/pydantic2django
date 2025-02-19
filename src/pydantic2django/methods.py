"""
Support for copying methods and properties from Pydantic to Django models.
"""
from inspect import getmembers, isdatadescriptor, isroutine, signature, Parameter
from typing import Any, cast, get_origin, get_args, List, Dict, Set, Type, Union, Callable, Optional
import functools
import inspect

from django.db import models
from django.apps import apps
from django.core.exceptions import ImproperlyConfigured
from pydantic import BaseModel

from .registry import normalize_model_name

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


def is_pydantic_model_type(type_hint: Any) -> bool:
    """
    Check if a type hint represents a Pydantic model or collection of Pydantic models.
    
    Args:
        type_hint: The type hint to check
        
    Returns:
        True if the type hint is a Pydantic model or collection containing Pydantic models
    """
    if inspect.isclass(type_hint) and issubclass(type_hint, BaseModel):
        return True
        
    # Check for collections (List, Dict, Set, etc.)
    origin = get_origin(type_hint)
    if origin in (list, List, set, Set, dict, Dict):
        args = get_args(type_hint)
        return any(is_pydantic_model_type(arg) for arg in args)
        
    return False


class PydanticModelConversionError(Exception):
    """Raised when a Pydantic model cannot be converted to a Django model."""
    pass


def convert_pydantic_to_django(value: Any, app_label: str = "django_llm", return_pydantic_model: bool = False) -> Any:
    """
    Convert a Pydantic model instance or collection to Django model instance(s).
    
    Args:
        value: The value to convert
        app_label: The Django app label to use for model lookup
        return_pydantic_model: If True, return the original Pydantic model when Django model not found
        
    Returns:
        Converted Django model instance(s) or the original value if not convertible
        
    Raises:
        PydanticModelConversionError: If a Django model is not found and return_pydantic_model is False
    """
    if isinstance(value, BaseModel):
        # Get the Django model class
        model_name = normalize_model_name(value.__class__.__name__)
        try:
            django_model = cast(Type[models.Model], apps.get_model(app_label, model_name.replace("Django", "")))
            # Convert to dict and create Django instance
            data = value.model_dump()
            return django_model.objects.create(**data)
        except (LookupError, ImproperlyConfigured) as e:
            if return_pydantic_model:
                return value
            raise PydanticModelConversionError(
                f"Could not convert {value.__class__.__name__} to Django model: "
                f"Model {model_name} not found in app {app_label}. "
                f"Original error: {str(e)}"
            ) from e
            
    # Handle collections
    if isinstance(value, list):
        return [convert_pydantic_to_django(item, app_label, return_pydantic_model) for item in value]
    if isinstance(value, set):
        return {convert_pydantic_to_django(item, app_label, return_pydantic_model) for item in value}
    if isinstance(value, dict):
        return {k: convert_pydantic_to_django(v, app_label, return_pydantic_model) for k, v in value.items()}
        
    return value


def wrap_method_for_conversion(method: Callable, return_type: Any, return_pydantic_model: bool = False) -> Callable:
    """
    Wrap a method to convert its Pydantic return value to Django model instance(s).
    
    Args:
        method: The method to wrap
        return_type: The method's return type annotation
        return_pydantic_model: If True, return Pydantic models when Django models not found
        
    Returns:
        Wrapped method that converts return values
    """
    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        result = method(*args, **kwargs)
        if is_pydantic_model_type(return_type):
            return convert_pydantic_to_django(result, return_pydantic_model=return_pydantic_model)
        return result
    return wrapper


def copy_method(method: Any, return_pydantic_model: bool = False) -> Any:
    """
    Copy a method while preserving its type (regular, class, or static method).
    Wraps methods that return Pydantic models to convert their return values.

    Args:
        method: The method to copy
        return_pydantic_model: If True, return Pydantic models when Django models not found

    Returns:
        The copied method with the same type
    """
    # Get the original function
    if is_classmethod(method):
        func = method.__get__(None, object).__func__
    elif is_staticmethod(method):
        func = method.__get__(None, object)
    else:
        func = method
        
    # Check return type annotation
    sig = signature(func)
    return_type = sig.return_annotation
    
    if return_type != Parameter.empty and is_pydantic_model_type(return_type):
        # Wrap the function to handle conversion
        wrapped = wrap_method_for_conversion(func, return_type, return_pydantic_model)
        
        # Restore the method type
        if is_classmethod(method):
            return classmethod(wrapped)
        elif is_staticmethod(method):
            return staticmethod(wrapped)
        return wrapped
        
    # No conversion needed
    if is_classmethod(method):
        return classmethod(func)
    elif is_staticmethod(method):
        return staticmethod(func)
    return func


def get_methods_and_properties(model: type[BaseModel], return_pydantic_model: bool = False) -> dict[str, Any]:
    """
    Extract methods and properties from a Pydantic model.

    Args:
        model: The Pydantic model class
        return_pydantic_model: If True, return Pydantic models when Django models not found

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
            attrs[name] = copy_method(member, return_pydantic_model)
            continue

        # Handle class variables and other descriptors
        if not name.startswith("__"):
            attrs[name] = member

    return attrs


def create_django_model_with_methods(
    name: str,
    pydantic_model: type[BaseModel],
    django_attrs: dict[str, Any],
    base_classes: list[type[models.Model]] | None = None,
    return_pydantic_model: bool = False,
) -> type[models.Model]:
    """
    Create a Django model class with methods and properties from a Pydantic model.

    Args:
        name: Name for the new model class
        pydantic_model: The source Pydantic model
        django_attrs: Base attributes for the Django model (fields, Meta, etc.)
        base_classes: List of base classes for the model (defaults to [models.Model])
        return_pydantic_model: If True, return Pydantic models when Django models not found

    Returns:
        A new Django model class with copied methods and properties
    """
    # Get methods and properties from Pydantic model
    copied_attrs = get_methods_and_properties(pydantic_model, return_pydantic_model)

    # Combine with Django attributes, letting Django attrs take precedence
    attrs = {**copied_attrs, **django_attrs}

    # Use provided base classes or default to models.Model
    bases = base_classes or [models.Model]

    # Create the model class
    model = type(name, tuple(bases), attrs)
    return cast(type[models.Model], model)
