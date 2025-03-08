"""
Support for copying methods and properties from Pydantic to Django models.
"""
import functools
import inspect
import logging
from collections.abc import Callable
from inspect import Parameter, getmembers, isdatadescriptor, isroutine, signature
from typing import Any, cast, get_args, get_origin

from django.apps import apps
from django.core.exceptions import FieldDoesNotExist, ImproperlyConfigured
from django.db import models
from pydantic import BaseModel

from .registry import normalize_model_name

logger = logging.getLogger(__name__)


class PydanticModelConversionError(Exception):
    """Raised when a Pydantic model cannot be converted to a Django model."""

    pass


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
    if origin in (list, list, set, set, dict, dict):
        args = get_args(type_hint)
        return any(is_pydantic_model_type(arg) for arg in args)

    return False


def serialize_class_instance(instance: Any) -> str | dict:
    """
    Serialize a class instance to a format suitable for storage in Django.

    Args:
        instance: The class instance to serialize

    Returns:
        A string or dict representation of the instance
    """
    if hasattr(instance, "to_json"):
        return instance.to_json()
    elif hasattr(instance, "to_dict"):
        return instance.to_dict()
    elif hasattr(instance, "__str__") and instance.__class__.__str__ is not object.__str__:
        # Use __str__ if it's been overridden
        return str(instance)
    elif hasattr(instance, "__dict__"):
        # Include class name for reconstruction
        return {
            "__class__": instance.__class__.__name__,
            "__module__": instance.__class__.__module__,
            "data": instance.__dict__,
        }
    else:
        return str(instance)


def deserialize_class_instance(data: str | dict, class_registry: dict[str, type] | None = None) -> Any:
    """
    Deserialize data back into a class instance.

    Args:
        data: The serialized data
        class_registry: Optional dictionary mapping class names to their types

    Returns:
        The reconstructed class instance
    """
    if isinstance(data, str):
        return data

    if not isinstance(data, dict):
        return data

    # If it's a serialized class instance
    if "__class__" in data and "__module__" in data:
        class_name = data["__class__"]
        module_name = data["__module__"]

        # Try to get the class from the registry first
        if class_registry and class_name in class_registry:
            cls = class_registry[class_name]
        else:
            # Try to import the class
            try:
                module = __import__(module_name, fromlist=[class_name])
                cls = getattr(module, class_name)
            except (ImportError, AttributeError) as e:
                logger.warning(f"Could not reconstruct class {class_name} from module {module_name}: {e}")
                return data

        # Create a new instance and update its dict
        try:
            instance = cls.__new__(cls)
            instance.__dict__.update(data["data"])
            return instance
        except Exception as e:
            logger.warning(f"Failed to reconstruct instance of {class_name}: {e}")
            return data

    return data


def convert_pydantic_to_django(
    value: Any,
    app_label: str,
    return_pydantic_model: bool = False,
    class_registry: dict[str, type] | None = None,
) -> Any:
    """
    Convert a Pydantic model instance or collection to Django model instance(s).

    Args:
        value: The value to convert
        app_label: The Django app label to use for model lookup
        return_pydantic_model: If True, return the original Pydantic model when Django model not found
        class_registry: Optional dictionary mapping class names to their types for custom class reconstruction

    Returns:
        Converted Django model instance(s) or the original value if not convertible

    Raises:
        PydanticModelConversionError: If a Django model is not found and return_pydantic_model is False
    """
    if isinstance(value, BaseModel):
        # Get the Django model class - normalize_model_name already adds the Django prefix
        model_name = normalize_model_name(value.__class__.__name__)
        logger.debug(f"Converting Pydantic model {value.__class__.__name__} to Django model {model_name}")
        try:
            django_model = cast(
                type[models.Model],
                apps.get_model(app_label, model_name),
            )
            logger.debug(f"Found Django model class: {django_model}")
            # Convert to dict and handle nested models
            data = value.model_dump()
            converted_data = {}
            m2m_data = {}

            # First, create all nested models
            for field_name, field_value in data.items():
                try:
                    field = django_model._meta.get_field(field_name)
                    logger.debug(f"Processing field {field_name} of type {type(field)}")

                    # Handle class instances that aren't Pydantic models
                    if isinstance(field_value, object) and not isinstance(
                        field_value,
                        (BaseModel, dict, list, set, str, int, float, bool, type(None)),
                    ):
                        converted_data[field_name] = serialize_class_instance(field_value)
                        continue

                    if isinstance(field_value, (BaseModel, dict)):
                        logger.debug(f"Found nested model in field {field_name}")
                        # Get the related model class for foreign key or one-to-one fields
                        if isinstance(field, (models.ForeignKey, models.OneToOneField)):
                            # Convert nested model data
                            if isinstance(field_value, BaseModel):
                                nested_instance = convert_pydantic_to_django(
                                    field_value,
                                    app_label,
                                    return_pydantic_model,
                                    class_registry,
                                )
                            else:
                                # Try to create the instance directly
                                nested_model_name = normalize_model_name(field.related_model.__name__)
                                logger.debug(f"Looking up Django model for nested model {nested_model_name}")
                                nested_django_model = apps.get_model(app_label, nested_model_name.replace("Django", ""))
                                logger.debug(f"Creating nested instance with data: {field_value}")
                                nested_instance = nested_django_model._default_manager.create(**field_value)
                            logger.debug(f"Created nested instance with ID {nested_instance.pk}")
                            converted_data[field_name] = nested_instance
                        else:
                            if return_pydantic_model:
                                converted_data[field_name] = field_value
                            else:
                                raise PydanticModelConversionError(f"Field {field_name} is not a relation field")
                    elif isinstance(field_value, (list, set)):
                        if isinstance(field, models.ManyToManyField):
                            logger.debug(f"Processing many-to-many field {field_name}")
                            # Handle many-to-many relationships after creation
                            converted_list = []
                            for item in field_value:
                                if isinstance(item, (BaseModel, dict)):
                                    logger.debug("Converting M2M item")
                                    if isinstance(item, BaseModel):
                                        nested_instance = convert_pydantic_to_django(
                                            item,
                                            app_label,
                                            return_pydantic_model,
                                            class_registry,
                                        )
                                    else:
                                        # Try to create the instance directly
                                        nested_model_name = normalize_model_name(field.related_model.__name__)
                                        nested_django_model = apps.get_model(
                                            app_label,
                                            nested_model_name.replace("Django", ""),
                                        )
                                        logger.debug(f"Creating M2M instance with data: {item}")
                                        nested_instance = nested_django_model._default_manager.create(**item)
                                    logger.debug(f"Created M2M instance with ID {nested_instance.pk}")
                                    converted_list.append(nested_instance)
                                else:
                                    converted_list.append(item)
                            m2m_data[field_name] = converted_list
                        else:
                            converted_data[field_name] = field_value
                    else:
                        converted_data[field_name] = field_value
                except (AttributeError, FieldDoesNotExist) as e:
                    logger.warning(f"Error processing field {field_name}: {str(e)}")
                    converted_data[field_name] = field_value

            logger.debug(f"Creating main instance with data: {converted_data}")
            # Create the main model with converted nested models
            instance = django_model._default_manager.create(**converted_data)

            # Set many-to-many relationships
            for field_name, field_value in m2m_data.items():
                logger.debug(f"Setting M2M relationship {field_name} with values: {field_value}")
                getattr(instance, field_name).set(field_value)

            logger.debug(f"Successfully created instance with ID {instance.pk}")
            return instance

        except (LookupError, ImproperlyConfigured) as e:
            logger.error(f"Error converting model: {str(e)}")
            if return_pydantic_model:
                return value
            raise PydanticModelConversionError(
                f"Could not convert {value.__class__.__name__} to Django model: "
                f"Model {model_name} not found in app {app_label}. "
                f"Original error: {str(e)}"
            ) from e

    # Handle collections
    if isinstance(value, list):
        return [convert_pydantic_to_django(item, app_label, return_pydantic_model, class_registry) for item in value]
    if isinstance(value, set):
        return {convert_pydantic_to_django(item, app_label, return_pydantic_model, class_registry) for item in value}
    if isinstance(value, dict):
        return {
            k: convert_pydantic_to_django(v, app_label, return_pydantic_model, class_registry) for k, v in value.items()
        }

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
            # Get the app_label from the instance's Meta class
            instance = args[0] if args else None
            app_label = instance._meta.app_label if instance else "tests"
            return convert_pydantic_to_django(result, app_label=app_label, return_pydantic_model=return_pydantic_model)
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

    # Create Meta class
    class Meta:
        app_label = "tests"
        db_table = f"tests_{name.lower()}"

    # Combine with Django attributes, letting Django attrs take precedence
    attrs = {
        "__module__": pydantic_model.__module__,
        "Meta": Meta,
        **copied_attrs,
        **django_attrs,
    }

    # Use provided base classes or default to models.Model
    bases = base_classes or [models.Model]

    # Create the model class
    model = type(name, tuple(bases), attrs)
    django_model = cast(type[models.Model], model)

    # Register the model with Django
    from django.apps import apps

    try:
        apps.get_registered_model("tests", name)
    except LookupError:
        apps.register_model("tests", django_model)

    # Skip table creation in tests since we're using an in-memory database
    # The tables will be created by Django's test runner
    return django_model
