import datetime
import inspect
import logging
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Optional,
    Union,
    get_args,
    get_origin,
)
from uuid import UUID

from django.db import models

# Pydantic types often used in mappings
from pydantic import BaseModel, EmailStr, IPvAnyAddress, Json
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

# Import the core definition
from ..core.defs import PythonType, TypeMappingDefinition

# Import FieldConversionResult
from ..core.factories import FieldConversionResult

# Import necessary utils
from ..pydantic.utils.introspection import is_pydantic_model_field_optional

logger = logging.getLogger(__name__)


def get_default_max_length(field_name: str, field_type: type[models.Field]) -> Optional[int]:
    """
    Determine a default max_length for specific Django field types based on Python type names.
    This function encapsulates the logic previously in TypeMappingDefinition.__post_init__.

    Args:
        field_name: The name of the field (used for logging/context)
        field_type: The Django field type (e.g., models.CharField)

    Returns:
        A default max_length integer or None if no default is applicable.
    """
    if field_type == models.CharField:
        # Basic default for CharFields if no specific type match
        return 255
    # Add more specific defaults based on python_type if needed here
    # For example:
    # if python_type_name == 'EmailStr': return 254
    return None


class TypeMapper:
    """
    Handles bidirectional mapping between Python/Pydantic types and Django field types.

    This class provides static methods for converting between Python types and Django field types,
    as well as determining appropriate field attributes like max_length.
    """

    # Define all type mappings as TypeMappingDefinition instances
    TYPE_MAPPINGS: list[TypeMappingDefinition] = [
        # Simple Types - Order matters for issubclass checks (bool before int)
        TypeMappingDefinition(str, models.CharField, max_length=255),
        TypeMappingDefinition(bool, models.BooleanField),  # Moved before int
        TypeMappingDefinition(int, models.IntegerField),
        TypeMappingDefinition(float, models.FloatField),
        TypeMappingDefinition(Decimal, models.DecimalField, field_kwargs={"max_digits": 19, "decimal_places": 10}),
        TypeMappingDefinition(datetime.datetime, models.DateTimeField),
        TypeMappingDefinition(datetime.date, models.DateField),
        TypeMappingDefinition(datetime.time, models.TimeField),
        TypeMappingDefinition(datetime.timedelta, models.DurationField),
        TypeMappingDefinition(UUID, models.UUIDField),
        TypeMappingDefinition(bytes, models.BinaryField),
        TypeMappingDefinition(Path, models.CharField, max_length=255),  # Store Path as CharField
        # Pydantic Specific Types
        TypeMappingDefinition(EmailStr, models.EmailField, max_length=254),  # Define directly
        TypeMappingDefinition(IPvAnyAddress, models.GenericIPAddressField),
        # Complex Types (handled as JSON by default)
        TypeMappingDefinition(dict, models.JSONField),  # Define directly
        TypeMappingDefinition(list, models.JSONField),
        TypeMappingDefinition(set, models.JSONField),
        TypeMappingDefinition(tuple, models.JSONField),
        TypeMappingDefinition(Json, models.JSONField),
        # Relationship Types are handled dynamically in get_mapping_for_type
    ]

    class UnsupportedTypeError(Exception):
        """Custom exception for unsupported types."""

        pass

    @classmethod
    def get_mapping_for_type(cls, python_type: Any) -> Optional[TypeMappingDefinition]:
        """
        Get the TypeMappingDefinition that matches the given Python type.

        Args:
            python_type: The Python type to find a mapping for.

        Returns:
            The matching TypeMappingDefinition or None if no match is found.
        """
        logger.debug(
            f"Searching for mapping for type: {python_type} (Origin: {get_origin(python_type)}, Args: {get_args(python_type)})"
        )

        # Handle Enums specifically
        try:
            if inspect.isclass(python_type) and issubclass(python_type, Enum):
                logger.debug(f"Type {python_type} is an Enum, generating enum field mapping.")
                return cls.enum_field(python_type)
        except TypeError:
            pass  # Not a class, continue checking

        # Handle Optional[T] -> Union[T, None]
        origin = get_origin(python_type)
        args = get_args(python_type)
        actual_type = python_type

        if origin is Union and type(None) in args and len(args) == 2:
            actual_type = next((arg for arg in args if arg is not type(None)), None)
            logger.debug(f"Type is Optional, searching for mapping for underlying type: {actual_type}")
            if actual_type is None:
                logger.warning(f"Cannot map Optional[NoneType] for type: {python_type}")
                return None
            # Re-evaluate origin and args based on the unwrapped actual_type
            origin = get_origin(actual_type)
            args = get_args(actual_type)

        # Handle Pydantic models specifically (potential relationships)
        try:
            if inspect.isclass(actual_type) and issubclass(actual_type, BaseModel):
                logger.debug(f"Type {actual_type} is a Pydantic BaseModel, creating ForeignKey mapping.")
                # Assume direct BaseModel reference is a ForeignKey
                return cls.foreign_key(actual_type)
        except TypeError:
            pass  # Not a class, continue checking

        # Handle List[BaseModel] or Dict[str, BaseModel] for ManyToMany
        if origin in (list, dict) and args:
            # For dict, check the value type (args[1]); for list, check the item type (args[0])
            inner_type_index = 1 if origin is dict else 0
            if len(args) > inner_type_index:
                inner_type = args[inner_type_index]
                # Unwrap Optional for inner type if needed
                inner_origin = get_origin(inner_type)
                inner_args = get_args(inner_type)
                if inner_origin is Union and type(None) in inner_args and len(inner_args) == 2:
                    inner_type = next((arg for arg in inner_args if arg is not type(None)), None)

                try:
                    if inspect.isclass(inner_type) and issubclass(inner_type, BaseModel):
                        logger.debug(f"Type {python_type} is Collection[BaseModel], creating ManyToMany mapping.")
                        return cls.many_to_many(python_type)  # Pass original type
                except TypeError:
                    pass  # Inner type not a class or suitable for issubclass

        # Iterate through predefined mappings
        for mapping in cls.TYPE_MAPPINGS:
            logger.debug(f"Checking mapping: {mapping.python_type} -> {mapping.django_field.__name__}")
            if mapping.matches_type(actual_type):
                logger.debug(f"Found matching mapping for {actual_type}: {mapping}")
                return mapping

        # Fallback for unhandled collection types like Dict, Set, Tuple containing non-BaseModel types
        # These often map well to JSONField
        if origin in (dict, list, set, tuple):
            logger.debug(f"Type {python_type} is a collection, falling back to JSONField mapping.")
            return cls.json_field(actual_type)

        # If no mapping found after all checks
        # Log using actual_type for clarity if it was unwrapped
        logger.warning(f"No direct mapping found for type: {actual_type}. It might be handled contextually.")
        return None

    # --- Class methods moved from TypeMappingDefinition --- #

    @classmethod
    def char_field(cls, python_type: PythonType, max_length: int = 255) -> "TypeMappingDefinition":
        """Create a CharField mapping with the specified max_length."""
        return TypeMappingDefinition(
            python_type=python_type,
            django_field=models.CharField,
            max_length=max_length,
        )

    @classmethod
    def text_field(cls, python_type: PythonType) -> "TypeMappingDefinition":
        """Create a TextField mapping."""
        return TypeMappingDefinition(python_type=python_type, django_field=models.TextField)

    @classmethod
    def json_field(cls, python_type: PythonType) -> "TypeMappingDefinition":
        """Create a JSONField mapping."""
        return TypeMappingDefinition(python_type=python_type, django_field=models.JSONField)

    @classmethod
    def email_field(cls, python_type: PythonType = EmailStr, max_length: int = 254) -> "TypeMappingDefinition":
        """Create an EmailField mapping with the specified max_length."""
        return TypeMappingDefinition(
            python_type=python_type,
            django_field=models.EmailField,
            max_length=max_length,
        )

    @classmethod
    def foreign_key(cls, python_type: PythonType) -> "TypeMappingDefinition":
        """Create a ForeignKey mapping."""
        return TypeMappingDefinition(
            python_type=python_type,
            django_field=models.ForeignKey,
            is_relationship=True,
            # on_delete=models.CASCADE, # Set in factory based on nullability
        )

    @classmethod
    def many_to_many(cls, python_type: PythonType) -> "TypeMappingDefinition":
        """Create a ManyToManyField mapping."""
        # Expects python_type to be like List[MyModel]
        return TypeMappingDefinition(
            python_type=python_type,
            django_field=models.ManyToManyField,
            is_relationship=True,
        )

    @classmethod
    def enum_field(cls, python_type: type[Enum]) -> TypeMappingDefinition:
        """Create a CharField or IntegerField mapping for an Enum based on its value types."""
        try:
            # Get enum values safely
            members = list(python_type)
            enum_values = [member.value for member in members]
            choices = [(member.value, member.name) for member in members]
        except Exception as e:
            logger.error(f"Could not process enum {python_type.__name__}: {e}")
            raise ValueError(f"Unsupported enum type: {python_type.__name__}. Could not extract members/values.") from e

        if not enum_values:
            raise ValueError(f"Enum {python_type.__name__} has no members.")

        # Determine the type of the enum values
        if all(isinstance(val, int) for val in enum_values):
            # Integer enum
            logger.debug(f"Enum {python_type.__name__} detected as IntegerField with choices.")
            return TypeMappingDefinition(
                python_type=python_type,
                django_field=models.IntegerField,
                field_kwargs={"choices": choices},
            )
        elif all(isinstance(val, str) for val in enum_values):
            # String enum
            max_length = max(len(str(val)) for val in enum_values) if enum_values else 10
            logger.debug(f"Enum {python_type.__name__} detected as CharField(max_length={max_length}) with choices.")
            return TypeMappingDefinition(
                python_type=python_type,
                django_field=models.CharField,
                max_length=max_length,
                field_kwargs={"choices": choices},
            )
        else:
            # Mixed types or other types - log warning and use TextField
            logger.warning(
                f"Enum {python_type.__name__} has mixed or unsupported value types ({set(type(v) for v in enum_values)}). "
                f"Falling back to TextField with string choices."
            )
            string_choices = [(str(member.value), member.name) for member in members]
            return TypeMappingDefinition(
                python_type=python_type,
                django_field=models.TextField,
                field_kwargs={"choices": string_choices},
            )

    # --- Other static methods --- #

    @classmethod
    def filter_by_django_field(cls, django_field: type[models.Field]) -> list[TypeMappingDefinition]:
        """Filter mappings by Django field type."""
        return [m for m in cls.TYPE_MAPPINGS if m.django_field == django_field]

    @classmethod
    def is_type_supported(cls, python_type: Any) -> bool:
        """Check if a Python type is directly supported by a mapping."""
        return cls.get_mapping_for_type(python_type) is not None

    @classmethod
    def get_all_mappings(cls) -> list[TypeMappingDefinition]:
        """Return all defined type mappings."""
        return cls.TYPE_MAPPINGS

    @classmethod
    def get_field_attributes(cls, python_type: Any) -> dict[str, Any]:
        """
        Get default Django field attributes based on the Python type.
        Currently focuses on null/blank based on Optional.
        """
        attributes = {}
        # Use the utility function from field_utils (will be moved)
        if is_pydantic_model_field_optional(python_type):
            attributes["null"] = True
            attributes["blank"] = True  # Often makes sense if null=True
        else:
            attributes["null"] = False
            attributes["blank"] = False  # Default for required fields

        # Add more attribute logic here (e.g., based on FieldInfo defaults)
        return attributes

    @classmethod
    def get_max_length(cls, field_name: str, field_type: type[models.Field]) -> Optional[int]:
        """
        Proxy method to get default max length.
        Delegates to the standalone function.
        """
        return get_default_max_length(field_name, field_type)

    @classmethod
    def instantiate_django_field(
        cls,
        mapping_definition: TypeMappingDefinition,
        field_info: FieldInfo,
        result: FieldConversionResult,
        field_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Instantiate the Django field based on the mapping definition and provided kwargs.
        This incorporates logic previously in TypeMappingDefinition.get_django_field.
        Modifies the passed FieldConversionResult object with the instantiated field
        and the final kwargs used.

        Args:
            mapping_definition: The TypeMappingDefinition describing the mapping.
            field_info: The Pydantic FieldInfo for additional context (e.g., default values).
            result: The FieldConversionResult object to populate.
            field_kwargs: Additional kwargs to pass to the field constructor (e.g., from factory).
        """
        final_kwargs = mapping_definition.field_kwargs.copy()  # Start with mapping defaults

        # Add/override with explicitly passed kwargs
        if field_kwargs:
            final_kwargs.update(field_kwargs)

        # -- Apply logic based on Pydantic FieldInfo --
        # Nullability based on Optional status
        is_optional = is_pydantic_model_field_optional(field_info.annotation)
        final_kwargs["null"] = is_optional
        # Often makes sense to set blank=True if null=True
        final_kwargs["blank"] = is_optional

        # Default value from Pydantic
        if field_info.default is not PydanticUndefined:
            # Only set default if it's not None when the field is already nullable
            if not (is_optional and field_info.default is None):
                final_kwargs["default"] = field_info.default
        elif field_info.default_factory is not None:
            # Django doesn't directly support default_factory like Pydantic
            # We might need special handling or ignore it for DB defaults
            logger.warning(
                f"Field {field_info} has default_factory - this is not directly translated to Django default."
            )

        # Max length for CharField/TextField from Pydantic
        # Pydantic v2 uses metadata for max_length
        pydantic_max_length = None
        if hasattr(field_info, "metadata"):
            for item in field_info.metadata:
                if hasattr(item, "max_length"):
                    pydantic_max_length = item.max_length
                    break

        if mapping_definition.django_field in (models.CharField, models.TextField):
            # Use Pydantic max_length if provided
            if pydantic_max_length is not None:
                final_kwargs["max_length"] = pydantic_max_length
            # Otherwise, use the mapping definition's max_length (e.g., for Path)
            elif mapping_definition.max_length is not None and "max_length" not in final_kwargs:
                final_kwargs["max_length"] = mapping_definition.max_length
            # Fallback for CharField if no length specified anywhere
            elif mapping_definition.django_field == models.CharField and "max_length" not in final_kwargs:
                logger.debug(
                    f"No max_length specified for CharField derived from {mapping_definition.python_type}, defaulting to 255."
                )
                final_kwargs["max_length"] = 255

        # Decimal field precision from Pydantic
        if mapping_definition.django_field == models.DecimalField:
            max_digits = final_kwargs.get("max_digits")
            decimal_places = final_kwargs.get("decimal_places")
            # Pydantic v2 uses metadata
            pydantic_max_digits = None
            pydantic_decimal_places = None
            if hasattr(field_info, "metadata"):
                for item in field_info.metadata:
                    if hasattr(item, "max_digits"):
                        pydantic_max_digits = item.max_digits
                    if hasattr(item, "decimal_places"):
                        pydantic_decimal_places = item.decimal_places

            if pydantic_max_digits is not None:
                final_kwargs["max_digits"] = pydantic_max_digits
            elif max_digits is None:
                final_kwargs["max_digits"] = 19  # Default if not set

            if pydantic_decimal_places is not None:
                final_kwargs["decimal_places"] = pydantic_decimal_places
            elif decimal_places is None:
                final_kwargs["decimal_places"] = 10  # Default if not set

        # -- Relationship specific kwargs --
        if mapping_definition.is_relationship:
            # on_delete (set based on nullability)
            if mapping_definition.django_field in (models.ForeignKey, models.OneToOneField):
                final_kwargs["on_delete"] = models.SET_NULL if final_kwargs.get("null", False) else models.CASCADE

            # related_name needs to be handled in the factory based on conflicts
            if "related_name" in final_kwargs:
                del final_kwargs["related_name"]  # Remove placeholder if present

            # 'to' field needs to be added in the factory
            if "to" in final_kwargs:
                del final_kwargs["to"]

        # Log the final kwargs before instantiation for debugging
        field_name = getattr(field_info, "alias", "?")  # Use alias if available for logging
        logger.debug(
            f"Instantiating {mapping_definition.django_field.__name__} for field '{field_name}' with final kwargs: {final_kwargs}"
        )

        try:
            # Instantiate the field and store it and kwargs in the result object
            result.django_field = mapping_definition.django_field(**final_kwargs)
            result.field_kwargs = final_kwargs  # Store the final kwargs

        except Exception as e:
            # Capture field name here for error message
            field_name_err = getattr(field_info, "alias", "?")
            logger.error(
                f"INSTANTIATION FAILED. Class: {mapping_definition.django_field.__name__}, Kwargs: {final_kwargs}",
                exc_info=True,
            )
            # Set error on the result object instead of raising
            result.error_str = f"Failed to instantiate Django field {mapping_definition.django_field.__name__} for '{field_name_err}'. Kwargs: {final_kwargs}. Error: {e}"
            result.django_field = None  # Ensure field is None on error
            result.field_kwargs = {}  # Clear kwargs on error

        # Return type is None, modification happens on result object
