"""
Provides a bidirectional mapping system between Django fields and Pydantic types/FieldInfo.

This module defines:
- `TypeMappingUnit`: Base class for defining a single bidirectional mapping rule.
- Specific subclasses of `TypeMappingUnit` for various field types.
- `BidirectionalTypeMapper`: A central registry and entry point for performing conversions.

It utilizes:
- `core.typing.TypeHandler`: For introspecting Pydantic type hints.
- `core.relationships.RelationshipConversionAccessor`: For resolving model-to-model relationships.
"""

import datetime
import inspect
import logging
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Optional, TypeVar, get_args, get_origin, Literal
from uuid import UUID

from django.db import models
from pydantic import BaseModel, EmailStr, HttpUrl, IPvAnyAddress
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

# Use absolute import path for relationships - Linter still complains, define directly
# from pydantic2django.core.relationships import (
#     RelationshipConversionAccessor, PydanticRelatedFieldType, PydanticListOfRelated
# )
from pydantic2django.core.relationships import RelationshipConversionAccessor

from .typing import TypeHandler

logger = logging.getLogger(__name__)

# Define relationship placeholders directly to avoid linter issues
PydanticRelatedFieldType = Any
PydanticListOfRelated = list[Any]

# Helper type variable - only used for annotation within TypeMappingUnit subclasses
T_DjangoField = TypeVar("T_DjangoField", bound=models.Field)
T_PydanticType = Any  # Pydantic types can be complex


class MappingError(Exception):
    """Custom exception for mapping errors."""

    pass


class TypeMappingUnit:
    """Base class defining a bidirectional mapping between a Python type and a Django Field."""

    python_type: type[T_PydanticType]
    django_field_type: type[models.Field]  # Use base class here

    def __init_subclass__(cls, **kwargs):
        """Ensure subclasses define the required types."""
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "python_type") or not hasattr(cls, "django_field_type"):
            raise NotImplementedError(
                "Subclasses of TypeMappingUnit must define 'python_type' and 'django_field_type' class attributes."
            )

    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        """Generate Django field constructor kwargs from Pydantic FieldInfo."""
        kwargs = {}
        if field_info:
            # Map common attributes
            if field_info.title:
                kwargs["verbose_name"] = field_info.title
            if field_info.description:
                kwargs["help_text"] = field_info.description
            if field_info.default is not PydanticUndefined and field_info.default is not None:
                # Skip None default if it matches null=True implicitly
                # Django doesn't handle callable defaults easily here
                if not callable(field_info.default):
                    kwargs["default"] = field_info.default
            elif field_info.default_factory is not None:
                logger.warning(
                    f"Pydantic field has default_factory, which is not directly mappable to Django default. "
                    f"Ignoring for {getattr(field_info, 'name', 'unknown field')}."
                )
            # Note: Frozen, ge, le etc. are validation rules, map separately if needed
        return kwargs

    def django_to_pydantic_field_info_kwargs(self, dj_field: models.Field) -> dict[str, Any]:
        """Generate Pydantic FieldInfo kwargs from a Django field instance."""
        kwargs = {}
        # Map common attributes
        if dj_field.verbose_name:
            kwargs["title"] = str(dj_field.verbose_name).capitalize()
        if dj_field.help_text:
            kwargs["description"] = str(dj_field.help_text)

        dj_default = dj_field.get_default()
        if dj_default is not models.fields.NOT_PROVIDED and dj_default is not None:
            if not callable(dj_default):
                kwargs["default"] = dj_default
            else:
                logger.debug(
                    f"Django field '{dj_field.name}' has a callable default ({dj_default}), "
                    "not mapping to Pydantic default."
                )
        elif dj_field.null:
            kwargs["default"] = None  # Explicit None default if nullable

        # Handle AutoField PKs -> frozen=True, default=None (even if not null in DB)
        is_auto_pk = dj_field.primary_key and isinstance(
            dj_field, (models.AutoField, models.BigAutoField, models.SmallAutoField)
        )
        if is_auto_pk:
            kwargs["frozen"] = True
            kwargs["default"] = None  # Ensure default is None for auto PKs

        return kwargs


# --- Specific Mapping Units --- #


class IntFieldMapping(TypeMappingUnit):
    python_type = int
    django_field_type = models.IntegerField

    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        return super().pydantic_to_django_kwargs(py_type, field_info)


class BigIntFieldMapping(TypeMappingUnit):
    python_type = int
    django_field_type = models.BigIntegerField

    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        return super().pydantic_to_django_kwargs(py_type, field_info)


class SmallIntFieldMapping(TypeMappingUnit):
    python_type = int
    django_field_type = models.SmallIntegerField

    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        return super().pydantic_to_django_kwargs(py_type, field_info)


class PositiveIntFieldMapping(TypeMappingUnit):
    python_type = int
    django_field_type = models.PositiveIntegerField

    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        return super().pydantic_to_django_kwargs(py_type, field_info)

    def django_to_pydantic_field_info_kwargs(self, dj_field: models.Field) -> dict[str, Any]:
        kwargs = super().django_to_pydantic_field_info_kwargs(dj_field)
        kwargs["ge"] = 0
        return kwargs

    # Pydantic to Django: No direct mapping for `ge` constraint to Django model field itself.


class PositiveSmallIntFieldMapping(PositiveIntFieldMapping):
    django_field_type = models.PositiveSmallIntegerField

    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        return super().pydantic_to_django_kwargs(py_type, field_info)


class PositiveBigIntFieldMapping(PositiveIntFieldMapping):
    django_field_type = models.PositiveBigIntegerField

    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        return super().pydantic_to_django_kwargs(py_type, field_info)


class AutoFieldMapping(TypeMappingUnit):
    python_type = int
    django_field_type = models.AutoField
    # Note: PK handling (frozen=True, default=None) is done in the base django_to_pydantic_field_info_kwargs

    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        return super().pydantic_to_django_kwargs(py_type, field_info)


class BigAutoFieldMapping(AutoFieldMapping):
    django_field_type = models.BigAutoField

    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        return super().pydantic_to_django_kwargs(py_type, field_info)


class SmallAutoFieldMapping(AutoFieldMapping):
    django_field_type = models.SmallAutoField

    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        return super().pydantic_to_django_kwargs(py_type, field_info)


class BoolFieldMapping(TypeMappingUnit):
    python_type = bool
    django_field_type = models.BooleanField

    # Add default handling
    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        kwargs = super().pydantic_to_django_kwargs(py_type, field_info)
        # BooleanFields often default to None/NULL in DB without explicit default,
        # but Pydantic bool defaults to False. Align by setting Django default=False.
        if "default" not in kwargs:
            kwargs["default"] = False
        return kwargs


class FloatFieldMapping(TypeMappingUnit):
    python_type = float
    django_field_type = models.FloatField

    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        return super().pydantic_to_django_kwargs(py_type, field_info)


class StrFieldMapping(TypeMappingUnit):  # Base for Char/Text
    python_type = str
    django_field_type = models.CharField  # Default to CharField

    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        kwargs = super().pydantic_to_django_kwargs(py_type, field_info)
        pyd_max_length = None
        if field_info and hasattr(field_info, "metadata"):
            for item in field_info.metadata:
                if hasattr(item, "max_length"):
                    pyd_max_length = item.max_length
                    break

        if pyd_max_length is not None:
            kwargs["max_length"] = pyd_max_length
        elif "max_length" not in kwargs:
            # Default max_length for CharField if not specified
            if self.django_field_type == models.CharField:
                kwargs["max_length"] = 255
        return kwargs

    def django_to_pydantic_field_info_kwargs(self, dj_field: models.Field) -> dict[str, Any]:
        kwargs = super().django_to_pydantic_field_info_kwargs(dj_field)
        if isinstance(dj_field, models.CharField) and dj_field.max_length is not None:
            kwargs["max_length"] = dj_field.max_length
        # Min length / pattern could be added by inspecting dj_field.validators
        return kwargs


class TextFieldMapping(StrFieldMapping):
    django_field_type = models.TextField

    # TextField doesn't typically have max_length in Django by default
    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        kwargs = super().pydantic_to_django_kwargs(py_type, field_info)
        # Check metadata for max_length
        pyd_max_length = None
        if field_info and hasattr(field_info, "metadata"):
            for item in field_info.metadata:
                if hasattr(item, "max_length"):
                    pyd_max_length = item.max_length
                    break
        # Remove max_length if mapping str to TextField unless explicitly set in field_info metadata
        if pyd_max_length is None:
            kwargs.pop("max_length", None)
        return kwargs


class EmailFieldMapping(StrFieldMapping):
    python_type = EmailStr
    django_field_type = models.EmailField

    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        kwargs = super().pydantic_to_django_kwargs(py_type, field_info)
        # Ensure default max_length for EmailField if not specified via metadata
        pyd_max_length = None
        if field_info and hasattr(field_info, "metadata"):
            for item in field_info.metadata:
                if hasattr(item, "max_length"):
                    pyd_max_length = item.max_length
                    break
        if pyd_max_length is None and "max_length" not in kwargs:
            kwargs["max_length"] = 254
        return kwargs


class SlugFieldMapping(StrFieldMapping):
    python_type = str  # Pydantic doesn't have a specific Slug type
    django_field_type = models.SlugField

    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        return super().pydantic_to_django_kwargs(py_type, field_info)


class URLFieldMapping(StrFieldMapping):
    python_type = HttpUrl
    django_field_type = models.URLField

    # Add default max_length for URLField
    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        kwargs = super().pydantic_to_django_kwargs(py_type, field_info)
        # Ensure default max_length for URLField if not specified via metadata
        pyd_max_length = None
        if field_info and hasattr(field_info, "metadata"):
            for item in field_info.metadata:
                if hasattr(item, "max_length"):
                    pyd_max_length = item.max_length
                    break
        if pyd_max_length is None and "max_length" not in kwargs:
            kwargs["max_length"] = 200  # Default for URLField matching test
        return kwargs


class IPAddressFieldMapping(StrFieldMapping):
    python_type = IPvAnyAddress  # Pydantic's type covers v4/v6
    django_field_type = models.GenericIPAddressField

    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        return super().pydantic_to_django_kwargs(py_type, field_info)


class FilePathFieldMapping(StrFieldMapping):
    python_type = Path
    django_field_type = models.FilePathField

    # Add default max_length for FilePathField
    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        kwargs = super().pydantic_to_django_kwargs(py_type, field_info)
        if "max_length" not in kwargs:
            kwargs["max_length"] = 100  # Default for FilePathField
        return kwargs


class FileFieldMapping(StrFieldMapping):
    # Map File/Image fields to str (URL/path) by default in Pydantic
    python_type = str
    django_field_type = models.FileField

    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        return super().pydantic_to_django_kwargs(py_type, field_info)


class ImageFieldMapping(FileFieldMapping):
    django_field_type = models.ImageField

    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        return super().pydantic_to_django_kwargs(py_type, field_info)


class UUIDFieldMapping(TypeMappingUnit):
    python_type = UUID
    django_field_type = models.UUIDField

    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        return super().pydantic_to_django_kwargs(py_type, field_info)


class DateTimeFieldMapping(TypeMappingUnit):
    python_type = datetime.datetime
    django_field_type = models.DateTimeField
    # auto_now / auto_now_add are Django specific, Pydantic uses default_factory typically
    # which we aren't mapping reliably from Django defaults.

    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        return super().pydantic_to_django_kwargs(py_type, field_info)


class DateFieldMapping(TypeMappingUnit):
    python_type = datetime.date
    django_field_type = models.DateField

    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        return super().pydantic_to_django_kwargs(py_type, field_info)


class TimeFieldMapping(TypeMappingUnit):
    python_type = datetime.time
    django_field_type = models.TimeField

    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        return super().pydantic_to_django_kwargs(py_type, field_info)


class DurationFieldMapping(TypeMappingUnit):
    python_type = datetime.timedelta
    django_field_type = models.DurationField

    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        return super().pydantic_to_django_kwargs(py_type, field_info)


class DecimalFieldMapping(TypeMappingUnit):
    python_type = Decimal
    django_field_type = models.DecimalField

    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        kwargs = super().pydantic_to_django_kwargs(py_type, field_info)
        # Pydantic v2 uses metadata for constraints
        max_digits = None
        decimal_places = None
        if field_info and hasattr(field_info, "metadata"):
            for item in field_info.metadata:
                if hasattr(item, "max_digits"):
                    max_digits = item.max_digits
                if hasattr(item, "decimal_places"):
                    decimal_places = item.decimal_places

        kwargs["max_digits"] = max_digits if max_digits is not None else 19  # Default
        kwargs["decimal_places"] = decimal_places if decimal_places is not None else 10  # Default
        return kwargs

    def django_to_pydantic_field_info_kwargs(self, dj_field: models.Field) -> dict[str, Any]:
        kwargs = super().django_to_pydantic_field_info_kwargs(dj_field)
        if isinstance(dj_field, models.DecimalField):
            if dj_field.max_digits is not None:
                kwargs["max_digits"] = dj_field.max_digits
            if dj_field.decimal_places is not None:
                kwargs["decimal_places"] = dj_field.decimal_places
        return kwargs


class BinaryFieldMapping(TypeMappingUnit):
    python_type = bytes
    django_field_type = models.BinaryField

    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        return super().pydantic_to_django_kwargs(py_type, field_info)


class JsonFieldMapping(TypeMappingUnit):
    # Map complex Python types (dict, list, tuple, set, Json) to JSONField
    # Map Django JSONField to Pydantic Any (or Json for stricter validation)
    python_type = Any  # Or could use Json type: from pydantic import Json
    django_field_type = models.JSONField

    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        return super().pydantic_to_django_kwargs(py_type, field_info)


class EnumFieldMapping(TypeMappingUnit):
    python_type = Enum  # Placeholder, actual enum type determined dynamically
    django_field_type = models.CharField  # Default, could be IntegerField

    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        kwargs = super().pydantic_to_django_kwargs(py_type, field_info)
        # Use the passed py_type directly
        # python_type = field_info.annotation if field_info else None # Old way
        origin_type = get_origin(py_type)

        if origin_type is Literal:
            literal_args = get_args(py_type)
            if not literal_args:
                logger.warning("Literal type has no arguments.")
                return kwargs  # Return base kwargs

            # Assume CharField for Literal, calculate max_length and choices
            choices = [(str(arg), str(arg)) for arg in literal_args]
            max_length = max(len(str(arg)) for arg in literal_args) if literal_args else 10

            # Directly modify and return kwargs here for Literal
            instance_kwargs = super().pydantic_to_django_kwargs(py_type, field_info)  # Get base kwargs again for safety
            instance_kwargs["max_length"] = max_length
            instance_kwargs["choices"] = choices
            # Explicitly set field type in instance kwargs? No, the mapper should handle type.
            # self.django_field_type = models.CharField # This modifies the class attribute, affecting subsequent calls!
            return instance_kwargs  # Return the specifically constructed kwargs for Literal

        elif py_type and inspect.isclass(py_type) and issubclass(py_type, Enum):
            # This block is now only for actual Enums
            members = list(py_type)
            enum_values = [member.value for member in members]
            choices = [(member.value, member.name) for member in members]
            kwargs["choices"] = choices

            if all(isinstance(val, int) for val in enum_values):
                self.django_field_type = models.IntegerField  # dynamically change!
                # Need to ensure IntegerFieldMapping handles choices if needed, or handle here
            else:
                self.django_field_type = models.CharField  # Default back to Char
                max_length = max(len(str(val)) for val in enum_values) if enum_values else 10
                kwargs["max_length"] = max_length
        else:
            logger.warning("Enum mapping used but type is not an Enum?")
        return kwargs

    def django_to_pydantic_field_info_kwargs(self, dj_field: models.Field) -> dict[str, Any]:
        kwargs = super().django_to_pydantic_field_info_kwargs(dj_field)
        # Pydantic can often infer Enum from type hint + choices are not needed in FieldInfo
        # We might need to construct the actual Enum type here or return metadata
        # For now, just return base info. The calling generator needs to handle Enum creation.
        if dj_field.choices:
            # Potentially add choices to schema_extra? Or let Pydantic handle it.
            # kwargs['json_schema_extra'] = {'choices': dj_field.choices}
            pass
        return kwargs


# Relationship Placeholders (Logic primarily in BidirectionalTypeMapper)
class ForeignKeyMapping(TypeMappingUnit):
    python_type = PydanticRelatedFieldType  # Placeholder
    django_field_type = models.ForeignKey

    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        return super().pydantic_to_django_kwargs(py_type, field_info)


class OneToOneFieldMapping(TypeMappingUnit):
    python_type = PydanticRelatedFieldType  # Placeholder
    django_field_type = models.OneToOneField

    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        return super().pydantic_to_django_kwargs(py_type, field_info)


class ManyToManyFieldMapping(TypeMappingUnit):
    python_type = PydanticListOfRelated  # Placeholder (List[Any])
    django_field_type = models.ManyToManyField

    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        # M2M doesn't use most standard kwargs like null, default from base
        # It needs specific handling in get_django_mapping for 'to' and 'blank'
        # Return minimal kwargs, mainly title/description from FieldInfo
        base_kwargs = super().pydantic_to_django_kwargs(py_type, field_info)
        allowed_m2m_keys = {"verbose_name", "help_text", "blank"}  # Add others if needed
        m2m_kwargs = {k: v for k, v in base_kwargs.items() if k in allowed_m2m_keys}
        # M2M fields should always have blank=True by default if mapping from Pydantic list
        m2m_kwargs.setdefault("blank", True)
        return m2m_kwargs


class BidirectionalTypeMapper:
    """Registry and entry point for bidirectional type mapping."""

    def __init__(self, relationship_accessor: Optional[RelationshipConversionAccessor] = None):
        self.relationship_accessor = relationship_accessor or RelationshipConversionAccessor()
        self._registry: list[type[TypeMappingUnit]] = self._build_registry()
        # Caches
        self._pydantic_cache: dict[Any, Optional[type[TypeMappingUnit]]] = {}
        self._django_cache: dict[type[models.Field], Optional[type[TypeMappingUnit]]] = {}

    def _build_registry(self) -> list[type[TypeMappingUnit]]:
        """Discover and order TypeMappingUnit subclasses."""
        # Order matters: More specific Django types first, BUT base types need to be matched correctly.
        # Place base types (Str, Int) before their more specific variants (Slug, Email, Auto)
        # Ensure exact matches are found before subclass matches override them.
        ordered_units = [
            # Base Primitives First (for exact matching)
            StrFieldMapping,  # Base for CharField, Text, Slug, Email etc.
            IntFieldMapping,  # Base for IntegerField, AutoField etc.
            BoolFieldMapping,
            FloatFieldMapping,
            # Other Base Types
            DecimalFieldMapping,
            UUIDFieldMapping,
            DateTimeFieldMapping,
            DateFieldMapping,
            TimeFieldMapping,
            DurationFieldMapping,
            BinaryFieldMapping,
            JsonFieldMapping,  # Catches dict, list, set, tuple, Any
            FilePathFieldMapping,  # Needs Path, maps to str subclass conceptually
            # Specific String Subclasses
            TextFieldMapping,
            EmailFieldMapping,
            URLFieldMapping,
            SlugFieldMapping,
            IPAddressFieldMapping,
            # File Fields (map Path/str)
            ImageFieldMapping,
            FileFieldMapping,
            # Specific Integer Subclasses
            BigIntFieldMapping,
            SmallIntFieldMapping,
            PositiveBigIntFieldMapping,
            PositiveSmallIntFieldMapping,
            PositiveIntFieldMapping,
            # PK Fields last among integers
            BigAutoFieldMapping,
            SmallAutoFieldMapping,
            AutoFieldMapping,
            # Relationships (handled separately in find method, but keep for completeness)
            ManyToManyFieldMapping,
            OneToOneFieldMapping,
            ForeignKeyMapping,
            # Enum handled dynamically by find method
        ]
        # TODO: Add validation to ensure no overlaps in django_field_type?
        return ordered_units

    def _find_unit_for_pydantic_type(self, py_type: Any) -> Optional[type[TypeMappingUnit]]:
        """Find the best matching mapping unit for a Pydantic/Python type."""
        # Handle caching
        # Use repr for cache key for complex types
        cache_key = repr(py_type) if not isinstance(py_type, type) else py_type
        if cache_key in self._pydantic_cache:
            return self._pydantic_cache[cache_key]

        result_unit = None  # Store result before caching

        # Specific checks first
        origin = get_origin(py_type)  # Get origin early for checks

        if inspect.isclass(py_type) and issubclass(py_type, Enum):
            result_unit = EnumFieldMapping
        elif inspect.isclass(py_type) and issubclass(py_type, BaseModel):
            # Check if it's a known relationship
            if self.relationship_accessor.is_source_model_known(py_type):
                # Assume ForeignKey for direct model ref, could be OneToOne
                # TODO: Differentiate FK/O2O from Pydantic side?
                result_unit = ForeignKeyMapping
            else:
                logger.warning(
                    f"Pydantic type {py_type.__name__} is a BaseModel but not found in relationship accessor."
                )
                # Fallback? Map to JSON? Or raise error?
                result_unit = JsonFieldMapping  # Fallback to JSON
        else:
            # Check for Literal before iterating registry
            if origin is Literal:
                result_unit = EnumFieldMapping  # Use Enum logic for choices
            else:
                # Simplify matching: One loop for exact match, then one loop for subclass match.
                # This avoids complex conditions within a single loop.

                # 1. Exact match pass
                for unit_cls in self._registry:
                    # Skip relationships and placeholders handled elsewhere
                    if unit_cls in (ForeignKeyMapping, OneToOneFieldMapping, ManyToManyFieldMapping, EnumFieldMapping):
                        continue
                    # Use direct equality which works for primitives (int, bool, str, float, bytes, etc.) and types like Any
                    if py_type == unit_cls.python_type:
                        result_unit = unit_cls
                        break  # Found exact match

                # 2. Subclass match pass (if no exact match found)
                if result_unit is None:
                    for unit_cls in self._registry:
                        if unit_cls in (
                            ForeignKeyMapping,
                            OneToOneFieldMapping,
                            ManyToManyFieldMapping,
                            EnumFieldMapping,
                        ):
                            continue
                        try:
                            # Check issubclass only if both are actual classes
                            if (
                                inspect.isclass(py_type)
                                and inspect.isclass(unit_cls.python_type)
                                and issubclass(py_type, unit_cls.python_type)
                            ):
                                # Important: Only assign if it's a *true* subclass, not the type itself
                                # Prevents mapping `int` to `BigAutoField` just because BigAutoField maps to `int`.
                                if py_type is not unit_cls.python_type:
                                    # Check if a more specific subclass mapping already exists (e.g. EmailStr vs str)
                                    # If py_type is EmailStr, we want EmailFieldMapping, not StrFieldMapping
                                    is_more_specific_match = False
                                    for check_unit in self._registry:
                                        if check_unit.python_type == py_type and issubclass(
                                            check_unit.django_field_type, unit_cls.django_field_type
                                        ):
                                            is_more_specific_match = True
                                            break
                                    if not is_more_specific_match:
                                        result_unit = unit_cls
                                        # Don't break here, allow potentially more specific subclass match later in registry?
                                        # Example: If Path matches StrFieldMapping, keep checking for FilePathFieldMapping.
                                        # Registry order should handle selecting the most specific subclass mapping.
                        except TypeError:
                            pass  # issubclass fails on non-classes

            # Handle complex types (dict, list, etc.) if no direct match found yet
            if result_unit is None and origin:  # Check origin was successfully retrieved
                args = get_args(py_type)
                # Check for List[BaseModel] -> ManyToMany
                if origin is list and args and inspect.isclass(args[0]) and issubclass(args[0], BaseModel):
                    if self.relationship_accessor.is_source_model_known(args[0]):
                        result_unit = ManyToManyFieldMapping
                    else:
                        logger.warning(f"List field with BaseModel {args[0].__name__} not in relationship accessor.")
                        result_unit = JsonFieldMapping  # Fallback collection to JSON
                else:
                    # Fallback for other collections
                    result_unit = JsonFieldMapping

        if result_unit is None:
            logger.warning(f"No specific mapping unit found for Python type: {py_type}")

        self._pydantic_cache[cache_key] = result_unit
        return result_unit

    def _find_unit_for_django_field(self, dj_field_type: type[models.Field]) -> Optional[type[TypeMappingUnit]]:
        """Find the most specific mapping unit based on Django field type MRO."""
        if dj_field_type in self._django_cache:
            return self._django_cache[dj_field_type]

        for unit_cls in self._registry:
            if issubclass(dj_field_type, unit_cls.django_field_type):
                self._django_cache[dj_field_type] = unit_cls
                return unit_cls

        logger.warning(f"No mapping unit found for Django field type: {dj_field_type.__name__}")
        self._django_cache[dj_field_type] = None
        return None

    def get_django_mapping(
        self, python_type: Any, field_info: Optional[FieldInfo] = None
    ) -> tuple[type[models.Field], dict[str, Any]]:
        """Get the corresponding Django Field type and constructor kwargs for a Python type."""
        processed_type_info = TypeHandler.process_field_type(python_type)
        original_py_type = python_type  # Preserve original for relationship checks
        is_optional = processed_type_info["is_optional"]
        is_list = processed_type_info["is_list"]

        # Determine the base Python type after unwrapping Optional/List
        base_py_type = original_py_type
        if is_optional:
            args = get_args(base_py_type)
            base_py_type = next((arg for arg in args if arg is not type(None)), Any)
        if is_list:
            # If also optional, base_py_type is already unwrapped Optional[List[T]] -> List[T]
            args = get_args(base_py_type)
            base_py_type = args[0] if args else Any

        # Find the mapping unit for the base Python type
        unit_cls = self._find_unit_for_pydantic_type(base_py_type)

        if not unit_cls:
            # Special check for list type where base_py_type might be Any
            # Need to check original type again if list
            if is_list:
                origin_check = get_origin(original_py_type)
                args_check = get_args(original_py_type)
                # Check for Optional[List[BaseModel]] case
                inner_list_type = original_py_type
                if is_optional:
                    inner_list_type = next((arg for arg in args_check if arg is not type(None)), Any)

                origin_list_check = get_origin(inner_list_type)
                args_list_check = get_args(inner_list_type)
                if origin_list_check is list and args_list_check:
                    inner_model = args_list_check[0]
                    if (
                        inspect.isclass(inner_model)
                        and issubclass(inner_model, BaseModel)
                        and self.relationship_accessor.is_source_model_known(inner_model)
                    ):
                        unit_cls = ManyToManyFieldMapping

        if not unit_cls:
            # Fallback to JSON? Or raise error?
            logger.warning(
                f"No mapping unit for base type {base_py_type} (derived from {original_py_type}), falling back to JSONField."
            )
            unit_cls = JsonFieldMapping
            # raise MappingError(f"Could not find mapping unit for Python type: {base_py_type}")

        instance_unit = unit_cls()  # Instantiate to call methods
        django_field_type = instance_unit.django_field_type
        # Pass base_py_type to kwargs method
        kwargs = instance_unit.pydantic_to_django_kwargs(base_py_type, field_info)

        # --- Handle Relationships --- #
        if unit_cls in (ForeignKeyMapping, OneToOneFieldMapping, ManyToManyFieldMapping):
            related_py_model = base_py_type  # This is the unwrapped model type

            target_django_model = self.relationship_accessor.get_django_model_for_pydantic(related_py_model)
            if not target_django_model:
                raise MappingError(
                    f"Cannot map relationship: No corresponding Django model found for Pydantic model "
                    f"{related_py_model.__name__} in RelationshipConversionAccessor."
                )
            # Use string representation (app_label.ModelName) if possible, else name
            model_ref = getattr(target_django_model._meta, "label_lower", target_django_model.__name__)
            kwargs["to"] = model_ref
            django_field_type = unit_cls.django_field_type  # M2MField, FK, O2O
            # Set on_delete for FK/O2O based on Optional status
            if unit_cls in (ForeignKeyMapping, OneToOneFieldMapping):
                kwargs["on_delete"] = models.SET_NULL if is_optional else models.CASCADE
        elif inspect.isclass(base_py_type) and issubclass(base_py_type, Enum):
            # EnumFieldMapping pydantic_to_django_kwargs dynamically sets field type
            django_field_type = instance_unit.django_field_type  # Update based on enum values

        # Apply nullability
        # M2M fields cannot be null in Django
        if django_field_type != models.ManyToManyField:
            kwargs["null"] = is_optional
            # Set blank based on null for simplicity (can be overridden)
            kwargs["blank"] = is_optional

        return django_field_type, kwargs

    def get_pydantic_mapping(self, dj_field: models.Field) -> tuple[Any, dict[str, Any]]:
        """Get the corresponding Pydantic type hint and FieldInfo kwargs for a Django Field."""
        dj_field_type = type(dj_field)
        unit_cls = self._find_unit_for_django_field(dj_field_type)

        if not unit_cls:
            logger.warning(f"No mapping unit for {dj_field_type.__name__}, falling back to Any.")
            pydantic_type = Optional[Any] if dj_field.null else Any
            return pydantic_type, {}  # No specific FieldInfo kwargs
            # raise MappingError(f"Could not find mapping unit for Django field type: {dj_field_type.__name__}")

        instance_unit = unit_cls()  # Instantiate
        base_pydantic_type = instance_unit.python_type
        field_info_kwargs = instance_unit.django_to_pydantic_field_info_kwargs(dj_field)

        final_pydantic_type = base_pydantic_type

        # --- Handle Relationships --- #
        if unit_cls in (ForeignKeyMapping, OneToOneFieldMapping, ManyToManyFieldMapping):
            related_dj_model = getattr(dj_field, "related_model", None)
            if not related_dj_model:
                raise MappingError(f"Cannot determine related Django model for field '{dj_field.name}'")

            # Get the *target* Pydantic model from the accessor
            target_pydantic_model = self.relationship_accessor.get_pydantic_model_for_django(related_dj_model)
            if not target_pydantic_model:
                logger.warning(
                    f"Cannot map relationship: No corresponding Pydantic model found for Django model "
                    f"'{related_dj_model._meta.label if hasattr(related_dj_model, '_meta') else related_dj_model.__name__}'. "
                    f"Using placeholder '{base_pydantic_type}'."
                )
                # Keep the placeholder type (Any or List[Any])
                final_pydantic_type = base_pydantic_type
            else:
                if unit_cls == ManyToManyFieldMapping:
                    final_pydantic_type = list[target_pydantic_model]
                else:  # FK or O2O
                    final_pydantic_type = target_pydantic_model
        # TODO: Handle Enum mapping back to specific Enum type? Requires dj_field.choices analysis

        # Apply optionality AFTER relationship resolution
        # M2M fields are never Optional on the Pydantic side representing the list itself
        if dj_field.null and unit_cls != ManyToManyFieldMapping:
            # Special case for Auto PKs handled in django_to_pydantic_field_info_kwargs
            is_auto_pk = dj_field.primary_key and isinstance(
                dj_field, (models.AutoField, models.BigAutoField, models.SmallAutoField)
            )
            if not is_auto_pk:
                final_pydantic_type = Optional[final_pydantic_type]
            else:
                # Auto PKs should already be Optional[int] via the base method + default=None
                final_pydantic_type = Optional[int]  # Ensure it's Optional[int]

        # Clean up redundant default=None for Optional fields, unless it's an AutoPK override
        if dj_field.null and field_info_kwargs.get("default") is None:
            # Exclude M2M here too
            if (
                not (
                    dj_field.primary_key
                    and isinstance(dj_field, (models.AutoField, models.BigAutoField, models.SmallAutoField))
                )
                and unit_cls != ManyToManyFieldMapping
            ):
                field_info_kwargs.pop("default", None)

        return final_pydantic_type, field_info_kwargs
