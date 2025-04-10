"""
Defines the individual mapping units used by the BidirectionalTypeMapper.

Each class maps a specific Python/Pydantic type to a Django Field type
and handles the conversion of relevant attributes between FieldInfo and Django kwargs.
"""

import datetime
import inspect
import logging
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional, TypeVar, get_args, get_origin
from uuid import UUID

from django.db import models
from pydantic import EmailStr, HttpUrl, IPvAnyAddress
from pydantic.fields import FieldInfo
from pydantic.types import StringConstraints
from pydantic_core import PydanticUndefined

logger = logging.getLogger(__name__)

# Define relationship placeholders directly to avoid linter issues
PydanticRelatedFieldType = Any
PydanticListOfRelated = list[Any]

# Helper type variable - only used for annotation within TypeMappingUnit subclasses
T_DjangoField = TypeVar("T_DjangoField", bound=models.Field)
T_PydanticType = Any  # Pydantic types can be complex


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

    @classmethod
    def matches(cls, py_type: Any, field_info: Optional[FieldInfo] = None) -> float:
        """
        Calculate a score indicating how well this unit matches the given Python type and FieldInfo.

        Args:
            py_type: The Python type to match against.
            field_info: Optional Pydantic FieldInfo for context.

        Returns:
            A float score (0.0 = no match, higher = better match).
            Base implementation scores:
            - 1.0 for exact type match (cls.python_type == py_type)
            - 0.5 for subclass match (issubclass(py_type, cls.python_type))
            - 0.0 otherwise
        """
        target_py_type = cls.python_type
        if py_type == target_py_type:
            return 1.0
        try:
            # Check issubclass only if both are actual classes and py_type is not Any
            if (
                py_type is not Any
                and inspect.isclass(py_type)
                and inspect.isclass(target_py_type)
                and issubclass(py_type, target_py_type)
            ):
                # Don't match if it's the same type (already handled by exact match)
                if py_type is not target_py_type:
                    return 0.5
        except TypeError:
            # issubclass fails on non-classes (like Any, List[int], etc.)
            pass
        return 0.0

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
                # Skip None default if it matches null=True implicitly - NO, include default=None if explicitly set
                # Django doesn't handle callable defaults easily here
                if not callable(field_info.default):
                    kwargs["default"] = field_info.default
            elif field_info.default is None:  # Explicitly check for None default
                kwargs["default"] = None  # Add default=None if present in FieldInfo
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

        # Stricter default handling
        if dj_field.has_default():
            dj_default = dj_field.get_default()
            if dj_default is not models.fields.NOT_PROVIDED:
                if callable(dj_default):
                    factory_set = False
                    # Map known callable defaults to factory
                    if dj_default is dict:
                        kwargs["default_factory"] = dict
                        factory_set = True
                    elif dj_default is list:
                        kwargs["default_factory"] = list
                        factory_set = True
                    # Add other known callable mappings if needed (e.g., set, datetime.now)
                    else:
                        logger.debug(
                            f"Django field '{dj_field.name}' has an unmapped callable default ({dj_default}), "
                            "not mapping to Pydantic default/default_factory."
                        )
                    # Do not add default= if factory was set
                    if factory_set:
                        kwargs.pop("default", None)
                # Only add non-None defaults if factory was NOT set.
                elif dj_default is not None:
                    kwargs["default"] = dj_default
        # Removed implicit default='' etc.

        # Handle AutoField PKs -> frozen=True, default=None (even if not null in DB)
        is_auto_pk = dj_field.primary_key and isinstance(
            dj_field, (models.AutoField, models.BigAutoField, models.SmallAutoField)
        )
        if is_auto_pk:
            kwargs["frozen"] = True
            kwargs["default"] = None  # Ensure default is None for auto PKs

        # Add choices to schema_extra if present
        if dj_field.choices:
            if "json_schema_extra" not in kwargs:
                kwargs["json_schema_extra"] = {}
            kwargs["json_schema_extra"]["choices"] = dj_field.choices
            # kwargs.pop("max_length", None) # Let specific mappings handle max_length
        elif dj_field.max_length is not None:
            kwargs["max_length"] = dj_field.max_length

        return kwargs


# --- Specific Mapping Units --- #


class IntFieldMapping(TypeMappingUnit):
    python_type = int
    django_field_type = models.IntegerField

    @classmethod
    def matches(cls, py_type: Any, field_info: Optional[FieldInfo] = None) -> float:
        """Prefer base IntField for plain int over AutoFields etc."""
        if py_type == int:
            # Slightly higher score than subclasses that also map int
            return 1.01
        return super().matches(py_type, field_info)

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
        kwargs = super().django_to_pydantic_field_info_kwargs(dj_field)  # Call super first
        kwargs["ge"] = 0  # Add ge constraint
        return kwargs  # Return updated kwargs

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

    @classmethod
    def matches(cls, py_type: Any, field_info: Optional[FieldInfo] = None) -> float:
        """Prefer mapping `str` to CharField when `max_length` is suggested by FieldInfo."""
        base_score = super().matches(py_type, field_info)
        if py_type == str:
            # Check if max_length is present in FieldInfo metadata
            has_max_length = False
            if field_info and hasattr(field_info, "metadata"):
                for item in field_info.metadata:
                    if hasattr(item, "max_length") and item.max_length is not None:
                        has_max_length = True
                        break
            # TODO: Also check field_info.json_schema_extra? Some libraries put constraints there.

            if has_max_length:
                return 1.1  # Higher score than exact match (1.0) for plain str if max_length is specified
            else:
                # If no max_length hint, give a lower score than TextFieldMapping
                return 0.9  # Lower score than TextFieldMapping's 1.0 for plain str
        return base_score  # Return base score for non-str types (e.g., subclass check)

    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        kwargs = super().pydantic_to_django_kwargs(py_type, field_info)
        pyd_max_length = None
        # Revert to checking metadata as FieldInfo has no direct max_length
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

    @classmethod
    def matches(cls, py_type: Any, field_info: Optional[FieldInfo] = None) -> float:
        """Prefer mapping `str` to TextField when `max_length` is NOT suggested by FieldInfo."""
        base_score = super(StrFieldMapping, cls).matches(py_type, field_info)  # Call base TypeMappingUnit.matches

        if py_type == str:
            # Check if max_length is present in FieldInfo metadata
            has_max_length = False
            if field_info and hasattr(field_info, "metadata"):
                for item in field_info.metadata:
                    if hasattr(item, "max_length") and item.max_length is not None:
                        has_max_length = True
                        break
            # TODO: Also check field_info.json_schema_extra?

            if not has_max_length:
                # Prefer TextField for plain str with no length constraint
                return 1.0  # Base exact match score
            else:
                # Lower score if max_length *is* specified, StrFieldMapping should win
                return 0.4  # Lower than StrFieldMapping's base subclass score (0.5)
        return base_score

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

    @classmethod
    def matches(cls, py_type: Any, field_info: Optional[FieldInfo] = None) -> float:
        """Strongly prefer EmailStr."""
        if py_type == cls.python_type:
            return 1.2  # Very high score for exact type
        # Don't match plain str to EmailField
        return 0.0
        # return super().matches(py_type, field_info) # Default base matching - Incorrect, inherits StrField logic

    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        kwargs = super().pydantic_to_django_kwargs(py_type, field_info)
        # EmailStr has its own validation, max_length often implicit
        # Let super handle title/desc, don't force max_length here
        # unless explicitly needed by project standard
        return kwargs


class SlugFieldMapping(StrFieldMapping):
    python_type = str  # Pydantic doesn't have a specific Slug type
    django_field_type = models.SlugField

    @classmethod
    def matches(cls, python_type: type, field_info: Optional[FieldInfo] = None) -> float:
        score = super().matches(python_type, field_info)
        if score == 0:
            return 0.0

        # Check for specific hints that suggest a SlugField
        if field_info:
            # Try extracting pattern from metadata (Pydantic V2+)
            pattern = None
            if field_info.metadata:
                # Look for StringConstraints or pattern string directly
                pattern_obj = next(
                    (m for m in field_info.metadata if isinstance(m, StringConstraints)),
                    None,
                )
                if pattern_obj:
                    pattern = pattern_obj.pattern
                elif isinstance(field_info.metadata[0], str) and field_info.metadata[0].startswith(
                    "^"
                ):  # Check if first metadata is a pattern string
                    pattern = field_info.metadata[0]
                logger.debug(f"SlugFieldMapping: Extracted pattern from metadata: {pattern}")

            # Try extracting pattern from constraints (Pydantic V1/V2)
            if pattern is None and getattr(field_info, "pattern", None):
                pattern = field_info.pattern  # Kept getattr for V1 compat, linter might warn
                logger.debug(f"SlugFieldMapping: Extracted pattern from field_info.pattern: {pattern}")

            # Check if the pattern matches the typical slug pattern
            # Note: Using direct string comparison. Raw string r"^[-\w]+$" might be clearer.
            if pattern == "^[-\\w]+$":  # Ensure backslash is escaped for comparison if needed by context
                logger.debug(f"SlugFieldMapping: Matched standard slug pattern: '{pattern}'")
                score += 0.3  # Strong indicator
            else:
                logger.debug(f"SlugFieldMapping: Pattern '{pattern}' did not match standard slug pattern '^[-\\w]+$'.")

            # Max length constraint check (common for slugs)
            if getattr(field_info, "max_length", None) is not None:
                # Slugs often have a max_length (like 50 for Django's default)
                score += 0.1

        # Check for 'slug' in the field name (weak indicator)
        if field_info and field_info.alias and "slug" in field_info.alias.lower():
            score += 0.05

        return round(score, 2)

    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        return super().pydantic_to_django_kwargs(py_type, field_info)


class URLFieldMapping(StrFieldMapping):
    python_type = HttpUrl
    django_field_type = models.URLField

    @classmethod
    def matches(cls, py_type: Any, field_info: Optional[FieldInfo] = None) -> float:
        """Strongly prefer HttpUrl."""
        if py_type == cls.python_type:
            return 1.2  # Very high score for exact type
        # Don't match plain str
        return 0.0
        # return super().matches(py_type, field_info) # Default base matching - Incorrect

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

    def django_to_pydantic_field_info_kwargs(self, dj_field: models.Field) -> dict[str, Any]:
        # Let super handle title/desc/max_length from Django field
        return super().django_to_pydantic_field_info_kwargs(dj_field)


class IPAddressFieldMapping(StrFieldMapping):
    python_type = IPvAnyAddress  # Pydantic's type covers v4/v6
    django_field_type = models.GenericIPAddressField

    @classmethod
    def matches(cls, py_type: Any, field_info: Optional[FieldInfo] = None) -> float:
        """Strongly prefer IPvAnyAddress."""
        if py_type == cls.python_type:
            return 1.2  # Very high score for exact type
        # Don't match plain str
        return 0.0
        # return super().matches(py_type, field_info) # Default base matching - Incorrect

    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        return super().pydantic_to_django_kwargs(py_type, field_info)

    def django_to_pydantic_field_info_kwargs(self, dj_field: models.Field) -> dict[str, Any]:
        kwargs = super().django_to_pydantic_field_info_kwargs(dj_field)
        # IP addresses don't have max_length in Pydantic equivalent
        kwargs.pop("max_length", None)
        return kwargs


class FilePathFieldMapping(StrFieldMapping):
    python_type = Path
    django_field_type = models.FilePathField

    @classmethod
    def matches(cls, py_type: Any, field_info: Optional[FieldInfo] = None) -> float:
        """Strongly prefer Path."""
        if py_type == cls.python_type:
            return 1.2  # Very high score for exact type
        # Don't match plain str
        return 0.0
        # return super().matches(py_type, field_info) # Default base matching - Incorrect

    # Add default max_length for FilePathField
    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        kwargs = super().pydantic_to_django_kwargs(py_type, field_info)
        if "max_length" not in kwargs:
            kwargs["max_length"] = 100  # Default for FilePathField
        return kwargs

    def django_to_pydantic_field_info_kwargs(self, dj_field: models.Field) -> dict[str, Any]:
        # FilePathField specific logic if any, else default
        return super().django_to_pydantic_field_info_kwargs(dj_field)


class FileFieldMapping(StrFieldMapping):
    # Map File/Image fields to str (URL/path) by default in Pydantic
    python_type = str
    django_field_type = models.FileField

    @classmethod
    def matches(cls, py_type: Any, field_info: Optional[FieldInfo] = None) -> float:
        """Prefer str if FieldInfo hints suggest a generic file/binary."""
        if py_type == str:
            # Check for format hint in json_schema_extra
            has_file_hint = False
            schema_extra = getattr(field_info, "json_schema_extra", None)
            if isinstance(schema_extra, dict):
                format_hint = schema_extra.get("format")
                # Prioritize if format is explicitly binary or file-like
                if format_hint in ("binary", "byte", "file"):  # Add other relevant hints?
                    has_file_hint = True

            if has_file_hint:
                return 1.15  # Higher than StrField/TextField/Slug
            else:
                # If str but no file hint, this unit should not match strongly.
                # It shouldn't override StrFieldMapping if max_length is present.
                # Return 0.0 to prevent interference.
                return 0.0
                # return StrFieldMapping.matches(py_type, field_info) # Incorrect, causes tie

        # Don't match non-str types
        return 0.0

    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        kwargs = super().pydantic_to_django_kwargs(py_type, field_info)
        kwargs.pop("max_length", None)  # File paths/URLs don't use Pydantic max_length
        return kwargs

    def django_to_pydantic_field_info_kwargs(self, dj_field: models.Field) -> dict[str, Any]:
        # Map FileField to str, let super handle kwargs (title, etc)
        kwargs = super().django_to_pydantic_field_info_kwargs(dj_field)
        kwargs.pop("max_length", None)  # File paths/URLs don't use Pydantic max_length
        # Add format hint for OpenAPI/JSON Schema
        if "json_schema_extra" not in kwargs:
            kwargs["json_schema_extra"] = {}
        kwargs["json_schema_extra"].setdefault("format", "binary")
        return kwargs


class ImageFieldMapping(FileFieldMapping):
    django_field_type = models.ImageField

    @classmethod
    def matches(cls, py_type: Any, field_info: Optional[FieldInfo] = None) -> float:
        """Prefer str if FieldInfo hints suggest an image."""
        if py_type == str:
            # Check for format hint in json_schema_extra
            schema_extra = getattr(field_info, "json_schema_extra", None)
            if isinstance(schema_extra, dict):
                format_hint = schema_extra.get("format")
                if format_hint == "image":
                    return 1.16  # Slightly higher than FileFieldMapping
            # If str but no image hint, this unit should not match.
            return 0.0
            # return super().matches(py_type, field_info) # Incorrect: Hits FileFieldMapping logic
        return 0.0  # Don't match non-str types

    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        return super().pydantic_to_django_kwargs(py_type, field_info)

    def django_to_pydantic_field_info_kwargs(self, dj_field: models.Field) -> dict[str, Any]:
        # Inherits FileFieldMapping logic (removes max_length)
        kwargs = super().django_to_pydantic_field_info_kwargs(dj_field)
        # Override format hint
        if "json_schema_extra" not in kwargs:
            kwargs["json_schema_extra"] = {}
        kwargs["json_schema_extra"]["format"] = "image"
        return kwargs


class UUIDFieldMapping(TypeMappingUnit):
    python_type = UUID
    django_field_type = models.UUIDField

    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        return super().pydantic_to_django_kwargs(py_type, field_info)

    def django_to_pydantic_field_info_kwargs(self, dj_field: models.Field) -> dict[str, Any]:
        kwargs = super().django_to_pydantic_field_info_kwargs(dj_field)
        kwargs.pop("max_length", None)  # UUIDs don't have max_length
        return kwargs


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

    def django_to_pydantic_field_info_kwargs(self, dj_field: models.Field) -> dict[str, Any]:
        # Call super for title/desc, but remove potential length constraints
        kwargs = super().django_to_pydantic_field_info_kwargs(dj_field)
        kwargs.pop("max_length", None)
        return kwargs


class JsonFieldMapping(TypeMappingUnit):
    # Map complex Python types (dict, list, tuple, set, Json) to JSONField
    # Map Django JSONField to Pydantic Any (or Json for stricter validation)
    python_type = Any  # Or could use Json type: from pydantic import Json
    django_field_type = models.JSONField

    @classmethod
    def matches(cls, py_type: Any, field_info: Optional[FieldInfo] = None) -> float:
        """Match collection types and Any as a fallback."""
        origin = get_origin(py_type)
        # Give a moderate score for common collection types (using origin OR type itself)
        if origin in (dict, list, set, tuple) or py_type in (dict, list, set, tuple):
            # TODO: Check if list contains BaseModels? M2M should handle that.
            # For now, assume non-model lists/collections map here.
            return 0.8
        # Give a slightly higher score for Any, acting as a preferred fallback over relationships
        if py_type == Any:
            return 0.2  # Higher than base match (0.1?), lower than specific types
        # If pydantic.Json is used, match it
        # Need to import Json from pydantic for this check
        # try:
        #     from pydantic import Json
        #     if py_type is Json:
        #          return 1.0
        # except ImportError:
        #     pass

        # Use super().matches() to handle potential subclass checks if needed?
        # No, JsonField is usually a fallback, direct type checks are sufficient.
        return 0.0  # Explicitly return 0.0 if not a collection or Any
        # return super().matches(py_type, field_info) # Default base matching (0.0 for most)

    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        return super().pydantic_to_django_kwargs(py_type, field_info)

    def django_to_pydantic_field_info_kwargs(self, dj_field: models.Field) -> dict[str, Any]:
        # Get base kwargs (handles title, default_factory for dict/list)
        kwargs = super().django_to_pydantic_field_info_kwargs(dj_field)
        # Remove max_length if present, JSON doesn't use it
        kwargs.pop("max_length", None)
        return kwargs


class EnumFieldMapping(TypeMappingUnit):
    python_type = Enum  # Placeholder, actual enum type determined dynamically
    django_field_type = models.CharField  # Default, could be IntegerField

    @classmethod
    def matches(cls, py_type: Any, field_info: Optional[FieldInfo] = None) -> float:
        """Match Enums and Literals."""
        origin = get_origin(py_type)
        if origin is Literal:
            return 1.2  # Strong match for Literal
        if inspect.isclass(py_type) and issubclass(py_type, Enum):
            return 1.2  # Strong match for Enum subclasses

        return super().matches(py_type, field_info)  # Default base matching (0.0 otherwise)

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
                self.django_field_type = models.IntegerField  # dynamically change! Does not work, need to signal back
                # Need to ensure IntegerFieldMapping handles choices if needed, or handle here
                kwargs.pop("max_length", None)  # No max_length for IntegerField
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
    python_type = PydanticRelatedFieldType  # Placeholder (Any)
    django_field_type = models.ForeignKey

    @classmethod
    def matches(cls, py_type: Any, field_info: Optional[FieldInfo] = None) -> float:
        # Should only be selected if py_type is a known BaseModel
        # The main logic handles this selection before scoring currently.
        # If scoring is used, give low score for base 'Any'
        if py_type == Any:
            return 0.05  # Very low score
        # Potential future enhancement: check if py_type is BaseModel? Requires RelationshipAccessor here.
        return super().matches(py_type, field_info)

    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        return super().pydantic_to_django_kwargs(py_type, field_info)


class OneToOneFieldMapping(TypeMappingUnit):
    python_type = PydanticRelatedFieldType  # Placeholder (Any)
    django_field_type = models.OneToOneField

    @classmethod
    def matches(cls, py_type: Any, field_info: Optional[FieldInfo] = None) -> float:
        if py_type == Any:
            return 0.05  # Very low score
        return super().matches(py_type, field_info)

    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        return super().pydantic_to_django_kwargs(py_type, field_info)


class ManyToManyFieldMapping(TypeMappingUnit):
    python_type = PydanticListOfRelated  # Placeholder (list[Any])
    django_field_type = models.ManyToManyField

    @classmethod
    def matches(cls, py_type: Any, field_info: Optional[FieldInfo] = None) -> float:
        # Should only be selected if py_type is List[KnownModel]
        # Handled before scoring loop. Give low score for base list[Any] match.
        origin = get_origin(py_type)
        if origin is list:
            args = get_args(py_type)
            inner_type = args[0] if args else Any
            if inner_type == Any:
                return 0.05  # Very low score for list[Any]
        # Check for bare list type
        if py_type == list:
            return 0.05  # Very low score for bare list
        return super().matches(py_type, field_info)

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
