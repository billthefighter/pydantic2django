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

# Import Django settings and translation
from django.conf import settings
from django.db import models
from django.utils import translation
from django.utils.encoding import force_str
from pydantic import EmailStr, HttpUrl, IPvAnyAddress
from pydantic.fields import FieldInfo
from pydantic.types import StringConstraints
from pydantic_core import PydanticUndefined

# Import BidirectionalTypeMapper for internal use within EnumFieldMapping
# from .bidirectional_mapper import BidirectionalTypeMapper

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

            # Only consider `default` if `default_factory` is None
            if field_info.default_factory is None:
                if field_info.default is not PydanticUndefined and field_info.default is not None:
                    # Django doesn't handle callable defaults easily here
                    if not callable(field_info.default):
                        kwargs["default"] = field_info.default
                elif field_info.default is None:  # Explicitly check for None default
                    kwargs["default"] = None  # Add default=None if present in FieldInfo
            # else: If default_factory is present, do not add a 'default' kwarg.
            # No warning needed as this is now expected behavior.

            # Note: Frozen, ge, le etc. are validation rules, map separately if needed
        return kwargs

    def django_to_pydantic_field_info_kwargs(self, dj_field: models.Field) -> dict[str, Any]:
        """Generate Pydantic FieldInfo kwargs from a Django field instance."""
        kwargs = {}
        field_name = getattr(dj_field, "name", "unknown_field")  # Get field name for logging

        # Title: Use verbose_name or generate from field name
        verbose_name = getattr(dj_field, "verbose_name", None)
        logger.debug(f"Processing field '{field_name}': verbose_name='{verbose_name}'")
        if verbose_name:
            # Ensure verbose_name is a string, handling lazy proxies
            kwargs["title"] = force_str(verbose_name).capitalize()
        elif field_name != "unknown_field" and isinstance(field_name, str):
            # Generate title from name if verbose_name is missing and name is a string
            generated_title = field_name.replace("_", " ").capitalize()
            kwargs["title"] = generated_title
            logger.debug(f"Generated title for '{field_name}': '{generated_title}'")
        # else: field name is None or 'unknown_field', no title generated by default

        # Description
        if dj_field.help_text:
            # Ensure help_text is a string, handling lazy proxies
            kwargs["description"] = force_str(dj_field.help_text)

        # Default value/factory handling
        if dj_field.has_default():
            dj_default = dj_field.get_default()
            if dj_default is not models.fields.NOT_PROVIDED:
                if callable(dj_default):
                    factory_set = False
                    if dj_default is dict:
                        kwargs["default_factory"] = dict
                        factory_set = True
                    elif dj_default is list:
                        kwargs["default_factory"] = list
                        factory_set = True
                    # Add other known callable mappings if needed
                    else:
                        logger.debug(
                            f"Django field '{dj_field.name}' has an unmapped callable default ({dj_default}), "
                            "not mapping to Pydantic default/default_factory."
                        )
                    if factory_set:
                        kwargs.pop("default", None)
                # Handle non-callable defaults
                # Map default={} back to default_factory=dict for JSONField
                elif dj_default == {}:
                    kwargs["default_factory"] = dict
                    kwargs.pop("default", None)
                elif dj_default == []:
                    kwargs["default_factory"] = list
                    kwargs.pop("default", None)
                elif dj_default is not None:
                    # Add non-None, non-callable, non-empty-collection defaults
                    logger.debug(
                        f"Processing non-callable default for '{field_name}'. Type: {type(dj_default)}, Value: {dj_default!r}"
                    )
                    # Apply force_str ONLY if the default value's type suggests it might be a lazy proxy string.
                    # A simple check is if 'proxy' is in the type name.
                    processed_default = dj_default
                    if "proxy" in type(dj_default).__name__:
                        try:
                            processed_default = force_str(dj_default)
                            logger.debug(
                                f"Applied force_str to potential lazy default for '{field_name}'. New value: {processed_default!r}"
                            )
                        except Exception as e:
                            logger.error(
                                f"Failed to apply force_str to default value for '{field_name}': {e}. Assigning raw default."
                            )
                            processed_default = dj_default  # Keep original on error

                    kwargs["default"] = processed_default
                    logger.debug(f"Assigned final default for '{field_name}': {kwargs.get('default')!r}")

        # Handle AutoField PKs -> frozen=True, default=None
        is_auto_pk = dj_field.primary_key and isinstance(
            dj_field, (models.AutoField, models.BigAutoField, models.SmallAutoField)
        )
        if is_auto_pk:
            kwargs["frozen"] = True
            kwargs["default"] = None

        # Handle choices (including processing labels and limiting)
        # Log choices *before* calling handle_choices
        if hasattr(dj_field, "choices") and dj_field.choices:
            try:
                # Log the raw choices from the Django field
                raw_choices_repr = repr(list(dj_field.choices))  # Materialize and get repr
                logger.debug(f"Field '{field_name}': Raw choices before handle_choices: {raw_choices_repr}")
            except Exception as log_err:
                logger.warning(f"Field '{field_name}': Error logging raw choices: {log_err}")

            self.handle_choices(dj_field, kwargs)
            # Log choices *after* handle_choices modified kwargs
            processed_choices_repr = repr(kwargs.get("json_schema_extra", {}).get("choices"))
            logger.debug(f"Field '{field_name}': Choices in kwargs after handle_choices: {processed_choices_repr}")

        # Handle non-choice max_length only if choices were NOT processed
        elif dj_field.max_length is not None:
            # Only add max_length if not choices - specific units can override
            kwargs["max_length"] = dj_field.max_length

        logger.debug(f"Base kwargs generated for '{field_name}': {kwargs}")
        return kwargs

    def handle_choices(self, dj_field: models.Field, kwargs: dict[str, Any]) -> None:
        """
        Handles Django field choices, ensuring lazy translation proxies are resolved.

        It processes the choices, forces string conversion on labels within an
        active translation context, limits the number of choices added to the schema,
        and stores them in `json_schema_extra`.
        """
        field_name = getattr(dj_field, "name", "unknown_field")
        processed_choices = []
        MAX_CHOICES_IN_SCHEMA = 30  # TODO: Make configurable
        limited_choices = []
        default_value = kwargs.get("default")  # Use potentially processed default
        default_included = False

        # --- Ensure Translation Context --- #
        active_translation = None
        if translation:
            try:
                # Get the currently active language to restore later
                current_language = translation.get_language()
                # Activate the default language (or a specific one like 'en')
                # This forces lazy objects to resolve using a consistent language.
                # Using settings.LANGUAGE_CODE assumes it's set correctly.
                default_language = getattr(settings, "LANGUAGE_CODE", "en")  # Fallback to 'en'
                active_translation = translation.override(default_language)
                logger.debug(
                    f"Activated translation override ('{default_language}') for processing choices of '{field_name}'"
                )
                active_translation.__enter__()  # Manually enter context
            except Exception as trans_err:
                logger.warning(f"Failed to activate translation context for '{field_name}': {trans_err}")
                active_translation = None  # Ensure it's None if activation failed
        else:
            logger.warning("Django translation module not available. Lazy choices might not resolve correctly.")

        # --- Process Choices (within potential translation context) --- #
        try:
            all_choices = list(dj_field.choices or [])
            for value, label in all_choices:
                logger.debug(
                    f"  Processing choice for '{field_name}': Value={value!r}, Label={label!r} (Type: {type(label)})"
                )
                try:
                    # Apply force_str defensively; should resolve lazy proxies if context is active
                    processed_label = force_str(label)
                    logger.debug(
                        f"  Processed label for '{field_name}': Value={value!r}, Label={processed_label!r} (Type: {type(processed_label)})"
                    )
                except Exception as force_str_err:
                    logger.error(
                        f"Error using force_str on label for '{field_name}' (value: {value!r}): {force_str_err}"
                    )
                    # Fallback: use repr or a placeholder if force_str fails completely
                    processed_label = f"<unresolved: {repr(label)}>"
                processed_choices.append((value, processed_label))

            # --- Limit Choices --- #
            if len(processed_choices) > MAX_CHOICES_IN_SCHEMA:
                logger.warning(
                    f"Limiting choices for '{field_name}' from {len(processed_choices)} to {MAX_CHOICES_IN_SCHEMA}"
                )
                if default_value is not None:
                    for val, lbl in processed_choices:
                        if val == default_value:
                            limited_choices.append((val, lbl))
                            default_included = True
                            break
                remaining_slots = MAX_CHOICES_IN_SCHEMA - len(limited_choices)
                if remaining_slots > 0:
                    for val, lbl in processed_choices:
                        if len(limited_choices) >= MAX_CHOICES_IN_SCHEMA:
                            break
                        if not (default_included and val == default_value):
                            limited_choices.append((val, lbl))
                final_choices_list = limited_choices
            else:
                final_choices_list = processed_choices

            # --- Store Choices --- #
            kwargs.setdefault("json_schema_extra", {})["choices"] = final_choices_list
            kwargs.pop("max_length", None)  # Remove max_length if choices are present
            logger.debug(f"Stored final choices in json_schema_extra for '{field_name}'")

        except Exception as e:
            logger.error(f"Error processing or limiting choices for field '{field_name}': {e}", exc_info=True)
            kwargs.pop("json_schema_extra", None)

        finally:
            # --- Deactivate Translation Context --- #
            if active_translation:
                try:
                    active_translation.__exit__(None, None, None)  # Manually exit context
                    logger.debug(f"Deactivated translation override for '{field_name}'")
                except Exception as trans_exit_err:
                    logger.warning(f"Error deactivating translation context for '{field_name}': {trans_exit_err}")


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

    @classmethod
    def matches(cls, py_type: Any, field_info: Optional[FieldInfo] = None) -> float:
        """Prefer mapping `int` to PositiveIntegerField when `ge=0` is suggested."""
        base_score = super().matches(py_type, field_info)  # Should be 0.0 or 0.5 normally
        if py_type != int:
            return base_score

        has_ge0_constraint = False
        if field_info and hasattr(field_info, "metadata"):
            for item in field_info.metadata:
                # Check for ge/gt constraints
                ge_value = getattr(item, "ge", None)
                gt_value = getattr(item, "gt", None)
                if ge_value == 0 or (gt_value is not None and gt_value >= -1):
                    # Consider gt > -1 (i.e. >=0) as positive indication too
                    has_ge0_constraint = True
                    break
        # Also check FieldInfo directly (older Pydantic?)
        if not has_ge0_constraint and field_info:
            ge_value = getattr(field_info, "ge", None)
            gt_value = getattr(field_info, "gt", None)
            if ge_value == 0 or (gt_value is not None and gt_value >= -1):
                has_ge0_constraint = True

        if has_ge0_constraint:
            # Higher score than IntFieldMapping (1.01) if ge=0 constraint present
            logger.debug("PositiveIntFieldMapping: Found ge=0 constraint, score -> 1.05")
            return 1.05
        else:
            # If no constraint, give a lower score than IntFieldMapping
            # Base score (0.5) for int is already lower than IntFieldMapping (1.01)
            logger.debug("PositiveIntFieldMapping: No ge=0 constraint, relying on base score.")
            # Return a slightly lower score than plain IntField to ensure IntField wins by default
            return 0.4  # Ensure it doesn't accidentally win over IntField (1.01)

    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        return super().pydantic_to_django_kwargs(py_type, field_info)

    def django_to_pydantic_field_info_kwargs(self, dj_field: models.Field) -> dict[str, Any]:
        kwargs = super().django_to_pydantic_field_info_kwargs(dj_field)
        kwargs["ge"] = 0
        return kwargs


class PositiveSmallIntFieldMapping(PositiveIntFieldMapping):
    django_field_type = models.PositiveSmallIntegerField

    @classmethod
    def matches(cls, py_type: Any, field_info: Optional[FieldInfo] = None) -> float:
        """Check for ge=0 constraint, but score slightly lower than base PositiveIntFieldMapping."""
        base_score = super().matches(
            py_type, field_info
        )  # Gets score from PositiveIntFieldMapping (e.g., 1.05 if constraint matches)
        # Reduce score slightly if constraint matched to allow base PositiveIntFieldMapping to win
        if base_score >= 1.05:
            logger.debug("PositiveSmallIntFieldMapping: Reducing score slightly from base Positive match.")
            return 1.04
        # If base score is low (no constraint match or not int), return that score.
        return base_score

    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        return super().pydantic_to_django_kwargs(py_type, field_info)


class PositiveBigIntFieldMapping(PositiveIntFieldMapping):
    django_field_type = models.PositiveBigIntegerField

    @classmethod
    def matches(cls, py_type: Any, field_info: Optional[FieldInfo] = None) -> float:
        """Check for ge=0 constraint, but score slightly lower than base PositiveIntFieldMapping."""
        base_score = super().matches(
            py_type, field_info
        )  # Gets score from PositiveIntFieldMapping (e.g., 1.05 if constraint matches)
        # Reduce score slightly if constraint matched to allow base PositiveIntFieldMapping to win
        if base_score >= 1.05:
            logger.debug("PositiveBigIntFieldMapping: Reducing score slightly from base Positive match.")
            return 1.04
        # If base score is low (no constraint match or not int), return that score.
        return base_score

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
        """Prefer mapping `str` to CharField ONLY when `max_length` is suggested."""
        base_score = super().matches(py_type, field_info)
        if py_type == str:
            has_max_length = False
            if field_info and hasattr(field_info, "metadata"):
                for item in field_info.metadata:
                    if hasattr(item, "max_length") and item.max_length is not None:
                        has_max_length = True
                        break

            if has_max_length:
                # High score if max_length specified
                return 1.1
            else:
                # Lower score than TextFieldMapping (1.0) if no max_length hint
                return 0.9
        return base_score

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
            if self.django_field_type == models.CharField:
                # Apply default only if CharField is selected and no length was specified
                kwargs["max_length"] = 255
        return kwargs

    def django_to_pydantic_field_info_kwargs(self, dj_field: models.Field) -> dict[str, Any]:
        kwargs = super().django_to_pydantic_field_info_kwargs(dj_field)
        if isinstance(dj_field, models.CharField) and dj_field.max_length is not None:
            kwargs["max_length"] = dj_field.max_length
        return kwargs


class TextFieldMapping(StrFieldMapping):
    django_field_type = models.TextField

    @classmethod
    def matches(cls, py_type: Any, field_info: Optional[FieldInfo] = None) -> float:
        """Prefer mapping `str` to TextField when `max_length` is NOT suggested."""
        base_score = super(StrFieldMapping, cls).matches(py_type, field_info)

        if py_type == str:
            has_max_length = False
            if field_info and hasattr(field_info, "metadata"):
                for item in field_info.metadata:
                    if hasattr(item, "max_length") and item.max_length is not None:
                        has_max_length = True
                        break

            if not has_max_length:
                # Prefer TextField (1.0) over CharField (0.9) if no length constraint
                return 1.0
            else:
                # Low score if max_length *is* specified
                return 0.4
        return base_score

    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        kwargs = super().pydantic_to_django_kwargs(py_type, field_info)
        pyd_max_length = None
        if field_info and hasattr(field_info, "metadata"):
            for item in field_info.metadata:
                if hasattr(item, "max_length"):
                    pyd_max_length = item.max_length
                    break
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
            return 1.2
        return 0.0

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
            # Django's default max_length for EmailField
            kwargs["max_length"] = 254
        return kwargs


class SlugFieldMapping(StrFieldMapping):
    python_type = str  # Pydantic doesn't have a specific Slug type
    django_field_type = models.SlugField

    @classmethod
    def matches(cls, python_type: type, field_info: Optional[FieldInfo] = None) -> float:
        # Don't match if it's not a string type initially
        if python_type != str:
            return 0.0

        score = 0.0  # Initialize score
        has_any_slug_hint = False
        pattern = None  # Initialize pattern

        if field_info:
            # Try extracting pattern from metadata (Pydantic V2+)
            if field_info.metadata:
                try:
                    constraint_meta = next(
                        (m for m in field_info.metadata if isinstance(m, StringConstraints) or hasattr(m, "pattern")),
                        None,
                    )
                except TypeError:  # Handle cases where metadata is not iterable
                    logger.warning("SlugFieldMapping: FieldInfo metadata was not iterable.")
                    constraint_meta = None

                if constraint_meta and hasattr(constraint_meta, "pattern"):
                    pattern = getattr(constraint_meta, "pattern", None)  # Use getattr safely
                elif (
                    field_info.metadata
                    and isinstance(field_info.metadata[0], str)
                    and field_info.metadata[0].startswith("^")
                ):
                    # Fallback: Check if the first metadata item is a pattern string itself
                    pattern = field_info.metadata[0]

            # Try extracting pattern from field_info attribute (less preferred now, V1 compat?)
            if pattern is None and hasattr(field_info, "pattern"):  # Use hasattr for safety
                field_pattern = getattr(field_info, "pattern", None)
                if field_pattern:
                    pattern = field_pattern

            # 1. Check pattern (Strongest indicator)
            # Compare against raw, single-escaped, and double-escaped pattern strings
            raw_slug_pattern = r"^[-\w]+$"  # Raw string
            single_escaped_slug_pattern = "^[-\\w]+$"  # Standard escaped string
            double_escaped_slug_pattern = "^[-\\\\w]+$"  # Double escaped as seen in metadata
            logger.debug(
                f"SlugFieldMapping: Comparing extracted pattern '{pattern}' (type: {type(pattern)}) with standard slug patterns."
            )

            if pattern in (raw_slug_pattern, single_escaped_slug_pattern, double_escaped_slug_pattern):
                logger.debug("SlugFieldMapping: Matched standard slug pattern. Score -> 1.3")
                score = 1.3  # High score for direct pattern match
                has_any_slug_hint = True
            elif pattern:
                logger.debug(f"SlugFieldMapping: Pattern '{pattern}' did not match standard slug patterns.")

            # 2. Check title/description/alias for 'slug' (Weaker indicator)
            if not has_any_slug_hint and "slug" in (
                f"{field_info.title or ''} {field_info.description or ''} {field_info.alias or ''}".lower()
            ):
                logger.debug("SlugFieldMapping: Found 'slug' in title/description/alias. Score -> 1.15")
                score = 1.15  # Higher than StrField (1.1), lower than pattern match (1.3)
                has_any_slug_hint = True

            # 3. Check max_length (Very weak indicator, mainly helps differentiate from TextField)
            has_max_length = False
            if hasattr(field_info, "metadata"):
                for item in field_info.metadata:
                    if hasattr(item, "max_length") and item.max_length is not None:
                        has_max_length = True
                        break

            if has_max_length and not has_any_slug_hint:
                # Slugs often have max_length, slightly prefer over TextField if no other hints
                # This score (1.05) is slightly above TextField (1.0)
                # but below StrField with max_length (1.1)
                logger.debug("SlugFieldMapping: Found max_length hint only. Score -> 1.05")
                score = 1.05

        final_score = round(score, 2)
        logger.debug(f"SlugFieldMapping final score for str with FieldInfo type {type(field_info)}: {final_score}")
        return final_score

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
    # Optionality should be determined by the field's null status, not the base mapping.
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
        # Start with base kwargs (which includes StrFieldMapping -> TypeMappingUnit)
        kwargs = super().django_to_pydantic_field_info_kwargs(dj_field)
        # Add OpenAPI format for file uploads
        kwargs.setdefault("json_schema_extra", {})["format"] = "binary"
        # File/Image fields typically map to str (URL/path), max_length isn't relevant
        kwargs.pop("max_length", None)
        logger.debug(f"FileFieldMapping kwargs for '{getattr(dj_field, 'name', 'unknown')}': {kwargs}")
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
        # Start with FileFieldMapping's kwargs
        kwargs = super().django_to_pydantic_field_info_kwargs(dj_field)
        # Override format to 'image' if possible
        if "json_schema_extra" in kwargs:
            kwargs["json_schema_extra"]["format"] = "image"
        else:  # Should not happen if FileFieldMapping worked correctly
            kwargs.setdefault("json_schema_extra", {})[
                "format"
            ] = "image"  # Use setdefault here too for consistency/safety
        logger.debug(f"ImageFieldMapping kwargs for '{getattr(dj_field, 'name', 'unknown')}': {kwargs}")
        return kwargs


class UUIDFieldMapping(TypeMappingUnit):
    python_type = UUID
    django_field_type = models.UUIDField

    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        return super().pydantic_to_django_kwargs(py_type, field_info)

    def django_to_pydantic_field_info_kwargs(self, dj_field: models.Field) -> dict[str, Any]:
        kwargs = super().django_to_pydantic_field_info_kwargs(dj_field)
        kwargs.pop("max_length", None)  # UUIDs don't have max_length
        # Do not map Django UUID default to Pydantic default
        kwargs.pop("default", None)
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

        # Use more standard defaults if not provided
        kwargs["max_digits"] = max_digits if max_digits is not None else 10  # Standard Default
        kwargs["decimal_places"] = decimal_places if decimal_places is not None else 2  # Standard Default
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
    """Handles fields with choices, mapping to Literal or Enum based on context.
    Django Field: Any field with `choices` set.
    Pydantic Type: Literal[...] or Enum.
    """

    # Use a generic Field as the base, as choices can be on various types
    django_field_type = models.Field
    # Base python type depends on the underlying field type (str, int, etc.)
    # This will be overridden dynamically based on the actual field.
    python_type = Any

    @classmethod
    def matches(cls, py_type: Any, field_info: Optional[FieldInfo] = None) -> float:
        # Check if it's a Literal type
        origin = get_origin(py_type)
        if origin is Literal:
            return 2.0  # High score for direct Literal match
        # Check if it's an Enum
        if inspect.isclass(py_type) and issubclass(py_type, Enum):
            return 1.9  # Slightly lower than Literal, but high
        return 0.0  # Don't match other types directly

    def django_to_pydantic_field_info_kwargs(self, dj_field: models.Field) -> dict[str, Any]:
        """Generate FieldInfo kwargs, adding choices to json_schema_extra."""
        # Start with base kwargs (title, description, default handling)
        kwargs = super().django_to_pydantic_field_info_kwargs(dj_field)

        # EnumFieldMapping specific logic (if any) could go here
        # For now, just rely on the base method
        logger.debug(f"EnumFieldMapping using base kwargs for '{dj_field.name}': {kwargs}")
        return kwargs

    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        kwargs = super().pydantic_to_django_kwargs(py_type, field_info)
        origin_type = get_origin(py_type)
        values = []
        field_type_hint = models.CharField  # Default hint

        if origin_type is Literal:
            literal_args = get_args(py_type)
            if not literal_args:
                logger.warning("Literal type has no arguments.")
                return kwargs
            values = [arg for arg in literal_args]
            kwargs["choices"] = [(str(v), str(v)) for v in values]
        elif py_type and inspect.isclass(py_type) and issubclass(py_type, Enum):
            members = list(py_type)
            values = [member.value for member in members]
            kwargs["choices"] = [(member.value, member.name) for member in members]
        else:
            logger.warning("Enum mapping used but type is not an Enum or Literal?")
            return kwargs  # Return base kwargs if type is somehow wrong

        if not values:
            logger.warning(f"No values found for Enum/Literal type: {py_type}")
            return kwargs

        # Determine field type and calculate max_length if needed
        if all(isinstance(val, int) for val in values):
            field_type_hint = models.IntegerField
            kwargs.pop("max_length", None)  # No max_length for IntegerField
            logger.debug(f"Enum/Literal {py_type} maps to IntegerField.")
        else:
            field_type_hint = models.CharField
            try:
                max_length = max(len(str(val)) for val in values)
            except TypeError:
                # Handle cases where values might be mixed types without good str representation?
                logger.warning(
                    f"Could not determine max_length for Enum/Literal {py_type} values: {values}. Defaulting to 255."
                )
                max_length = 255
            kwargs["max_length"] = max_length
            logger.debug(f"Enum/Literal {py_type} maps to CharField with max_length={max_length}.")

        # Add the hint to the kwargs to be used by the caller
        kwargs["_field_type_hint"] = field_type_hint

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

    def django_to_pydantic_field_info_kwargs(self, dj_field: models.Field) -> dict[str, Any]:
        kwargs = super().django_to_pydantic_field_info_kwargs(dj_field)
        kwargs["default_factory"] = list  # Ensure default is an empty list
        return kwargs


# --- Generic Foreign Key Placeholder --- #


class GenericForeignKeyMappingUnit(TypeMappingUnit):
    """Represents a Generic Foreign Key relationship concept.

    This unit doesn't map directly to a single Django field type on the parent
    model or a single Pydantic type in the traditional sense. It's identified
    structurally (e.g., List[Union[ModelA, ModelB]]) by the BidirectionalTypeMapper,
    which signals the generator to create the necessary intermediate model and GFK fields
    (ContentType FK, object_id, parent FK).
    """

    # Use placeholders that are unlikely to be matched directly
    python_type = list[Any]  # Conceptual representation (List[Union[...]])
    django_field_type = models.Field  # Placeholder, not directly used

    @classmethod
    def matches(cls, py_type: Any, field_info: Optional[FieldInfo] = None) -> float:
        """GFK relationships are typically detected structurally, not via simple type matching.
        This unit should generally not be selected via scoring.
        """
        return 0.0  # Do not participate in standard scoring selection

    def pydantic_to_django_kwargs(self, py_type: Any, field_info: Optional[FieldInfo] = None) -> dict[str, Any]:
        """Returns base kwargs. GFK details are signaled separately.
        Actual GFK fields (ContentType, object_id) are handled by the generator.
        """
        # Return base attributes like verbose_name/help_text if applicable
        kwargs = super().pydantic_to_django_kwargs(py_type, field_info)
        # Remove attributes not relevant to the GFK *placeholder* on the parent
        kwargs.pop("default", None)
        kwargs.pop("max_length", None)
        kwargs.pop("choices", None)
        return kwargs

    def django_to_pydantic_field_info_kwargs(self, dj_field: models.Field) -> dict[str, Any]:
        """Mapping *from* a GFK conceptual field back to Pydantic is complex.
        Typically results in a List[Union[...]]. Return base kwargs.
        """
        kwargs = super().django_to_pydantic_field_info_kwargs(dj_field)
        # Represent as a list, likely defaulting to an empty list
        kwargs["default_factory"] = list
        # Remove attributes not relevant to the Pydantic List[Union[...]] representation
        kwargs.pop("max_length", None)
        kwargs.pop("max_digits", None)
        kwargs.pop("decimal_places", None)
        kwargs.pop("ge", None)
        kwargs.pop("le", None)
        # Choices might be relevant if the Union members have constraints, but complex to map here.
        # Keep title/description if they exist.
        return kwargs
