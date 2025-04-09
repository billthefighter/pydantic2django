"""
Provides the `DjangoFieldMapper` class for converting Django model fields to Pydantic types and `FieldInfo`.

This mapper aims to handle common Django field attributes and map them to their
closest Pydantic `Field` arguments. However, it does not cover all possible
Pydantic `Field` arguments.

Handled Pydantic arguments (based on Django field attributes):
- `default`: From `dj_field.get_default()` (non-callable), or `None` if `null=True`.
- `title`: From `dj_field.verbose_name`.
- `description`: From `dj_field.help_text`.
- `frozen`: Set `True` for auto-PKs.
- `ge`: Set `0` for Positive* IntegerFields.
- `max_digits`: From `dj_field.max_digits`.
- `decimal_places`: From `dj_field.decimal_places`.
- `max_length`: From `dj_field.max_length`.

Arguments NOT handled (no direct Django equivalent or not implemented):
- Aliasing (`alias`, `validation_alias`, `serialization_alias`)
- `examples`
- `exclude`
- `discriminator`
- `deprecated`
- `json_schema_extra`
- `validate_default`
- `repr`, `init`, `init_var`, `kw_only`
- `pattern` (could potentially map from validators)
- `strict`
- `coerce_numbers_to_str`
- `gt`, `lt`, `le` (could potentially map from validators)
- `multiple_of`
- `allow_inf_nan`
- `min_length` (could potentially map from validators)
- `union_mode`, `fail_fast`
- `**extra`

Relationship fields (ForeignKey, OneToOneField, ManyToManyField) are mapped
to placeholder types (`Any`, `List[Any]`) and require the caller (e.g., a model
generator) to resolve the specific related Pydantic model type.
"""
import datetime
import logging
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

from django.db import models

# Use pydantic v2 imports
from pydantic import EmailStr, Field, HttpUrl
from pydantic.fields import FieldInfo

logger = logging.getLogger(__name__)

# Placeholder types for relationships - calling code (e.g., model generator) needs to resolve these
# Using ForwardRef might be an option, but Any is simpler for the mapper itself.
PydanticRelatedFieldType = Any
PydanticListOfRelated = list[Any]  # Placeholder for M2M


def extract_base_field_info(dj_field: models.Field) -> dict[str, Any]:
    """
    Extract common Pydantic FieldInfo arguments from a Django field instance.

    Args:
        dj_field: The Django model field instance.

    Returns:
        A dictionary of potential Pydantic FieldInfo arguments.
    """
    kwargs = {}
    # Default value
    # Django default can be a value or a callable. Pydantic's 'default' expects a value.
    # 'default_factory' could handle callables, but mapping DB defaults (potentially complex functions)
    # to Pydantic's default_factory isn't always straightforward or safe.
    # We will map direct values and skip callables for now.
    dj_default = dj_field.get_default()
    if dj_default is not models.fields.NOT_PROVIDED and dj_default is not None:
        if not callable(dj_default):
            kwargs["default"] = dj_default
        else:
            logger.debug(
                f"Django field '{dj_field.name}' has a callable default ({dj_default}), "
                "not mapping to Pydantic default/default_factory."
            )
    elif dj_field.null:
        # If the field is nullable and has no specific DB default, Pydantic's default is None
        kwargs["default"] = None
    # else: default remains PydanticUndefined, meaning the field is required if not Optional

    # Description/Help Text
    if dj_field.help_text:
        kwargs["description"] = str(dj_field.help_text)  # Ensure it's a string

    # Title/Verbose Name
    if dj_field.verbose_name:
        # Pydantic Field doesn't have 'verbose_name', 'title' is the closest equivalent
        kwargs["title"] = str(dj_field.verbose_name).capitalize()

    # Read Only / Exclude Flags (Example for AutoFields)
    # The exact handling (frozen, exclude, validation/serialization alias) depends heavily
    # on whether you're generating input models, output models, or both.
    if dj_field.primary_key and isinstance(dj_field, (models.AutoField, models.BigAutoField, models.SmallAutoField)):
        # Common pattern: PKs are often read-only after creation.
        # 'frozen=True' makes the Pydantic field immutable after initialization.
        # 'exclude=True' might be used for input models.
        # Let's use 'frozen' as a starting point for PKs.
        kwargs["frozen"] = True
        # Ensure default is None or not set for PKs unless explicitly provided,
        # as DB usually generates it. Pydantic needs 'default=None' for frozen fields
        # that aren't required during init. If it's nullable, it's already None.
        # If non-null, it must be provided or have a default/default_factory.
        # Since DB generates it, making it Optional[int] with default=None is safest.
        kwargs["default"] = None  # Override any other default logic for PKs

    # Note: Django's `blank` attribute primarily relates to form/admin validation
    # (allowing empty input) and doesn't directly map to Pydantic validation rules
    # in the same way `null` (database nullability) does. We prioritize `null` for type hinting.

    # TODO: Handle 'choices' -> Literal or Enum? This is complex due to dynamic choices. Start without it.

    return kwargs


def extract_char_field_info(dj_field: models.Field) -> dict[str, Any]:
    """Extract FieldInfo for CharField and subclasses."""
    kwargs = extract_base_field_info(dj_field)
    # Ensure it's a CharField or subclass before accessing max_length
    if isinstance(dj_field, models.CharField) and dj_field.max_length is not None:
        kwargs["max_length"] = dj_field.max_length
    # Min_length doesn't have a direct Django equivalent.
    # Regex validation could potentially be mapped if validators are used.
    return kwargs


def extract_decimal_field_info(dj_field: models.Field) -> dict[str, Any]:
    """Extract FieldInfo for DecimalField."""
    kwargs = extract_base_field_info(dj_field)
    # Ensure it's a DecimalField before accessing decimal attributes
    if isinstance(dj_field, models.DecimalField):
        if dj_field.max_digits is not None:
            kwargs["max_digits"] = dj_field.max_digits
        if dj_field.decimal_places is not None:
            kwargs["decimal_places"] = dj_field.decimal_places
    return kwargs


def extract_relation_field_info(dj_field: models.Field) -> dict[str, Any]:
    """Extract base FieldInfo for relationship fields (FK, M2M, O2O)."""
    kwargs = extract_base_field_info(dj_field)
    # The actual related Pydantic type needs to be determined by the caller.
    # Add a marker? For now, the placeholder type is the main indicator.
    return kwargs


def extract_file_field_info(dj_field: models.Field) -> dict[str, Any]:
    """Extract FieldInfo for FileField/ImageField (maps to str/URL by default)."""
    # FileFields often have max_length like CharFields
    kwargs = extract_char_field_info(dj_field)
    # By default, we map FileField/ImageField to a string (URL or path).
    # Pydantic models for upload might use `bytes` or `UploadFile`.
    # Pydantic models for output often use `HttpUrl` or `str`.
    # `str` is a safe default. Could refine to HttpUrl if appropriate.
    # If we need specific FileField attributes, add isinstance(dj_field, models.FileField) check here
    return kwargs


class DjangoMappingDefinition:
    """Defines a mapping from a Django field type to Pydantic type and FieldInfo."""

    def __init__(
        self,
        django_field_type: type[models.Field],
        pydantic_type: Any,
        field_info_extractor: Callable[[models.Field], dict[str, Any]] = extract_base_field_info,
        is_relation: bool = False,  # Mark relationship fields
    ):
        self.django_field_type = django_field_type
        self.pydantic_type = pydantic_type
        self.field_info_extractor = field_info_extractor
        self.is_relation = is_relation

    def get_pydantic_type_and_field_info(self, dj_field: models.Field) -> tuple[Any, Optional[FieldInfo]]:
        """
        Extract Pydantic type and FieldInfo for the given Django field instance based on this definition.

        Args:
            dj_field: The Django model field instance.

        Returns:
            Tuple of (Pydantic Type, Optional[FieldInfo]).

        Raises:
            TypeError: If the provided field instance doesn't match the expected Django type.
        """
        # Note: We rely on find_mapping_definition using issubclass, so dj_field
        # *should* be an instance of self.django_field_type or its subclass.
        # The original check was stricter but redundant given the find logic.
        # if not isinstance(dj_field, self.django_field_type):
        #     raise TypeError(f"Mapping definition expected Django field of type {self.django_field_type.__name__}, "
        #                     f"but received instance of {type(dj_field).__name__}")

        pydantic_base_type = self.pydantic_type
        field_info_kwargs = self.field_info_extractor(dj_field)

        # Handle nullability: Wrap type in Optional if null=True
        final_pydantic_type = pydantic_base_type
        is_optional = dj_field.null

        # Special case: Primary keys are often implicitly Optional[int] = None in Pydantic
        # even if null=False in DB, because they are generated by the DB.
        is_auto_pk = dj_field.primary_key and isinstance(
            dj_field, (models.AutoField, models.BigAutoField, models.SmallAutoField)
        )
        if is_auto_pk and not is_optional:
            logger.debug(f"Auto PK field '{dj_field.name}' (null=False) mapped to Optional type for Pydantic.")
            is_optional = True
            # Ensure default=None is set, overriding base extractor if needed
            field_info_kwargs["default"] = None
            field_info_kwargs["frozen"] = True  # Reiterate PK is typically frozen

        if is_optional:
            # Ensure default=None is set if nullable, unless another default is present
            # The base extractor already handles this, but double-check
            if "default" not in field_info_kwargs:
                field_info_kwargs["default"] = None
            # `Union[T, None]` is equivalent to `Optional[T]`
            final_pydantic_type = Optional[pydantic_base_type]

        # Create FieldInfo instance only if there are specific constraints/metadata
        pydantic_field_info = None
        if field_info_kwargs:
            # Filter out default=None if the type is already Optional, Pydantic handles this implicitly
            if is_optional and field_info_kwargs.get("default") is None:
                # Keep default=None if specifically set (e.g., for PK override) unless it was the automatic one
                # This logic needs care: Pydantic treats `Field(default=None)` differently from no default on Optional field.
                # Generally, avoid redundant `default=None` for Optional fields.
                # Let's remove it unless it's the PK case where we explicitly set it.
                if not is_auto_pk:
                    field_info_kwargs.pop("default", None)

            # Only create FieldInfo if there are *actually* remaining kwargs
            if field_info_kwargs:
                try:
                    pydantic_field_info = Field(**field_info_kwargs)
                except Exception as e:
                    logger.error(
                        f"Failed to create Pydantic FieldInfo for Django field '{dj_field.name}' "
                        f"with kwargs {field_info_kwargs}: {e}",
                        exc_info=True,
                    )
                    # Fallback: return type without FieldInfo
                    pydantic_field_info = None
                    field_info_kwargs = {}  # Clear kwargs as they failed

        logger.debug(
            f"Mapping Django field '{dj_field.name}' ({type(dj_field).__name__}, null={dj_field.null}) "
            f"to Pydantic type '{final_pydantic_type}' with FieldInfo args: {field_info_kwargs}"
        )

        return final_pydantic_type, pydantic_field_info


class DjangoFieldMapper:
    """Maps Django model fields to Pydantic types and FieldInfo objects."""

    # The order matters for finding the most specific match (subclasses before base classes).
    DJ_FIELD_MAPPINGS: list[DjangoMappingDefinition] = [
        # --- Most Specific Types First ---
        # AutoFields (PKs) - Handled slightly differently due to DB generation
        DjangoMappingDefinition(models.BigAutoField, int),  # Base type int, handled by is_auto_pk logic
        DjangoMappingDefinition(models.SmallAutoField, int),  # Base type int, handled by is_auto_pk logic
        DjangoMappingDefinition(models.AutoField, int),  # Base type int, handled by is_auto_pk logic
        # Specific String Types
        DjangoMappingDefinition(models.EmailField, EmailStr, extract_char_field_info),
        DjangoMappingDefinition(models.SlugField, str, extract_char_field_info),
        # Consider mapping URLField to HttpUrl if validation is desired
        DjangoMappingDefinition(models.URLField, HttpUrl, extract_char_field_info),
        # File/Image Fields (Map to str/URL by default)
        DjangoMappingDefinition(models.ImageField, str, extract_file_field_info),
        DjangoMappingDefinition(models.FileField, str, extract_file_field_info),
        DjangoMappingDefinition(models.FilePathField, Path, extract_char_field_info),
        # Specific Numeric Types
        DjangoMappingDefinition(models.DecimalField, Decimal, extract_decimal_field_info),
        DjangoMappingDefinition(models.PositiveBigIntegerField, int, lambda f: {**extract_base_field_info(f), "ge": 0}),
        DjangoMappingDefinition(
            models.PositiveSmallIntegerField, int, lambda f: {**extract_base_field_info(f), "ge": 0}
        ),
        DjangoMappingDefinition(models.PositiveIntegerField, int, lambda f: {**extract_base_field_info(f), "ge": 0}),
        # --- General Types ---
        # Numeric
        DjangoMappingDefinition(models.FloatField, float),
        DjangoMappingDefinition(models.SmallIntegerField, int),
        DjangoMappingDefinition(models.BigIntegerField, int),
        DjangoMappingDefinition(models.IntegerField, int),  # Catch-all integer
        # Boolean
        DjangoMappingDefinition(models.BooleanField, bool),
        # String
        DjangoMappingDefinition(models.TextField, str, extract_base_field_info),  # Usually no max_length
        DjangoMappingDefinition(models.CharField, str, extract_char_field_info),  # Base character field
        # Temporal
        DjangoMappingDefinition(models.DateTimeField, datetime.datetime),
        DjangoMappingDefinition(models.DateField, datetime.date),
        DjangoMappingDefinition(models.TimeField, datetime.time),
        DjangoMappingDefinition(models.DurationField, datetime.timedelta),
        # Other Data Types
        DjangoMappingDefinition(models.UUIDField, UUID),
        # Pydantic has IPvAnyAddress, but str covers IPv4/v6 and is simpler
        DjangoMappingDefinition(models.GenericIPAddressField, str, extract_char_field_info),
        DjangoMappingDefinition(models.BinaryField, bytes),
        # JSONField can hold anything; Any or perhaps Dict/List if more context available
        DjangoMappingDefinition(
            models.JSONField, Any
        ),  # Pydantic's Json type could be used for stricter JSON validation
        # --- Relationship Types (Placeholders) ---
        # These MUST come after specific fields they might inherit from (if any)
        # The calling code needs to resolve PydanticRelatedFieldType/PydanticListOfRelated
        DjangoMappingDefinition(
            models.ManyToManyField, PydanticListOfRelated, extract_relation_field_info, is_relation=True
        ),
        DjangoMappingDefinition(
            models.OneToOneField, PydanticRelatedFieldType, extract_relation_field_info, is_relation=True
        ),
        DjangoMappingDefinition(
            models.ForeignKey, PydanticRelatedFieldType, extract_relation_field_info, is_relation=True
        ),
    ]

    # Cache for resolved mappings per Django field class
    _mapping_cache: dict[type[models.Field], Optional[DjangoMappingDefinition]] = {}

    @classmethod
    def find_mapping_definition(cls, dj_field_type: type[models.Field]) -> Optional[DjangoMappingDefinition]:
        """
        Find the most specific mapping definition for a Django field type using the MRO.
        Caches the result.
        """
        if dj_field_type in cls._mapping_cache:
            return cls._mapping_cache[dj_field_type]

        # Check mappings in the defined order. Since specific types are first,
        # the first match via issubclass should be the most specific one.
        for mapping in cls.DJ_FIELD_MAPPINGS:
            if issubclass(dj_field_type, mapping.django_field_type):
                cls._mapping_cache[dj_field_type] = mapping
                logger.debug(
                    f"Found mapping for {dj_field_type.__name__}: uses definition for {mapping.django_field_type.__name__}"
                )
                return mapping

        # No mapping found for this type or any parent type in our list
        cls._mapping_cache[dj_field_type] = None
        logger.debug(f"No mapping definition found for Django field type: {dj_field_type.__name__}")
        return None

    @classmethod
    def get_pydantic_type_and_field_info(cls, dj_field: models.Field) -> tuple[Any, Optional[FieldInfo]]:
        """
        Get the Pydantic type and FieldInfo corresponding to a Django model field instance.

        Args:
            dj_field: The Django model field instance.

        Returns:
            A tuple containing:
                - The corresponding Pydantic type (e.g., int, str, Optional[datetime.datetime], Any).
                - A Pydantic FieldInfo object with constraints/metadata, or None if not needed.
        """
        dj_field_type = type(dj_field)
        mapping_def = cls.find_mapping_definition(dj_field_type)

        if mapping_def:
            try:
                pydantic_type, pydantic_field_info = mapping_def.get_pydantic_type_and_field_info(dj_field)

                # Post-processing for relations: Log and confirm placeholder
                if mapping_def.is_relation:
                    relation_type = "Unknown Relation"
                    related_model_name = "Unknown"
                    # Safely access relation attributes
                    related_model = getattr(dj_field, "related_model", None)
                    if related_model:
                        # related_model can be a model class or a string ("self" or "app.Model")
                        if isinstance(related_model, str):
                            related_model_name = related_model
                        else:
                            # Check if it looks like a Model class (has _meta)
                            if hasattr(related_model, "_meta"):
                                related_model_name = related_model.__name__
                            else:
                                related_model_name = repr(related_model)  # Fallback

                    if isinstance(dj_field, models.ForeignKey):
                        relation_type = "ForeignKey"
                    elif isinstance(dj_field, models.OneToOneField):
                        relation_type = "OneToOneField"
                    elif isinstance(dj_field, models.ManyToManyField):
                        relation_type = "ManyToManyField"

                    logger.info(
                        f"Field '{dj_field.name}' is a {relation_type} to '{related_model_name}'. "
                        f"Returning placeholder type '{pydantic_type}'. Caller must resolve this."
                    )
                    # The caller (model generator) needs to use related_model_name
                    # to find the corresponding Pydantic model and substitute the placeholder.

                return pydantic_type, pydantic_field_info

            except Exception as e:
                logger.error(
                    f"Error processing Django field '{dj_field.name}' ({dj_field_type.__name__}) "
                    f"using mapping definition for {mapping_def.django_field_type.__name__}: {e}",
                    exc_info=True,
                )
                # Fallback to Any, respecting nullability
                logger.warning(f"Falling back to 'Any' for Django field '{dj_field.name}' due to processing error.")
                fallback_type = Optional[Any] if dj_field.null else Any
                return fallback_type, None

        else:
            # No mapping definition found at all
            logger.warning(
                f"No explicit Pydantic mapping found for Django field type: {dj_field_type.__name__} "
                f"(field name: '{dj_field.name}'). Falling back to 'Any'."
            )
            # Fallback to Any, respecting nullability
            fallback_type = Optional[Any] if dj_field.null else Any
            return fallback_type, None

    @classmethod
    def is_relation(cls, dj_field: models.Field) -> bool:
        """Check if the Django field maps to a relationship placeholder."""
        mapping_def = cls.find_mapping_definition(type(dj_field))
        return mapping_def.is_relation if mapping_def else False


# --- Example Usage ---
# Needs Django setup (settings configured or django.setup()) to work.
# from django.conf import settings
# from django import setup
#
# if not settings.configured:
#     settings.configure(
#         INSTALLED_APPS=['django.contrib.contenttypes', 'django.contrib.auth'], # Minimal apps
#         DATABASES={'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': ':memory:'}}
#     )
#     setup()
#
# class OtherModel(models.Model):
#      data = models.CharField(max_length=10)
#      class Meta:
#          app_label = 'testapp_dj2pyd' # Use unique app label
#
# class ExampleModel(models.Model):
#     # Django Fields
#     pk_id = models.AutoField(primary_key=True)
#     name = models.CharField(max_length=100, help_text="The user's name")
#     email = models.EmailField(null=True, blank=True, unique=True, default=None)
#     age = models.IntegerField(default=30)
#     score = models.DecimalField(max_digits=5, decimal_places=2, null=True)
#     is_active = models.BooleanField(default=True)
#     created_at = models.DateTimeField(auto_now_add=True) # implies non-null, non-editable
#     updated_at = models.DateField(auto_now=True, null=True) # implies non-editable
#     duration = models.DurationField(null=True)
#     unique_code = models.UUIDField(default=UUID('d9a3e2a9-4b9b-4c7b-8d3a-8e1fca6c2e1d'))
#     notes = models.TextField(blank=True, null=True) # TextField can be null
#     # Relationship Fields
#     related_one = models.ForeignKey(OtherModel, on_delete=models.CASCADE, null=True, related_name='examples_fkey')
#     related_m2m = models.ManyToManyField(OtherModel, related_name='examples_m2m')
#     related_o2o = models.OneToOneField(OtherModel, on_delete=models.SET_NULL, null=True, related_name='example_o2o')
#
#     class Meta:
#         app_label = 'testapp_dj2pyd' # Needs an app label
#
# logger.info("--- Testing DjangoFieldMapper ---")
#
# for field in ExampleModel._meta.get_fields():
#      if isinstance(field, models.Field): # Process only actual model fields
#          try:
#              pyd_type, pyd_field_info = DjangoFieldMapper.get_pydantic_type_and_field_info(field)
#              field_info_repr = repr(pyd_field_info) if pyd_field_info else "None"
#              logger.info(f"Django Field '{field.name}' ({type(field).__name__}):")
#              logger.info(f"  -> Pydantic Type: {pyd_type}")
#              logger.info(f"  -> Pydantic FieldInfo: {field_info_repr}")
#          except Exception as e:
#              logger.error(f"Error testing field '{field.name}': {e}", exc_info=True)
#
# logger.info("--- Testing Done ---")
