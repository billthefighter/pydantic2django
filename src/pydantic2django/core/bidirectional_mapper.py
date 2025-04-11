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

import inspect
import logging
from types import UnionType
from typing import Any, Literal, Optional, Union, get_args, get_origin

from django.db import models
from pydantic import BaseModel
from pydantic.fields import FieldInfo

# Use absolute import path for relationships - Linter still complains, define directly
# from pydantic2django.core.relationships import (
#     RelationshipConversionAccessor, PydanticRelatedFieldType, PydanticListOfRelated
# )
from pydantic2django.core.relationships import RelationshipConversionAccessor

from .mapping_units import (  # Import the mapping units
    AutoFieldMapping,
    BigAutoFieldMapping,
    BigIntFieldMapping,
    BinaryFieldMapping,
    BoolFieldMapping,
    DateFieldMapping,
    DateTimeFieldMapping,
    DecimalFieldMapping,
    DurationFieldMapping,
    EmailFieldMapping,
    EnumFieldMapping,
    FileFieldMapping,
    FilePathFieldMapping,
    FloatFieldMapping,
    ForeignKeyMapping,
    ImageFieldMapping,
    IntFieldMapping,
    IPAddressFieldMapping,
    JsonFieldMapping,
    ManyToManyFieldMapping,
    OneToOneFieldMapping,
    PositiveBigIntFieldMapping,
    PositiveIntFieldMapping,
    PositiveSmallIntFieldMapping,
    SlugFieldMapping,
    SmallAutoFieldMapping,
    SmallIntFieldMapping,
    StrFieldMapping,
    TextFieldMapping,
    TimeFieldMapping,
    TypeMappingUnit,
    URLFieldMapping,
    UUIDFieldMapping,
)
from .typing import TypeHandler

logger = logging.getLogger(__name__)

# Define relationship placeholders directly to avoid linter issues
# PydanticRelatedFieldType = Any # Now imported from mapping_units
# PydanticListOfRelated = list[Any] # Now imported from mapping_units

# Helper type variable - only used for annotation within TypeMappingUnit subclasses
# T_DjangoField = TypeVar("T_DjangoField", bound=models.Field) # Defined in mapping_units
# T_PydanticType = Any # Defined in mapping_units


class MappingError(Exception):
    """Custom exception for mapping errors."""

    pass


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
        # Order matters less for selection now, but still useful for tie-breaking?
        # References mapping units imported from .mapping_units
        ordered_units = [
            # Specific PKs first (subclass of IntField)
            BigAutoFieldMapping,
            SmallAutoFieldMapping,
            AutoFieldMapping,
            # Specific Numerics (subclass of IntField/FloatField/DecimalField)
            PositiveBigIntFieldMapping,
            PositiveSmallIntFieldMapping,
            PositiveIntFieldMapping,
            # Specific Strings (subclass of CharField/TextField)
            EmailFieldMapping,
            URLFieldMapping,
            SlugFieldMapping,
            IPAddressFieldMapping,
            FilePathFieldMapping,  # Needs Path, but Django field is specific
            # File Fields (map Path/str, Django fields are specific)
            ImageFieldMapping,  # Subclass of FileField
            FileFieldMapping,
            # Other specific types before bases
            UUIDFieldMapping,
            JsonFieldMapping,  # Before generic collections/Any might map elsewhere
            # Base Relationship types (before fields they might inherit from like FK < Field)
            ManyToManyFieldMapping,
            OneToOneFieldMapping,
            ForeignKeyMapping,
            # General Base Types LAST
            DecimalFieldMapping,
            DateTimeFieldMapping,
            DateFieldMapping,
            TimeFieldMapping,
            DurationFieldMapping,
            BinaryFieldMapping,
            FloatFieldMapping,
            BoolFieldMapping,
            # Str/Text: Order now primarily determined by `matches` score overrides
            TextFieldMapping,
            StrFieldMapping,
            # Specific Int types first
            BigIntFieldMapping,  # Map int to BigInt before Int
            SmallIntFieldMapping,
            IntFieldMapping,
            # Enum handled dynamically by find method
            EnumFieldMapping,  # Include EnumFieldMapping here for the loop
        ]
        # Remove duplicates just in case
        seen = set()
        unique_units = []
        for unit in ordered_units:
            if unit not in seen:
                unique_units.append(unit)
                seen.add(unit)
        return unique_units

    def _find_unit_for_pydantic_type(
        self, py_type: Any, field_info: Optional[FieldInfo] = None
    ) -> Optional[type[TypeMappingUnit]]:
        """
        Find the best mapping unit for a given Pydantic type and FieldInfo.
        Uses a scoring system based on the `matches` classmethod of each unit.
        Handles Optional unwrapping and caching.
        """
        original_type_for_cache = py_type  # Use the original type as the cache key

        # --- Unwrap Optional ---
        origin = get_origin(py_type)
        if origin is Optional:
            args = get_args(py_type)
            # Get the first non-None type argument
            type_to_match = next((arg for arg in args if arg is not type(None)), Any)
            logger.debug(f"Unwrapped Optional[{type_to_match.__name__}] to {type_to_match.__name__}")
        # Handle X | None syntax (UnionType)
        elif origin is UnionType:
            args = get_args(py_type)
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) == 1:  # If it's just `T | None`
                type_to_match = non_none_args[0]
                logger.debug(f"Unwrapped Union[{py_type}] with None to {type_to_match}")
            else:  # Keep the original UnionType if it's Union[A, B, ...]
                type_to_match = py_type
                logger.debug(f"Keeping UnionType {py_type} as is for matching.")
        else:
            type_to_match = py_type  # Use the original type if not Optional or simple T | None

        logger.debug(
            f"Final type_to_match for scoring: {type_to_match} (origin: {get_origin(type_to_match)}, args: {get_args(type_to_match)})"
        )

        # --- Cache Check ---
        # Re-enable caching
        cache_key = (original_type_for_cache, field_info)
        if cache_key in self._pydantic_cache:
            # logger.debug(f"Cache hit for {cache_key}")
            return self._pydantic_cache[cache_key]
        # logger.debug(f"Cache miss for {cache_key}")

        # --- Initialization ---
        best_unit: Optional[type[TypeMappingUnit]] = None
        highest_score = 0.0
        scores: dict[str, float | str] = {}  # Store scores for debugging

        # --- Union Type Special Handling (Before Scoring Loop) ---
        unwrapped_origin = get_origin(type_to_match)
        unwrapped_args = get_args(type_to_match)

        # Check if it's a Union containing BaseModels
        if unwrapped_origin in (Union, UnionType) and unwrapped_args:
            union_models = []
            other_types_in_union = False
            for arg in unwrapped_args:
                if arg is type(None):
                    continue  # Skip NoneType
                is_known = False
                if inspect.isclass(arg) and issubclass(arg, BaseModel):
                    is_known = self.relationship_accessor.is_source_model_known(arg)
                    logger.debug(f"Checking Union arg: {arg.__name__}, is_known: {is_known}")  # Added logging
                else:
                    logger.debug(f"Checking Union arg: {arg}, not a known BaseModel subclass.")  # Added logging

                if is_known:
                    union_models.append(arg)
                else:
                    other_types_in_union = True

            # If the Union *only* contains known BaseModels (and optionally None)
            if union_models and not other_types_in_union:
                logger.debug(
                    f"Detected Union of known BaseModels: {[m.__name__ for m in union_models]}. "
                    f"Signaling for multi-FK generation."
                )
                # Signal the generator via kwargs - use a placeholder unit for now
                # The actual field type doesn't matter much here, JSONField is fine.
                best_unit = JsonFieldMapping  # Use JsonFieldMapping as a placeholder unit
                self._pydantic_cache[cache_key] = best_unit
                return best_unit  # Return immediately, skipping scoring

            # Handle Union[Literal[...], str]
            elif (
                not union_models
                and any(t == str for t in unwrapped_args)
                and any(get_origin(t) is Literal for t in unwrapped_args)
            ):
                logger.debug("Detected Union[Literal..., str]. Selecting TextFieldMapping.")
                best_unit = TextFieldMapping  # Prefer TextField if str is present
                self._pydantic_cache[cache_key] = best_unit
                return best_unit  # Return immediately

            # Add other specific Union handling logic here if needed
            # e.g., Union[List[...], str] -> JsonField (handled by fallback below)

        # --- Relationship Check (Specific Model Types - Moved After Basic Union Check) --- #
        # Check if the unwrapped type is a known Pydantic model for relationships
        # This should now catch single BaseModels, not Unions of them.
        try:
            is_known_model = (
                inspect.isclass(type_to_match)
                and issubclass(type_to_match, BaseModel)
                and self.relationship_accessor.is_source_model_known(type_to_match)
            )
        except TypeError:
            is_known_model = False

        if is_known_model:
            # Directly select ForeignKeyMapping for known related models, bypassing scoring loop.
            # O2O handling might require specific FieldInfo hints or separate logic if needed.
            logger.debug(
                f"Type {type_to_match.__name__} is a known related model. Selecting ForeignKeyMapping directly."
            )
            best_unit = ForeignKeyMapping
            # Update cache and return immediately
            self._pydantic_cache[cache_key] = best_unit
            return best_unit
        # else:
        # This should generally be handled by get_django_mapping before calling this,
        # but include a basic check here for robustness or direct calls.
        # Prioritize O2O/FK based on some criteria? Pydantic field name or convention?
        # For now, let scoring handle it or assume get_django_mapping sorts it out.
        # We might need to add hints via FieldInfo or naming conventions if conflicts arise.
        # logger.debug(f"Type {type_to_match.__name__} is a known related model. Checking FK/O2O scores.")
        # Let FK/O2O scores compete in the main loop below.
        # pass  # Continue to scoring loop

        # M2M (List[KnownModel]) is handled in get_django_mapping, not here.

        # --- Scoring Loop (Only if not a known related model or specific Union) --- #
        # Use type_to_match (unwrapped) for matching
        if best_unit is None:  # Only run scoring if a unit wasn't picked already
            for unit_cls in self._registry:
                try:  # Add try-except around matches call for robustness
                    # Pass the unwrapped type to matches
                    score = unit_cls.matches(type_to_match, field_info)
                    if score > 0:  # Log all positive scores
                        scores[unit_cls.__name__] = score  # Store score regardless of whether it's the highest
                        logger.debug(
                            f"Scoring {unit_cls.__name__}.matches({type_to_match}, {field_info=}) -> {score}"
                        )  # Added logging
                    if score > highest_score:
                        highest_score = score
                        best_unit = unit_cls
                        # Store the winning score as well - Moved above
                        # scores[unit_cls.__name__] = score  # Overwrite if it was a lower score before
                    # elif score > 0:  # Log non-winning positive scores too - Moved above
                    # Only add if not already present (first positive score encountered)
                    # scores.setdefault(unit_cls.__name__, score)
                except Exception as e:
                    logger.error(f"Error calling {unit_cls.__name__}.matches for {type_to_match}: {e}", exc_info=True)
                    scores[unit_cls.__name__] = f"ERROR: {e}"  # Log error in scores dict

        # Sort scores for clearer logging (highest first)
        sorted_scores = dict(
            sorted(scores.items(), key=lambda item: item[1] if isinstance(item[1], (int, float)) else -1, reverse=True)
        )
        logger.debug(
            f"Scores for {original_type_for_cache} (unwrapped: {type_to_match}, {field_info=}): {sorted_scores}"
        )
        if best_unit:  # Added logging
            logger.debug(f"Selected best unit: {best_unit.__name__} with score {highest_score}")  # Added logging
        else:  # Added logging
            logger.debug("No best unit found based on scoring.")  # Added logging

        # --- Handle Fallbacks (Collections/Any) --- #
        if best_unit is None and highest_score == 0.0:
            logger.debug(f"Entering fallback logic for {type_to_match}")  # Added logging
            # Use type_to_match (unwrapped) and its origin
            unwrapped_origin = get_origin(type_to_match)

            # 1. Check for standard collection types
            if unwrapped_origin in (dict, list, set, tuple) or type_to_match in (dict, list, set, tuple):
                logger.debug(f"Type {type_to_match} is a standard collection, checking JsonFieldMapping score.")
                # Re-check score for JsonFieldMapping explicitly if it wasn't the best_unit
                json_score = JsonFieldMapping.matches(type_to_match, field_info)
                if json_score > 0:
                    best_unit = JsonFieldMapping
                    logger.debug(f"Selected JsonFieldMapping as fallback for collection {type_to_match}")
                else:  # Should not happen if JsonFieldMapping.matches covers collections
                    logger.warning(f"JsonFieldMapping did not match fallback collection type: {type_to_match}")
            # 2. Check for Any type
            elif type_to_match is Any:
                logger.debug("Type is Any, checking JsonFieldMapping score.")
                # Check score first in case Any has a specific override somewhere
                json_score = JsonFieldMapping.matches(type_to_match, field_info)
                if json_score > 0:  # JsonFieldMapping.matches should return > 0 for Any
                    best_unit = JsonFieldMapping
                    logger.debug("Selected JsonFieldMapping as fallback for Any type.")
                else:  # Should not happen with current JsonFieldMapping.matches
                    logger.warning("JsonFieldMapping did not match Any type.")
            # 3. Check if it's an unhandled UnionType
            elif unwrapped_origin is UnionType:
                logger.debug(f"Type {type_to_match} is an unhandled UnionType, falling back to JsonFieldMapping.")
                best_unit = JsonFieldMapping  # Explicitly map unhandled Unions to JSON
            else:
                logger.debug(f"Type {type_to_match} did not match standard fallbacks (Collection, Any, UnionType).")

        # Final Logging
        if best_unit is None:
            logger.warning(
                f"No specific mapping unit found for Python type: {original_type_for_cache} (unwrapped to {type_to_match}) with field_info: {field_info}"
            )
            # Log cache state before potential fallback write
            logger.debug(f"Cache keys before fallback write: {list(self._pydantic_cache.keys())}")

        # Re-enable cache write
        self._pydantic_cache[cache_key] = best_unit  # Cache using original key
        return best_unit

    def _find_unit_for_django_field(self, dj_field_type: type[models.Field]) -> Optional[type[TypeMappingUnit]]:
        """Find the most specific mapping unit based on Django field type MRO and registry order."""
        # Revert to simpler single pass using refined registry order.
        if dj_field_type in self._django_cache:
            return self._django_cache[dj_field_type]

        # Filter registry to exclude EnumFieldMapping unless it's specifically needed? No, registry order handles it.
        # Ensure EnumFieldMapping isn't incorrectly picked before Str/Int if choices are present.
        # The registry order should have Str/Int base mappings *after* EnumFieldMapping if EnumFieldMapping
        # only maps Enum/Literal python types. But dj_field_type matching is different.
        # If a CharField has choices, we want EnumFieldMapping logic, not StrFieldMapping.
        registry_for_django = self._registry  # Use the full registry for now

        for unit_cls in registry_for_django:
            # Special check: If field has choices, prioritize EnumFieldMapping if applicable type
            # This is handled by get_pydantic_mapping logic already, not needed here.

            if issubclass(dj_field_type, unit_cls.django_field_type):
                # Found the first, most specific match based on registry order
                # Example: PositiveIntegerField is subclass of IntegerField. If PositiveIntFieldMapping
                # comes first in registry, it will be matched correctly.
                self._django_cache[dj_field_type] = unit_cls
                return unit_cls

        # Fallback if no unit explicitly handles it (should be rare)
        logger.warning(
            f"No specific mapping unit found for Django field type: {dj_field_type.__name__}, check registry order."
        )
        self._django_cache[dj_field_type] = None
        return None

    def get_django_mapping(
        self,
        python_type: Any,
        field_info: Optional[FieldInfo] = None,
        parent_pydantic_model: Optional[type[BaseModel]] = None,  # Add parent model for self-ref check
    ) -> tuple[type[models.Field], dict[str, Any]]:
        """Get the corresponding Django Field type and constructor kwargs for a Python type."""
        processed_type_info = TypeHandler.process_field_type(python_type)
        original_py_type = python_type
        is_optional = processed_type_info["is_optional"]
        is_list = processed_type_info["is_list"]

        unit_cls = None  # Initialize unit_cls
        base_py_type = original_py_type  # Start with original
        union_details = None  # Store details if it's a Union[BaseModel,...]

        # --- Check for M2M case FIRST ---
        if is_list:
            # Get the type inside the list, handling Optional[List[T]]
            list_inner_type = original_py_type
            if is_optional:
                args_check = get_args(list_inner_type)
                list_inner_type = next((arg for arg in args_check if arg is not type(None)), Any)

            # Now get the type *inside* the list
            list_args = get_args(list_inner_type)  # Should be List[T]
            inner_type = list_args[0] if list_args else Any

            # Is the inner type a known related BaseModel?
            if (
                inspect.isclass(inner_type)
                and issubclass(inner_type, BaseModel)
                and self.relationship_accessor.is_source_model_known(inner_type)
            ):
                unit_cls = ManyToManyFieldMapping
                base_py_type = inner_type  # Set base_py_type for relationship handling below
                logger.debug(f"Detected List[RelatedModel] ({inner_type.__name__}), mapping to ManyToManyField.")
            # Else: If it's a list of non-models, let _find_unit_for_pydantic_type handle it (likely JsonFieldMapping)

        # --- If not M2M, find unit for the base (non-list) type ---
        if unit_cls is None:
            # Determine the base Python type after unwrapping Optional (List already handled conceptually)
            if is_optional:
                # If was Optional[List[...]] this was handled above
                # If Optional[SimpleType], unwrap here
                if not is_list:
                    args = get_args(base_py_type)  # base_py_type is original here
                    base_py_type = next((arg for arg in args if arg is not type(None)), Any)
                # If just List[...], base_py_type is still original List type, _find_unit... will handle
            else:
                # If not optional, base_py_type remains original_py_type
                pass

            # Find the mapping unit for the base Python type (which might be Model for FK, or simple type)
            # Pass field_info here!
            unit_cls = self._find_unit_for_pydantic_type(base_py_type, field_info)

        # --- Handle Union[BaseModel,...] Signaling --- #
        # Check if the original type was a Union of *multiple* known BaseModels
        # This section should ONLY handle the multi-FK case, not Optional[Model]
        union_details = None  # Initialize here
        check_origin = get_origin(original_py_type)
        check_args = get_args(original_py_type)

        # Check if it's Union[...] or T | U
        if check_origin in (Union, UnionType) and check_args:
            union_models = []
            other_types_in_union = False
            has_none_type = False

            for arg in check_args:
                if arg is type(None):
                    has_none_type = True
                    continue
                if (
                    inspect.isclass(arg)
                    and issubclass(arg, BaseModel)
                    and self.relationship_accessor.is_source_model_known(arg)
                ):
                    union_models.append(arg)
                else:
                    other_types_in_union = True

            # CRITICAL CHECK: Only trigger multi-FK if >1 model and no other types
            if len(union_models) > 1 and not other_types_in_union:
                logger.debug(
                    f"Detected Union of MULTIPLE ({len(union_models)}) known BaseModels. Signaling for multi-FK."
                )
                union_details = {
                    "type": "multi_fk",
                    "models": union_models,
                    "is_optional": has_none_type,  # Reflect if None was present
                }
                # Switch unit to JSON placeholder *only* for multi-FK case
                unit_cls = JsonFieldMapping
                logger.debug(f"Prepared union_details for multi-FK: {union_details}")
                # Set is_optional based on None presence for final kwargs adjustment
                is_optional = has_none_type
            elif len(union_models) == 1 and not other_types_in_union and has_none_type:
                # This is just Optional[KnownModel], ensure it wasn't lost.
                # _find_unit_for_pydantic_type should have handled unwrapping and
                # selected ForeignKeyMapping. Let's double-check unit_cls.
                if unit_cls not in (ForeignKeyMapping, OneToOneFieldMapping):
                    logger.warning(
                        f"Detected Optional[KnownModel] {union_models[0].__name__}, but unit_cls is "
                        f"{(unit_cls.__name__ if unit_cls else 'None')}. Expected a relationship unit."
                        f" Re-evaluating with unwrapped type {union_models[0]}..."
                    )
                    # Attempt to re-find the unit using the known model type
                    potential_unit = self._find_unit_for_pydantic_type(union_models[0], field_info)
                    if potential_unit in (ForeignKeyMapping, OneToOneFieldMapping):
                        unit_cls = potential_unit
                        logger.info(f"Corrected unit_cls to {unit_cls.__name__} for Optional[KnownModel].")
                    else:
                        logger.error(
                            f"Failed to correct unit_cls for Optional[KnownModel] {union_models[0].__name__}. "
                            f"Found {potential_unit.__name__ if potential_unit else 'None'} instead."
                        )
                else:
                    logger.debug(
                        f"Correctly identified Optional[{union_models[0].__name__}]. Proceeding with {unit_cls.__name__}."
                    )
                # Ensure is_optional reflects the Optional wrapper
                is_optional = True  # It came from Optional[Model] or Union[Model, None]
                # union_details remains None
            # else: This is not a Union containing ONLY BaseModels (and optional None), proceed normally

        # --- Fallback and Final Checks --- #
        # If M2M was detected OR a Relationship unit was found (FK/O2O) OR multi-FK was signaled,
        # unit_cls should be set. Otherwise, check for fallback.
        if not unit_cls:
            # If _find_unit_for_pydantic_type returned None, fallback to JSON
            logger.warning(
                f"No mapping unit found by scoring for base type {base_py_type} "
                f"(derived from {original_py_type}), falling back to JSONField."
            )
            unit_cls = JsonFieldMapping
            # Consider raising MappingError if even JSON doesn't fit?
            # raise MappingError(f"Could not find mapping unit for Python type: {base_py_type}")

        instance_unit = unit_cls()  # Instantiate to call methods

        # --- Determine Django Field Type ---
        # Start with the type defined on the selected unit class
        django_field_type = instance_unit.django_field_type

        # --- Get Kwargs (before potentially overriding field type for Enums) ---
        kwargs = instance_unit.pydantic_to_django_kwargs(base_py_type, field_info)

        # --- Add Union Details if applicable --- #
        if union_details:
            kwargs["_union_details"] = union_details
            # Skip relationship handling below if we are doing multi-FK
            logger.debug("Added _union_details to kwargs, skipping standard relationship handling.")
        else:
            # --- Special Handling for Enums/Literals (Only if not multi-FK union) --- #
            if unit_cls is EnumFieldMapping:
                # Check the hint returned by the kwargs method
                field_type_hint = kwargs.pop("_field_type_hint", None)
                if field_type_hint and isinstance(field_type_hint, type) and issubclass(field_type_hint, models.Field):
                    if field_type_hint == models.IntegerField and django_field_type != models.IntegerField:
                        logger.debug(
                            f"Overriding Enum default {django_field_type.__name__} with IntegerField based on values."
                        )
                        django_field_type = models.IntegerField
                    elif field_type_hint == models.CharField and django_field_type != models.CharField:
                        logger.debug(
                            f"Overriding Enum default {django_field_type.__name__} with CharField based on values."
                        )
                        django_field_type = models.CharField
                    # Ensure max_length is removed if type becomes IntegerField
                    if django_field_type == models.IntegerField:
                        kwargs.pop("max_length", None)
                else:
                    logger.warning("EnumFieldMapping selected but failed to get valid field type hint from kwargs.")
            # else:
            # For non-enum units, the field type is determined by the class attribute
            # django_field_type = instance_unit.django_field_type # Already assigned

            # --- Handle Relationships (Only if not multi-FK union) --- #
            # This section needs to run *after* unit selection but *before* final nullability checks
            if unit_cls in (ForeignKeyMapping, OneToOneFieldMapping, ManyToManyFieldMapping):
                # Ensure base_py_type is the related model (set during M2M check or found by find_unit for FK/O2O)
                related_py_model = base_py_type

                if not (inspect.isclass(related_py_model) and issubclass(related_py_model, BaseModel)):
                    raise MappingError(
                        f"Relationship mapping unit {unit_cls.__name__} selected, but base type {related_py_model} is not a Pydantic BaseModel."
                    )

                # Check for self-reference BEFORE trying to get the Django model
                is_self_ref = parent_pydantic_model is not None and related_py_model == parent_pydantic_model

                if is_self_ref:
                    model_ref = "self"
                    # Get the target Django model name for logging/consistency if possible, but use 'self'
                    target_django_model = self.relationship_accessor.get_django_model_for_pydantic(related_py_model)
                    logger.debug(
                        f"Detected self-reference for {related_py_model.__name__} "
                        f"(Django: {getattr(target_django_model, '__name__', 'N/A')}), using 'self'."
                    )
                else:
                    target_django_model = self.relationship_accessor.get_django_model_for_pydantic(related_py_model)
                    if not target_django_model:
                        raise MappingError(
                            f"Cannot map relationship: No corresponding Django model found for Pydantic model "
                            f"{related_py_model.__name__} in RelationshipConversionAccessor."
                        )
                    # Use string representation (app_label.ModelName) if possible, else name
                    model_ref = getattr(target_django_model._meta, "label_lower", target_django_model.__name__)

                kwargs["to"] = model_ref
                django_field_type = unit_cls.django_field_type  # Re-confirm M2MField, FK, O2O type
                # Set on_delete for FK/O2O based on Optional status
                if unit_cls in (ForeignKeyMapping, OneToOneFieldMapping):
                    # Default to PROTECT for non-optional, SET_NULL for optional
                    kwargs["on_delete"] = models.SET_NULL if is_optional else models.PROTECT
                # M2M blank=True is handled by ManyToManyFieldMapping.pydantic_to_django_kwargs

        # --- Final Adjustments (Nullability, etc.) --- #
        # Apply nullability. M2M fields cannot be null in Django.
        # Apply nullability BEFORE potentially overriding blank for multi-FK
        if django_field_type != models.ManyToManyField:
            kwargs["null"] = is_optional
            # Explicitly set blank based on optionality.
            # Override if blank is not already set OR if it was set to False/None.
            # Preserve blank=True if set by a unit (e.g., M2M).
            if kwargs.get("blank") is not True:
                kwargs["blank"] = is_optional

        # If it's a multi-FK union, force null=True, blank=True regardless of Optional status
        # because only one FK can be set.
        if union_details and union_details.get("type") == "multi_fk":
            kwargs["null"] = True
            kwargs["blank"] = True
            logger.debug("Forcing null=True, blank=True for multi-FK union placeholder.")

        return django_field_type, kwargs

    def get_pydantic_mapping(self, dj_field: models.Field) -> tuple[Any, dict[str, Any]]:
        """Get the corresponding Pydantic type hint and FieldInfo kwargs for a Django Field."""
        dj_field_type = type(dj_field)
        is_optional = dj_field.null
        is_choices = bool(dj_field.choices)

        # --- Find base unit (ignoring choices for now) ---
        # _find_unit_for_django_field uses registry order and issubclass
        base_unit_cls = self._find_unit_for_django_field(dj_field_type)

        if not base_unit_cls:
            logger.warning(f"No base mapping unit for {dj_field_type.__name__}, falling back to Any.")
            pydantic_type = Optional[Any] if is_optional else Any
            return pydantic_type, {}

        base_instance_unit = base_unit_cls()
        base_pydantic_type = base_instance_unit.python_type  # Get the base python type from the unit

        # --- Determine Final Pydantic Type (including choices, relationships, optionality) ---
        final_pydantic_type = base_pydantic_type

        # 1. Handle Choices -> Literal or Enum (Let generator handle Enum creation)
        if is_choices:
            # Keep the base python type (str/int). Kwargs will contain choices.
            # The generator component should create Literal[values...] or an Enum.
            logger.debug(f"Field '{dj_field.name}' has choices. Base Python type {final_pydantic_type} retained.")
            pass  # Keep final_pydantic_type as base_pydantic_type

        # 2. Handle Relationships
        elif base_unit_cls in (ForeignKeyMapping, OneToOneFieldMapping, ManyToManyFieldMapping):
            related_dj_model = getattr(dj_field, "related_model", None)
            if not related_dj_model:
                raise MappingError(f"Cannot determine related Django model for field '{dj_field.name}'")

            target_pydantic_model = self.relationship_accessor.get_pydantic_model_for_django(related_dj_model)
            if not target_pydantic_model:
                logger.warning(
                    f"Cannot map relationship: No corresponding Pydantic model found for Django model "
                    f"'{related_dj_model._meta.label if hasattr(related_dj_model, '_meta') else related_dj_model.__name__}'. "
                    f"Using placeholder '{base_pydantic_type}'."
                )
                final_pydantic_type = base_pydantic_type  # Keep placeholder (e.g., Any)
            else:
                if base_unit_cls == ManyToManyFieldMapping:
                    # Use list[] directly for cleaner annotations if possible
                    final_pydantic_type = list[target_pydantic_model]
                else:  # FK or O2O
                    final_pydantic_type = target_pydantic_model
                logger.debug(f"Mapped relationship field '{dj_field.name}' to Pydantic type: {final_pydantic_type}")

        # 3. AutoPK override (after relationship resolution)
        is_auto_pk = dj_field.primary_key and isinstance(
            dj_field, (models.AutoField, models.BigAutoField, models.SmallAutoField)
        )
        if is_auto_pk:
            # Auto PKs map to Optional[int], even if null=False in DB
            # Because they are not provided when creating new instances via Pydantic.
            final_pydantic_type = Optional[int]
            logger.debug(f"Mapped AutoPK field '{dj_field.name}' to {final_pydantic_type}")
            # Ensure is_optional flag reflects this for FieldInfo kwargs logic below
            is_optional = True

        # 4. Apply Optional[...] wrapper if necessary
        # Apply optionality AFTER relationship/Literal/AutoPK resolution
        # Do not wrap M2M lists or already Optional AutoPKs in Optional[] again.
        if is_optional and not is_auto_pk and base_unit_cls != ManyToManyFieldMapping:
            # Check if it's *already* Optional (e.g., from AutoPK override)
            origin = get_origin(final_pydantic_type)
            args = get_args(final_pydantic_type)
            is_already_optional = origin is Optional or origin is UnionType and type(None) in args

            if not is_already_optional:
                final_pydantic_type = Optional[final_pydantic_type]
                logger.debug(f"Wrapped type for '{dj_field.name}' in Optional: {final_pydantic_type}")

        # --- Generate FieldInfo Kwargs --- #
        # Use EnumFieldMapping logic for kwargs ONLY if choices exist, otherwise use base unit
        kwargs_unit_cls = EnumFieldMapping if is_choices else base_unit_cls
        if not kwargs_unit_cls:  # Should not happen if base_unit_cls was found
            logger.error(f"Could not determine kwargs unit for {dj_field.name}")
            instance_unit = base_instance_unit  # Fallback to base instance
        else:
            instance_unit = kwargs_unit_cls()

        field_info_kwargs = instance_unit.django_to_pydantic_field_info_kwargs(dj_field)

        # Clean up redundant `default=None` for Optional fields.
        # Pydantic v2 implicitly handles `default=None` for `Optional[T]` types.
        # Keep `default=None` only if it was an AutoPK (where frozen=True is also set).
        if is_optional and field_info_kwargs.get("default") is None:
            if not is_auto_pk:  # AutoPKs need default=None explicitly alongside frozen=True
                field_info_kwargs.pop("default", None)
                logger.debug(f"Removed redundant default=None for Optional field '{dj_field.name}'")

        return final_pydantic_type, field_info_kwargs
