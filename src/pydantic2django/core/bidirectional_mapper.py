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

import dataclasses
import inspect
import logging
from types import UnionType
from typing import Any, Literal, Optional, Union, cast, get_args, get_origin

from django.db import models
from pydantic import BaseModel
from pydantic.fields import FieldInfo

# Use absolute import path for relationships - Linter still complains, define directly
# from pydantic2django.core.relationships import (
#     RelationshipConversionAccessor, PydanticRelatedFieldType, PydanticListOfRelated
# )
from pydantic2django.core.relationships import RelationshipConversionAccessor
from pydantic2django.core.typing import TypeHandler

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

        # --- Literal Type Check (using original type) --- #
        original_origin = get_origin(original_type_for_cache)
        if original_origin is Literal:
            logger.debug(f"Type {original_type_for_cache} is Literal. Selecting EnumFieldMapping directly.")
            best_unit = EnumFieldMapping
            self._pydantic_cache[cache_key] = best_unit
            return best_unit

        # --- Prioritize Collection Types -> JSON --- #
        # Use the unwrapped origin for this check
        # unwrapped_origin = get_origin(type_to_match)
        # if unwrapped_origin in (list, dict, set, tuple):
        #     logger.debug(f"Type {type_to_match} is a collection. Selecting JsonFieldMapping directly.")
        #     best_unit = JsonFieldMapping
        #     self._pydantic_cache[cache_key] = best_unit
        #     return best_unit

        # --- Initialization --- #
        best_unit: Optional[type[TypeMappingUnit]] = None
        highest_score = 0.0
        scores: dict[str, float | str] = {}  # Store scores for debugging

        # --- Relationship Check (Specific Model Types and Lists of Models) BEFORE Scoring --- #
        # Check if the type_to_match itself is a known model
        try:
            is_direct_known_model = (
                inspect.isclass(type_to_match)
                and (issubclass(type_to_match, BaseModel) or dataclasses.is_dataclass(type_to_match))
                and self.relationship_accessor.is_source_model_known(type_to_match)
            )
        except TypeError:
            is_direct_known_model = False

        if is_direct_known_model:
            logger.debug(
                f"Type {type_to_match.__name__} is a known related model. Selecting ForeignKeyMapping directly."
            )
            best_unit = ForeignKeyMapping
            self._pydantic_cache[cache_key] = best_unit
            return best_unit

        # Check if it's a list/set of known models (potential M2M)
        unwrapped_origin = get_origin(type_to_match)
        unwrapped_args = get_args(type_to_match)
        if unwrapped_origin in (list, set) and unwrapped_args:  # Check for list or set
            inner_type = unwrapped_args[0]
            try:
                is_list_of_known_models = (
                    inspect.isclass(inner_type)
                    and (issubclass(inner_type, BaseModel) or dataclasses.is_dataclass(inner_type))
                    and self.relationship_accessor.is_source_model_known(inner_type)
                )
            except TypeError:
                is_list_of_known_models = False
                logger.error(f"TypeError checking if {inner_type} is a known model list item.", exc_info=True)

            logger.debug(
                f"Checking list/set: unwrapped_origin={unwrapped_origin}, inner_type={inner_type}, is_list_of_known_models={is_list_of_known_models}"
            )
            if is_list_of_known_models:
                logger.debug(
                    f"Type {type_to_match} is a list/set of known models ({inner_type.__name__}). Selecting ManyToManyFieldMapping directly."
                )
                best_unit = ManyToManyFieldMapping
                self._pydantic_cache[cache_key] = best_unit
                return best_unit
            else:
                logger.debug(
                    f"Type {type_to_match} is a list/set, but inner type {inner_type} is not a known model. Proceeding."
                )

        # --- Specific Union Handling BEFORE Scoring --- #
        unwrapped_args = get_args(type_to_match)
        # Check for non-model Unions (Model unions handled in get_django_mapping signal)
        if unwrapped_origin in (Union, UnionType) and unwrapped_args:
            logger.debug(f"Evaluating specific Union type {type_to_match} args: {unwrapped_args} before scoring.")
            has_str = any(arg is str for arg in unwrapped_args)
            has_collection_or_any = any(
                get_origin(arg) in (dict, list, set, tuple) or arg is Any
                for arg in unwrapped_args
                if arg is not type(None)
            )
            # Don't handle Union[ModelA, ModelB] here, that needs the signal mechanism
            is_model_union = any(
                inspect.isclass(arg) and (issubclass(arg, BaseModel) or dataclasses.is_dataclass(arg))
                for arg in unwrapped_args
                if arg is not type(None)
            )

            if not is_model_union:
                if has_str and not has_collection_or_any:
                    logger.debug(f"Union {type_to_match} contains str, selecting TextFieldMapping directly.")
                    best_unit = TextFieldMapping
                    self._pydantic_cache[cache_key] = best_unit
                    return best_unit
                elif has_collection_or_any:
                    logger.debug(f"Union {type_to_match} contains complex types, selecting JsonFieldMapping directly.")
                    best_unit = JsonFieldMapping
                    self._pydantic_cache[cache_key] = best_unit
                    return best_unit
                # Else: Union of simple types (e.g., int | float) - let scoring handle it.
                else:
                    logger.debug(f"Union {type_to_match} is non-model, non-str/complex. Proceeding to scoring.")
            else:
                logger.debug(
                    f"Union {type_to_match} contains models. Proceeding to scoring (expecting JsonField fallback)."
                )

        # --- Scoring Loop (Only if not a known related model or specific Union handled above) --- #
        # Use type_to_match (unwrapped) for matching
        # --- EDIT: Removed redundant check `if best_unit is None:` --- #
        # This loop now runs only if no direct selection happened above.
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
            logger.debug(f"Re-evaluating fallback/handling for {type_to_match}")
            unwrapped_origin = get_origin(type_to_match)
            unwrapped_args = get_args(type_to_match)

            # 1. Check standard collections first (MOVED FROM TOP)
            if unwrapped_origin in (dict, list, set, tuple) or type_to_match in (dict, list, set, tuple):
                # Re-check list/set here to ensure it wasn't a list of known models handled above
                if unwrapped_origin in (list, set) and unwrapped_args:
                    inner_type = unwrapped_args[0]
                    try:
                        is_list_of_known_models_fallback = (
                            inspect.isclass(inner_type)
                            and (issubclass(inner_type, BaseModel) or dataclasses.is_dataclass(inner_type))
                            and self.relationship_accessor.is_source_model_known(inner_type)
                        )
                    except TypeError:
                        is_list_of_known_models_fallback = False

                    if not is_list_of_known_models_fallback:
                        logger.debug(f"Type {type_to_match} is a non-model collection, selecting JsonFieldMapping.")
                        best_unit = JsonFieldMapping
                    # else: It was a list of known models, should have been handled earlier. Log warning?
                    else:
                        logger.warning(f"List of known models {type_to_match} reached fallback logic unexpectedly.")
                        # Default to M2M as a safe bet?
                        best_unit = ManyToManyFieldMapping
                # Handle dict/tuple
                elif unwrapped_origin in (dict, tuple) or type_to_match in (dict, tuple):
                    logger.debug(f"Type {type_to_match} is a dict/tuple collection, selecting JsonFieldMapping.")
                    best_unit = JsonFieldMapping
            # 2. Check for Any
            elif type_to_match is Any:
                logger.debug("Type is Any, selecting JsonFieldMapping.")
                best_unit = JsonFieldMapping

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
        gfk_details = None

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

            # --- GFK Check: Is the inner type a Union of known models? ---
            inner_origin = get_origin(inner_type)
            inner_args = get_args(inner_type)
            if inner_origin in (Union, UnionType) and inner_args:
                union_models = []
                other_types = [
                    arg
                    for arg in inner_args
                    if not (
                        inspect.isclass(arg)
                        and (issubclass(arg, BaseModel) or dataclasses.is_dataclass(arg))
                        and self.relationship_accessor.is_source_model_known(arg)
                    )
                ]
                union_models = [arg for arg in inner_args if arg not in other_types]

                if union_models and not other_types:
                    logger.debug(f"Detected GFK List[Union[...]] with models: {union_models}")
                    gfk_details = {
                        "type": "gfk",
                        "models": union_models,
                        "is_optional": is_optional,
                    }
                    unit_cls = JsonFieldMapping
                    base_py_type = original_py_type

            if unit_cls is None:
                # --- M2M Check: Is the inner type a known related BaseModel OR Dataclass? ---
                if (
                    inspect.isclass(inner_type)
                    and (issubclass(inner_type, BaseModel) or dataclasses.is_dataclass(inner_type))
                    and self.relationship_accessor.is_source_model_known(inner_type)
                ):
                    unit_cls = ManyToManyFieldMapping
                    base_py_type = inner_type
                    logger.debug(f"Detected List[RelatedModel] ({inner_type.__name__}), mapping to ManyToManyField.")
                else:
                    # --- Fallback for other lists ---
                    unit_cls = JsonFieldMapping
                    base_py_type = original_py_type
                    logger.debug(f"Detected List of non-models ({original_py_type}), mapping directly to JSONField.")

        # --- If not a list, find unit for the base (non-list) type ---
        if unit_cls is None:
            # --- Handle Union[BaseModel,...] Signaling FIRST --- #
            simplified_base_type = processed_type_info["type_obj"]
            simplified_origin = get_origin(simplified_base_type)
            simplified_args = get_args(simplified_base_type)

            logger.debug(
                f"Checking simplified type for Union[Model,...]: {simplified_base_type!r} (Origin: {simplified_origin})"
            )
            # Log the is_optional flag determined by TypeHandler
            logger.debug(f"TypeHandler returned is_optional: {is_optional} for original type: {original_py_type!r}")

            # Check if the simplified origin is Union[...] or T | U
            if simplified_origin in (Union, UnionType) and simplified_args:
                union_models = []
                other_types_in_union = []

                for arg in simplified_args:
                    # We already unwrapped Optional, so no need to check for NoneType here
                    logger.debug(f"-- Checking simplified Union arg: {arg!r}")

                    # Check if arg is a known BaseModel or Dataclass
                    is_class = inspect.isclass(arg)
                    # Need try-except for issubclass with non-class types
                    is_pyd_model = False
                    is_dc = False
                    is_known_by_accessor = False
                    if is_class:
                        try:
                            is_pyd_model = issubclass(arg, BaseModel)
                            is_dc = dataclasses.is_dataclass(arg)
                            # Only check accessor if it's a model type
                            if is_pyd_model or is_dc:
                                is_known_by_accessor = self.relationship_accessor.is_source_model_known(arg)
                        except TypeError:
                            # issubclass might fail if arg is not a class (e.g., a type alias)
                            pass  # Keep flags as False

                    logger.debug(
                        f"    is_class: {is_class}, is_pyd_model: {is_pyd_model}, is_dc: {is_dc}, is_known_by_accessor: {is_known_by_accessor}"
                    )

                    is_known_model_or_dc = is_class and (is_pyd_model or is_dc) and is_known_by_accessor

                    if is_known_model_or_dc:
                        logger.debug(f"    -> Added {arg.__name__} to union_models")  # More specific logging
                        union_models.append(arg)
                    else:
                        # Make sure we don't add NoneType here if Optional wasn't fully handled upstream somehow
                        if arg is not type(None):
                            logger.debug(f"    -> Added {arg!r} to other_types_in_union")  # More specific logging
                            other_types_in_union.append(arg)

                # --- EDIT: Only set union_details IF ONLY models were found ---
                # Add logging just before the check
                logger.debug(
                    f"Finished Union arg loop. union_models: {[m.__name__ for m in union_models]}, other_types: {other_types_in_union}"
                )
                if union_models and not other_types_in_union:
                    logger.debug(
                        f"Detected Union containing ONLY known models: {union_models}. Generating _union_details signal."
                    )
                    union_details = {
                        "type": "multi_fk",
                        "models": union_models,
                        "is_optional": is_optional,  # Use the flag determined earlier
                    }
                    # Log the created union_details
                    logger.debug(f"Generated union_details: {union_details!r}")
                    # Set unit_cls to JsonFieldMapping for model unions
                    unit_cls = JsonFieldMapping
                    base_py_type = original_py_type
                    logger.debug("Setting unit_cls to JsonFieldMapping for model union")

            # --- Now, find the unit for the (potentially complex) base type --- #
            # Only find unit if not already set (e.g. by model union handling)
            if unit_cls is None:
                # Determine the type to use for finding the unit.
                # If it was M2M or handled List, unit_cls is already set.
                # Otherwise, use the processed type_obj which handles Optional/Annotated.
                type_for_unit_finding = processed_type_info["type_obj"]
                logger.debug(f"Type used for finding unit (after Union check): {type_for_unit_finding!r}")

                # Use the simplified base type after processing Optional/Annotated
                base_py_type = type_for_unit_finding
                logger.debug(f"Finding unit for base type: {base_py_type!r} with field_info: {field_info}")
                unit_cls = self._find_unit_for_pydantic_type(base_py_type, field_info)

        # --- Check if a unit was found --- #
        if not unit_cls:
            # If _find_unit_for_pydantic_type returned None, fallback to JSON
            logger.warning(
                f"No mapping unit found by scoring for base type {base_py_type} "
                f"(derived from {original_py_type}), falling back to JSONField."
            )
            unit_cls = JsonFieldMapping
            # Consider raising MappingError if even JSON doesn't fit?
            # raise MappingError(f"Could not find mapping unit for Python type: {base_py_type}")

        # >> Add logging to check selected unit <<
        logger.info(f"Selected Unit for {original_py_type}: {unit_cls.__name__ if unit_cls else 'None'}")

        instance_unit = unit_cls()  # Instantiate to call methods

        # --- Determine Django Field Type ---
        # Start with the type defined on the selected unit class
        django_field_type = instance_unit.django_field_type

        # --- Get Kwargs (before potentially overriding field type for Enums) ---
        kwargs = instance_unit.pydantic_to_django_kwargs(base_py_type, field_info)

        # --- Add Union or GFK Details if applicable --- #
        if union_details:
            logger.info("Adding _union_details to kwargs.")
            kwargs["_union_details"] = union_details
            kwargs["null"] = union_details.get("is_optional", False)
            kwargs["blank"] = union_details.get("is_optional", False)
        elif gfk_details:
            logger.info("Adding _gfk_details to kwargs.")
            kwargs["_gfk_details"] = gfk_details
            # GFK fields are placeholder JSONFields, nullability is based on Optional status
            kwargs["null"] = is_optional
            kwargs["blank"] = is_optional
        else:
            logger.debug("union_details and gfk_details are None, skipping addition to kwargs.")
            # --- Special Handling for Enums/Literals (Only if not multi-FK/GFK union) --- #
            if unit_cls is EnumFieldMapping:
                field_type_hint = kwargs.pop("_field_type_hint", None)
                if field_type_hint and isinstance(field_type_hint, type) and issubclass(field_type_hint, models.Field):
                    # Directly use the hinted field type if valid
                    logger.debug(
                        f"Using hinted field type {field_type_hint.__name__} from EnumFieldMapping for {base_py_type}."
                    )
                    django_field_type = field_type_hint
                    # Ensure max_length is removed if type becomes IntegerField
                    if django_field_type == models.IntegerField:
                        kwargs.pop("max_length", None)
                else:
                    logger.warning("EnumFieldMapping selected but failed to get valid field type hint from kwargs.")

            # --- Handle Relationships (Only if not multi-FK union) --- #
            # This section needs to run *after* unit selection but *before* final nullability checks
            if unit_cls in (ForeignKeyMapping, OneToOneFieldMapping, ManyToManyFieldMapping):
                # Ensure base_py_type is the related model (set during M2M check or found by find_unit for FK/O2O)
                related_py_model = base_py_type

                # Check if it's a known Pydantic BaseModel OR a known Dataclass
                is_pyd_or_dc = inspect.isclass(related_py_model) and (
                    issubclass(related_py_model, BaseModel) or dataclasses.is_dataclass(related_py_model)
                )
                if not is_pyd_or_dc:
                    raise MappingError(
                        f"Relationship mapping unit {unit_cls.__name__} selected, but base type {related_py_model} is not a known Pydantic model or Dataclass."
                    )

                # Check for self-reference BEFORE trying to get the Django model
                is_self_ref = parent_pydantic_model is not None and related_py_model == parent_pydantic_model

                if is_self_ref:
                    model_ref = "self"
                    # Get the target Django model name for logging/consistency if possible, but use 'self'
                    # Check if the related model is a Pydantic BaseModel or a dataclass
                    if inspect.isclass(related_py_model) and issubclass(related_py_model, BaseModel):
                        target_django_model = self.relationship_accessor.get_django_model_for_pydantic(
                            cast(type[BaseModel], related_py_model)
                        )
                    elif dataclasses.is_dataclass(related_py_model):
                        target_django_model = self.relationship_accessor.get_django_model_for_dataclass(
                            related_py_model
                        )
                    else:
                        # This case should ideally not be reached due to earlier checks, but handle defensively
                        target_django_model = None
                        logger.warning(
                            f"Self-reference check: related_py_model '{related_py_model}' is neither BaseModel nor dataclass."
                        )

                    logger.debug(
                        f"Detected self-reference for {related_py_model.__name__ if inspect.isclass(related_py_model) else related_py_model} "
                        f"(Django: {getattr(target_django_model, '__name__', 'N/A')}), using 'self'."
                    )
                else:
                    # Get target Django model based on source type (Pydantic or Dataclass)
                    target_django_model = None
                    # Ensure related_py_model is actually a type before issubclass check
                    if inspect.isclass(related_py_model) and issubclass(related_py_model, BaseModel):
                        # Cast to satisfy type checker, as we've confirmed it's a BaseModel subclass here
                        target_django_model = self.relationship_accessor.get_django_model_for_pydantic(
                            cast(type[BaseModel], related_py_model)
                        )
                    elif dataclasses.is_dataclass(related_py_model):
                        target_django_model = self.relationship_accessor.get_django_model_for_dataclass(
                            related_py_model
                        )

                    if not target_django_model:
                        raise MappingError(
                            f"Cannot map relationship: No corresponding Django model found for source model "
                            f"{related_py_model.__name__} in RelationshipConversionAccessor."
                        )
                    # Use lowercase label for internal consistency with existing expectations
                    model_ref = getattr(target_django_model._meta, "label_lower", target_django_model.__name__)

                kwargs["to"] = model_ref
                django_field_type = unit_cls.django_field_type  # Re-confirm M2MField, FK, O2O type
                # Set on_delete for FK/O2O based on Optional status
                if unit_cls in (ForeignKeyMapping, OneToOneFieldMapping):
                    # Default to CASCADE for non-optional, SET_NULL for optional (matching test expectation)
                    kwargs["on_delete"] = (
                        models.SET_NULL if is_optional else models.CASCADE
                    )  # Changed PROTECT to CASCADE

        # --- Final Adjustments (Nullability, etc.) --- #
        # Apply nullability. M2M fields cannot be null in Django.
        # Do not override nullability if it was already forced by a multi-FK union
        if django_field_type != models.ManyToManyField and not union_details:
            kwargs["null"] = is_optional
            # Explicitly set blank based on optionality.
            # Simplified logic: Mirror the null assignment directly
            kwargs["blank"] = is_optional

        logger.debug(
            f"FINAL RETURN from get_django_mapping: Type={django_field_type}, Kwargs={kwargs}"
        )  # Added final state logging
        return django_field_type, kwargs

    def get_pydantic_mapping(self, dj_field: models.Field) -> tuple[Any, dict[str, Any]]:
        """Get the corresponding Pydantic type hint and FieldInfo kwargs for a Django Field."""
        dj_field_type = type(dj_field)
        is_optional = dj_field.null
        is_choices = bool(dj_field.choices)

        # --- Find base unit (ignoring choices for now) ---
        # Find the mapping unit based on the specific Django field type MRO
        # This gives us the correct underlying Python type (str, int, etc.)
        base_unit_cls = self._find_unit_for_django_field(dj_field_type)

        if not base_unit_cls:
            logger.warning(f"No base mapping unit for {dj_field_type.__name__}, falling back to Any.")
            pydantic_type = Optional[Any] if is_optional else Any
            return pydantic_type, {}

        base_instance_unit = base_unit_cls()
        # Get the base Pydantic type from this unit
        final_pydantic_type = base_instance_unit.python_type

        # --- Determine Final Pydantic Type Adjustments --- #
        # (Relationships, AutoPK, Optional wrapper)

        # Handle choices FIRST to determine the core type before Optional wrapping
        if is_choices:
            # Default to base type, override if valid choices found
            final_pydantic_type = base_instance_unit.python_type
            if dj_field.choices:  # Explicit check before iteration
                try:
                    choice_values = tuple(choice[0] for choice in dj_field.choices)
                    if choice_values:  # Ensure the tuple is not empty
                        final_pydantic_type = Literal[choice_values]  # type: ignore
                        logger.debug(f"Mapped choices for '{dj_field.name}' to Pydantic type: {final_pydantic_type}")
                    else:
                        logger.warning(
                            f"Field '{dj_field.name}' has choices defined, but extracted values are empty. Falling back."
                        )
                        # Keep final_pydantic_type as base type
                except Exception as e:
                    logger.warning(f"Failed to extract choices for field '{dj_field.name}'. Error: {e}. Falling back.")
                    # Keep final_pydantic_type as base type
            # If dj_field.choices was None/empty initially, final_pydantic_type remains the base type
        else:
            # Get the base Pydantic type from this unit if not choices
            final_pydantic_type = base_instance_unit.python_type

        # 1. Handle Relationships first, as they determine the core type
        if base_unit_cls in (ForeignKeyMapping, OneToOneFieldMapping, ManyToManyFieldMapping):
            related_dj_model = getattr(dj_field, "related_model", None)
            if not related_dj_model:
                raise MappingError(f"Cannot determine related Django model for field '{dj_field.name}'")

            # Resolve 'self' reference
            if related_dj_model == "self":
                # We need the Django model class that dj_field belongs to.
                # This info isn't directly passed, so this approach might be limited.
                # Assuming self-reference points to the same type hierarchy for now.
                # A better solution might need the model context passed down.
                logger.warning(
                    f"Handling 'self' reference for field '{dj_field.name}'. Mapping might be incomplete without parent model context."
                )
                # Attempt to get Pydantic model mapped to the field's owner model if possible (heuristically)
                # This is complex and potentially fragile.
                # For now, let's use a placeholder or raise an error if needed strictly.
                # Sticking with the base type (e.g., Any or int for PK) might be safer without context.
                # Use the base type (likely PK int/uuid) as the fallback type here
                target_pydantic_model = base_instance_unit.python_type
                logger.debug(f"Using Any as placeholder for 'self' reference '{dj_field.name}'")
            else:
                target_pydantic_model = self.relationship_accessor.get_pydantic_model_for_django(related_dj_model)

            if not target_pydantic_model or target_pydantic_model is Any:
                if related_dj_model != "self":  # Avoid redundant warning for self
                    logger.warning(
                        f"Cannot map relationship: No corresponding Pydantic model found for Django model "
                        f"'{related_dj_model._meta.label if hasattr(related_dj_model, '_meta') else related_dj_model.__name__}'. "
                        f"Using placeholder '{final_pydantic_type}'."
                    )
                # Keep final_pydantic_type as the base unit's python_type (e.g., int for FK)
            else:
                if base_unit_cls == ManyToManyFieldMapping:
                    final_pydantic_type = list[target_pydantic_model]
                else:  # FK or O2O
                    # Keep the PK type (e.g., int) if target model not found,
                    # otherwise use the target Pydantic model type.
                    final_pydantic_type = target_pydantic_model  # This should now be the related model type
                logger.debug(f"Mapped relationship field '{dj_field.name}' to Pydantic type: {final_pydantic_type}")

        # 2. AutoPK override (after relationship resolution)
        is_auto_pk = dj_field.primary_key and isinstance(
            dj_field, (models.AutoField, models.BigAutoField, models.SmallAutoField)
        )
        if is_auto_pk:
            final_pydantic_type = Optional[int]
            logger.debug(f"Mapped AutoPK field '{dj_field.name}' to {final_pydantic_type}")
            is_optional = True  # AutoPKs are always optional in Pydantic input

        # 3. Apply Optional[...] wrapper if necessary (AFTER relationship/AutoPK)
        # Do not wrap M2M lists or already Optional AutoPKs in Optional[] again.
        # Also, don't wrap if the type is already Literal (choices handled Optionality) - NO, wrap Literal too if null=True
        if is_optional and not is_auto_pk:  # Check if is_choices? No, optional applies to literal too.
            origin = get_origin(final_pydantic_type)
            args = get_args(final_pydantic_type)
            is_already_optional = origin is Optional or origin is UnionType and type(None) in args

            if not is_already_optional:
                final_pydantic_type = Optional[final_pydantic_type]
                logger.debug(f"Wrapped type for '{dj_field.name}' in Optional: {final_pydantic_type}")

        # --- Generate FieldInfo Kwargs --- #
        # Use EnumFieldMapping logic for kwargs ONLY if choices exist,
        # otherwise use the base unit determined earlier. # --> NO, always use base unit for kwargs now. Literal type handles choices.
        # kwargs_unit_cls = EnumFieldMapping if is_choices else base_unit_cls # OLD logic
        instance_unit = base_unit_cls()  # Use the base unit (e.g., StrFieldMapping) for base kwargs

        field_info_kwargs = instance_unit.django_to_pydantic_field_info_kwargs(dj_field)

        # --- Explicitly cast title (verbose_name) and description (help_text) --- #
        if field_info_kwargs.get("title") is not None:
            field_info_kwargs["title"] = str(field_info_kwargs["title"])
            logger.debug(f"Ensured title is str for '{dj_field.name}': {field_info_kwargs['title']}")
        if field_info_kwargs.get("description") is not None:
            field_info_kwargs["description"] = str(field_info_kwargs["description"])
            logger.debug(f"Ensured description is str for '{dj_field.name}': {field_info_kwargs['description']}")
        # --- End Casting --- #

        # --- Keep choices in json_schema_extra even when using Literal ---
        # This preserves the (value, label) mapping as metadata alongside the Literal type.
        if (
            is_choices
            and "json_schema_extra" in field_info_kwargs
            and "choices" in field_info_kwargs["json_schema_extra"]
        ):
            logger.debug(f"Kept choices in json_schema_extra for Literal field '{dj_field.name}'")
        elif is_choices:
            logger.debug(
                f"Field '{dj_field.name}' has choices, but they weren't added to json_schema_extra by the mapping unit."
            )

        # Clean up redundant `default=None` for Optional fields handled by Pydantic v2.
        # Do not force-add default=None; only keep explicit defaults (e.g., for AutoPK).
        if is_optional:
            if field_info_kwargs.get("default") is None and not is_auto_pk:
                # Remove implicit default=None for Optional fields
                field_info_kwargs.pop("default", None)
                logger.debug(f"Removed redundant default=None for Optional field '{dj_field.name}'")
            elif is_auto_pk and "default" not in field_info_kwargs:
                # Keep default=None for AutoPK if not already set
                field_info_kwargs["default"] = None
                logger.debug(f"Set default=None for AutoPK Optional field '{dj_field.name}'")

        logger.debug(
            f"Final Pydantic mapping for '{dj_field.name}': Type={final_pydantic_type}, Kwargs={field_info_kwargs}"
        )
        return final_pydantic_type, field_info_kwargs
