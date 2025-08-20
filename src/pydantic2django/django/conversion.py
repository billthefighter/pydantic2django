"""
Provides functionality to convert Django model instances to Pydantic models.
"""
import datetime as dt
import json
import logging
from typing import Annotated, Any, ForwardRef, Generic, Optional, TypeVar, Union, cast, get_args, get_origin
from uuid import UUID

from django.contrib.contenttypes.fields import GenericForeignKey
from django.db import models
from django.db.models.fields.related import ForeignKey, ManyToManyField, OneToOneField, RelatedField
from django.db.models.fields.reverse_related import (
    ForeignObjectRel,
    ManyToManyRel,
    ManyToOneRel,
    OneToOneRel,
)
from django.db.models.query import QuerySet
from django.utils.timezone import get_default_timezone, is_aware, make_aware
from pydantic import BaseModel, Field, TypeAdapter, create_model
from pydantic.types import Json

# from .mapping import TypeMapper # Removed old import
# Add imports for new mapper and accessor
from pydantic2django.core.bidirectional_mapper import BidirectionalTypeMapper
from pydantic2django.core.relationships import RelationshipConversionAccessor

# Potentially useful imports from the project (adjust as needed)
# from .mapping import TypeMapper # Might not be directly needed if we create reverse mapping here
# from .typing import ...
# from ..core.utils import ...


logger = logging.getLogger(__name__)

PydanticModelT = TypeVar("PydanticModelT", bound=BaseModel)
DjangoModelT = TypeVar("DjangoModelT", bound=models.Model)


# --- Helper Functions ---
def _is_pydantic_json_annotation(annotation: Any) -> bool:
    """Checks if an annotation represents pydantic.Json, handling Optional and Annotated."""
    origin = get_origin(annotation)
    args = get_args(annotation)

    # Direct Json type
    if annotation is Json:
        return True

    # Optional[Json]
    if origin is Union and type(None) in args:
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1 and non_none_args[0] is Json:
            return True

    # Annotated[Json, ...]
    if origin is Annotated:
        base_type = args[0] if args else None
        if base_type is Json:
            return True

    # Optional[Annotated[Json, ...]]
    if origin is Union and type(None) in args:
        non_none_arg = next((arg for arg in args if arg is not type(None)), None)
        if non_none_arg and get_origin(non_none_arg) is Annotated:
            annotated_args = get_args(non_none_arg)
            base_type = annotated_args[0] if annotated_args else None
            if base_type is Json:
                return True

    # Check cases like Union[Json, str], etc.
    if origin is Union:
        # Direct Json in Union
        if Json in args:
            return True
        # Annotated[Json, ...] in Union
        if any(
            get_origin(arg) is Annotated and get_args(arg)[0] is Json for arg in args if get_origin(arg) is Annotated
        ):
            return True

    # If none of the above match, it's not considered a Json annotation for serialization purposes.
    logger.debug(f"Annotation {annotation!r} determined NOT to be pydantic.Json type for serialization purposes.")
    return False


# --- Metadata Extraction --- (Moved Before Usage)

# GeneratedModelCache = dict[type[models.Model], type[BaseModel] | ForwardRef]  # type: ignore[misc]
# Redefine cache to use model name (str) as key for consistency
GeneratedModelCache = dict[str, Union[type[BaseModel], ForwardRef]]


# --- Conversion Functions ---


def django_to_pydantic(
    db_obj: DjangoModelT,
    pydantic_model: type[PydanticModelT],
    *,
    exclude: set[str] | None = None,
    depth: int = 0,  # Add depth to prevent infinite recursion
    max_depth: int = 3,  # Set a default max depth
) -> PydanticModelT:
    """
    Converts a Django model instance to a Pydantic model instance.

    Args:
        db_obj: The Django model instance to convert.
        pydantic_model: The target Pydantic model class.
        exclude: A set of field names to exclude from the conversion.
        depth: Current recursion depth (internal use).
        max_depth: Maximum recursion depth for related models.

    Returns:
        An instance of the target Pydantic model populated with data
        from the Django model instance.

    Raises:
        ValueError: If conversion fails or recursion depth is exceeded.
        AttributeError: If a field expected by Pydantic doesn't exist on the Django model.
    """
    if depth > max_depth:
        logger.warning(
            f"Maximum recursion depth ({max_depth}) exceeded for {pydantic_model.__name__} from {db_obj.__class__.__name__}"
        )
        # Decide how to handle this: raise error or return None/partial data?
        # For now, let's raise an error to be explicit.
        raise ValueError(f"Maximum recursion depth ({max_depth}) exceeded.")

    # Extract metadata if not provided
    # if django_metadata is None: # Removed reference to django_metadata
    #     django_metadata = _extract_django_model_metadata(db_obj.__class__)

    exclude = exclude or set()
    data: dict[str, Any] = {}
    # Use pre-extracted metadata if available - This logic needs adjustment as _extract_django_model_metadata is not defined
    # metadata_source = _extract_django_model_metadata(db_obj.__class__) # Directly call extraction

    pydantic_fields = pydantic_model.model_fields

    for field_name, pydantic_field in pydantic_fields.items():
        if field_name in exclude:
            logger.debug(f"Skipping excluded field '{field_name}'")
            continue

        logger.debug(
            f"Processing Pydantic field: {field_name} (Depth: {depth}) for Django model {db_obj.__class__.__name__}"
        )

        # Check if field exists on Django model instance
        if not hasattr(db_obj, field_name):
            logger.warning(f"Field '{field_name}' not found on Django model {db_obj.__class__.__name__}. Skipping.")
            continue

        django_value = getattr(db_obj, field_name)
        pydantic_annotation = pydantic_field.annotation
        origin = get_origin(pydantic_annotation)
        args = get_args(pydantic_annotation)
        # meta = metadata_source.get(field_name) # Removed metadata_source

        # Get Django field object for type checking
        try:
            django_field = db_obj._meta.get_field(field_name)
        except models.FieldDoesNotExist:
            django_field = None
            logger.debug(f"Could not find Django field for Pydantic field '{field_name}'. Handling as simple field.")

        # --- 1. Handle Relations ---
        # if meta and meta.is_relation: # Replaced meta check
        if django_field and isinstance(django_field, RelatedField):
            if depth >= max_depth:
                logger.warning(
                    f"Max depth ({max_depth}) reached for relation '{field_name}' at depth {depth}. Assigning PK(s) or None/[] based on relation type."
                )
                # Replace meta.is_m2m check
                if isinstance(django_field, ManyToManyField):
                    data[field_name] = []  # Default for max depth M2M is empty list
                else:  # FK or O2O
                    data[field_name] = None  # Default for max depth FK/O2O is None
                continue  # Relation handled (depth limit), move to next field

            # --- Attempt Recursive Conversion ---\n            target_related_pydantic_model: Optional[type[BaseModel]] = None

            # Determine target Pydantic model for FK/O2O
            # Replace meta.is_fk or meta.is_o2o check
            if isinstance(django_field, (ForeignKey, OneToOneField)):
                # Simplified logic: get potential model from annotation
                potential_model_type = next((t for t in args if isinstance(t, type) and issubclass(t, BaseModel)), None)
                if (
                    not potential_model_type
                    and isinstance(pydantic_annotation, type)
                    and issubclass(pydantic_annotation, BaseModel)
                ):
                    potential_model_type = pydantic_annotation
                # Add ForwardRef resolution attempt if needed (simplified)
                if isinstance(next((t for t in args if t is not type(None)), pydantic_annotation), ForwardRef):
                    # Ensure we use the *resolved* annotation if the original was ForwardRef
                    resolved_annotation = pydantic_model.model_fields[field_name].annotation
                    potential_model_type = next(
                        (
                            t
                            for t in get_args(resolved_annotation)  # Check args of resolved Optional[Model] etc.
                            if isinstance(t, type) and issubclass(t, BaseModel)
                        ),
                        # Fallback check if resolved annotation is directly the model (not Optional[Model])
                        resolved_annotation
                        if isinstance(resolved_annotation, type) and issubclass(resolved_annotation, BaseModel)
                        else None,
                    )

                target_related_pydantic_model = potential_model_type

                if target_related_pydantic_model:
                    logger.debug(
                        f"Handling FK/O2O relationship for '{field_name}' -> {target_related_pydantic_model.__name__}"
                    )
                    related_obj = django_value
                    if related_obj:
                        try:
                            data[field_name] = django_to_pydantic(
                                related_obj,
                                target_related_pydantic_model,
                                exclude=exclude,
                                depth=depth + 1,
                                max_depth=max_depth,
                            )
                        except ValueError as e:
                            logger.error(f"Failed converting related object in FK/O2O '{field_name}': {e}")
                            data[field_name] = None  # Assign None on conversion failure (e.g. nested depth limit)
                    else:
                        data[field_name] = None  # Django FK/O2O was None
                    continue  # Relation handled (recursion success/failure/was None), move to next field
                else:
                    # Could not determine target Pydantic model, assign PK
                    logger.warning(
                        f"Could not determine target Pydantic model for FK/O2O '{field_name}'. Assigning PK."
                    )
                    # Extract PK using field_name_id convention
                    data[field_name] = getattr(db_obj, f"{field_name}_id", None)
                    continue  # Relation handled (assigned PK), move to next field

            # Determine target Pydantic model for M2M
            # Replace meta.is_m2m check
            elif isinstance(django_field, ManyToManyField):
                # Simplified logic: get potential model from List[Model] or List[ForwardRef]
                target_related_pydantic_model = None
                if origin is list and args:
                    potential_model_type = args[0]
                    if isinstance(potential_model_type, type) and issubclass(potential_model_type, BaseModel):
                        target_related_pydantic_model = potential_model_type
                    elif isinstance(potential_model_type, ForwardRef):
                        # Ensure we use the *resolved* annotation if the original was ForwardRef
                        resolved_annotation = pydantic_model.model_fields[field_name].annotation
                        resolved_args = get_args(resolved_annotation)
                        if (
                            resolved_args
                            and isinstance(resolved_args[0], type)
                            and issubclass(resolved_args[0], BaseModel)
                        ):
                            target_related_pydantic_model = resolved_args[0]

                if target_related_pydantic_model:
                    logger.debug(
                        f"Handling M2M relationship for '{field_name}' -> List[{target_related_pydantic_model.__name__}]"
                    )
                    related_manager = django_value
                    converted_related = []
                    if hasattr(related_manager, "all"):
                        try:
                            for related_obj in related_manager.all():
                                try:
                                    converted_related.append(
                                        django_to_pydantic(
                                            related_obj,
                                            target_related_pydantic_model,
                                            exclude=exclude,
                                            depth=depth + 1,
                                            max_depth=max_depth,
                                        )
                                    )
                                except ValueError as e:
                                    logger.error(f"Failed converting related object in M2M '{field_name}': {e}")
                                    continue  # Skip item on depth error
                        except Exception as e:
                            logger.error(f"Error accessing M2M manager for '{field_name}': {e}", exc_info=True)
                    else:
                        logger.warning(
                            f"Expected a manager for M2M field '{field_name}' but got {type(django_value)}. Assigning empty list."
                        )
                    data[field_name] = converted_related
                    continue  # Relation handled (recursion), move to next field
                else:
                    # Could not determine target Pydantic model, assign list of PKs
                    logger.warning(
                        f"Could not determine target Pydantic model for M2M '{field_name}'. Assigning list of PKs."
                    )
                    related_manager = django_value
                    pk_list = []
                    if hasattr(related_manager, "all"):
                        try:
                            pk_list = list(related_manager.values_list("pk", flat=True))
                        except Exception as e:
                            logger.error(f"Error getting PKs for M2M field '{field_name}': {e}", exc_info=True)
                    else:
                        logger.warning(
                            f"Expected a manager for M2M field '{field_name}' but got {type(django_value)}. Assigning empty PK list."
                        )
                    data[field_name] = pk_list
                    continue  # Relation handled (assigned PK list), move to next field

        # --- 2. Handle FileField/ImageField ---
        # if meta and issubclass(meta.django_field_type, models.FileField): # Replaced meta check
        if django_field and isinstance(django_field, models.FileField):
            field_value = getattr(db_obj, field_name)  # Use original django_value which is the FieldFile
            # Check if Pydantic field is required
            is_pydantic_required = pydantic_field.is_required()
            # Get URL if field has a value, otherwise None
            url_value = field_value.url if field_value else None

            if url_value is None and is_pydantic_required:
                # If Django value is None/empty but Pydantic requires a string, assign empty string
                data[field_name] = ""
                logger.debug(
                    f"Handling FileField/ImageField '{field_name}' -> Assigned empty string for required field."
                )
            else:
                # Otherwise, assign the URL (which might be None if Pydantic field is Optional)
                data[field_name] = url_value
                logger.debug(f"Handling FileField/ImageField '{field_name}' -> URL: {data.get(field_name)}")
            continue  # FileField handled, move to next field

        # --- 3. Handle JSONField ---
        # if meta and meta.django_field_type == models.JSONField: # Replaced meta check
        if django_field and isinstance(django_field, models.JSONField):
            logger.debug(f"Field '{field_name}' is a Django JSONField. Attempting serialization.")
            logger.debug(f"Field '{field_name}': Django value type: {type(django_value)}, value: {django_value!r}")
            value_to_assign = None
            if django_value is not None:
                try:
                    # Always serialize if the source is JSONField
                    value_to_assign = json.dumps(django_value)
                    logger.debug(f"Field '{field_name}': Serialized value: {value_to_assign!r}")
                except TypeError as e:
                    logger.error(f"Failed to serialize JSON for field '{field_name}': {e}", exc_info=True)
                    # Fallback: assign raw value if serialization fails? Or None?
                    # Assigning raw value might still lead to validation error later.
                    # Let's assign None for now if serialization fails.
                    value_to_assign = None
            data[field_name] = value_to_assign
            logger.debug(f"Field '{field_name}': Assigning serialized/fallback value: {data.get(field_name)!r}")
            continue  # JSONField handled

        # --- Check Pydantic Annotation (secondary check for non-JSONField Django types) ---
        # This path is now less likely to be hit if source is JSONField
        is_target_pydantic_json = _is_pydantic_json_annotation(pydantic_annotation)
        if is_target_pydantic_json and not (
            django_field and isinstance(django_field, models.JSONField)
        ):  # Avoid re-processing JSONField
            logger.debug(
                f"Field '{field_name}' (Django type: {type(django_field).__name__ if django_field else 'Unknown'}) has pydantic.Json annotation. Attempting serialization."
            )
            # Same serialization logic as above
            value_to_assign = None
            if django_value is not None:
                try:
                    value_to_assign = json.dumps(django_value)
                except TypeError as e:
                    logger.error(
                        f"Failed to serialize non-JSONField '{field_name}' for pydantic.Json target: {e}", exc_info=True
                    )
                    value_to_assign = None
            data[field_name] = value_to_assign
            continue

        # --- 4. Handle Simple Fields (Default) ---
        logger.debug(f"Handling simple/property/other field '{field_name}' with value: {django_value!r}")
        data[field_name] = django_value

    # Instantiate the Pydantic model with the collected data
    try:
        instance = pydantic_model(**data)
        logger.info(
            f"Successfully converted {db_obj.__class__.__name__} instance (PK: {db_obj.pk}) to {pydantic_model.__name__}"
        )
        return cast(PydanticModelT, instance)
    except Exception as e:
        logger.error(f"Failed to instantiate Pydantic model {pydantic_model.__name__} with data {data}", exc_info=True)
        raise ValueError(
            f"Failed to create Pydantic model {pydantic_model.__name__} from Django instance {db_obj}: {e}"
        ) from e


# --- Dynamic Pydantic Model Generation ---


def generate_pydantic_class(
    django_model_cls: type[models.Model],
    mapper: BidirectionalTypeMapper,
    *,
    model_name: Optional[str] = None,
    cache: Optional[GeneratedModelCache] = None,
    depth: int = 0,
    max_depth: int = 3,
    pydantic_base: Optional[type[BaseModel]] = None,
    exclude: Optional[set[str]] = None,
) -> "Union[type[BaseModel], ForwardRef]":
    """
    Dynamically generates a Pydantic model class from a Django model class,
    using pre-extracted metadata if provided.

    Args:
        django_model_cls: The Django model class to convert.
        mapper: The BidirectionalTypeMapper instance for mapping between Django and Pydantic types.
        model_name: Optional explicit name for the generated Pydantic model.
                    Defaults to f"{django_model_cls.__name__}Pydantic".
        cache: A dictionary to cache generated models and prevent recursion errors.
               Must be provided for recursive generation. Keys are model names.
        depth: Current recursion depth.
        max_depth: Maximum recursion depth for related models.
        pydantic_base: Optional base for generated Pydantic model.
        exclude: Field names to exclude during generation.

    Returns:
        A dynamically created Pydantic model class or a ForwardRef if max_depth is hit.

    Raises:
        ValueError: If maximum recursion depth is exceeded or generation fails.
        TypeError: If a field type cannot be mapped.
    """
    logger.info(f"*** Running generate_pydantic_class [v_lazy_fix] for {django_model_cls.__name__} ***")

    if cache is None:
        cache = {}

    # Ensure exclude is a set
    exclude_set = exclude or set()

    pydantic_model_name = model_name or f"{django_model_cls.__name__}Pydantic"
    logger.debug(
        f"Generating Pydantic model '{pydantic_model_name}' for Django model '{django_model_cls.__name__}' (Depth: {depth})"
    )

    # --- Cache Check --- (Check if ACTUAL model is already cached)
    if pydantic_model_name in cache:
        cached_item = cache[pydantic_model_name]
        if isinstance(cached_item, type) and issubclass(cached_item, BaseModel):
            logger.debug(f"Cache hit (actual class) for name '{pydantic_model_name}' (Depth: {depth})")
            return cached_item
        elif isinstance(cached_item, ForwardRef):
            # If a ForwardRef is in the cache, it means we hit max_depth earlier or are in a loop
            logger.debug(f"Cache hit (ForwardRef) for name '{pydantic_model_name}' (Depth: {depth})")
            return cached_item

    # --- Max Depth Check --- (Return ForwardRef, but DON'T cache it here)
    if depth > max_depth:
        logger.warning(
            f"Max recursion depth ({max_depth}) reached for {django_model_cls.__name__}. Returning ForwardRef."
        )
        # Store the ForwardRef in the cache ONLY if max depth is reached
        # This prevents premature caching before dependencies are potentially resolved.
        forward_ref = ForwardRef(pydantic_model_name)
        cache[pydantic_model_name] = forward_ref  # Cache the ForwardRef when hitting max depth
        return forward_ref

    # --- Process Fields (Recursively generate dependencies FIRST) ---
    # Metadata extraction is now handled within the loop using the mapper

    field_definitions: dict[str, tuple[Any, Any]] = {}
    # Use set[Any] for dependencies to avoid complex Union typing with ForwardRef
    model_dependencies: set[Any] = set()

    # TEMPORARILY place a ForwardRef to handle self-references during field processing
    # This will be replaced by the actual model later.
    temp_forward_ref = ForwardRef(pydantic_model_name)
    cache[pydantic_model_name] = temp_forward_ref
    logger.debug(f"Placed TEMPORARY ForwardRef '{pydantic_model_name}' in cache for self-ref handling (Depth: {depth})")

    # Iterate through Django model fields directly
    for dj_field in django_model_cls._meta.get_fields(include_hidden=False):
        # Skip reverse relations and non-concrete fields immediately
        if isinstance(
            dj_field, (ForeignObjectRel, OneToOneRel, ManyToOneRel, ManyToManyRel, GenericForeignKey)
        ) or not getattr(dj_field, "concrete", False):
            logger.debug(f"Skipping non-concrete/reverse field '{dj_field.name}' of type {type(dj_field).__name__}")
            continue

        field_name = dj_field.name
        # --- Check Exclusion BEFORE processing ---
        if field_name in exclude_set:
            logger.debug(f"  Skipping excluded field: {field_name}")
            continue

        logger.debug(f"  Processing field: {field_name} ({type(dj_field).__name__}) using mapper...")

        # --- Use Mapper to get Pydantic Type and FieldInfo --- #
        try:
            initial_type, field_info_kwargs = mapper.get_pydantic_mapping(dj_field)
            # Initialize final_type here to avoid unbound error later
            final_type = initial_type
        except Exception as e:
            logger.error(f"  Error getting initial Pydantic mapping for field '{field_name}': {e}", exc_info=True)
            continue

        # --- Handle Relationships (Recursive Generation) --- #
        is_relation_field = isinstance(dj_field, RelatedField)
        related_model_cls = getattr(dj_field, "related_model", None)
        is_self_ref = related_model_cls == "self" or related_model_cls == django_model_cls

        if is_relation_field and related_model_cls:
            # Check depth *before* recursive call
            if depth >= max_depth:
                # Max depth reached - Use initial type (likely PK representation from mapper)
                logger.warning(
                    f"Max recursion depth ({max_depth}) reached for {related_model_cls.__name__} relation '{field_name}'. Using initial type/PK representation."
                )
                # final_type is already initial_type from above
                is_relation_field = False  # Treat as non-relation for dependency tracking
            else:
                # Depth not exceeded, proceed with recursive generation logic
                try:  # Outer try for recursive generation logic
                    resolved_related_model_cls: Optional[type[models.Model]] = None
                    if related_model_cls == "self":
                        resolved_related_model_cls = django_model_cls
                    elif isinstance(related_model_cls, type) and issubclass(related_model_cls, models.Model):
                        resolved_related_model_cls = related_model_cls

                    needs_recursive_call = False
                    if resolved_related_model_cls:
                        # Check if it's a Django model to determine if recursion makes sense
                        if isinstance(resolved_related_model_cls, type) and issubclass(
                            resolved_related_model_cls, models.Model
                        ):
                            target_pydantic_name = f"{resolved_related_model_cls.__name__}Pydantic"
                            # Prevent infinite loop on self-reference if already processing this model
                            if not (
                                is_self_ref
                                and target_pydantic_name in cache
                                and cache[target_pydantic_name] == temp_forward_ref
                            ):
                                needs_recursive_call = True

                    if needs_recursive_call and resolved_related_model_cls:
                        logger.debug(
                            f"    Recursively generating/fetching for related model: {resolved_related_model_cls.__name__}"
                        )
                        related_pydantic_model_ref = generate_pydantic_class(
                            resolved_related_model_cls,
                            mapper,
                            model_name=None,  # Generate default name
                            cache=cache,
                            depth=depth + 1,
                            max_depth=max_depth,
                            pydantic_base=pydantic_base,
                            exclude=exclude,
                        )

                        # Check if recursion returned a valid type and try re-mapping
                        if (
                            isinstance(related_pydantic_model_ref, type)
                            or type(related_pydantic_model_ref) is ForwardRef
                        ):
                            model_dependencies.add(related_pydantic_model_ref)  # type: ignore[arg-type]
                            # --- Re-evaluate the type AFTER recursive call ---
                            try:  # Inner try for re-mapping
                                updated_type, updated_kwargs = mapper.get_pydantic_mapping(dj_field)
                                final_type = updated_type  # Update final_type ONLY if re-map succeeds
                                field_info_kwargs = updated_kwargs  # Update kwargs too
                                logger.debug(f"    Re-evaluated type for '{field_name}' after recursion: {final_type}")
                            except Exception as re_map_error:
                                logger.warning(
                                    f"    Failed to re-evaluate mapping for '{field_name}' after recursion, using initial type. Error: {re_map_error}"
                                )
                                # Keep final_type = initial_type (already set) on re-map failure
                        else:
                            # Recursion failed to return a valid type
                            logger.warning(
                                f"Recursive call for relation '{field_name}' did not return Model or ForwardRef. Type: {type(related_pydantic_model_ref)}. Using initial type."
                            )
                            # Keep final_type = initial_type (already set)

                    elif is_relation_field:  # Recursion skipped (e.g., self-ref or other condition)
                        # Use the initial type determined by the mapper earlier
                        logger.debug(
                            f"    Skipping recursive call for relation '{field_name}' (e.g. self-ref or other condition). Using initial type."
                        )
                        ref_to_add = get_args(initial_type)[0] if get_origin(initial_type) is list else initial_type
                        model_dependencies.add(ref_to_add)  # type: ignore[arg-type]
                        # Keep final_type = initial_type (already set)

                except Exception as e:  # Except for outer try (recursive generation logic)
                    logger.error(
                        f"Error during recursive generation logic for field '{field_name}': {e}", exc_info=True
                    )
                    # Keep final_type = initial_type and continue to next field definition
                    continue  # Skip field definition steps on this error
        # else: # Not a relation field
        # final_type is already initial_type

        # --- Final Type Adjustment and Field Definition ---
        # 'final_type' should now always be defined (initialized to initial_type)

        # The mapper should have handled Optional wrapping based on dj_field.null
        # The mapper's field_info_kwargs should contain defaults etc.
        field_instance: Any = ...  # Default to required

        # Determine Field definition based on potentially updated kwargs
        if field_info_kwargs:
            try:
                # --- Explicitly resolve lazy proxies in choices within json_schema_extra --- #
                if "json_schema_extra" in field_info_kwargs and isinstance(
                    field_info_kwargs["json_schema_extra"], dict
                ):
                    choices_data = field_info_kwargs["json_schema_extra"].get("choices")
                    if isinstance(choices_data, list):
                        resolved_choices = []
                        for choice_item in choices_data:
                            if isinstance(choice_item, (list, tuple)) and len(choice_item) == 2:
                                value, label = choice_item
                                # Force label to string if it's a lazy proxy or potentially other types
                                resolved_choices.append((value, str(label)))
                            else:
                                resolved_choices.append(choice_item)  # Keep invalid item as is for now
                        field_info_kwargs["json_schema_extra"]["choices"] = resolved_choices
                        logger.debug(f"    Resolved lazy proxies in choices for field '{field_name}'")

                # Ensure optional Django fields are not required in Pydantic if no explicit default provided
                if dj_field.null and "default" not in field_info_kwargs:
                    field_info_kwargs["default"] = None
                field_instance = Field(**field_info_kwargs)
            except TypeError as field_exc:
                logger.error(f"Invalid FieldInfo kwargs for '{field_name}': {field_info_kwargs}. Error: {field_exc}")
                # Fallback: Check Django field nullability
                field_instance = None if dj_field.null else ...
        else:
            # No field info kwargs: Check Django field nullability
            field_instance = None if dj_field.null else ...

        field_definitions[field_name] = (final_type, field_instance)
        logger.debug(f"  Defined field '{field_name}': Type={final_type}, Definition={field_instance!r}")

    # --- Create the Pydantic Model Class ---
    model_base = pydantic_base or BaseModel
    try:
        model_cls = create_model(
            pydantic_model_name,
            __base__=model_base,
            **field_definitions,  # type: ignore
        )
        logger.info(f"Successfully created Pydantic model class '{pydantic_model_name}'")

        # --- IMPORTANT: Update cache with the *actual* class, replacing the temp ForwardRef ---
        cache[pydantic_model_name] = model_cls
        logger.debug(f"Updated cache for '{pydantic_model_name}' with actual class object.")

        # --- ALSO IMPORTANT: Register the mapping in the accessor ---
        if mapper and mapper.relationship_accessor:
            # Assume generate_pydantic_class only deals with Pydantic for now
            mapper.relationship_accessor.map_relationship(source_model=model_cls, django_model=django_model_cls)
            logger.debug(f"Registered mapping for {model_cls.__name__} <-> {django_model_cls.__name__} in accessor.")

        return model_cls

    except Exception as e:
        logger.error(f"Failed to create Pydantic model '{pydantic_model_name}' using create_model: {e}", exc_info=True)
        # If creation fails, remove the temporary ForwardRef if it exists
        if cache.get(pydantic_model_name) is temp_forward_ref:
            del cache[pydantic_model_name]
        raise ValueError(f"Failed to create Pydantic model '{pydantic_model_name}'") from e


class DjangoPydanticConverter(Generic[DjangoModelT]):
    """
    Manages the conversion lifecycle between a Django model instance
    and a dynamically generated Pydantic model.

    Handles:
    1. Generating a Pydantic model class definition from a Django model class.
    2. Converting a Django model instance to an instance of the generated Pydantic model.
    3. Converting a Pydantic model instance back to a saved Django model instance.
    """

    def __init__(
        self,
        django_model_or_instance: Union[type[DjangoModelT], DjangoModelT],
        *,
        max_depth: int = 3,
        exclude: Optional[set[str]] = None,
        pydantic_base: Optional[type[BaseModel]] = None,
        model_name: Optional[str] = None,
    ):
        """
        Initializes the converter.

        Args:
            django_model_or_instance: Either the Django model class or an instance.
            max_depth: Maximum recursion depth for generating/converting related models.
            exclude: Field names to exclude during conversion *to* Pydantic.
                     Note: Exclusion during generation is not yet implemented here,
                     but `generate_pydantic_class` could be adapted if needed.
            pydantic_base: Optional base for generated Pydantic model.
            model_name: Optional name for generated Pydantic model.
        """
        if isinstance(django_model_or_instance, models.Model):
            self.django_model_cls: type[DjangoModelT] = django_model_or_instance.__class__
            self.initial_django_instance: Optional[DjangoModelT] = django_model_or_instance
        elif issubclass(django_model_or_instance, models.Model):
            # Ensure the type checker understands self.django_model_cls is Type[DjangoModelT]
            # where DjangoModelT is the specific type bound to the class instance.
            self.django_model_cls: type[DjangoModelT] = django_model_or_instance  # Type should be consistent now
            self.initial_django_instance = None
        else:
            raise TypeError("Input must be a Django Model class or instance.")

        self.max_depth = max_depth
        self.exclude = exclude or set()
        self.pydantic_base = pydantic_base or BaseModel
        self.model_name = model_name  # Store optional name
        # Initialize dependencies
        self.relationship_accessor = RelationshipConversionAccessor()
        self.mapper = BidirectionalTypeMapper(self.relationship_accessor)
        # Register the initial model with the accessor (if needed for self-refs immediately)
        # self.relationship_accessor.map_relationship(source_model=???, django_model=self.django_model_cls) # Need Pydantic type?

        # Use the correctly defined cache type (name -> model/ref)
        self._generation_cache: GeneratedModelCache = {}
        # _django_metadata is no longer needed

        # Generate the Pydantic class definition immediately
        generated_cls_or_ref = self._generate_pydantic_class()

        # --- Resolve ForwardRef if necessary --- #
        resolved_cls: type[BaseModel]
        if isinstance(generated_cls_or_ref, ForwardRef):
            logger.warning(
                f"Initial generation for {self.django_model_cls.__name__} resulted in a ForwardRef."
                f" Attempting resolution via model_rebuild on potentially dependent models in cache."
            )
            # Try resolving by rebuilding all models in the cache that are actual classes
            # This assumes dependencies were generated and cached correctly.
            # Rebuild cached models to attempt resolving ForwardRefs
            for name, model_or_ref in self._generation_cache.items():
                if isinstance(model_or_ref, type) and issubclass(model_or_ref, BaseModel):
                    try:
                        model_or_ref.model_rebuild(force=True)
                        # logger.debug(f"Rebuilt cached model {name} to potentially resolve ForwardRefs")
                        # Successfully rebuilt a cached model; continue rebuilding others
                    except Exception as e:
                        logger.warning(f"Failed to rebuild cached model {name} during ForwardRef resolution: {e}")

            # After attempting rebuilds, try accessing the resolved type from the cache again
            final_cached_item = self._generation_cache.get(generated_cls_or_ref.__forward_arg__)
            if isinstance(final_cached_item, type) and issubclass(final_cached_item, BaseModel):
                logger.info(f"Successfully resolved ForwardRef for {generated_cls_or_ref.__forward_arg__}")
                resolved_cls = final_cached_item
            else:
                # If still not resolved, raise an error as we cannot proceed
                raise TypeError(
                    f"Failed to resolve ForwardRef '{generated_cls_or_ref.__forward_arg__}' for the main model {self.django_model_cls.__name__} after generation and rebuild attempts."
                )
        elif isinstance(generated_cls_or_ref, type) and issubclass(generated_cls_or_ref, BaseModel):
            # Already a valid class
            resolved_cls = generated_cls_or_ref
        else:
            # Should not happen if _generate_pydantic_class works correctly
            raise TypeError(
                f"_generate_pydantic_class returned an unexpected type: {type(generated_cls_or_ref)} for {self.django_model_cls.__name__}"
            )

        # Assign the resolved class
        self.pydantic_model_cls: type[BaseModel] = resolved_cls

        #          logger.debug(f"Skipping update_forward_refs for ForwardRef in cache: {item_name}")

        # Rebuild the main model AFTER resolving refs in the cache.
        # No need for _types_namespace if update_forward_refs worked.
        # Now it should be safe to call rebuild on the resolved class
        try:
            self.pydantic_model_cls.model_rebuild(force=True)
            logger.info(f"Rebuilt main model {self.pydantic_model_cls.__name__} after generation.")
        except Exception as e:
            # Log warning if final rebuild still fails
            logger.warning(
                f"Failed to rebuild main model {self.pydantic_model_cls.__name__} after resolution: {e}",
                exc_info=True,
            )

    # Helper map for TypeAdapter in to_django
    _INTERNAL_TYPE_TO_PYTHON_TYPE = {
        "AutoField": int,
        "BigAutoField": int,
        "IntegerField": int,
        "UUIDField": UUID,
        "CharField": str,
        "TextField": str,
        # Add other PK types as needed
    }

    def _generate_pydantic_class(self) -> Union[type[BaseModel], ForwardRef]:
        """Internal helper to generate or retrieve the Pydantic model class."""
        # Pass the stored exclude set during generation
        generated_cls: Union[type[BaseModel], ForwardRef] = generate_pydantic_class(
            django_model_cls=self.django_model_cls,
            mapper=self.mapper,
            model_name=self.model_name,
            cache=self._generation_cache,
            depth=0,
            max_depth=self.max_depth,
            pydantic_base=self.pydantic_base,
            exclude=self.exclude,
        )

        # Allow ForwardRef through, otherwise check for BaseModel subclass
        # This check is good, but the type hint of the function needs to be updated
        # if not isinstance(generated_cls, ForwardRef):
        #     if not isinstance(generated_cls, type) or not issubclass(generated_cls, BaseModel):
        #         raise TypeError(f"Generated type is not a valid {self.pydantic_base.__name__} or ForwardRef.")

        # The return type must accommodate ForwardRef
        # Cast here is less ideal than fixing the return type hint of _generate_pydantic_class
        # However, casting here allows the rest of the __init__ to proceed assuming BaseModel
        # We need to ensure the call site handles the ForwardRef possibility properly later.
        # For now, we assume generation is successful for the main class.
        # if isinstance(generated_cls, ForwardRef):
        #     # This case should ideally not happen for the top-level model unless max_depth=0
        #     # Or if there's an immediate self-reference issue not caught.
        #     # Log a warning, but proceed. The rebuild step might resolve it.
        #     logger.warning(f"_generate_pydantic_class returned a ForwardRef for the main model {self.django_model_cls.__name__}. Rebuild might be necessary.")
        # We need a type[BaseModel] for the rest of the init. This is problematic.
        # Let's rely on the rebuild step to fix this. If it fails, it will raise later.
        # Return the ForwardRef for now, but the attribute type hint needs fixing.
        # return generated_cls # TYPE HINT MISMATCH HERE

        return generated_cls

    def to_pydantic(self, db_obj: Optional[DjangoModelT] = None) -> PydanticModelT:
        """
        Converts a Django model instance to an instance of the generated Pydantic model.

        Args:
            db_obj: The Django model instance to convert. If None, attempts to use
                    the instance provided during initialization.

        Returns:
            An instance of the generated Pydantic model (subclass of BaseModel).

        Raises:
            ValueError: If no Django instance is available or conversion fails.
        """
        target_db_obj = db_obj or self.initial_django_instance
        if target_db_obj is None:
            raise ValueError("A Django model instance must be provided for conversion.")

        if not isinstance(target_db_obj, self.django_model_cls):
            raise TypeError(f"Provided instance is not of type {self.django_model_cls.__name__}")

        logger.info(
            f"Converting {self.django_model_cls.__name__} instance (PK: {target_db_obj.pk}) to {self.pydantic_model_cls.__name__}"
        )

        # We know self.pydantic_model_cls is Type[BaseModel] from _generate_pydantic_class
        # Pass the pre-extracted metadata
        result = django_to_pydantic(
            target_db_obj,
            self.pydantic_model_cls,  # Now guaranteed to be type[BaseModel] by __init__
            exclude=self.exclude,
            max_depth=self.max_depth,
        )
        return cast(PydanticModelT, result)

    def _determine_target_django_instance(
        self, pydantic_instance: BaseModel, update_instance: Optional[DjangoModelT]
    ) -> DjangoModelT:
        """Determines the target Django instance (update existing or create new)."""
        if update_instance:
            if not isinstance(update_instance, self.django_model_cls):
                raise TypeError(f"update_instance is not of type {self.django_model_cls.__name__}")
            logger.debug(f"Updating provided Django instance (PK: {update_instance.pk})")
            return update_instance
        elif self.initial_django_instance:
            logger.debug(f"Updating initial Django instance (PK: {self.initial_django_instance.pk})")
            # Re-fetch to ensure we have the latest state? Maybe not necessary if we overwrite all fields.
            return self.initial_django_instance
        else:
            # Check if Pydantic instance has a PK to determine if it represents an existing object
            pk_field = self.django_model_cls._meta.pk
            if pk_field is None:
                raise ValueError(f"Model {self.django_model_cls.__name__} does not have a primary key.")
            assert pk_field is not None  # Help type checker
            pk_field_name = pk_field.name

            pk_value = getattr(pydantic_instance, pk_field_name, None)
            if pk_value is not None:
                try:
                    target_django_instance = cast(DjangoModelT, self.django_model_cls.objects.get(pk=pk_value))
                    logger.debug(f"Found existing Django instance by PK ({pk_value}) from Pydantic data.")
                    return target_django_instance
                except self.django_model_cls.DoesNotExist:
                    logger.warning(
                        f"PK ({pk_value}) found in Pydantic data, but no matching Django instance exists. Creating new."
                    )
                    return cast(DjangoModelT, self.django_model_cls())
            else:
                logger.debug("Creating new Django instance.")
                return cast(DjangoModelT, self.django_model_cls())

    def _assign_fk_o2o_field(
        self, target_django_instance: DjangoModelT, django_field: RelatedField, pydantic_value: Any
    ):
        """Assigns a value to a ForeignKey or OneToOneField.
        NOTE: This method still uses the passed django_field object directly,
        as it already contains the necessary info like related_model and attname.
        It doesn't need to re-fetch metadata for the field itself, but uses it for PK type info.
        """
        related_model_cls_ref = django_field.related_model
        related_model_cls: type[models.Model]
        if related_model_cls_ref == "self":
            related_model_cls = self.django_model_cls
        elif isinstance(related_model_cls_ref, type) and issubclass(related_model_cls_ref, models.Model):
            related_model_cls = related_model_cls_ref
        else:
            raise TypeError(
                f"Unexpected related_model type for field '{django_field.name}': {type(related_model_cls_ref)}"
            )

        related_pk_field = related_model_cls._meta.pk
        if related_pk_field is None:
            raise ValueError(
                f"Related model {related_model_cls.__name__} for field '{django_field.name}' has no primary key."
            )
        related_pk_name = related_pk_field.name

        if pydantic_value is None:
            # Check if field is nullable before setting None
            if not django_field.null and not isinstance(django_field, OneToOneField):
                logger.warning(f"Attempting to set non-nullable FK/O2O '{django_field.name}' to None. Skipping.")
                return
            setattr(target_django_instance, django_field.name, None)
        elif isinstance(pydantic_value, BaseModel):
            # Assume nested Pydantic model has PK, fetch related Django obj
            related_pk = getattr(pydantic_value, related_pk_name, None)
            if related_pk is not None:
                try:
                    related_obj = related_model_cls.objects.get(pk=related_pk)
                    setattr(target_django_instance, django_field.name, related_obj)
                except related_model_cls.DoesNotExist:
                    logger.error(f"Related object for '{django_field.name}' with PK {related_pk} not found.")
                    if django_field.null:
                        setattr(target_django_instance, django_field.name, None)
                    else:
                        raise ValueError(
                            f"Cannot save '{django_field.name}': Related object with PK {related_pk} not found and field is not nullable."
                        ) from None
            else:
                logger.error(
                    f"Cannot set FK '{django_field.name}': Nested Pydantic model missing PK '{related_pk_name}'."
                )
                if django_field.null:
                    setattr(target_django_instance, django_field.name, None)
                else:
                    raise ValueError(
                        f"Cannot save non-nullable FK '{django_field.name}': Nested Pydantic model missing PK."
                    ) from None

        elif isinstance(pydantic_value, dict):
            # Handle dictionary input - extract PK
            related_pk = pydantic_value.get(related_pk_name)
            if related_pk is not None:
                try:
                    # Adapt and validate the extracted PK
                    internal_type_name = django_field.target_field.get_internal_type()
                    python_type = self._INTERNAL_TYPE_TO_PYTHON_TYPE.get(internal_type_name)
                    adapted_pk: Any = None
                    if python_type:
                        pk_adapter = TypeAdapter(python_type)
                        adapted_pk = pk_adapter.validate_python(related_pk)
                    else:
                        logger.warning(
                            f"Could not determine specific Python type for PK internal type '{internal_type_name}' on field '{django_field.name}'. Assigning raw PK value from dict."
                        )
                        adapted_pk = related_pk

                    # Set the FK using the ID field (attname)
                    fk_field_name = django_field.attname
                    setattr(target_django_instance, fk_field_name, adapted_pk)
                except Exception as e:
                    logger.error(
                        f"Failed to adapt PK '{related_pk}' extracted from dict for FK field '{django_field.name}': {e}"
                    )
                    if django_field.null or isinstance(django_field, OneToOneField):
                        setattr(target_django_instance, django_field.attname, None)
                        setattr(target_django_instance, django_field.name, None)  # Clear object too
                    else:
                        raise ValueError(
                            f"Invalid PK value '{related_pk}' in dict for non-nullable FK field '{django_field.name}'."
                        ) from e
            else:
                logger.error(
                    f"Cannot set FK '{django_field.name}': Dictionary input missing PK '{related_pk_name}'. Dict: {pydantic_value}"
                )
                if django_field.null or isinstance(django_field, OneToOneField):
                    setattr(target_django_instance, django_field.attname, None)
                    setattr(target_django_instance, django_field.name, None)  # Clear object too
                else:
                    raise ValueError(f"Cannot save non-nullable FK '{django_field.name}': Dictionary input missing PK.")

        else:  # Assume pydantic_value is the PK itself
            try:
                # Use TypeAdapter for robust PK conversion
                target_type = getattr(django_field.target_field, "target_field", django_field.target_field)
                # Get the internal type for adaptation (e.g., UUID, int, str)
                # Note: Using get_internal_type() might be less reliable than direct type check for adaptation
                # Consider a map or checking target_field class directly
                internal_type_name = target_type.get_internal_type()
                python_type = self._INTERNAL_TYPE_TO_PYTHON_TYPE.get(internal_type_name)

                if python_type:
                    pk_adapter = TypeAdapter(python_type)
                    adapted_pk = pk_adapter.validate_python(pydantic_value)
                else:
                    logger.warning(
                        f"Could not determine specific Python type for PK internal type '{internal_type_name}' on field '{django_field.name}'. Assigning raw value."
                    )
                    adapted_pk = pydantic_value

            except Exception as e:
                logger.error(f"Failed to adapt PK value '{pydantic_value}' for FK field '{django_field.name}': {e}")
                if django_field.null or isinstance(django_field, OneToOneField):
                    adapted_pk = None
                    setattr(target_django_instance, django_field.name, None)  # Clear the object too
                else:
                    raise ValueError(f"Invalid PK value type for non-nullable FK field '{django_field.name}'.") from e

            fk_field_name = django_field.attname  # Use attname to set the ID directly
            setattr(target_django_instance, fk_field_name, adapted_pk)
        logger.debug(f"Assigned FK/O2O '{django_field.name}'")

    def _assign_datetime_field(
        self, target_django_instance: DjangoModelT, django_field: models.DateTimeField, pydantic_value: Any
    ):
        """Assigns a value to a DateTimeField, handling timezone awareness."""
        # TODO: Add more robust timezone handling based on Django settings (USE_TZ)
        is_field_aware = getattr(django_field, "is_aware", False)  # Approximation

        if isinstance(pydantic_value, dt.datetime):
            current_value_aware = is_aware(pydantic_value)
            # Assume field needs aware if Django's USE_TZ is True (needs better check)
            if not current_value_aware and is_field_aware:
                try:
                    default_tz = get_default_timezone()
                    pydantic_value = make_aware(pydantic_value, default_tz)
                    logger.debug(f"Made naive datetime timezone-aware for field '{django_field.name}'")
                except Exception as e:
                    logger.warning(
                        f"Failed to make datetime timezone-aware for field '{django_field.name}'. Value: {pydantic_value}, Error: {e}"
                    )
                    # Decide if we should proceed with naive datetime or raise error/skip

        setattr(target_django_instance, django_field.name, pydantic_value)
        logger.debug(f"Assigned DateTimeField '{django_field.name}'")

    def _assign_file_field(
        self, target_django_instance: DjangoModelT, django_field: models.FileField, pydantic_value: Any
    ):
        """Handles assignment for FileField/ImageField (currently limited)."""
        if pydantic_value is None:
            setattr(target_django_instance, django_field.name, None)
            logger.debug(f"Set FileField/ImageField '{django_field.name}' to None.")
        elif isinstance(pydantic_value, str):
            # Avoid overwriting if the string likely represents the existing file
            current_file = getattr(target_django_instance, django_field.name, None)
            if current_file:
                matches = False
                if hasattr(current_file, "url") and current_file.url == pydantic_value:
                    matches = True
                elif hasattr(current_file, "name") and current_file.name == pydantic_value:
                    matches = True

                if matches:
                    logger.debug(
                        f"Skipping assignment for FileField/ImageField '{django_field.name}': value matches existing file."
                    )
                    return  # Skip assignment

            logger.warning(
                f"Skipping assignment for FileField/ImageField '{django_field.name}' from string value '{pydantic_value}'. Direct assignment/update from URL/string not supported."
            )
        else:
            # Allow assignment if it's not None or string (e.g., UploadedFile object)
            setattr(target_django_instance, django_field.name, pydantic_value)
            logger.debug(f"Assigned non-string value to FileField/ImageField '{django_field.name}'")

    def _process_pydantic_fields(
        self, target_django_instance: DjangoModelT, pydantic_instance: BaseModel
    ) -> dict[str, Any]:
        """Iterates through Pydantic fields and assigns values to the Django instance."""
        m2m_data = {}
        # Use model_dump(mode='python') which might return rich types
        # Convert specific rich types to string *before* assigning
        pydantic_data = pydantic_instance.model_dump(mode="python")

        for field_name, pydantic_value in pydantic_data.items():
            # Explicitly convert potentially problematic types here
            value_to_process = pydantic_value
            # Check type names instead of isinstance for robustness
            type_name = type(pydantic_value).__name__
            if type_name in ("HttpUrl", "EmailStr", "IPvAnyAddress", "IPv4Address", "IPv6Address"):
                logger.debug(f"Explicitly converting Pydantic type {type_name} to string for field '{field_name}'")
                value_to_process = str(pydantic_value)
            # Add other necessary pre-conversions here (e.g., Enums?)

            # Now call assign_field_value with the potentially converted value
            m2m_result = self._assign_field_value(target_django_instance, field_name, value_to_process)
            if m2m_result:
                m2m_data[m2m_result[0]] = m2m_result[1]

        return m2m_data

    def _assign_field_value(
        self, target_django_instance: DjangoModelT, field_name: str, pydantic_value: Any
    ) -> Optional[tuple[str, Any]]:
        """
        Assigns a single field value from Pydantic to the Django instance.
        Uses the Django model's _meta API to get field info.
        Returns M2M data to be processed later, or None.
        """
        # --- Refactored: Get Django field directly using _meta --- #
        try:
            django_field = target_django_instance._meta.get_field(field_name)
        except models.FieldDoesNotExist:
            # Field exists on Pydantic model but not on Django model
            logger.warning(
                f"Field '{field_name}' exists on Pydantic model but not found on Django model {self.django_model_cls.__name__}. Skipping assignment."
            )
            return None  # Skip assignment for this Pydantic-only field

        # Check if the field is concrete (e.g., not a reverse relation managed elsewhere)
        if not getattr(django_field, "concrete", False):
            logger.debug(f"Skipping non-concrete field '{field_name}'.")
            return None

        # --- Ensure it's a concrete Field before accessing Field-specific attributes ---
        if not isinstance(django_field, models.Field):
            logger.warning(
                f"Field '{field_name}' is not an instance of models.Field (type: {type(django_field).__name__}). Skipping."
            )
            return None

        try:
            # --- Skip non-editable fields based on field properties --- #
            is_pk = django_field.primary_key
            # AutoFields are implicitly not editable after creation if instance already has PK
            is_auto_field = isinstance(django_field, (models.AutoField, models.BigAutoField, models.SmallAutoField))
            is_editable = getattr(django_field, "editable", True)  # Default to True if editable attr doesn't exist

            # Skip if field is marked non-editable, or if it's an AutoField on an existing instance
            if not is_editable or (is_auto_field and target_django_instance.pk is not None):
                logger.debug(f"Skipping non-editable or auto-field '{field_name}'.")
                return None
            # Explicitly skip PK update on existing instances (should be covered above, but safe check)
            if is_pk and target_django_instance.pk is not None:
                logger.debug(f"Skipping primary key field '{field_name}' during update.")
                return None

            # --- Handle field types based on isinstance checks --- #

            # M2M handled separately after save - return data for later processing
            if isinstance(django_field, models.ManyToManyField):
                logger.debug(f"Deferring M2M assignment for '{field_name}'")
                return (field_name, pydantic_value)

            # FK/O2O handled by helper
            elif isinstance(django_field, (models.ForeignKey, models.OneToOneField)):
                # Cast to RelatedField for the helper function type hint
                self._assign_fk_o2o_field(target_django_instance, cast(RelatedField, django_field), pydantic_value)

            # JSONField
            elif isinstance(django_field, models.JSONField):
                # Django's JSONField usually handles dict/list directly
                setattr(target_django_instance, field_name, pydantic_value)
                logger.debug(f"Assigned JSONField '{field_name}'")

            # FileField/ImageField
            elif isinstance(django_field, models.FileField):
                # Use helper, casting for type hint clarity
                self._assign_file_field(target_django_instance, cast(models.FileField, django_field), pydantic_value)

            # Datetime fields might need timezone handling (use helper)
            elif isinstance(django_field, models.DateTimeField):
                self._assign_datetime_field(
                    target_django_instance, cast(models.DateTimeField, django_field), pydantic_value
                )

            else:  # Other simple fields (CharField, IntegerField, etc.)
                # The value should have been pre-processed in the calling loop
                value_to_assign = pydantic_value
                # Log the value being assigned to be sure
                logger.debug(
                    f"Assigning simple field '{field_name}': Type={type(value_to_assign).__name__}, Value={value_to_assign!r}"
                )
                setattr(target_django_instance, field_name, value_to_assign)
                # logger.debug(f"Assigned simple field '{field_name}'") # Slightly redundant log removed

        except Exception as e:
            logger.error(f"Error assigning field '{field_name}': {e}", exc_info=True)
            # Re-raise as a ValueError to be caught by the main `to_django` method
            raise ValueError(f"Failed to process field '{field_name}' for saving.") from e

        return None  # Indicate no M2M data for this field (it was handled or is simple)

    def _assign_m2m_fields(self, target_django_instance: DjangoModelT, m2m_data: dict[str, Any]):
        """Assigns ManyToMany field values using stored metadata."""
        if not m2m_data:
            return

        logger.debug("Assigning M2M relationships...")
        for field_name, pydantic_m2m_list in m2m_data.items():
            if pydantic_m2m_list is None:  # Allow clearing M2M
                pydantic_m2m_list = []

            if not isinstance(pydantic_m2m_list, list):
                raise ValueError(f"M2M field '{field_name}' expects a list, got {type(pydantic_m2m_list)}")

            # Get metadata for the M2M field
            # meta = self._django_metadata.get(field_name)
            # if not meta or not meta.is_m2m or not meta.related_model:
            #     logger.error(f"Could not find valid M2M metadata for field '{field_name}'. Skipping assignment.")
            #     continue

            # --- Refactored: Get Django field and related model directly using _meta --- #
            try:
                django_field = target_django_instance._meta.get_field(field_name)
                # Ensure it is actually an M2M field
                if not isinstance(django_field, models.ManyToManyField):
                    logger.error(
                        f"Field '{field_name}' found but is not a ManyToManyField (type: {type(django_field).__name__}). Skipping M2M assignment."
                    )
                    continue

                # Resolve related model ('self' or actual model class)
                related_model_cls_ref = django_field.related_model
                related_model_cls: type[models.Model]
                if related_model_cls_ref == "self":
                    related_model_cls = self.django_model_cls
                elif isinstance(related_model_cls_ref, type) and issubclass(related_model_cls_ref, models.Model):
                    related_model_cls = related_model_cls_ref
                else:
                    # This path indicates an issue with the Django model definition itself
                    raise TypeError(
                        f"Unexpected related_model type for M2M field '{django_field.name}': {type(related_model_cls_ref)}"
                    )

            except models.FieldDoesNotExist:
                logger.error(
                    f"Could not find M2M field '{field_name}' on Django model {target_django_instance.__class__.__name__}. Skipping assignment."
                )
                continue
            except TypeError as e:
                # Catch potential error from issubclass if related_model_cls_ref is unexpected type
                logger.error(f"Error resolving related model for M2M field '{field_name}': {e}", exc_info=True)
                continue

            try:
                # Get the manager instance from the target Django object
                manager = getattr(target_django_instance, field_name)

                # Get PK name from the resolved related model
                related_pk_field = related_model_cls._meta.pk
                if related_pk_field is None:
                    # Should not happen for valid Django models
                    raise ValueError(
                        f"Related model {related_model_cls.__name__} for M2M field '{field_name}' has no primary key defined in its _meta."
                    )
                related_pk_name = related_pk_field.name

                # --- Process the list of Pydantic items (models, dicts, or PKs) --- #

                # Update type hint to include QuerySet
                related_objs_or_pks: Union[list[models.Model], list[Any], QuerySet[models.Model]] = []
                if not pydantic_m2m_list:
                    pass  # Handled by manager.set([])
                elif all(isinstance(item, BaseModel) for item in pydantic_m2m_list):
                    # List of Pydantic models, extract PKs
                    related_pks = [getattr(item, related_pk_name, None) for item in pydantic_m2m_list]
                    valid_pks = [pk for pk in related_pks if pk is not None]
                    if len(valid_pks) != len(pydantic_m2m_list):
                        logger.warning(
                            f"Some related Pydantic models for M2M field '{field_name}' were missing PKs or had None PK."
                        )
                    # Query for existing Django objects using valid PKs
                    related_objs_or_pks = list(related_model_cls.objects.filter(pk__in=valid_pks))
                    if len(related_objs_or_pks) != len(valid_pks):
                        logger.warning(
                            f"Could not find all related Django objects for M2M field '{field_name}' based on Pydantic model PKs. Found {len(related_objs_or_pks)} out of {len(valid_pks)}."
                        )
                elif all(isinstance(item, dict) for item in pydantic_m2m_list):
                    # List of dictionaries, extract PKs based on the related model's PK name
                    try:
                        related_instances_pks = []
                        for related_instance_dict in pydantic_m2m_list:
                            pk = related_instance_dict.get(related_pk_name)
                            if pk is not None:
                                related_instances_pks.append(pk)
                            else:
                                logger.warning(
                                    f"Could not find PK '{related_pk_name}' in related instance data: {related_instance_dict}"
                                )
                        target_queryset = related_model_cls.objects.filter(pk__in=related_instances_pks)
                        # Assign the queryset to be used by the final set() call
                        related_objs_or_pks = target_queryset
                    except Exception as e:
                        logger.error(f"Failed to extract PKs from dictionary list for M2M field '{field_name}': {e}")
                        # Optionally re-raise or handle error appropriately
                        # If we don't assign, related_objs_or_pks remains empty, clearing the relation

                elif all(not isinstance(item, (BaseModel, dict)) for item in pydantic_m2m_list):
                    # Assume list of PKs if not BaseModels or dicts, use TypeAdapter for conversion
                    try:
                        internal_type_str = related_pk_field.get_internal_type()
                        python_pk_type = self._INTERNAL_TYPE_TO_PYTHON_TYPE.get(internal_type_str)
                        if python_pk_type:
                            pk_adapter = TypeAdapter(list[python_pk_type])
                            adapted_pks = pk_adapter.validate_python(pydantic_m2m_list)
                            related_objs_or_pks = adapted_pks
                        else:
                            logger.warning(
                                f"Unsupported PK internal type '{internal_type_str}' for M2M field '{field_name}'. Passing raw list to manager.set()."
                            )
                            related_objs_or_pks = pydantic_m2m_list
                    except Exception as e:
                        logger.error(f"Failed to adapt PK list for M2M field '{field_name}': {e}")
                        raise ValueError(f"Invalid PK list type for M2M field '{field_name}'.") from e
                else:
                    # Mixed list of Pydantic models and PKs? Handle error or try to process?
                    raise ValueError(
                        f"M2M field '{field_name}' received a mixed list of items (models and non-models). This is not supported."
                    )

                manager.set(related_objs_or_pks)  # .set() handles list of objects or PKs
                logger.debug(f"Set M2M field '{field_name}'")
            except Exception as e:
                logger.error(f"Error setting M2M field '{field_name}': {e}", exc_info=True)
                raise ValueError(f"Failed to set M2M field '{field_name}' on {target_django_instance}: {e}") from e

    def to_django(self, pydantic_instance: BaseModel, update_instance: Optional[DjangoModelT] = None) -> DjangoModelT:
        """
        Converts a Pydantic model instance back to a Django model instance,
        updating an existing one or creating a new one.

        Args:
            pydantic_instance: The Pydantic model instance containing the data.
            update_instance: An optional existing Django instance to update.
                             If None, attempts to update the initial instance (if provided),
                             otherwise creates a new Django instance.

        Returns:
            The saved Django model instance.

        Raises:
            TypeError: If the pydantic_instance is not of the expected type or related models are incorrect.
            ValueError: If saving fails, PKs are invalid, or required fields are missing.
        """
        if not isinstance(pydantic_instance, self.pydantic_model_cls):
            raise TypeError(f"Input must be an instance of {self.pydantic_model_cls.__name__}")

        logger.info(
            f"Attempting to save data from {self.pydantic_model_cls.__name__} instance back to Django model {self.django_model_cls.__name__}"
        )

        # 1. Determine the target Django instance
        target_django_instance = self._determine_target_django_instance(pydantic_instance, update_instance)

        # 2. Process Pydantic fields and assign to Django instance (defer M2M)
        try:
            m2m_data = self._process_pydantic_fields(target_django_instance, pydantic_instance)
        except ValueError as e:
            # Catch errors during field assignment
            logger.error(f"Error processing Pydantic fields for {self.django_model_cls.__name__}: {e}", exc_info=True)
            raise  # Re-raise the ValueError

        # 3. Save the main instance
        try:
            # Consider moving full_clean() call after M2M assignment if it causes issues here.
            # target_django_instance.full_clean(exclude=[...]) # Option to exclude fields causing early validation issues
            target_django_instance.save()
            logger.info(f"Saved basic fields for Django instance (PK: {target_django_instance.pk})")

        except Exception as e:  # Catch save errors (e.g., database constraints)
            logger.exception(
                f"Failed initial save for Django instance (PK might be {target_django_instance.pk}) of model {self.django_model_cls.__name__}"
            )
            raise ValueError(f"Django save operation failed for {target_django_instance}: {e}") from e

        # 4. Handle M2M Assignment (After Save)
        try:
            self._assign_m2m_fields(target_django_instance, m2m_data)
        except ValueError as e:
            # Catch errors during M2M assignment
            logger.error(
                f"Error assigning M2M fields for {self.django_model_cls.__name__} (PK: {target_django_instance.pk}): {e}",
                exc_info=True,
            )
            # Decide if we should re-raise or just log. Re-raising seems safer.
            raise

        # 5. Run final validation
        try:
            target_django_instance.full_clean()
            logger.info(
                f"Successfully validated and saved Pydantic data to Django instance (PK: {target_django_instance.pk})"
            )
        except Exception as e:  # Catch validation errors
            logger.exception(
                f"Django validation failed after saving and M2M assignment for instance (PK: {target_django_instance.pk}) of model {self.django_model_cls.__name__}"
            )
            # It's already saved, but invalid state. Raise the validation error.
            raise ValueError(f"Django validation failed for {target_django_instance}: {e}") from e

        return target_django_instance

    # Add helper properties/methods?
    @property
    def generated_pydantic_model(self) -> type[BaseModel]:
        """Returns the generated Pydantic model class.

        This is guaranteed to be a resolved model class after successful initialization.
        """
        # This property should return the potentially ForwardRef attribute
        return self.pydantic_model_cls  # Return the actual attribute
