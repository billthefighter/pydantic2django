"""
Provides functionality to convert Django model instances to Pydantic models.
"""
import dataclasses
import datetime
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
from django.utils.timezone import get_default_timezone, is_aware, make_aware
from pydantic import BaseModel, Field, TypeAdapter, create_model

# from .mapping import TypeMapper # Removed old import
# Add imports for new mapper and accessor
from ..core.bidirectional_mapper import BidirectionalTypeMapper, MappingError
from ..core.relationships import RelationshipConversionAccessor

# Potentially useful imports from the project (adjust as needed)
# from .mapping import TypeMapper # Might not be directly needed if we create reverse mapping here
# from .typing import ...
# from ..core.utils import ...


logger = logging.getLogger(__name__)

PydanticModelT = TypeVar("PydanticModelT", bound=BaseModel)
DjangoModelT = TypeVar("DjangoModelT", bound=models.Model)


# --- Helper Functions ---
def _is_pydantic_json_annotation(annotation: Any) -> bool:
    """Checks if a type annotation ultimately resolves to pydantic.Json."""
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Union:  # Handle Optional[T]
        # Check if it's Optional[Json] or Optional[Annotated[..., Json]]
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            return _is_pydantic_json_annotation(non_none_args[0])
        else:
            # Union of multiple types, not considered Json for this check
            return False

    if origin is Annotated:
        # Check if Json is in the metadata
        logger.debug(f"Checking Annotated type. Args: {args}")
        # The first argument is the underlying type, the rest is metadata
        metadata = args[1:]
        logger.debug(f"Checking Annotated metadata: {metadata}")
        found_json_in_metadata = False
        for item in metadata:
            # Log details about the item being checked
            item_name = getattr(item, "__name__", "N/A")
            logger.debug(f"  Metadata item: {item!r}, Type: {type(item)}, Name: {item_name}")
            # Check __name__ as pydantic.Json might not be the exact same object
            if item_name == "Json":
                logger.debug("  Found Json marker by name in metadata!")
                found_json_in_metadata = True
                break  # Found it
        if not found_json_in_metadata:
            logger.debug("Did not find Json in Annotated metadata.")
        return found_json_in_metadata

    # Direct check: Is the annotation itself pydantic.Json?
    # This might be less reliable if Json is used within other constructs like Annotated
    # Check __name__ as pydantic.Json might not be the exact same object
    is_direct_json = getattr(annotation, "__name__", None) == "Json"
    if is_direct_json:
        logger.debug(f"Annotation {annotation} directly identified as Json by name.")
    return is_direct_json


# --- Metadata Extraction --- (Moved Before Usage)

# GeneratedModelCache = dict[type[models.Model], type[BaseModel] | ForwardRef]  # type: ignore[misc]
# Redefine cache to use model name (str) as key for consistency
GeneratedModelCache = dict[str, Union[type[BaseModel], ForwardRef]]


@dataclasses.dataclass
class DjangoFieldMetadata:
    """Stores extracted metadata about a single Django model field."""

    field_name: str
    django_field: models.Field
    django_field_type: type[models.Field]
    is_relation: bool = False
    is_fk: bool = False
    is_o2o: bool = False
    is_m2m: bool = False
    is_self_ref: bool = False
    is_nullable: bool = False
    is_editable: bool = True
    is_pk: bool = False
    related_model: Optional[type[models.Model]] = None
    default: Any = models.NOT_PROVIDED
    # python_type: Optional[Any] = None # Maybe add later if TypeMapper is integrated


def _extract_django_model_metadata(
    django_model_cls: type[models.Model],
) -> dict[str, DjangoFieldMetadata]:
    """Extracts metadata for all concrete fields of a Django model."""
    metadata_map: dict[str, DjangoFieldMetadata] = {}
    logger.debug(f"Extracting metadata for Django model: {django_model_cls.__name__}")

    for field in django_model_cls._meta.get_fields(include_hidden=False):
        # Skip reverse relations and non-concrete fields immediately
        if isinstance(
            field, (ForeignObjectRel, OneToOneRel, ManyToOneRel, ManyToManyRel, GenericForeignKey)
        ) or not getattr(field, "concrete", False):
            logger.debug(f"Skipping non-concrete/reverse field '{field.name}' of type {type(field).__name__}")
            continue

        field_name = field.name
        django_field_type = type(field)
        is_relation = isinstance(field, RelatedField)
        related_model = None
        is_self_ref = False

        if is_relation:
            related_model_cls_ref = field.related_model
            if related_model_cls_ref == "self":
                related_model = django_model_cls
                is_self_ref = True
            elif isinstance(related_model_cls_ref, type) and issubclass(related_model_cls_ref, models.Model):
                related_model = related_model_cls_ref
            else:
                # This case might occur with abstract models or complex setups, log warning
                logger.warning(
                    f"Could not resolve related model for field '{field_name}' ({type(related_model_cls_ref)}). Treating as non-relational for metadata."
                )
                is_relation = False  # Treat as simple if related model can't be determined

        metadata_map[field_name] = DjangoFieldMetadata(
            field_name=field_name,
            django_field=field,
            django_field_type=django_field_type,
            is_relation=is_relation,
            is_fk=isinstance(field, ForeignKey),
            is_o2o=isinstance(field, OneToOneField),
            is_m2m=isinstance(field, ManyToManyField),
            is_self_ref=is_self_ref,
            related_model=related_model,
            is_nullable=getattr(field, "null", False),
            is_editable=getattr(field, "editable", True),
            is_pk=field.primary_key,
            default=getattr(field, "default", models.NOT_PROVIDED),
        )
        logger.debug(f"Extracted metadata for field '{field_name}': {metadata_map[field_name]}")

    return metadata_map


# --- Conversion Functions ---


def django_to_pydantic(
    db_obj: DjangoModelT,
    pydantic_model: type[PydanticModelT],
    *,
    exclude: set[str] | None = None,
    depth: int = 0,  # Add depth to prevent infinite recursion
    max_depth: int = 3,  # Set a default max depth
    django_metadata: Optional[dict[str, DjangoFieldMetadata]] = None,  # Allow passing pre-extracted metadata
) -> PydanticModelT:
    """
    Converts a Django model instance to a Pydantic model instance.

    Args:
        db_obj: The Django model instance to convert.
        pydantic_model: The target Pydantic model class.
        exclude: A set of field names to exclude from the conversion.
        depth: Current recursion depth (internal use).
        max_depth: Maximum recursion depth for related models.
        django_metadata: Optional pre-extracted metadata for the Django model's fields.
                         If None, it will be extracted.

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
    if django_metadata is None:
        django_metadata = _extract_django_model_metadata(db_obj.__class__)

    data = {}
    exclude_set = exclude or set()

    pydantic_fields = pydantic_model.model_fields

    for field_name, pydantic_field in pydantic_fields.items():
        if field_name in exclude_set:
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

        # Get metadata for the corresponding Django field, if it exists
        meta = django_metadata.get(field_name)

        if meta:
            logger.debug(
                f"Found Django metadata for '{field_name}': Type={meta.django_field_type.__name__}, Relation={meta.is_relation}"
            )
        else:
            logger.debug(
                f"No direct Django metadata found for '{field_name}'. Assuming property/method or Pydantic-only field."
            )

        # --- Handle Relationships using Metadata ---
        if meta and meta.is_relation:
            # 1. ManyToManyField / List[BaseModel]
            # Check if the Pydantic annotation is List[BaseModel]
            is_pydantic_list_of_models = False
            related_pydantic_model_in_list: Optional[type[BaseModel]] = None
            if origin is list and args:
                # Check if the argument is a Pydantic BaseModel class
                arg_type = args[0]
                if isinstance(arg_type, type) and issubclass(arg_type, BaseModel):
                    is_pydantic_list_of_models = True
                    related_pydantic_model_in_list = arg_type
                elif isinstance(arg_type, ForwardRef):  # Handle List[ForwardRef(...)]
                    is_pydantic_list_of_models = True  # Assume it *will* be a BaseModel
                    # Cannot get class name easily here, just proceed

            if meta.is_m2m and is_pydantic_list_of_models:
                target_model = related_pydantic_model_in_list  # May be None if ForwardRef
                # If ForwardRef, we must proceed assuming it resolves to BaseModel
                if target_model or (isinstance(args[0], ForwardRef)):
                    logger.debug(
                        f"Handling M2M relationship for '{field_name}' -> {target_model.__name__ if target_model else args[0]}"
                    )
                    related_manager = django_value  # Django related manager
                    converted_related = []
                    try:
                        related_queryset = related_manager.all()
                        for related_obj in related_queryset:
                            # We need the *actual* Pydantic class for the recursive call.
                            # If target_model is None (was ForwardRef), we *cannot* proceed here.
                            # This indicates an issue in model generation/resolution order.
                            # However, the django_to_pydantic function assumes the target pydantic_model is already resolved.
                            # We'll rely on the caller (or tests) providing the correct, resolved `pydantic_model`.
                            # Let's assume related_pydantic_model_in_list MUST be the resolved class here.
                            if not target_model:
                                # If we are here, args[0] was a ForwardRef that SHOULD have been resolved
                                # before django_to_pydantic was called with this pydantic_model definition.
                                # Attempt to resolve it now based on the parent model's namespace? Risky.
                                # Let's rely on the provided pydantic_model having resolved annotations.
                                # Re-evaluate the annotation from the provided *resolved* model.
                                resolved_annotation = pydantic_model.model_fields[field_name].annotation
                                resolved_origin = get_origin(resolved_annotation)
                                resolved_args = get_args(resolved_annotation)
                                if (
                                    resolved_origin is list
                                    and resolved_args
                                    and isinstance(resolved_args[0], type)
                                    and issubclass(resolved_args[0], BaseModel)
                                ):
                                    target_model = resolved_args[0]
                                else:
                                    logger.error(
                                        f"Could not resolve ForwardRef in List for M2M field '{field_name}'. Skipping related object conversion."
                                    )
                                    continue  # Skip this related object

                            try:
                                converted_related.append(
                                    django_to_pydantic(
                                        related_obj,
                                        target_model,  # Use the resolved model type
                                        exclude=exclude,
                                        depth=depth + 1,
                                        max_depth=max_depth,
                                        django_metadata=None,  # Let recursive call extract metadata
                                    )
                                )
                            except ValueError as e:
                                logger.error(f"Failed converting related object in M2M '{field_name}': {e}")
                                continue  # Skip item on depth error
                    except Exception as e:
                        logger.error(f"Error accessing M2M manager for '{field_name}': {e}", exc_info=True)
                        converted_related = []
                    data[field_name] = converted_related
                else:
                    logger.warning(
                        f"Could not determine target Pydantic model for M2M field '{field_name}'. Assigning raw manager."
                    )
                    data[field_name] = django_value  # Fallback?

            # 2. ForeignKey or OneToOneField / Optional[BaseModel] or BaseModel
            elif meta.is_fk or meta.is_o2o:
                # Determine the target Pydantic model from the annotation
                target_pydantic_model: Optional[type[BaseModel]] = None
                # Need to handle ForwardRef annotations as well
                annotation_to_check = pydantic_annotation
                if isinstance(annotation_to_check, ForwardRef):
                    # If the main annotation is a ForwardRef, we assume it resolves to BaseModel
                    # We cannot determine the *actual* model here, rely on caller providing resolved model
                    # Attempt resolution like in M2M case:
                    resolved_annotation = pydantic_model.model_fields[field_name].annotation
                    if isinstance(resolved_annotation, type) and issubclass(resolved_annotation, BaseModel):
                        target_pydantic_model = resolved_annotation
                    elif get_origin(resolved_annotation) is Union:  # Check Optional[ForwardRef] -> Optional[Model]
                        resolved_args = get_args(resolved_annotation)
                        potential_model = next((t for t in resolved_args if t is not type(None)), None)
                        if isinstance(potential_model, type) and issubclass(potential_model, BaseModel):
                            target_pydantic_model = potential_model
                    else:
                        logger.warning(
                            f"Could not resolve ForwardRef annotation '{annotation_to_check}' for FK/O2O field '{field_name}'. Cannot convert relation."
                        )
                        # Decide on fallback: assign PK or None? Assign PK for now.
                        data[field_name] = getattr(db_obj, f"{field_name}_id", None)

                elif (
                    origin is Union and len(args) == 2 and type(None) in args
                ):  # Optional[Model] or Optional[ForwardRef]
                    potential_model_type = next((t for t in args if t is not type(None)), None)
                    if isinstance(potential_model_type, type) and issubclass(potential_model_type, BaseModel):
                        target_pydantic_model = potential_model_type
                    elif isinstance(potential_model_type, ForwardRef):
                        # Attempt resolution like above
                        resolved_annotation = pydantic_model.model_fields[field_name].annotation
                        resolved_args = get_args(resolved_annotation)
                        potential_model = next((t for t in resolved_args if t is not type(None)), None)
                        if isinstance(potential_model, type) and issubclass(potential_model, BaseModel):
                            target_pydantic_model = potential_model
                        else:
                            logger.warning(
                                f"Could not resolve ForwardRef inside Optional for FK/O2O field '{field_name}'. Cannot convert relation."
                            )
                            data[field_name] = None  # Assign None for Optional field if resolution fails

                elif isinstance(pydantic_annotation, type) and issubclass(pydantic_annotation, BaseModel):  # Model
                    target_pydantic_model = pydantic_annotation

                if target_pydantic_model:
                    logger.debug(f"Handling FK/O2O relationship for '{field_name}' -> {target_pydantic_model.__name__}")
                    related_obj = django_value  # The related Django instance or None
                    if related_obj:
                        try:
                            data[field_name] = django_to_pydantic(
                                related_obj,
                                target_pydantic_model,
                                exclude=exclude,
                                depth=depth + 1,
                                max_depth=max_depth,
                                django_metadata=None,  # Let recursive call extract metadata
                            )
                        except ValueError as e:
                            logger.error(f"Failed converting related object in FK/O2O '{field_name}': {e}")
                            data[field_name] = None  # Set to None if conversion fails due to depth
                    else:
                        data[field_name] = None  # Related object was None in Django
                elif data.get(field_name) is None:  # Ensure field is added if not handled above
                    # Pydantic field is not a nested model, assign raw Django value (likely PK)
                    logger.debug(
                        f"Treating related field '{field_name}' as simple value (e.g., PK). Value: {django_value!r}"
                    )
                    # Check if django_value is a related manager or instance before accessing _id
                    if isinstance(django_value, models.Model):
                        # For FK/O2O where Pydantic expects PK, assign the PK
                        pk_name = getattr(django_value._meta.pk, "name", "pk")
                        data[field_name] = getattr(django_value, pk_name, None)
                    elif django_value is None:  # Handle case where FK is None
                        data[field_name] = None
                    else:  # Assign raw value if not a model instance (e.g., property?)
                        data[field_name] = django_value

            else:
                # This case might occur if metadata says it's a relation but Pydantic annotation doesn't match
                # or if it's an unhandled relation type (GenericForeignKey?)
                logger.warning(
                    f"Mismatch or unhandled relation type for field '{field_name}'. Django meta: {meta}, Pydantic type: {pydantic_annotation}. Assigning raw value: {django_value!r}"
                )
                data[field_name] = django_value

        # --- Handle Simple Fields (or fields without direct Django metadata) ---
        else:
            # Check specifically for FileField and ImageField subclasses
            if meta and issubclass(meta.django_field_type, models.FileField):
                field_value = getattr(db_obj, field_name)
                data[field_name] = field_value.url if field_value else None
                logger.debug(f"Handling FileField/ImageField '{field_name}' -> URL: {data[field_name]}")

            elif meta and meta.django_field_type == models.JSONField:
                type_to_check = pydantic_annotation
                is_pydantic_json_type = _is_pydantic_json_annotation(type_to_check)  # Use module-level helper

                logger.debug(f"Field '{field_name}': Annotation={type_to_check}, IsJsonType={is_pydantic_json_type}")
                logger.debug(f"Field '{field_name}': Django value type: {type(django_value)}, value: {django_value!r}")

                if is_pydantic_json_type:
                    # Target is pydantic.Json, expects string or bytes
                    logger.debug(f"Field '{field_name}': Pydantic type IS Json. Attempting serialization.")
                    value_to_assign = None
                    if django_value is not None:
                        try:
                            # Serialize the Python object from Django JSONField into a JSON string
                            value_to_assign = json.dumps(django_value)
                            logger.debug(f"Field '{field_name}': Serialized value: {value_to_assign!r}")
                        except TypeError as e:
                            logger.error(f"Failed to serialize JSON for field '{field_name}': {e}", exc_info=True)
                            value_to_assign = None  # Assign None on serialization failure
                    data[field_name] = value_to_assign
                    logger.debug(f"Field '{field_name}': Assigning JSON string: {data[field_name]!r}")
                else:
                    # Target is likely dict, list, Any - assign the Python object directly
                    logger.debug(f"Field '{field_name}': Pydantic type IS NOT Json. Assigning raw value.")
                    data[field_name] = django_value
                    logger.debug(f"Field '{field_name}': Assigning raw value: {data[field_name]!r}")
            else:
                # Includes simple types, properties, Pydantic-only fields
                logger.debug(
                    f"Handling simple/property/Pydantic-only field '{field_name}' with value: {django_value!r}"
                )
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
    django_metadata: Optional[dict[str, DjangoFieldMetadata]] = None,
) -> Union[type[BaseModel], ForwardRef]:
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
        django_metadata: Optional pre-extracted metadata for the model's fields.
                         If None, it will be extracted.

    Returns:
        A dynamically created Pydantic model class or a ForwardRef if max_depth is hit.

    Raises:
        ValueError: If maximum recursion depth is exceeded or generation fails.
        TypeError: If a field type cannot be mapped.
    """
    if cache is None:
        cache = {}
        logger.debug(f"Initializing generation cache for {django_model_cls.__name__}")

    pydantic_model_name = model_name or f"{django_model_cls.__name__}Pydantic"
    logger.debug(
        f"Generating Pydantic model '{pydantic_model_name}' for Django model '{django_model_cls.__name__}' (Depth: {depth})"
    )

    # --- Cache Check ---
    # Check if the actual model or a ForwardRef is already in the cache
    if pydantic_model_name in cache:
        cached_item = cache[pydantic_model_name]
        # If it's a real class (not ForwardRef), return it directly
        if isinstance(cached_item, type) and issubclass(cached_item, BaseModel):
            logger.debug(f"Cache hit (actual class) for name '{pydantic_model_name}' (Depth: {depth})")
            return cached_item
        # If it's a ForwardRef (meaning we are potentially in a recursion loop or hit max depth earlier)
        elif isinstance(cached_item, ForwardRef):
            logger.debug(f"Cache hit (ForwardRef) for name '{pydantic_model_name}' (Depth: {depth})")
            return cached_item  # Return the existing ForwardRef

    # --- Max Depth Check ---
    if depth > max_depth:
        logger.warning(
            f"Max recursion depth ({max_depth}) reached for {django_model_cls.__name__}. Returning ForwardRef."
        )
        # Ensure ForwardRef is placed in cache if max depth is hit *before* processing this model
        if pydantic_model_name not in cache:
            forward_ref = ForwardRef(pydantic_model_name)
            cache[pydantic_model_name] = forward_ref
            return forward_ref
        else:  # Should already be a ForwardRef if we hit this path after the cache check above
            return cache[pydantic_model_name]  # type: ignore

    # --- Initial Cache Setup ---
    forward_ref = ForwardRef(pydantic_model_name)
    cache[pydantic_model_name] = forward_ref
    logger.debug(f"Placed ForwardRef '{pydantic_model_name}' in cache (Depth: {depth})")

    # Extract metadata if not provided
    if django_metadata is None:
        django_metadata = _extract_django_model_metadata(django_model_cls)

    field_definitions: dict[str, tuple[Any, Any]] = {}
    model_dependencies: set[Union[type[BaseModel], ForwardRef]] = set()

    for field_name, meta in django_metadata.items():
        logger.debug(f"  Processing field: {field_name} ({meta.django_field_type.__name__}) ...")
        python_type: Any = None
        field_info_kwargs: dict = {}

        # --- Handle Relationships using Metadata ---
        if meta.is_relation and meta.related_model:
            related_model_cls = meta.related_model
            logger.debug(f"  Relation field '{field_name}' -> {related_model_cls.__name__} (Depth: {depth})")
            try:
                related_pydantic_model_ref = generate_pydantic_class(
                    related_model_cls,
                    mapper,
                    model_name=None,
                    cache=cache,
                    depth=depth + 1,
                    max_depth=max_depth,
                    pydantic_base=pydantic_base,
                    django_metadata=None,
                )
                model_dependencies.add(related_pydantic_model_ref)
                if meta.is_m2m:
                    python_type = list[related_pydantic_model_ref]
                    field_info_kwargs = {"default_factory": list}
                else:
                    python_type = related_pydantic_model_ref
            except ValueError as e:
                logger.error(f"  Failed generating related model for '{field_name}' due to depth: {e}")
                python_type = Any
                meta.is_nullable = True
        # --- Handle Simple Fields using BidirectionalTypeMapper ---
        else:
            try:
                python_type, field_info_kwargs = mapper.get_pydantic_mapping(meta.django_field)
                logger.debug(
                    f"  Mapped simple field '{field_name}' to Pydantic type: {python_type}, kwargs: {field_info_kwargs}"
                )
            except MappingError as e:
                logger.warning(f"  Mapping failed for field '{field_name}': {e}. Falling back to Any.")
                python_type = Any
                field_info_kwargs = {}
            except Exception as e:
                logger.error(f"  Unexpected error mapping field '{field_name}': {e}", exc_info=True)
                python_type = Any
                field_info_kwargs = {}
        # --- Final Type Adjustment and Field Definition ---
        if python_type is not None:
            final_type = Optional[python_type] if meta.is_nullable else python_type
            if "default" in field_info_kwargs or "default_factory" in field_info_kwargs:
                field_instance = Field(**field_info_kwargs)
            elif meta.is_m2m:
                field_instance = Field(default_factory=list, **field_info_kwargs)
            elif meta.is_nullable:
                field_instance = Field(default=None, **field_info_kwargs)
            else:
                if field_info_kwargs:
                    field_instance = Field(**field_info_kwargs)
                else:
                    field_instance = ...
            field_definitions[field_name] = (final_type, field_instance)
            logger.debug(f"  Defined field '{field_name}': Type={final_type}, Definition={field_instance!r}")

    # --- Create the Pydantic Model Class ---
    model_base = pydantic_base or BaseModel
    try:
        model_cls = create_model(
            pydantic_model_name,
            __base__=model_base,
            **field_definitions,
        )
        logger.info(f"Successfully created Pydantic model class '{pydantic_model_name}'")

        # --- IMPORTANT: Update cache with the *actual* class ---
        cache[pydantic_model_name] = model_cls
        logger.debug(f"Updated cache for '{pydantic_model_name}' with actual class object.")

        return model_cls

    except Exception as e:
        logger.error(f"Failed to create Pydantic model '{pydantic_model_name}' using create_model: {e}", exc_info=True)
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
        self._django_metadata: dict[str, DjangoFieldMetadata] = _extract_django_model_metadata(self.django_model_cls)

        # Generate the Pydantic class definition immediately
        self.pydantic_model_cls = self._generate_pydantic_class()

        # Rebuild the generated model to resolve forward references
        # Pass the cache containing generated models/refs as the namespace
        try:
            self.pydantic_model_cls.model_rebuild(force=True, _types_namespace=self._generation_cache)
            logger.info(f"Rebuilt {self.pydantic_model_cls.__name__} to resolve ForwardRefs.")
        except Exception as e:
            logger.warning(f"Failed to rebuild {self.pydantic_model_cls.__name__} post-generation: {e}", exc_info=True)

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

    def _generate_pydantic_class(self) -> type[BaseModel]:
        """Generates the Pydantic model class for the Django model."""
        logger.info(f"Generating Pydantic class for {self.django_model_cls.__name__}")
        # Reset cache for this generation process
        self._generation_cache = {}
        # Pass the pre-extracted metadata AND the mapper instance
        generated_type = generate_pydantic_class(
            self.django_model_cls,
            mapper=self.mapper,
            cache=self._generation_cache,
            max_depth=self.max_depth,
            pydantic_base=self.pydantic_base,
            model_name=self.model_name,
            django_metadata=self._django_metadata,  # Pass stored metadata
        )

        if isinstance(generated_type, ForwardRef):
            logger.error(
                f"Pydantic model generation for {self.django_model_cls.__name__} resulted in an unresolved ForwardRef, likely due to exceeding max_depth ({self.max_depth})."
            )
            raise TypeError(
                f"Could not fully resolve Pydantic model for {self.django_model_cls.__name__} due to recursion depth."
            )
        elif not issubclass(generated_type, self.pydantic_base):
            # This case should ideally not happen if generate_pydantic_class works correctly
            logger.error(f"Generated type is not a {self.pydantic_base.__name__} subclass: {type(generated_type)}")
            raise TypeError(f"Generated type is not a valid {self.pydantic_base.__name__}.")

        # The generated_type should be a subclass of BaseModel (or the specified base)
        return generated_type

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

        # Use the existing django_to_pydantic function
        # We know self.pydantic_model_cls is Type[BaseModel] from _generate_pydantic_class
        # Pass the pre-extracted metadata
        result = django_to_pydantic(
            target_db_obj,
            self.pydantic_model_cls,
            exclude=self.exclude,
            max_depth=self.max_depth,
            django_metadata=self._django_metadata,  # Pass stored metadata
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
                        )
            else:
                logger.error(
                    f"Cannot set FK '{django_field.name}': Nested Pydantic model missing PK '{related_pk_name}'."
                )
                if django_field.null:
                    setattr(target_django_instance, django_field.name, None)
                else:
                    raise ValueError(
                        f"Cannot save non-nullable FK '{django_field.name}': Nested Pydantic model missing PK."
                    )

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

        if isinstance(pydantic_value, datetime.datetime):
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

    def _assign_field_value(
        self, target_django_instance: DjangoModelT, field_name: str, pydantic_value: Any
    ) -> Optional[tuple[str, Any]]:
        """
        Assigns a single field value from Pydantic to the Django instance using stored metadata.
        Returns M2M data to be processed later, or None.
        """
        # Get metadata for the field
        meta = self._django_metadata.get(field_name)

        if not meta:
            # Field exists on Pydantic model but not found in Django metadata
            # (could be Pydantic-only field, or metadata extraction issue)
            logger.warning(
                f"Field '{field_name}' exists on Pydantic model but no corresponding metadata found for Django model {self.django_model_cls.__name__}. Skipping."
            )
            return None

        # Use the actual Django field object from metadata
        django_field = meta.django_field

        try:
            # --- Skip non-editable fields based on metadata ---
            # PK check already implicitly handled by is_editable=False for AutoFields
            if not meta.is_editable:
                logger.debug(f"Skipping non-editable field '{field_name}'.")
                return None
            # Explicit check for PK update (although should be caught by is_editable)
            if meta.is_pk and target_django_instance.pk is not None:
                logger.debug(f"Skipping primary key field '{field_name}' during update.")
                return None

            # --- Handle field types based on metadata ---
            if meta.is_m2m:
                logger.debug(f"Deferring M2M assignment for '{field_name}'")
                return (field_name, pydantic_value)  # Return M2M data

            elif meta.is_fk or meta.is_o2o:
                # Cast to RelatedField for the helper function type hint
                self._assign_fk_o2o_field(target_django_instance, cast(RelatedField, django_field), pydantic_value)

            elif meta.django_field_type == models.JSONField:
                # Django's JSONField usually handles dict/list directly
                setattr(target_django_instance, field_name, pydantic_value)
                logger.debug(f"Assigned JSONField '{field_name}'")
            elif meta.django_field_type in (models.FileField, models.ImageField):
                self._assign_file_field(target_django_instance, cast(models.FileField, django_field), pydantic_value)

            else:  # Other simple fields
                # Potential place for TypeAdapter for more robustness if needed
                setattr(target_django_instance, field_name, pydantic_value)
                logger.debug(f"Assigned simple field '{field_name}'")

        except Exception as e:
            logger.error(f"Error assigning field '{field_name}': {e}", exc_info=True)
            # Re-raise as a ValueError to be caught by the main method
            raise ValueError(f"Failed to process field '{field_name}' for saving.") from e

        return None  # Indicate no M2M data for this field

    def _process_pydantic_fields(
        self, target_django_instance: DjangoModelT, pydantic_instance: BaseModel
    ) -> dict[str, Any]:
        """Iterates through Pydantic fields and assigns values to the Django instance."""
        m2m_data = {}
        pydantic_data = pydantic_instance.model_dump()

        for field_name, pydantic_value in pydantic_data.items():
            m2m_result = self._assign_field_value(target_django_instance, field_name, pydantic_value)
            if m2m_result:
                m2m_data[m2m_result[0]] = m2m_result[1]

        return m2m_data

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
            meta = self._django_metadata.get(field_name)
            if not meta or not meta.is_m2m or not meta.related_model:
                logger.error(f"Could not find valid M2M metadata for field '{field_name}'. Skipping assignment.")
                continue

            try:
                manager = getattr(target_django_instance, field_name)
                django_field = cast(ManyToManyField, meta.django_field)  # Use field from metadata
                related_model_cls = meta.related_model  # Use related model from metadata

                related_pk_field = related_model_cls._meta.pk
                if related_pk_field is None:
                    raise ValueError(
                        f"Related model {related_model_cls.__name__} for M2M field '{field_name}' has no primary key."
                    )
                related_pk_name = related_pk_field.name

                related_objs_or_pks: Union[list[models.Model], list[Any]] = []
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
                        # Use set() for efficient addition
                        getattr(target_django_instance, field_name).set(target_queryset)
                    except Exception as e:
                        logger.error(f"Failed to extract PKs from dictionary list for M2M field '{field_name}': {e}")
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
        """Returns the generated Pydantic model class."""
        return self.pydantic_model_cls
