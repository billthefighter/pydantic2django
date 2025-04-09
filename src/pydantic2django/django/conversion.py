"""
Provides functionality to convert Django model instances to Pydantic models.
"""
import dataclasses
import datetime
import json
import logging
from typing import Any, ForwardRef, Generic, Optional, TypeVar, Union, cast, get_args, get_origin, Type
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
from pydantic import BaseModel, Field, Json, TypeAdapter, create_model
from pydantic_core import PydanticUndefined
from pydantic.fields import FieldInfo

from .mapping import TypeMapper

# Potentially useful imports from the project (adjust as needed)
# from .mapping import TypeMapper # Might not be directly needed if we create reverse mapping here
# from .typing import ...
# from ..core.utils import ...


logger = logging.getLogger(__name__)

PydanticModelT = TypeVar("PydanticModelT", bound=BaseModel)
DjangoModelT = TypeVar("DjangoModelT", bound=models.Model)

# --- Metadata Extraction --- (Moved Before Usage)

GeneratedModelCache = dict[type[models.Model], type[BaseModel] | ForwardRef]  # type: ignore[misc]


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
            if meta.is_m2m and origin is list and args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
                related_pydantic_model = args[0]
                logger.debug(f"Handling M2M relationship for '{field_name}' -> {related_pydantic_model.__name__}")
                related_manager = django_value  # Django related manager
                converted_related = []
                try:
                    # Using try-except to catch potential manager errors if prefetch_related wasn't used optimally
                    related_queryset = related_manager.all()
                    for related_obj in related_queryset:
                        try:
                            converted_related.append(
                                django_to_pydantic(
                                    related_obj,
                                    related_pydantic_model,
                                    exclude=exclude,
                                    depth=depth + 1,
                                    max_depth=max_depth,
                                    django_metadata=None,  # Let recursive call extract metadata for related model
                                )
                            )
                        except ValueError as e:
                            logger.error(f"Failed converting related object in M2M '{field_name}': {e}")
                            continue  # Skip item on depth error
                except Exception as e:
                    logger.error(f"Error accessing M2M manager for '{field_name}': {e}", exc_info=True)
                    # Decide what to do: empty list or raise error? Empty list for now.
                    converted_related = []
                data[field_name] = converted_related

            # 2. ForeignKey or OneToOneField / Optional[BaseModel] or BaseModel
            elif meta.is_fk or meta.is_o2o:
                related_pydantic_model: type[BaseModel] | None = None
                if origin is Union and type(None) in args and len(args) == 2:  # Optional[Model]
                    potential_model = next((arg for arg in args if arg is not type(None)), None)
                    if potential_model and isinstance(potential_model, type) and issubclass(potential_model, BaseModel):
                        related_pydantic_model = potential_model
                elif (
                    pydantic_annotation
                    and isinstance(pydantic_annotation, type)
                    and issubclass(pydantic_annotation, BaseModel)
                ):  # Model
                    related_pydantic_model = pydantic_annotation

                if related_pydantic_model:
                    logger.debug(
                        f"Handling FK/O2O relationship for '{field_name}' -> {related_pydantic_model.__name__}"
                    )
                    related_obj = django_value  # The related Django instance or None
                    if related_obj:
                        try:
                            data[field_name] = django_to_pydantic(
                                related_obj,
                                related_pydantic_model,
                                exclude=exclude,
                                depth=depth + 1,
                                max_depth=max_depth,
                                django_metadata=None,  # Let recursive call extract metadata for related model
                            )
                        except ValueError as e:
                            logger.error(f"Failed converting related object in FK/O2O '{field_name}': {e}")
                            data[field_name] = None  # Set to None if conversion fails due to depth
                    else:
                        data[field_name] = None  # Related object was None in Django
                else:
                    # Pydantic field is not a BaseModel, treat as simple field (e.g., FK's pk)
                    logger.debug(
                        f"Treating related field '{field_name}' as simple value (e.g., PK). Value: {django_value!r}"
                    )
                    data[field_name] = django_value  # Assign the raw value (likely PK)
            else:
                # Should not happen if meta.is_relation is True and meta handles M2M/FK/O2O
                logger.warning(
                    f"Unhandled relation type for field '{field_name}' with metadata: {meta}. Assigning raw value."
                )
                data[field_name] = django_value

        # --- Handle Simple Fields (or fields without direct Django metadata) ---
        else:
            # Use metadata if available to guide conversion
            # Check specifically for FileField and ImageField subclasses
            if meta and issubclass(meta.django_field_type, models.FileField):
                # If the field has a value (file is set), try to get its URL. Otherwise, None.
                field_value = getattr(db_obj, field_name)
                data[field_name] = field_value.url if field_value else None
                logger.debug(f"Handling FileField/ImageField '{field_name}' -> URL: {data[field_name]}")
            elif meta and meta.django_field_type == models.JSONField:
                # Pydantic Json type expects string/bytes, check if needed
                # Check if Json is part of the annotation, handling Optional[Json[...]]
                is_pydantic_json_type = False
                if origin in (Union, Optional) and args:
                    is_pydantic_json_type = any(get_origin(arg) is Json for arg in args if get_origin(arg) is not None)
                elif get_origin(pydantic_annotation) is Json:
                    is_pydantic_json_type = True
                # Compatibility check for older Pydantic where Json might not have origin
                elif pydantic_annotation is Json or getattr(pydantic_annotation, "__origin__", None) is Json:
                    is_pydantic_json_type = True

                if is_pydantic_json_type:
                    if django_value is None:
                        data[field_name] = None
                    else:
                        try:
                            # Dump only if Pydantic target involves Json type
                            data[field_name] = json.dumps(django_value)
                            logger.debug(
                                f"Handling JSONField '{field_name}' -> Serialized JSON string for Pydantic Json type"
                            )
                        except TypeError as e:
                            logger.error(f"Failed to serialize JSON for field '{field_name}': {e}", exc_info=True)
                            data[field_name] = None  # Or handle error appropriately
                else:
                    # Target Pydantic type likely dict/list, assign directly
                    data[field_name] = django_value
                    logger.debug(
                        f"Handling JSONField '{field_name}' -> Assigning raw value ({type(django_value)}) to Pydantic field"
                    )
            else:
                # Includes cases where meta is None (property, Pydantic-only field)
                # or meta is for a simple type (CharField, IntegerField etc)
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
        # Consider wrapping the exception for more context
        raise ValueError(
            f"Failed to create Pydantic model {pydantic_model.__name__} from Django instance {db_obj}: {e}"
        ) from e


# --- Dynamic Pydantic Model Generation ---

# Mapping from Django Field types to Python/Pydantic types
# REMOVED - We will now use TypeMapper from mapping.py
# DJANGO_FIELD_TO_PYDANTIC_TYPE = { ... }


def generate_pydantic_class(
    django_model_cls: type[models.Model],
    *,
    model_name: Optional[str] = None,
    cache: Optional[GeneratedModelCache] = None,
    depth: int = 0,
    max_depth: int = 3,
    pydantic_base: Optional[type[BaseModel]] = None,
    django_metadata: Optional[dict[str, DjangoFieldMetadata]] = None,
) -> Union[Type[BaseModel], ForwardRef]:
    """
    Dynamically generates a Pydantic model class from a Django model class,
    using pre-extracted metadata if provided.

    Args:
        django_model_cls: The Django model class to convert.
        model_name: Optional explicit name for the generated Pydantic model.
                    Defaults to f"{django_model_cls.__name__}Pydantic".
        cache: A dictionary to cache generated models and prevent recursion errors.
               Must be provided for recursive generation.
        depth: Current recursion depth.
        max_depth: Maximum recursion depth for related models.
        pydantic_base: Optional base for generated Pydantic model.
        django_metadata: Optional pre-extracted metadata for the model's fields.
                         If None, it will be extracted.

    Returns:
        A dynamically created Pydantic model class.

    Raises:
        ValueError: If maximum recursion depth is exceeded or generation fails.
        TypeError: If a field type cannot be mapped.
    """
    if cache is None:
        cache = {}
        logger.debug(f"Initializing generation cache for {django_model_cls.__name__}")

    if django_model_cls in cache:
        logger.debug(f"Cache hit for {django_model_cls.__name__} (Depth: {depth})")
        return cache[django_model_cls]

    if depth > max_depth:
        logger.warning(
            f"Max recursion depth ({max_depth}) reached for {django_model_cls.__name__}. Returning ForwardRef."
        )
        ref_name = model_name or f"{django_model_cls.__name__}Pydantic"
        return ForwardRef(ref_name)

    # Extract metadata if not provided
    if django_metadata is None:
        django_metadata = _extract_django_model_metadata(django_model_cls)

    pydantic_model_name = model_name or f"{django_model_cls.__name__}Pydantic"
    logger.debug(
        f"Generating Pydantic model '{pydantic_model_name}' for Django model '{django_model_cls.__name__}' (Depth: {depth})"
    )

    # Use name for caching forward refs and final models
    if pydantic_model_name in cache:
        logger.debug(f"Cache hit for name '{pydantic_model_name}' (Depth: {depth})")
        return cache[pydantic_model_name]

    forward_ref = ForwardRef(pydantic_model_name)
    cache[pydantic_model_name] = forward_ref  # Cache the ForwardRef by name

    field_definitions: dict[str, tuple[Any, Any]] = {}

    # Use extracted metadata
    for field_name, meta in django_metadata.items():
        pydantic_field_default = PydanticUndefined
        if meta.default is not models.NOT_PROVIDED:
            if callable(meta.default):
                logger.warning(
                    f"Field '{field_name}' has callable default {meta.default}, cannot directly translate. Ignoring default."
                )
            else:
                pydantic_field_default = meta.default

        python_type: Any = None

        # --- Handle Relationships using Metadata ---
        if meta.is_relation and meta.related_model:
            related_model_cls = meta.related_model
            logger.debug(
                f"Processing relation field '{field_name}' -> {related_model_cls.__name__} (Type: {'M2M' if meta.is_m2m else 'FK/O2O'}, SelfRef: {meta.is_self_ref}, Depth: {depth})"
            )

            try:
                # Recursively generate the related Pydantic model/ForwardRef
                # Pass None for django_metadata to force extraction in recursive call if needed
                related_pydantic_model_ref = generate_pydantic_class(
                    related_model_cls,
                    cache=cache,
                    depth=depth + 1,
                    max_depth=max_depth,
                    pydantic_base=pydantic_base,
                    django_metadata=None,  # Let recursive call handle its own metadata
                )

                if meta.is_m2m:
                    python_type = list[related_pydantic_model_ref]  # type: ignore
                    # Set default for M2M list
                    pydantic_field_default = Field(default_factory=list)
                else:  # FK or O2O
                    python_type = related_pydantic_model_ref

            except ValueError as e:
                logger.error(f"Failed generating related model for '{field_name}': {e}")
                python_type = Any
                # If relation generation failed, make the field optional in Pydantic
                # to avoid blocking the parent model generation.
                meta.is_nullable = True  # Override nullability if sub-generation fails

        # --- Handle Simple Fields using TypeMapper --- (No change needed here initially)
        else:
            logger.debug(
                f"Attempting to map simple field '{field_name}' of type {meta.django_field_type.__name__} using TypeMapper"
            )
            # Use TypeMapper to get the base Python type
            mapping_definition = TypeMapper.get_mapping_for_type(meta.django_field_type)

            if mapping_definition and not mapping_definition.is_relationship:
                python_type = mapping_definition.python_type
                logger.debug(f"--> TypeMapper mapped {meta.django_field_type.__name__} to {python_type}")
            elif mapping_definition and mapping_definition.is_relationship:
                # This shouldn't happen if metadata extraction is correct, but handle defensively
                logger.warning(
                    f"TypeMapper returned a relationship mapping for field '{field_name}' ({meta.django_field_type.__name__}) in simple field handler. Falling back to Any."
                )
                python_type = Any
            else:
                logger.warning(
                    f"TypeMapper could not find mapping for Django field '{field_name}' of type {meta.django_field_type.__name__}. Falling back to 'Any'."
                )
                python_type = Any

        # --- Final Type Adjustment and Field Definition ---
        if python_type is not None:
            final_type = Optional[python_type] if meta.is_nullable else python_type

            # Prepare arguments for pydantic.Field
            field_args = {}
            description = getattr(meta.django_field, "help_text", None)
            if description:
                field_args["description"] = str(description)  # Ensure it's a string

            # Determine the default value or factory
            if meta.is_m2m:
                # Use default_factory for lists/M2M
                field_args["default_factory"] = list
            elif pydantic_field_default is not PydanticUndefined:
                # Use explicit Django default if available and not None for nullable
                if meta.is_nullable and pydantic_field_default is None:
                    field_args["default"] = None
                elif not (
                    meta.is_nullable and pydantic_field_default is None
                ):  # Avoid setting default=None if already handled above
                    field_args["default"] = pydantic_field_default
            elif meta.is_nullable:
                # No explicit default, but field is nullable
                field_args["default"] = None
            # else: Field is required, no default needed in args

            # Create the field definition using Ellipsis for required fields without other args,
            # otherwise use Field(**field_args)
            if (
                not field_args
                and not meta.is_nullable
                and pydantic_field_default is PydanticUndefined
                and not meta.is_m2m
            ):
                default_or_field = ...  # Required field with no description/default
            else:
                default_or_field = Field(**field_args)

            field_definitions[field_name] = (final_type, default_or_field)
        else:
            logger.error(f"Could not determine Python type for field '{field_name}'. Skipping.")

    # Create the Pydantic model
    try:
        created_model = create_model(
            pydantic_model_name,
            __base__=pydantic_base or BaseModel,
            __module__=__name__,  # Explicitly set module for ForwardRef resolution
            **field_definitions,  # Unpack the dictionary here # type: ignore[arg-type]
            # Ignore potential type checker issues with create_model dynamic kwargs
        )
        logger.info(f"Successfully created Pydantic model '{pydantic_model_name}'")

        # Update cache with the actual model, replacing the ForwardRef (using name key)
        cache[pydantic_model_name] = created_model

        # Update ForwardRef cache entry ONLY if it's still a ForwardRef
        cached_item = cache.get(django_model_cls)
        if isinstance(cached_item, ForwardRef):
            cached_item.__forward_arg__ = created_model  # Update the forward ref
            logger.debug(f"Updated ForwardRef for {pydantic_model_name} in cache.")
        elif cached_item is not created_model:  # Check if cache was unexpectedly overwritten
            logger.warning(
                f"Cache for {django_model_cls.__name__} was unexpectedly updated during generation. Final model: {type(created_model)}"
            )
            # Ensure cache holds the final model
            cache[django_model_cls] = created_model

        return created_model
    except Exception as e:
        logger.exception(f"Failed to create Pydantic model '{pydantic_model_name}' with fields {field_definitions}")
        # Clean cache entry if creation failed after inserting ForwardRef
        if pydantic_model_name in cache and cache[pydantic_model_name] is forward_ref:
            del cache[pydantic_model_name]
        raise TypeError(f"Failed to create Pydantic model {pydantic_model_name}: {e}") from e


# Type alias for the generation cache to improve clarity
# Maps model name (str) to the generated type or a ForwardRef
GeneratedModelNameCache = dict[str, Union[Type[BaseModel], ForwardRef]]


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
        # Use the new Name Cache type
        self._generation_cache: GeneratedModelNameCache = {}
        self._django_metadata: dict[str, DjangoFieldMetadata] = _extract_django_model_metadata(self.django_model_cls)

        # Generate the Pydantic class definition immediately
        self.pydantic_model_cls = self._generate_pydantic_class()

        # Rebuild the generated model to resolve forward references
        try:
            self.pydantic_model_cls.model_rebuild(
                force=True, _types_namespace=globals()  # Use globals() for namespace resolution
            )
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
        # Pass the pre-extracted metadata
        generated_type = generate_pydantic_class(
            self.django_model_cls,
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

            elif meta.django_field_type == models.DateTimeField:
                self._assign_datetime_field(
                    target_django_instance, cast(models.DateTimeField, django_field), pydantic_value
                )

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
                        related_pks = [item.get(related_pk_name) for item in pydantic_m2m_list]
                        valid_pks = [pk for pk in related_pks if pk is not None]
                        if len(valid_pks) != len(pydantic_m2m_list):
                            logger.warning(
                                f"Some related dictionaries for M2M field '{field_name}' were missing PKs ('{related_pk_name}') or had None PK."
                            )
                        related_objs_or_pks = valid_pks # Manager.set() can handle PKs
                    except Exception as e:
                         logger.error(f"Failed to extract PKs from dictionary list for M2M field '{field_name}': {e}\")
                         raise ValueError(f"Invalid dictionary list for M2M field '{field_name}\'.\") from e

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
