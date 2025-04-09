"""
Provides functionality to convert Django model instances to Pydantic models.
"""
import datetime
import json
import logging
from decimal import Decimal
from typing import Any, ForwardRef, Optional, TypeVar, Union, get_args, get_origin, Dict, Type, List
from uuid import UUID

from django.db import models
from django.db.models.fields.related import ForeignKey, ManyToManyField, OneToOneField, RelatedField
from django.db.models.fields.reverse_related import ManyToManyRel, ManyToOneRel, OneToOneRel
from pydantic import BaseModel, EmailStr, Field, IPvAnyAddress, Json, create_model, FieldInfo
from pydantic_core import PydanticUndefined

# Potentially useful imports from the project (adjust as needed)
# from .mapping import TypeMapper
# from .typing import ...
# from ..core.utils import ...


logger = logging.getLogger(__name__)

PydanticModelT = TypeVar("PydanticModelT", bound=BaseModel)
DjangoModelT = TypeVar("DjangoModelT", bound=models.Model)


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

    data = {}
    exclude_set = exclude or set()

    pydantic_fields = pydantic_model.model_fields  # Pydantic v2

    for field_name, pydantic_field in pydantic_fields.items():
        if field_name in exclude_set:
            continue

        logger.debug(f"Processing field: {field_name} (Depth: {depth})")

        # Check if field exists on Django model
        if not hasattr(db_obj, field_name):
            # If the Pydantic field is required and missing on Django model, this is an issue.
            # If it's optional, we might be able to skip or use default.
            # Pydantic's validation will catch missing required fields later.
            logger.warning(f"Field '{field_name}' not found on Django model {db_obj.__class__.__name__}. Skipping.")
            continue

        django_value = getattr(db_obj, field_name)
        pydantic_annotation = pydantic_field.annotation

        # --- Handle Relationships ---
        origin = get_origin(pydantic_annotation)
        args = get_args(pydantic_annotation)

        try:
            django_field = db_obj._meta.get_field(field_name)
        except models.FieldDoesNotExist:
            # This might be a property or a method result on the Django model
            # Or it might be a field only defined on the Pydantic model
            logger.debug(
                f"'{field_name}' not a direct model field on {db_obj.__class__.__name__}. Assuming property/method or Pydantic-only."
            )
            django_field = None  # Indicate it's not a standard DB field

        # 1. ManyToManyField / List[BaseModel]
        if isinstance(django_field, ManyToManyField) and origin is list and args and issubclass(args[0], BaseModel):
            related_pydantic_model = args[0]
            logger.debug(f"Handling ManyToMany relationship for '{field_name}' -> {related_pydantic_model.__name__}")
            related_manager = django_value  # This is the RelatedManager
            converted_related = []
            for related_obj in related_manager.all():
                try:
                    converted_related.append(
                        django_to_pydantic(
                            related_obj,
                            related_pydantic_model,
                            exclude=exclude,  # Pass exclude along
                            depth=depth + 1,
                            max_depth=max_depth,
                        )
                    )
                except ValueError as e:  # Catch max depth error
                    logger.error(f"Failed converting related object in M2M '{field_name}': {e}")
                    # Decide: skip this item or re-raise? Skip for now.
                    continue
            data[field_name] = converted_related

        # 2. ForeignKey or OneToOneField / Optional[BaseModel] or BaseModel
        elif isinstance(django_field, RelatedField) and not isinstance(django_field, ManyToManyField):
            # Determine the actual Pydantic type (handling Optional[...])
            related_pydantic_model: type[BaseModel] | None = None
            if origin is Union and type(None) in args and len(args) == 2:  # Optional[Model]
                potential_model = next((arg for arg in args if arg is not type(None)), None)
                if potential_model and issubclass(potential_model, BaseModel):
                    related_pydantic_model = potential_model
            elif pydantic_annotation and issubclass(pydantic_annotation, BaseModel):  # Model
                related_pydantic_model = pydantic_annotation

            if related_pydantic_model:
                logger.debug(
                    f"Handling ForeignKey/OneToOne relationship for '{field_name}' -> {related_pydantic_model.__name__}"
                )
                related_obj = django_value  # This is the related Django instance or None
                if related_obj:
                    try:
                        data[field_name] = django_to_pydantic(
                            related_obj,
                            related_pydantic_model,
                            exclude=exclude,  # Pass exclude along
                            depth=depth + 1,
                            max_depth=max_depth,
                        )
                    except ValueError as e:  # Catch max depth error
                        logger.error(f"Failed converting related object in FK/O2O '{field_name}': {e}")
                        data[field_name] = None  # Set to None if conversion fails due to depth
                else:
                    data[field_name] = None
            else:
                # Pydantic field is not a BaseModel, treat as simple field (e.g., FK's pk)
                logger.debug(f"Treating related field '{field_name}' as simple value (e.g., PK).")
                data[field_name] = django_value  # Assign the raw value (likely PK)

        # --- Handle Simple Fields ---
        else:
            # Handle specific field types before direct assignment
            if isinstance(django_field, models.FileField):
                # For FileField/ImageField, Pydantic usually expects a URL or path string.
                # If the field has a value, try getting its URL, otherwise None.
                if django_value and hasattr(django_value, "url"):
                    data[field_name] = django_value.url
                else:
                    data[field_name] = None  # Or empty string? Pydantic should handle Optional[str]
                logger.debug(f"Handling {type(django_field).__name__} '{field_name}' -> URL: {data[field_name]}")
            elif isinstance(django_field, models.JSONField) and not isinstance(django_value, (str, bytes, bytearray)):
                # Pydantic's Json type expects a string/bytes, not a pre-parsed dict/list.
                # We need to dump the Python object back to a JSON string.
                # Handle potential None value from Django.
                if django_value is None:
                    data[field_name] = None
                else:
                    try:
                        data[field_name] = json.dumps(django_value)
                    except TypeError as e:
                        logger.error(f"Failed to serialize JSON for field '{field_name}': {e}", exc_info=True)
                        # Decide handling: raise error, set None, or pass raw value?
                        # Setting to None for now to avoid Pydantic validation error immediately.
                        data[field_name] = None
                logger.debug(f"Handling JSONField '{field_name}' -> Serialized JSON string")
            else:
                # Directly assign other simple values. Pydantic will validate types.
                logger.debug(f"Handling simple field '{field_name}' with value: {django_value!r}")
                data[field_name] = django_value

    # Instantiate the Pydantic model with the collected data
    try:
        instance = pydantic_model(**data)
        logger.info(
            f"Successfully converted {db_obj.__class__.__name__} instance (PK: {db_obj.pk}) to {pydantic_model.__name__}"
        )
        return instance
    except Exception as e:
        logger.error(f"Failed to instantiate Pydantic model {pydantic_model.__name__} with data {data}", exc_info=True)
        # Consider wrapping the exception for more context
        raise ValueError(
            f"Failed to create Pydantic model {pydantic_model.__name__} from Django instance {db_obj}: {e}"
        ) from e


# --- Dynamic Pydantic Model Generation ---

# Mapping from Django Field types to Python/Pydantic types
# Needs careful handling, especially for complex types and relationships
DJANGO_FIELD_TO_PYDANTIC_TYPE = {
    models.AutoField: int,
    models.BigAutoField: int,
    models.SmallIntegerField: int,
    models.IntegerField: int,
    models.BigIntegerField: int,
    models.PositiveSmallIntegerField: int,
    models.PositiveIntegerField: int,
    models.FloatField: float,
    models.BooleanField: bool,
    models.NullBooleanField: Optional[bool],  # Deprecated in Django, but handle if present
    models.CharField: str,
    models.TextField: str,
    models.SlugField: str,
    models.EmailField: EmailStr,
    models.GenericIPAddressField: IPvAnyAddress,
    models.URLField: str,  # Could use Pydantic's AnyUrl
    models.DateField: datetime.date,
    models.DateTimeField: datetime.datetime,
    models.DurationField: datetime.timedelta,
    models.TimeField: datetime.time,
    models.DecimalField: Decimal,
    models.UUIDField: UUID,
    models.JSONField: Json,  # Pydantic's Json type handles serialization/deserialization
    models.BinaryField: bytes,
    # FileField/ImageField typically map to URL/path strings in Pydantic
    models.FileField: Optional[str],
    models.FilePathField: Optional[str],
    models.ImageField: Optional[str],
    # Relationships are handled separately
    models.ForeignKey: Any,  # Placeholder, resolved dynamically
    models.OneToOneField: Any,  # Placeholder, resolved dynamically
    models.ManyToManyField: Any,  # Placeholder, resolved dynamically
}

GeneratedModelCache = Dict[Type[models.Model], Type[BaseModel] | ForwardRef]


def generate_pydantic_class(
    django_model_cls: Type[models.Model],
    *,
    model_name: Optional[str] = None,
    cache: Optional[GeneratedModelCache] = None,
    depth: int = 0,
    max_depth: int = 3,  # Same default as django_to_pydantic
    # exclude: Optional[set[str]] = None, # TODO: Add exclude support?
) -> Type[BaseModel] | ForwardRef:  # Allow ForwardRef in return type
    """
    Dynamically generates a Pydantic model class from a Django model class.

    Args:
        django_model_cls: The Django model class to convert.
        model_name: Optional explicit name for the generated Pydantic model.
                    Defaults to f"{django_model_cls.__name__}Pydantic".
        cache: A dictionary to cache generated models and prevent recursion errors.
               Must be provided for recursive generation.
        depth: Current recursion depth.
        max_depth: Maximum recursion depth for related models.
        # exclude: Optional set of field names to exclude.

    Returns:
        A dynamically created Pydantic model class.

    Raises:
        ValueError: If maximum recursion depth is exceeded or generation fails.
        TypeError: If a field type cannot be mapped.
    """
    if cache is None:
        # Initialize cache if this is the top-level call
        cache = {}
        logger.debug(f"Initializing generation cache for {django_model_cls.__name__}")

    if django_model_cls in cache:
        logger.debug(f"Cache hit for {django_model_cls.__name__} (Depth: {depth})")
        # Type assertion to help linter understand cache value can be BaseModel or ForwardRef
        cached_item = cache[django_model_cls]
        # The type checker might still struggle here depending on its sophistication
        # but this expresses intent.
        if isinstance(cached_item, ForwardRef):
            return cached_item  # type: ignore # Explicitly ignore if checker complains
        # else: # It must be Type[BaseModel] based on cache type hint
        #     return cached_item
        # Simplified return:
        return cached_item

    if depth > max_depth:
        logger.warning(
            f"Max recursion depth ({max_depth}) reached for {django_model_cls.__name__}. Returning ForwardRef."
        )
        ref_name = model_name or f"{django_model_cls.__name__}Pydantic"
        # Ensure the ForwardRef string is properly quoted internally for Pydantic resolution
        # The __name__ gives the current module, assuming the generated model 'lives' here.
        # If models are generated across modules, this might need adjustment.
        # Correct format for ForwardRef string seems to be just the name if in the same module,
        # or 'module.name' if not. Pydantic resolves based on context.
        # Let's try just the name, assuming resolution within the generated scope.
        # return ForwardRef(f"'{ref_name}'") # Extra quotes might be wrong
        return ForwardRef(ref_name)

    pydantic_model_name = model_name or f"{django_model_cls.__name__}Pydantic"
    logger.debug(
        f"Generating Pydantic model '{pydantic_model_name}' for Django model '{django_model_cls.__name__}' (Depth: {depth})"
    )

    # Pre-register in cache with ForwardRef to handle recursion
    forward_ref = ForwardRef(pydantic_model_name)
    cache[django_model_cls] = forward_ref

    field_definitions: Dict[str, tuple[Any, Any]] = {}  # Use Any for type part temporarily
    # exclude_set = exclude or set()

    # Iterate through concrete fields and relations
    for field in django_model_cls._meta.get_fields(include_hidden=False):
        # if field.name in exclude_set:
        #     continue

        field_name = field.name
        # Handle potential AttributeError if field doesn't have 'null' (e.g., reverse relations)
        # Although get_fields(include_hidden=False) should mostly avoid these
        is_nullable = getattr(field, "null", False)
        default_value = PydanticUndefined  # Use PydanticUndefined to signal no default unless specified

        # Get Django default if present
        django_default = getattr(field, "default", models.NOT_PROVIDED)
        pydantic_field_default = PydanticUndefined  # Default to Pydantic's undefined
        if django_default is not models.NOT_PROVIDED:
            if callable(django_default):
                logger.warning(
                    f"Field '{field_name}' has callable default {django_default}, cannot directly translate. Ignoring default."
                )
                # Pydantic's default_factory could potentially be used if the callable is simple,
                # but for general Django callables, it's safer to ignore for the dynamic model definition.
            else:
                pydantic_field_default = django_default

        python_type: Any = None  # Use Any initially

        # --- Handle Relationships ---
        if isinstance(field, (ForeignKey, OneToOneField)):
            # Handle self-referential relationships ('self')
            related_model_cls = field.related_model
            if related_model_cls == "self":
                related_model_cls = django_model_cls
                logger.debug(
                    f"Processing self-referential FK/O2O field '{field_name}' -> {related_model_cls.__name__} (Depth: {depth})"
                )
            else:
                logger.debug(f"Processing FK/O2O field '{field_name}' -> {related_model_cls.__name__} (Depth: {depth})")  # type: ignore

            try:
                related_pydantic_model_ref = generate_pydantic_class(
                    related_model_cls, cache=cache, depth=depth + 1, max_depth=max_depth  # type: ignore
                )
                python_type = related_pydantic_model_ref
            except ValueError as e:  # Catch max depth error from recursive call
                logger.error(f"Failed generating related model for '{field_name}': {e}")
                python_type = Any  # Fallback
                is_nullable = True

        elif isinstance(field, ManyToManyField):
            # Handle self-referential relationships ('self')
            related_model_cls = field.related_model
            if related_model_cls == "self":
                related_model_cls = django_model_cls
                logger.debug(
                    f"Processing self-referential M2M field '{field_name}' -> {related_model_cls.__name__} (Depth: {depth})"
                )
            else:
                logger.debug(f"Processing M2M field '{field_name}' -> {related_model_cls.__name__} (Depth: {depth})")  # type: ignore

            try:
                related_pydantic_model_ref = generate_pydantic_class(
                    related_model_cls, cache=cache, depth=depth + 1, max_depth=max_depth  # type: ignore
                )
                # M2M is always a list of related models
                python_type = List[related_pydantic_model_ref]  # type: ignore
                pydantic_field_default = Field(default_factory=list)
                is_nullable = False  # List itself isn't nullable unless outer Optional
            except ValueError as e:
                logger.error(f"Failed generating related model for M2M '{field_name}': {e}")
                python_type = List[Any]  # Fallback
                pydantic_field_default = Field(default_factory=list)
                is_nullable = False

        # --- Handle Reverse Relations ---
        elif isinstance(field, (OneToOneRel, ManyToOneRel, ManyToManyRel)):
            logger.debug(f"Skipping reverse relation field '{field_name}'")
            continue  # Exclude reverse relations

        # --- Handle Simple Fields ---
        else:
            # Find the most specific matching type in the map
            # This relies on DJANGO_FIELD_TO_PYDANTIC_TYPE being ordered reasonably
            # or ensuring exact type matches where needed.
            # Using isinstance() allows for inheritance (e.g., BigIntegerField matches IntegerField)
            # Order might matter if fields inherit (e.g., place EmailField before CharField if needed)
            # Let's refine the lookup to be slightly safer
            field_type = type(field)
            mapped_type = DJANGO_FIELD_TO_PYDANTIC_TYPE.get(field_type)

            if mapped_type:
                python_type = mapped_type
                logger.debug(f"Mapping simple field '{field_name}' ({field_type.__name__}) -> {python_type}")
            else:
                # Maybe check MRO? For now, log warning and use Any
                logger.warning(
                    f"Cannot find exact Pydantic type mapping for Django field '{field_name}' of type {field_type.__name__}. Checking base classes..."
                )
                # Simple MRO check (optional optimization)
                for base in field_type.__mro__[1:]:
                    if base in DJANGO_FIELD_TO_PYDANTIC_TYPE:
                        python_type = DJANGO_FIELD_TO_PYDANTIC_TYPE[base]
                        logger.warning(f"--> Found mapping via base class {base.__name__}: {python_type}")
                        break
                if python_type is None:
                    logger.error(f"--> No mapping found via MRO for {field_type.__name__}. Falling back to 'Any'.")
                    python_type = Any

        # --- Final Type Adjustment (Optional) and Default Handling ---
        if python_type is not None:  # Ensure we found *some* type
            final_type = Optional[python_type] if is_nullable else python_type

            # Determine the second element of the tuple for create_model
            default_or_field = ...  # Ellipsis for required fields

            if pydantic_field_default is not PydanticUndefined:
                # If nullable and default is None, Pydantic Optional[T] = None handles it
                if is_nullable and pydantic_field_default is None:
                    default_or_field = None
                # Special case for M2M default_factory=list
                elif isinstance(pydantic_field_default, FieldInfo) and pydantic_field_default.default_factory is list:
                    default_or_field = Field(default_factory=list)
                else:
                    # Use Field() to specify a non-None default value
                    default_or_field = Field(default=pydantic_field_default)
            elif is_nullable:
                # Optional field with no other default -> defaults to None
                default_or_field = None
            # else: required field, default_or_field remains Ellipsis (...)

            field_definitions[field_name] = (final_type, default_or_field)

        else:
            # This case should ideally not be reached if fallback to Any works
            logger.error(f"Internal error: Could not determine Pydantic type for field '{field_name}'. Skipping.")

    # Create the Pydantic model
    try:
        # Set __module__ to the current module (__name__)
        # This helps Pydantic resolve ForwardRefs correctly within this module.
        created_model = create_model(
            pydantic_model_name,
            __base__=BaseModel,
            __module__=__name__,  # Explicitly set module for ForwardRef resolution
            **field_definitions  # type: ignore
            # Ignore potential type checker issues with create_model dynamic kwargs
        )
        logger.info(f"Successfully created Pydantic model '{pydantic_model_name}'")

        # Update cache with the actual model, replacing the ForwardRef
        cache[django_model_cls] = created_model

        # Update ForwardRefs if Pydantic supports it (requires Pydantic v1.9+)
        # This resolves circular dependencies stored as ForwardRefs earlier
        # Needs testing if this specific call is correct/needed for Pydantic v2+
        # Pydantic v2's create_model might handle this internally better.
        # from pydantic.v1.typing import update_forward_refs # Pydantic v1 style
        # update_forward_refs(created_model) # May need adjustment for Pydantic v2
        # Let's rely on Pydantic v2's improved resolution for now.

        return created_model
    except Exception as e:
        logger.exception(f"Failed to create Pydantic model '{pydantic_model_name}' with fields {field_definitions}")
        # Clean cache entry if creation failed after inserting ForwardRef
        if django_model_cls in cache and cache[django_model_cls] is forward_ref:
            del cache[django_model_cls]
        raise TypeError(f"Failed to create Pydantic model {pydantic_model_name}: {e}") from e


def convert_django_to_dynamic_pydantic(
    db_obj: DjangoModelT,
    *,
    exclude: set[str] | None = None,
    max_depth: int = 3,
) -> BaseModel:
    """
    Converts a Django model instance to a dynamically generated Pydantic model instance.

    Args:
        db_obj: The Django model instance to convert.
        exclude: A set of field names to exclude from the conversion.
        max_depth: Maximum recursion depth for related models during data population.

    Returns:
        An instance of a dynamically generated Pydantic model populated with data
        from the Django model instance.

    Raises:
        ValueError: If conversion fails.
        TypeError: If Pydantic model generation fails.
    """
    logger.info(f"Starting dynamic conversion for {db_obj.__class__.__name__} instance (PK: {db_obj.pk})")
    generation_cache: GeneratedModelCache = {}
    try:
        # 1. Generate the Pydantic class definition recursively
        pydantic_class = generate_pydantic_class(db_obj.__class__, cache=generation_cache, max_depth=max_depth)

        # 2. Populate the generated Pydantic class using django_to_pydantic
        # We need to pass the *generated* class to django_to_pydantic
        pydantic_instance = django_to_pydantic(
            db_obj,
            pydantic_class,  # Use the generated class
            exclude=exclude,
            max_depth=max_depth,
            # Note: django_to_pydantic uses its own depth tracking for data population
        )
        return pydantic_instance

    except (ValueError, TypeError) as e:
        logger.error(f"Dynamic conversion failed for {db_obj.__class__.__name__} (PK: {db_obj.pk}): {e}", exc_info=True)
        raise  # Re-raise the caught exception


# Example usage (can be removed or moved to tests):
# if __name__ == '__main__':
#     # Assume setup for Django environment is done elsewhere
#     # from .models import MyDjangoModel # Import your Django model
#
#     try:
#         django_instance = MyDjangoModel.objects.get(pk=1)
#         dynamic_pydantic_instance = convert_django_to_dynamic_pydantic(django_instance)
#         print("Dynamic conversion successful:")
#         print(f"Generated Model Type: {type(dynamic_pydantic_instance)}")
#         print(dynamic_pydantic_instance.model_dump_json(indent=2))
#
#         # Example with exclusion (if implemented in django_to_pydantic)
#         # dynamic_pydantic_instance_excluded = convert_django_to_dynamic_pydantic(
#         #     django_instance,
#         #     exclude={'some_field', 'related_m2m_field'}
#         # )
#         # print("\\nDynamic conversion with exclusions:")
#         # print(dynamic_pydantic_instance_excluded.model_dump_json(indent=2))
#
#     except MyDjangoModel.DoesNotExist:
#         print("Django model instance not found.")
#     except Exception as e:
#         print(f"An error occurred during dynamic conversion: {e}")
