import dataclasses
import logging
from typing import Optional, TypeVar, Union, get_args, get_origin

from django.db import models

from ..core.context import ModelContext  # Assuming this exists and is correct

# Core imports
from ..core.factories import (
    BaseFieldFactory,
    BaseModelFactory,
    ConversionCarrier,
    FieldConversionResult,
)
from ..core.relationships import RelationshipConversionAccessor  # Assuming this exists

# Django mapping and utils
from ..django.mapping import TypeMapper, TypeMappingDefinition

# from ..django.utils.naming import sanitize_related_name # Not used directly?
# from ..django.models import Pydantic2DjangoBaseClass # Base model not used here

# Dataclass utils? (If needed, TBD)
# from .utils import ...

# Define DataclassType alias if not globally defined (e.g., in core.defs)
# For now, define locally:
DataclassType = TypeVar("DataclassType")

logger = logging.getLogger(__name__)

# --- Dataclass Specific Factories ---


class DataclassFieldFactory(BaseFieldFactory[dataclasses.Field]):
    """Creates Django model fields from dataclass fields."""

    relationship_accessor: Optional[RelationshipConversionAccessor]  # Added accessor

    def __init__(self, relationship_accessor: Optional[RelationshipConversionAccessor] = None):
        self.relationship_accessor = relationship_accessor
        super().__init__()  # Call base __init__

    def create_field(
        self, field_info: dataclasses.Field, model_name: str, carrier: ConversionCarrier[DataclassType]
    ) -> FieldConversionResult:
        """
        Maps a dataclasses.Field object to a Django models.Field instance.
        Uses TypeMapper for the Django field mapping.
        Relies on field_info.metadata['django'] for specific overrides.
        """
        field_name = field_info.name
        original_field_type = field_info.type
        metadata = field_info.metadata or {}  # Ensure metadata is a dict
        django_meta_options = metadata.get("django", {})

        # Initialize result
        result = FieldConversionResult(field_info=field_info, field_name=field_name)
        logger.debug(f"Processing field {model_name}.{field_name}: Type={original_field_type}, Metadata={metadata}")

        try:
            # 1. Analyze type (similar to Pydantic factory, maybe refactor to core util?)
            origin = get_origin(original_field_type)
            args = get_args(original_field_type)
            is_optional = origin is Union and type(None) in args
            field_type = (
                next((arg for arg in args if arg is not type(None)), original_field_type)
                if is_optional
                else original_field_type
            )
            # Update origin/args if we unwrapped Optional
            if is_optional:
                origin = get_origin(field_type)
                args = get_args(field_type)

            logger.debug(f"  -> Analyzed: Optional={is_optional}, Core Type={field_type}, Origin={origin}, Args={args}")

            # --- Get Mapping (includes relationship detection) ---
            mapping_definition = TypeMapper.get_mapping_for_type(field_type)
            target_dataclass_model = None

            if not mapping_definition:
                if dataclasses.is_dataclass(field_type):
                    logger.debug(f"  -> Type {field_type.__name__} is a dataclass, treating as ForeignKey.")
                    mapping_definition = TypeMappingDefinition(
                        python_type=field_type, django_field=models.ForeignKey, is_relationship=True
                    )
                    target_dataclass_model = field_type
                elif origin in (list, set) and args and dataclasses.is_dataclass(args[0]):
                    logger.debug(
                        f"  -> Type {original_field_type} is a collection of dataclasses, treating as ManyToMany."
                    )
                    mapping_definition = TypeMappingDefinition(
                        python_type=original_field_type, django_field=models.ManyToManyField, is_relationship=True
                    )
                    target_dataclass_model = args[0]
                else:
                    logger.warning(
                        f"Could not map '{model_name}.{field_name}' of type {field_type}. Treating as context field."
                    )
                    result.context_field = field_info
                    return result
            elif mapping_definition.is_relationship:
                # If TypeMapper found a relationship, determine the target dataclass
                if mapping_definition.django_field in (models.ForeignKey, models.OneToOneField):
                    target_dataclass_model = field_type
                elif mapping_definition.django_field == models.ManyToManyField and args:
                    target_dataclass_model = args[0]
                # Ensure it actually is a dataclass
                if not target_dataclass_model or not dataclasses.is_dataclass(target_dataclass_model):
                    result.error_str = f"TypeMapper identified relationship, but target type {target_dataclass_model} is not a dataclass."
                    logger.warning(result.error_str)
                    result.context_field = field_info
                    return result

            logger.debug(
                f"  -> Mapping: {mapping_definition.django_field.__name__}, IsRel: {mapping_definition.is_relationship}"
            )

            # 3. Prepare Django field kwargs from metadata and defaults
            # Start with mapping definition defaults
            field_kwargs = mapping_definition.field_kwargs.copy()
            # Update with explicit Django options from metadata
            field_kwargs.update(django_meta_options)

            # Null/Blank based on Optional status (use field_type which might be Optional[T])
            type_mapper_attrs = TypeMapper.get_field_attributes(original_field_type)
            # Only update null/blank if not explicitly set in metadata
            if "null" not in field_kwargs:
                field_kwargs["null"] = type_mapper_attrs.get("null", False)
            if "blank" not in field_kwargs:
                field_kwargs["blank"] = type_mapper_attrs.get("blank", False)
                # Ensure blank=True if null=True for non-char/text fields (common pattern)
                is_char_or_text = mapping_definition.django_field in (models.CharField, models.TextField)
                if field_kwargs["null"] and not is_char_or_text:
                    field_kwargs["blank"] = True

            # Default value from dataclasses.Field
            if field_info.default is not dataclasses.MISSING and "default" not in field_kwargs:
                # Avoid mutable defaults
                if not isinstance(field_info.default, (list, dict, set)):
                    field_kwargs["default"] = field_info.default
                else:
                    logger.warning(
                        f"Field {model_name}.{field_name} has mutable default {field_info.default}. Skipping default."
                    )
            elif field_info.default_factory is not dataclasses.MISSING and "default" not in field_kwargs:
                # Django doesn't support default_factory directly
                logger.warning(
                    f"Field {model_name}.{field_name} uses default_factory. Set Django default via metadata if needed."
                )

            # --- Handle Relationship Kwargs ---
            if mapping_definition.is_relationship:
                if not self.relationship_accessor:
                    result.error_str = (
                        f"Relationship field '{field_name}' found, but no RelationshipAccessor provided to factory."
                    )
                    logger.error(result.error_str)
                    result.context_field = field_info
                    return result
                if not target_dataclass_model:  # Should be set if is_relationship is true
                    result.error_str = f"Internal error: Relationship mapping found but target dataclass model not identified for '{field_name}'."
                    logger.error(result.error_str)
                    result.context_field = field_info
                    return result

                # Use helper to get 'to' and 'related_name'
                rel_specific_kwargs, error = self._handle_relationship_kwargs(
                    carrier=carrier,
                    field_name=field_name,
                    target_dataclass_model=target_dataclass_model,
                    source_model_name=model_name,
                    base_rel_kwargs=field_kwargs,  # Pass current kwargs to get related_name override
                )
                if error:
                    result.error_str = error
                    result.context_field = field_info
                    return result
                field_kwargs.update(rel_specific_kwargs)

                # 'on_delete' (already handled partially in Pydantic, refine here)
                if (
                    mapping_definition.django_field in (models.ForeignKey, models.OneToOneField)
                    and "on_delete" not in field_kwargs
                ):
                    field_kwargs["on_delete"] = models.SET_NULL if field_kwargs.get("null", False) else models.CASCADE
                elif mapping_definition.django_field == models.ManyToManyField:
                    field_kwargs.pop("on_delete", None)  # Ensure no on_delete for M2M

            # --- Specific Field Type Attrs (e.g., max_length) ---
            elif mapping_definition.django_field == models.CharField:
                if "max_length" not in field_kwargs:
                    # Use mapping definition max_length or default
                    field_kwargs["max_length"] = mapping_definition.max_length or 255
            elif mapping_definition.django_field == models.DecimalField:
                if "max_digits" not in field_kwargs or "decimal_places" not in field_kwargs:
                    result.error_str = f"DecimalField '{model_name}.{field_name}' requires 'max_digits' and 'decimal_places' in metadata."
                    logger.error(result.error_str)
                    result.context_field = field_info
                    return result

            # --- Instantiate ---
            try:
                django_field_class = mapping_definition.django_field
                # Remove placeholder relationship args if they weren't resolved by helper
                to_value = field_kwargs.get("to")
                if mapping_definition.is_relationship and isinstance(to_value, str) and "Placeholder" in to_value:
                    result.error_str = f"Relationship 'to' field for '{field_name}' could not be resolved."
                    logger.error(result.error_str)
                    result.context_field = field_info
                    return result

                logger.debug(f"  -> Instantiating: models.{django_field_class.__name__}(**{field_kwargs})")
                result.django_field = django_field_class(**field_kwargs)
            except Exception as e:
                error_msg = f"Failed to instantiate Django field for {model_name}.{field_name}: {e}"
                logger.error(error_msg, exc_info=True)
                result.error_str = error_msg
                result.context_field = field_info  # Fallback to context
                result.django_field = None

            return result

        except Exception as e:
            # Catch-all for unexpected errors
            error_msg = f"Unexpected error converting dataclass field '{model_name}.{field_name}': {e}"
            logger.error(error_msg, exc_info=True)
            result.error_str = error_msg
            result.context_field = field_info  # Fallback to context
            result.django_field = None
            return result

    def _handle_relationship_kwargs(
        self,
        carrier: ConversionCarrier[DataclassType],
        field_name: str,
        target_dataclass_model: type,
        source_model_name: str,
        base_rel_kwargs: dict,  # Kwargs potentially containing user related_name override
    ) -> tuple[dict, Optional[str]]:  # Returns (resolved_kwargs, error_string)
        """Resolves 'to' and 'related_name' using the RelationshipAccessor."""
        resolved_kwargs = {}
        error = None

        if not self.relationship_accessor:
            # This case should ideally be caught before calling this helper
            error = f"Internal Error: _handle_relationship_kwargs called without a RelationshipAccessor for field '{field_name}'."
            logger.error(error)
            return {}, error

        # --- Resolve 'to' ---
        target_django_model = self.relationship_accessor.get_django_model_for_dataclass(target_dataclass_model)
        if not target_django_model:
            error = f"Target Django model for dataclass '{target_dataclass_model.__name__}' (field '{field_name}') not found in RelationshipAccessor."
            logger.warning(error)
            return {}, error

        target_model_label = f"{target_django_model._meta.app_label}.{target_django_model.__name__}"
        resolved_kwargs["to"] = target_model_label

        # --- Resolve 'related_name' ---
        # Check for override in metadata first
        user_related_name = base_rel_kwargs.get("related_name")
        related_name_base = user_related_name if user_related_name else f"{source_model_name.lower()}_{field_name}_set"

        # Import sanitize_related_name locally or move import to top
        from ..django.utils.naming import sanitize_related_name

        final_related_name = sanitize_related_name(str(related_name_base), target_django_model.__name__, field_name)

        # Ensure uniqueness
        target_related_names = carrier.used_related_names_per_target.setdefault(target_django_model.__name__, set())
        unique_related_name = final_related_name
        counter = 1
        while unique_related_name in target_related_names:
            unique_related_name = f"{final_related_name}_{counter}"
            counter += 1
        target_related_names.add(unique_related_name)
        resolved_kwargs["related_name"] = unique_related_name
        logger.debug(f"[REL] Dataclass Field '{field_name}': Assigning related_name='{unique_related_name}'")

        return resolved_kwargs, error


class DataclassModelFactory(BaseModelFactory[DataclassType, dataclasses.Field]):
    """Dynamically creates Django model classes from dataclasses."""

    # Cache specific to Dataclass models
    _converted_models: dict[str, ConversionCarrier[DataclassType]] = {}

    relationship_accessor: Optional[RelationshipConversionAccessor]  # Optional for now

    def __init__(
        self,
        field_factory: DataclassFieldFactory,
        relationship_accessor: Optional[RelationshipConversionAccessor] = None,
    ):
        self.relationship_accessor = relationship_accessor  # Store if provided
        super().__init__(field_factory=field_factory)

    # Overrides the base method to add caching
    def make_django_model(self, carrier: ConversionCarrier[DataclassType]) -> None:
        """Creates a Django model from a dataclass, checking cache first."""
        # Initial validation specific to dataclasses
        source_model = carrier.source_model
        if not dataclasses.is_dataclass(source_model):
            logger.error(f"DataclassFactory: Cannot create model - Source {carrier.model_key} is not a dataclass.")
            carrier.django_model = None
            carrier.invalid_fields.append(("_source_type", "Input is not a dataclass"))
            return

        model_key = carrier.model_key()
        logger.debug(f"DataclassFactory: Attempting to create Django model for {model_key}")

        # --- Check Cache ---
        if model_key in self._converted_models and not carrier.existing_model:
            logger.debug(f"DataclassFactory: Using cached conversion result for {model_key}")
            cached_carrier = self._converted_models[model_key]
            # Copy results
            carrier.django_fields = cached_carrier.django_fields.copy()
            carrier.relationship_fields = cached_carrier.relationship_fields.copy()
            carrier.context_fields = cached_carrier.context_fields.copy()
            carrier.invalid_fields = cached_carrier.invalid_fields.copy()
            carrier.django_meta_class = cached_carrier.django_meta_class
            carrier.django_model = cached_carrier.django_model
            carrier.model_context = cached_carrier.model_context
            carrier.used_related_names_per_target.update(cached_carrier.used_related_names_per_target)
            return

        # --- Call Base Implementation for Core Logic ---
        super().make_django_model(carrier)

        # --- Cache Result ---
        if carrier.django_model and not carrier.existing_model:
            logger.debug(f"DataclassFactory: Caching conversion result for {model_key}")
            # self._converted_models[model_key] = replace(carrier) # Linter issue?
            self._converted_models[carrier.model_key()] = carrier  # Direct assign # Call model_key()
        elif not carrier.django_model:
            logger.error(
                f"DataclassFactory: Failed to create Django model for {model_key}. Invalid fields: {carrier.invalid_fields}"
            )

    # --- Implementation of Abstract Methods ---

    def _process_source_fields(self, carrier: ConversionCarrier[DataclassType]):
        """Iterate through dataclass fields and convert them."""
        source_model = carrier.source_model
        model_name = source_model.__name__

        # Guard again just in case, though checked in make_django_model
        if not dataclasses.is_dataclass(source_model):
            logger.error(f"_process_source_fields called with non-dataclass: {source_model}")
            return

        for field_info in dataclasses.fields(source_model):
            field_name = field_info.name

            conversion_result = self.field_factory.create_field(
                field_info=field_info, model_name=model_name, carrier=carrier
            )

            if conversion_result.django_field:
                if isinstance(
                    conversion_result.django_field, (models.ForeignKey, models.ManyToManyField, models.OneToOneField)
                ):
                    carrier.relationship_fields[field_name] = conversion_result.django_field
                else:
                    carrier.django_fields[field_name] = conversion_result.django_field
            elif conversion_result.context_field:
                carrier.context_fields[field_name] = conversion_result.context_field
            elif conversion_result.error_str:
                carrier.invalid_fields.append((field_name, conversion_result.error_str))
            else:
                error = f"Dataclass field factory returned unexpected result for {model_name}.{field_name}: {conversion_result}"
                logger.error(error)
                carrier.invalid_fields.append((field_name, error))

    def _build_model_context(self, carrier: ConversionCarrier[DataclassType]):
        """Builds the ModelContext specifically for dataclass source models."""
        # Implementation requires ModelContext to be fully generic
        if not carrier.source_model or not carrier.django_model:
            logger.debug("Skipping context build: missing source or django model.")
            return

        try:
            model_context = ModelContext[DataclassType](  # Use generic type hint
                django_model=carrier.django_model, source_class=carrier.source_model
            )
            for field_name, field_info in carrier.context_fields.items():
                if isinstance(field_info, dataclasses.Field):
                    origin = get_origin(field_info.type)
                    args = get_args(field_info.type)
                    optional = origin is Union and type(None) in args
                    field_type_str = repr(field_info.type)
                    # Call add_field with correct signature
                    model_context.add_field(field_name=field_name, field_type_str=field_type_str, is_optional=optional)
                else:
                    logger.warning(
                        f"Context field '{field_name}' is not a dataclasses.Field ({type(field_info)}), cannot add to ModelContext."
                    )
            carrier.model_context = model_context
            logger.debug(f"Successfully built ModelContext for {carrier.model_key}")
        except Exception as e:
            logger.error(f"Failed to build ModelContext for {carrier.model_key}: {e}", exc_info=True)
            carrier.model_context = None

    # --- Removed Methods (Now in Base Class) ---
    # _handle_field_collisions
    # _create_django_meta
    # _assemble_django_model_class
