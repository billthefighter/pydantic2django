import logging
from typing import Optional, get_args, get_origin

from django.db import models
from pydantic import BaseModel
from pydantic.fields import FieldInfo

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
from ..django.utils.naming import sanitize_related_name  # get_related_model_name not needed directly here

# Pydantic utils
from .utils.introspection import get_model_fields, is_pydantic_model_field_optional

logger = logging.getLogger(__name__)

# --- Pydantic Specific Factories ---


class PydanticFieldFactory(BaseFieldFactory[FieldInfo]):
    """Creates Django fields from Pydantic fields (FieldInfo)."""

    available_relationships: RelationshipConversionAccessor

    def __init__(self, available_relationships: RelationshipConversionAccessor):
        """Initializes with the relationship accessor needed for resolving FKs/M2Ms."""
        self.available_relationships = available_relationships
        # No call to super().__init__() needed as BaseFieldFactory.__init__ does nothing now

    def create_field(
        self, field_info: FieldInfo, model_name: str, carrier: ConversionCarrier[type[BaseModel]]
    ) -> FieldConversionResult:
        """
        Convert a Pydantic FieldInfo to a Django field.
        Implements the abstract method from BaseFieldFactory.
        """
        # Use alias first, then the actual key from model_fields as name
        # field_name is passed as an argument, but let's re-derive from field_info if alias exists
        # field_name = field_info.alias or field_info.name # <-- This was the error source
        # Correction: Use the name passed in, which is derived from the dict key or alias already
        # Keep the passed 'model_name' argument for context. field_info doesn't hold the name directly.
        field_name = field_info.alias or next(
            (k for k, v in carrier.source_model.model_fields.items() if v is field_info), "<unknown>"
        )

        # Initialize result with the source field info and determined name
        result = FieldConversionResult(field_info=field_info, field_name=field_name)

        try:
            # Handle potential 'id' field conflict
            if id_field := self._handle_id_field(field_name, field_info):
                result.django_field = id_field
                return result  # ID field handled, return early

            # Get field type from annotation
            field_type = field_info.annotation
            if field_type is None:
                logger.warning(f"Field '{model_name}.{field_name}' has no annotation, treating as context field.")
                result.context_field = field_info  # Store original FieldInfo for context
                return result

            # Get the field mapping from TypeMapper
            mapping_definition = TypeMapper.get_mapping_for_type(field_type)

            if not mapping_definition:
                logger.warning(
                    f"Could not map '{model_name}.{field_name}' of type {field_type} to a Django field, treating as context field."
                )
                result.context_field = field_info  # Store original FieldInfo
                return result

            # Process relationship fields first
            if mapping_definition.is_relationship:
                # Delegate to helper method, passing necessary context
                result = self._handle_relationship_field(result, field_info, model_name, carrier, mapping_definition)
                # If relationship handler decided it's a context field or failed, return
                # No need to check django_field here, status is in context_field/error_str
                if result.context_field or result.error_str:
                    return result
                # If relationship handling succeeded, result.django_field should be set
                elif not result.django_field:
                    # This indicates an unexpected failure in _handle_relationship_field
                    if not result.error_str:
                        result.error_str = f"Relationship handler failed for '{field_name}' without error message."
                    logger.error(result.error_str)
                    result.context_field = field_info  # Fallback to context
                    return result
            else:
                # Instantiate regular Django field using TypeMapper's logic
                try:
                    # Call instantiate_django_field, passing result to be modified
                    TypeMapper.instantiate_django_field(
                        mapping_definition=mapping_definition,
                        field_info=field_info,
                        result=result,  # Pass the result object
                        field_kwargs={},  # Start with empty kwargs, TypeMapper adds defaults
                    )
                    # Check if instantiation failed (error would be set on result)
                    if result.error_str:
                        logger.error(
                            f"Failed to instantiate Django field for '{model_name}.{field_name}': {result.error_str}"
                        )
                        result.context_field = field_info  # Fallback to context
                        result.django_field = None
                        return result  # Return the result with error state

                except Exception as e:
                    # This is an unexpected error *calling* instantiate_django_field
                    error_msg = f"Unexpected error calling TypeMapper.instantiate_django_field for '{model_name}.{field_name}': {e}"
                    logger.error(error_msg, exc_info=True)
                    result.error_str = error_msg
                    result.context_field = field_info  # Fallback to context field on error
                    result.django_field = None
                    return result

            # --- Generate Field Definition String (if successful) ---
            if result.django_field:
                try:
                    from ..django.utils.serialization import generate_field_definition_string

                    # We need the final kwargs used to instantiate the field.
                    # For non-relationship fields, TypeMapper.instantiate_django_field handles this internally,
                    # but doesn't expose the final kwargs easily. We need to reconstruct or capture them.
                    # For relationship fields, _handle_relationship_field builds `rel_kwargs`.

                    # Temporary Solution: Assume result.field_kwargs is populated by handlers
                    # This requires _handle_relationship_field and TypeMapper.instantiate_django_field
                    # to store the final kwargs in result.field_kwargs.
                    # We'll need to modify those if they don't already.

                    if result.field_kwargs:
                        result.field_definition_str = generate_field_definition_string(
                            type(result.django_field), result.field_kwargs, carrier.meta_app_label
                        )
                    else:
                        logger.warning(
                            f"Could not generate definition string for '{model_name}.{field_name}': final kwargs not found in result."
                        )
                        # Attempt basic serialization as fallback (might be incomplete)
                        from ..django.utils.serialization import FieldSerializer

                        result.field_definition_str = FieldSerializer.serialize_field(result.django_field)

                except Exception as e:
                    logger.error(
                        f"Failed to generate field definition string for '{model_name}.{field_name}': {e}",
                        exc_info=True,
                    )
                    # Keep the instantiated field but mark definition as failed?
                    result.field_definition_str = f"# Error generating definition: {e}"

            # Return success or context/error from relationship handler/instantiation
            return result

        except Exception as e:
            # Catch-all for unexpected errors during conversion
            error_msg = f"Unexpected error converting field '{model_name}.{field_name}': {e}"
            logger.error(error_msg, exc_info=True)
            result.error_str = error_msg
            result.context_field = field_info  # Fallback to context
            result.django_field = None
            return result

    def _handle_id_field(self, field_name: str, field_info: FieldInfo) -> Optional[models.Field]:
        """Handle potential ID field naming conflicts (logic moved from original factory)."""
        if field_name.lower() == "id":
            field_type = field_info.annotation
            # Default to AutoField unless explicitly int or str (matching original logic)
            field_class = models.AutoField
            field_kwargs = {"primary_key": True, "verbose_name": "ID"}
            if field_type is int:
                pass  # AutoField is fine
            elif field_type is str:
                # Pydantic usually uses UUID for string IDs, Django uses CharField or UUIDField.
                # Let's default to CharField for simplicity, user can override via metadata if needed.
                field_class = models.CharField
                field_kwargs["max_length"] = 255  # Default length
            elif field_type is UUID:
                field_class = models.UUIDField
            else:
                logger.warning(f"Field 'id' has unusual type {field_type}, using AutoField primary key.")

            # Apply verbose_name from Pydantic if present
            if field_info.title:
                field_kwargs["verbose_name"] = field_info.title

            logger.debug(f"Handling field '{field_name}' as primary key using {field_class.__name__}")
            return field_class(**field_kwargs)
        return None

    def _handle_relationship_field(
        self,
        result: FieldConversionResult,  # Pass result to modify
        field_info: FieldInfo,
        source_model_name: str,
        carrier: ConversionCarrier[type[BaseModel]],
        mapping_definition: TypeMappingDefinition,
    ) -> FieldConversionResult:
        """Handles creation of ForeignKey, ManyToManyField, OneToOneField."""
        field_type = field_info.annotation
        field_name = result.field_name

        # Determine target Pydantic model type
        is_optional = is_pydantic_model_field_optional(field_type)
        actual_type = field_type
        if is_optional:
            # Ensure we handle potential None if get_args returns empty or non-Union
            args = get_args(field_type)
            actual_type = next((arg for arg in args if arg is not type(None)), None)
            if actual_type is None:
                result.error_str = f"Optional field '{field_name}' type could not be resolved (Args: {args})"
                logger.warning(result.error_str)
                result.context_field = field_info
                return result

        origin = get_origin(actual_type)
        args = get_args(actual_type)
        target_pydantic_model: Optional[type[BaseModel]] = None

        # Check if target is a BaseModel subclass
        possible_target = None
        if origin in (list, set) and args:
            possible_target = args[0]
        elif origin is dict and len(args) == 2:
            possible_target = args[1]
        elif origin is None:
            possible_target = actual_type

        if isinstance(possible_target, type) and issubclass(possible_target, BaseModel):
            target_pydantic_model = possible_target
        else:
            result.error_str = f"Could not determine target Pydantic model for relationship field '{field_name}' (type: {field_type}, possible_target: {possible_target})"
            logger.warning(result.error_str)
            result.context_field = field_info  # Treat as context if target unclear
            return result

        # Check if target model is available in relationship manager
        if not self.available_relationships.is_source_model_known(target_pydantic_model):
            result.error_str = f"Target Pydantic model '{target_pydantic_model.__name__}' for field '{field_name}' is not known to RelationshipConversionAccessor. Ensure it was discovered."
            logger.warning(result.error_str)
            result.context_field = field_info  # Treat as context if target not discoverable
            return result

        # Get corresponding Django model
        target_django_model = self.available_relationships.get_django_model_for_pydantic(target_pydantic_model)
        if not target_django_model:
            # This might happen if the Django model hasn't been created yet (dependency order)
            result.error_str = f"Target Django model for '{target_pydantic_model.__name__}' (field '{field_name}') not yet available. Check discovery/generation order."
            logger.warning(result.error_str)
            result.context_field = field_info  # Treat as context
            return result

        # Determine final kwargs for the relationship field
        # Start with attributes determined by TypeMapper (null, blank, default etc based on Optional)
        # Note: We pass field_type (which might be Optional[T]) to get correct null/blank
        rel_kwargs = TypeMapper.get_field_attributes(field_type)

        # --- 'to' argument ---
        target_model_label = f"{target_django_model._meta.app_label}.{target_django_model.__name__}"
        rel_kwargs["to"] = target_model_label

        # --- 'related_name' ---
        # Check Pydantic Field(..., related_name=...)
        user_related_name = (
            field_info.json_schema_extra.get("related_name") if isinstance(field_info.json_schema_extra, dict) else None
        )

        related_name_base = user_related_name if user_related_name else f"{source_model_name.lower()}_{field_name}_set"
        # Ensure related_name is valid Python identifier and explicitly convert base to string
        final_related_name = sanitize_related_name(str(related_name_base), target_django_model.__name__, field_name)

        # Ensure uniqueness within the target model scope using the carrier's tracker
        target_related_names = carrier.used_related_names_per_target.setdefault(target_django_model.__name__, set())
        unique_related_name = final_related_name
        counter = 1
        while unique_related_name in target_related_names:
            unique_related_name = f"{final_related_name}_{counter}"
            counter += 1
        target_related_names.add(unique_related_name)
        rel_kwargs["related_name"] = unique_related_name
        logger.debug(f"[REL] Field '{field_name}': Assigning related_name='{unique_related_name}'")

        # --- 'on_delete' (FK/O2O only) ---
        django_field_class = mapping_definition.django_field
        if django_field_class in (models.ForeignKey, models.OneToOneField):
            # Default based on nullability (matches TypeMapper.instantiate_django_field logic)
            rel_kwargs["on_delete"] = models.SET_NULL if rel_kwargs.get("null", False) else models.CASCADE
        else:
            # M2M fields don't have on_delete
            rel_kwargs.pop("on_delete", None)

        # --- Instantiate ---
        try:
            logger.debug(
                f"Instantiating relationship {django_field_class.__name__} for '{field_name}' with kwargs: {rel_kwargs}"
            )
            result.django_field = django_field_class(**rel_kwargs)
            result.field_kwargs = rel_kwargs  # Store the final kwargs used
        except Exception as e:
            error_msg = f"Failed to instantiate relationship field '{field_name}': {e}"
            logger.error(error_msg, exc_info=True)
            result.error_str = error_msg
            result.context_field = field_info  # Fallback to context
            result.django_field = None
            # Don't return here, let the main create_field handle returning the error result

        return result


class PydanticModelFactory(BaseModelFactory[type[BaseModel], FieldInfo]):  # Use Type[BaseModel]
    """Creates Django models from Pydantic models."""

    # Cache specific to Pydantic models
    _converted_models: dict[str, ConversionCarrier[type[BaseModel]]] = {}

    relationship_accessor: RelationshipConversionAccessor

    def __init__(self, field_factory: PydanticFieldFactory, relationship_accessor: RelationshipConversionAccessor):
        self.relationship_accessor = relationship_accessor
        super().__init__(field_factory=field_factory)

    # Overrides the base method to add caching
    def make_django_model(self, carrier: ConversionCarrier[type[BaseModel]]) -> None:
        """Creates a Django model from Pydantic, checking cache first."""
        model_key = carrier.model_key()
        logger.debug(f"PydanticFactory: Attempting to create Django model for {model_key}")

        # --- Check Cache ---
        if model_key in self._converted_models and not carrier.existing_model:
            logger.debug(f"PydanticFactory: Using cached conversion result for {model_key}")
            cached_carrier = self._converted_models[model_key]
            # Copy results onto the passed-in carrier
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

        # --- Register Relationship Mapping (if successful) ---
        if carrier.source_model and carrier.django_model:
            logger.debug(
                f"PydanticFactory: Registering mapping for {carrier.source_model.__name__} -> {carrier.django_model.__name__}"
            )
            self.relationship_accessor.map_relationship(
                source_model=carrier.source_model, django_model=carrier.django_model
            )

        # --- Cache Result ---
        if carrier.django_model and not carrier.existing_model:
            logger.debug(f"PydanticFactory: Caching conversion result for {model_key}")
            # Store a copy using replace to avoid modification issues
            # self._converted_models[model_key] = replace(carrier) # Linter issue?
            self._converted_models[carrier.model_key()] = carrier  # Direct assign # Call model_key()
        elif not carrier.django_model:
            logger.error(
                f"PydanticFactory: Failed to create Django model for {model_key}. Invalid fields: {carrier.invalid_fields}"
            )

    # --- Implementation of Abstract Methods ---

    def _process_source_fields(self, carrier: ConversionCarrier[type[BaseModel]]):
        """Iterate through Pydantic fields and convert them using the field factory."""
        source_model = carrier.source_model
        model_name = source_model.__name__  # For logging/context

        for field_name_original, field_info in get_model_fields(source_model).items():
            field_name = field_info.alias or field_name_original  # Use alias for Django field name

            if field_name == "id" and carrier.existing_model:
                logger.debug(f"Skipping 'id' field for existing model update: {carrier.existing_model.__name__}")
                continue

            conversion_result = self.field_factory.create_field(
                field_info=field_info, model_name=model_name, carrier=carrier
            )

            if conversion_result.django_field:
                if isinstance(
                    conversion_result.django_field, (models.ForeignKey, models.ManyToManyField, models.OneToOneField)
                ):
                    carrier.relationship_fields[field_name] = conversion_result.django_field
                    # Also store the definition string
                    if conversion_result.field_definition_str:
                        carrier.django_field_definitions[field_name] = conversion_result.field_definition_str
                else:
                    carrier.django_fields[field_name] = conversion_result.django_field
                    # Also store the definition string
                    if conversion_result.field_definition_str:
                        carrier.django_field_definitions[field_name] = conversion_result.field_definition_str
            elif conversion_result.context_field:
                carrier.context_fields[field_name] = conversion_result.context_field
            elif conversion_result.error_str:
                carrier.invalid_fields.append((field_name, conversion_result.error_str))
            else:
                error = f"Field factory returned unexpected result for {model_name}.{field_name_original}: {conversion_result}"
                logger.error(error)
                carrier.invalid_fields.append((field_name, error))

    def _build_pydantic_model_context(self, carrier: ConversionCarrier[type[BaseModel]]):
        """Builds the ModelContext specifically for Pydantic source models."""
        # Renamed to match base class expectation
        self._build_model_context(carrier)

    # Actual implementation of the abstract method
    def _build_model_context(self, carrier: ConversionCarrier[type[BaseModel]]):
        """Builds the ModelContext specifically for Pydantic source models."""
        if not carrier.source_model or not carrier.django_model:
            logger.debug("Skipping context build: missing source or django model.")
            return

        try:
            # Pass source_model to the generic ModelContext
            model_context = ModelContext["BaseModel"](  # Explicitly type hint if needed
                django_model=carrier.django_model,
                source_class=carrier.source_model,
            )
            for field_name, field_info in carrier.context_fields.items():
                if isinstance(field_info, FieldInfo) and field_info.annotation is not None:
                    optional = is_pydantic_model_field_optional(field_info.annotation)
                    field_type_str = repr(field_info.annotation)
                    # Call add_field with correct signature
                    model_context.add_field(field_name=field_name, field_type_str=field_type_str, is_optional=optional)
                elif isinstance(field_info, FieldInfo):
                    logger.warning(f"Context field '{field_name}' has no annotation, cannot add to ModelContext.")
                else:
                    logger.warning(
                        f"Context field '{field_name}' is not a FieldInfo ({type(field_info)}), cannot add to ModelContext."
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


# Helper function (example - might live elsewhere, e.g., in __init__ or a builder class)
def create_pydantic_factory(relationship_accessor: RelationshipConversionAccessor) -> PydanticModelFactory:
    field_factory = PydanticFieldFactory(available_relationships=relationship_accessor)
    model_factory = PydanticModelFactory(field_factory=field_factory, relationship_accessor=relationship_accessor)
    return model_factory


# Import UUID here as it's used in _handle_id_field
from uuid import UUID
