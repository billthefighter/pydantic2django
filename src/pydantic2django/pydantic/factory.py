import logging
from typing import Optional, cast
from uuid import UUID  # Added for _handle_id_field

from django.apps import apps  # Added apps
from django.db import models
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from ..core.bidirectional_mapper import BidirectionalTypeMapper, MappingError  # Import the new mapper
from ..core.context import ModelContext  # Assuming this exists and is correct

# Core imports
from ..core.factories import (
    BaseFieldFactory,
    BaseModelFactory,
    ConversionCarrier,
    FieldConversionResult,
)
from ..core.relationships import RelationshipConversionAccessor

# Remove old TypeMapper import
# from ..django.mapping import TypeMapper, TypeMappingDefinition
from ..django.utils.naming import sanitize_related_name
from ..django.utils.serialization import FieldSerializer, generate_field_definition_string  # Import serialization utils

# Pydantic utils
from .utils.introspection import get_model_fields, is_pydantic_model_field_optional

logger = logging.getLogger(__name__)

# --- Pydantic Specific Factories ---


class PydanticFieldFactory(BaseFieldFactory[FieldInfo]):
    """Creates Django fields from Pydantic fields (FieldInfo)."""

    # Dependencies injected
    relationship_accessor: RelationshipConversionAccessor
    bidirectional_mapper: BidirectionalTypeMapper

    def __init__(
        self, relationship_accessor: RelationshipConversionAccessor, bidirectional_mapper: BidirectionalTypeMapper
    ):
        """Initializes with dependencies."""
        self.relationship_accessor = relationship_accessor
        self.bidirectional_mapper = bidirectional_mapper
        # No super().__init__() needed

    def create_field(
        self, field_info: FieldInfo, model_name: str, carrier: ConversionCarrier[type[BaseModel]]
    ) -> FieldConversionResult:
        """
        Convert a Pydantic FieldInfo to a Django field instance.
        Implements the abstract method from BaseFieldFactory.
        Uses BidirectionalTypeMapper and local instantiation.
        """
        # Use alias first, then the actual key from model_fields as name
        field_name = field_info.alias or next(
            (k for k, v in carrier.source_model.model_fields.items() if v is field_info), "<unknown>"
        )

        # Initialize result with the source field info and determined name
        result = FieldConversionResult(field_info=field_info, field_name=field_name)

        try:
            # Handle potential 'id' field conflict
            if id_field := self._handle_id_field(field_name, field_info):
                result.django_field = id_field
                # Need to capture kwargs for serialization if possible
                # For now, assume default kwargs for ID fields
                # TODO: Extract actual kwargs used in _handle_id_field
                result.field_kwargs = {"primary_key": True}
                if isinstance(id_field, models.CharField):
                    result.field_kwargs["max_length"] = getattr(id_field, "max_length", 255)
                elif isinstance(id_field, models.UUIDField):
                    pass  # No extra kwargs needed typically
                else:  # AutoField
                    pass  # No extra kwargs needed typically

                result.field_definition_str = self._generate_field_def_string(result, carrier.meta_app_label)
                return result  # ID field handled, return early

            # Get field type from annotation
            field_type = field_info.annotation
            if field_type is None:
                logger.warning(f"Field '{model_name}.{field_name}' has no annotation, treating as context field.")
                result.context_field = field_info
                return result

            # --- Use BidirectionalTypeMapper --- #
            try:
                django_field_class, constructor_kwargs = self.bidirectional_mapper.get_django_mapping(
                    python_type=field_type, field_info=field_info
                )
            except MappingError as e:
                # Handle errors specifically from the mapper (e.g., missing relationship)
                logger.error(f"Mapping error for '{model_name}.{field_name}' (type: {field_type}): {e}")
                result.error_str = str(e)
                result.context_field = field_info  # Treat as context on mapping error
                return result
            except Exception as e:
                # Handle unexpected errors during mapping lookup
                logger.error(
                    f"Unexpected error getting Django mapping for '{model_name}.{field_name}': {e}", exc_info=True
                )
                result.error_str = f"Unexpected mapping error: {e}"
                result.context_field = field_info
                return result

            # Store raw kwargs before modifications/checks
            result.raw_mapper_kwargs = constructor_kwargs.copy()

            # --- Check for Multi-FK Union Signal --- #
            union_details = constructor_kwargs.pop("_union_details", None)
            if union_details and isinstance(union_details, dict):
                logger.info(f"Detected multi-FK union signal for '{field_name}'. Deferring field generation.")
                # Store the original field name and the details for the generator
                carrier.pending_multi_fk_unions.append((field_name, union_details))
                # Store remaining kwargs (null, blank for placeholder) in raw_kwargs if needed? Already done.
                # Do not set django_field or field_definition_str
                return result  # Return early, deferring generation

            # --- Handle Relationships Specifically (Adjust Kwargs) --- #
            # Check if it's a relationship type *after* getting mapping AND checking for union signal
            is_relationship = issubclass(
                django_field_class, (models.ForeignKey, models.OneToOneField, models.ManyToManyField)
            )

            if is_relationship:
                # Apply specific relationship logic (like related_name uniqueness)
                # The mapper should have set 'to' and basic 'on_delete'
                if "to" not in constructor_kwargs:
                    # This indicates an issue in the mapper or relationship accessor setup
                    result.error_str = f"Mapper failed to determine 'to' for relationship field '{field_name}'."
                    logger.error(result.error_str)
                    result.context_field = field_info
                    return result

                # Sanitize and ensure unique related_name
                # Check Pydantic Field(..., json_schema_extra={"related_name": ...})
                user_related_name = (
                    field_info.json_schema_extra.get("related_name")
                    if isinstance(field_info.json_schema_extra, dict)
                    else None
                )
                target_django_model_str = constructor_kwargs["to"]  # Mapper returns string like app_label.ModelName

                # Try to get the actual target model class to pass to sanitize_related_name if possible
                # This relies on the target model being importable/available
                target_model_cls = None
                target_model_cls_name_only = target_django_model_str  # Default fallback
                try:
                    app_label, model_cls_name = target_django_model_str.split(".")
                    target_model_cls = apps.get_model(app_label, model_cls_name)  # Use apps.get_model
                    target_model_cls_name_only = model_cls_name  # Use name from split
                except Exception:
                    logger.warning(
                        f"Could not get target model class for '{target_django_model_str}' when generating related_name for '{field_name}'. Using model name string."
                    )
                    # Fallback: try splitting by dot just for name, otherwise use whole string
                    target_model_cls_name_only = target_django_model_str.split(".")[-1]

                related_name_base = (
                    user_related_name
                    if user_related_name
                    else f"{carrier.source_model.__name__.lower()}_{field_name}_set"
                )
                final_related_name_base = sanitize_related_name(
                    str(related_name_base),
                    target_model_cls.__name__ if target_model_cls else target_model_cls_name_only,
                    field_name,
                )

                # Ensure uniqueness using carrier's tracker
                target_model_key_for_tracker = (
                    target_model_cls.__name__ if target_model_cls else target_django_model_str
                )
                target_related_names = carrier.used_related_names_per_target.setdefault(
                    target_model_key_for_tracker, set()
                )
                unique_related_name = final_related_name_base
                counter = 1
                while unique_related_name in target_related_names:
                    unique_related_name = f"{final_related_name_base}_{counter}"
                    counter += 1
                target_related_names.add(unique_related_name)
                constructor_kwargs["related_name"] = unique_related_name
                logger.debug(f"[REL] Field '{field_name}': Assigning related_name='{unique_related_name}'")

                # Re-confirm on_delete (mapper should set default based on Optional)
                if (
                    django_field_class in (models.ForeignKey, models.OneToOneField)
                    and "on_delete" not in constructor_kwargs
                ):
                    is_optional = is_pydantic_model_field_optional(field_type)
                    constructor_kwargs["on_delete"] = models.SET_NULL if is_optional else models.CASCADE
                elif django_field_class == models.ManyToManyField:
                    constructor_kwargs.pop("on_delete", None)
                    # M2M doesn't use null=True, mapper handles this
                    constructor_kwargs.pop("null", None)
                    constructor_kwargs["blank"] = constructor_kwargs.get("blank", True)  # M2M usually blank=True

            # --- Perform Instantiation Locally --- #
            try:
                logger.debug(
                    f"Instantiating {django_field_class.__name__} for '{field_name}' with kwargs: {constructor_kwargs}"
                )
                result.django_field = django_field_class(**constructor_kwargs)
                result.field_kwargs = constructor_kwargs  # Store final kwargs
            except Exception as e:
                error_msg = f"Failed to instantiate Django field '{field_name}' (type: {django_field_class.__name__}) with kwargs {constructor_kwargs}: {e}"
                logger.error(error_msg, exc_info=True)
                result.error_str = error_msg
                result.context_field = field_info  # Fallback to context
                return result

            # --- Generate Field Definition String --- #
            result.field_definition_str = self._generate_field_def_string(result, carrier.meta_app_label)

            return result  # Success

        except Exception as e:
            # Catch-all for unexpected errors during conversion
            error_msg = f"Unexpected error converting field '{model_name}.{field_name}': {e}"
            logger.error(error_msg, exc_info=True)
            result.error_str = error_msg
            result.context_field = field_info  # Fallback to context
            return result

    def _generate_field_def_string(self, result: FieldConversionResult, app_label: str) -> str:
        """Generates the field definition string safely."""
        if not result.django_field:
            return "# Field generation failed"
        try:
            if result.field_kwargs:
                return generate_field_definition_string(type(result.django_field), result.field_kwargs, app_label)
            else:
                logger.warning(
                    f"Could not generate definition string for '{result.field_name}': final kwargs not found in result. Using basic serialization."
                )
                return FieldSerializer.serialize_field(result.django_field)
        except Exception as e:
            logger.error(
                f"Failed to generate field definition string for '{result.field_name}': {e}",
                exc_info=True,
            )
            return f"# Error generating definition: {e}"

    def _handle_id_field(self, field_name: str, field_info: FieldInfo) -> Optional[models.Field]:
        """Handle potential ID field naming conflicts (logic moved from original factory)."""
        if field_name.lower() == "id":
            field_type = field_info.annotation
            # Default to AutoField unless explicitly specified by type
            field_class = models.AutoField
            field_kwargs = {"primary_key": True, "verbose_name": "ID"}

            # Use mapper to find appropriate Django PK field if type is specified
            # But only override AutoField if it's clearly not a standard int sequence
            pk_field_class_override = None
            if field_type is UUID:
                pk_field_class_override = models.UUIDField
                field_kwargs.pop("verbose_name")  # UUIDField doesn't need verbose_name='ID'
            elif field_type is str:
                # Default Pydantic str ID to CharField PK
                pk_field_class_override = models.CharField
                field_kwargs["max_length"] = 255  # Default length
            elif field_type is int:
                pass  # Default AutoField is fine
            elif field_type:
                # Check if mapper finds a specific non-auto int field (e.g., BigIntegerField)
                try:
                    mapped_cls, mapped_kwargs = self.bidirectional_mapper.get_django_mapping(field_type, field_info)
                    if issubclass(mapped_cls, models.IntegerField) and not issubclass(mapped_cls, models.AutoField):
                        pk_field_class_override = mapped_cls
                        field_kwargs.update(mapped_kwargs)
                        # Ensure primary_key=True is set
                        field_kwargs["primary_key"] = True
                    elif not issubclass(mapped_cls, models.AutoField):
                        logger.warning(
                            f"Field 'id' has type {field_type} mapping to non-integer {mapped_cls.__name__}. Using AutoField PK."
                        )
                except MappingError:
                    logger.warning(f"Field 'id' has unmappable type {field_type}. Using AutoField PK.")

            if pk_field_class_override:
                field_class = pk_field_class_override
            else:
                # Stick with AutoField, apply title if present
                if field_info.title:
                    field_kwargs["verbose_name"] = field_info.title

            logger.debug(f"Handling field '{field_name}' as primary key using {field_class.__name__}")
            # Instantiate the ID field
            try:
                return field_class(**field_kwargs)
            except Exception as e:
                logger.error(
                    f"Failed to instantiate ID field {field_class.__name__} with kwargs {field_kwargs}: {e}",
                    exc_info=True,
                )
                # Fallback to basic AutoField? Or let error propagate?
                # Let's return None and let the main create_field handle error reporting
                return None
        return None

    # --- Removed _handle_relationship_field --- #
    # Logic is now integrated into create_field using the mapper


class PydanticModelFactory(BaseModelFactory[type[BaseModel], FieldInfo]):
    """Creates Django models from Pydantic models."""

    # Cache specific to Pydantic models
    _converted_models: dict[str, ConversionCarrier[type[BaseModel]]] = {}

    relationship_accessor: RelationshipConversionAccessor
    # No need for field_factory instance here if Base class handles it

    def __init__(self, field_factory: PydanticFieldFactory, relationship_accessor: RelationshipConversionAccessor):
        """Initialize with field factory and relationship accessor."""
        self.relationship_accessor = relationship_accessor
        # Pass the field_factory up to the base class
        super().__init__(field_factory=field_factory)

    # Overrides the base method to add caching and relationship mapping
    def make_django_model(self, carrier: ConversionCarrier[type[BaseModel]]) -> None:
        """Creates a Django model from Pydantic, checking cache first and mapping relationships."""
        model_key = carrier.model_key()
        logger.debug(f"PydanticFactory: Attempting to create Django model for {model_key}")

        # --- Check Cache --- #
        if model_key in self._converted_models and not carrier.existing_model:
            logger.debug(f"PydanticFactory: Using cached conversion result for {model_key}")
            cached_carrier = self._converted_models[model_key]
            # Update the passed-in carrier with cached results
            carrier.__dict__.update(cached_carrier.__dict__)
            # Ensure used_related_names is properly updated (dict update might not merge sets correctly)
            for target, names in cached_carrier.used_related_names_per_target.items():
                carrier.used_related_names_per_target.setdefault(target, set()).update(names)
            return

        # --- Call Base Implementation for Core Logic --- #
        # This calls _process_source_fields, _assemble_django_model_class etc.
        super().make_django_model(carrier)

        # --- Register Relationship Mapping (if successful) --- #
        if carrier.source_model and carrier.django_model:
            logger.debug(
                f"PydanticFactory: Registering mapping for {carrier.source_model.__name__} -> {carrier.django_model.__name__}"
            )
            self.relationship_accessor.map_relationship(
                source_model=carrier.source_model, django_model=carrier.django_model
            )

        # --- Cache Result --- #
        if carrier.django_model and not carrier.existing_model:
            logger.debug(f"PydanticFactory: Caching conversion result for {model_key}")
            # Store a copy to prevent modification issues? Simple assignment for now.
            self._converted_models[model_key] = carrier
        elif not carrier.django_model:
            logger.error(
                f"PydanticFactory: Failed to create Django model for {model_key}. Invalid fields: {carrier.invalid_fields}"
            )

    # --- Implementation of Abstract Methods --- #

    def _process_source_fields(self, carrier: ConversionCarrier[type[BaseModel]]):
        """Iterate through Pydantic fields and convert them using the field factory."""
        source_model = carrier.source_model
        model_name = source_model.__name__

        for field_name_original, field_info in get_model_fields(source_model).items():
            field_name = field_info.alias or field_name_original

            # Skip 'id' field if updating an existing model definition
            # Note: _handle_id_field in field factory handles primary key logic
            if field_name.lower() == "id" and carrier.existing_model:
                logger.debug(f"Skipping 'id' field for existing model update: {carrier.existing_model.__name__}")
                continue

            # Cast needed because BaseFactory uses generic TFieldInfo
            field_factory_typed = cast(PydanticFieldFactory, self.field_factory)
            conversion_result = field_factory_typed.create_field(
                field_info=field_info, model_name=model_name, carrier=carrier
            )

            # Store results in the carrier
            if conversion_result.django_field:
                # Store definition string first
                if conversion_result.field_definition_str:
                    carrier.django_field_definitions[field_name] = conversion_result.field_definition_str
                else:
                    logger.warning(f"Missing field definition string for successfully created field '{field_name}'")

                # Store the field instance itself
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
                # Should not happen if FieldConversionResult is used correctly
                error = f"Field factory returned unexpected empty result for {model_name}.{field_name_original}"
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
            model_context = ModelContext(  # Removed generic type hint for base class compatibility
                django_model=carrier.django_model,
                source_class=carrier.source_model,
            )
            for field_name, field_info in carrier.context_fields.items():
                if isinstance(field_info, FieldInfo) and field_info.annotation is not None:
                    optional = is_pydantic_model_field_optional(field_info.annotation)
                    # Use repr() for field_type_str as expected by ModelContext.add_field
                    field_type_str = repr(field_info.annotation)
                    model_context.add_field(
                        field_name=field_name,
                        field_type_str=field_type_str,  # Pass string representation
                        is_optional=optional,
                        annotation=field_info.annotation,  # Keep annotation if needed elsewhere
                    )
                elif isinstance(field_info, FieldInfo):
                    logger.warning(f"Context field '{field_name}' has no annotation, cannot add to ModelContext.")
                else:
                    logger.warning(
                        f"Context field '{field_name}' is not a FieldInfo ({type(field_info)}), cannot add to ModelContext."
                    )
            carrier.model_context = model_context
            logger.debug(f"Successfully built ModelContext for {carrier.model_key()}")  # Call model_key()
        except Exception as e:
            logger.error(f"Failed to build ModelContext for {carrier.model_key()}: {e}", exc_info=True)
            carrier.model_context = None


# Helper function (example - might live elsewhere, e.g., in __init__ or a builder class)
def create_pydantic_factory(
    relationship_accessor: RelationshipConversionAccessor, bidirectional_mapper: BidirectionalTypeMapper
) -> PydanticModelFactory:
    """Helper to create the Pydantic factory stack with dependencies."""
    field_factory = PydanticFieldFactory(
        relationship_accessor=relationship_accessor, bidirectional_mapper=bidirectional_mapper
    )
    model_factory = PydanticModelFactory(field_factory=field_factory, relationship_accessor=relationship_accessor)
    return model_factory
