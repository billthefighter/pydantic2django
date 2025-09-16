import dataclasses
import inspect  # Import inspect for introspection
import logging
import sys
from typing import Optional, TypeVar, Union, get_args, get_origin, get_type_hints

from django.apps import apps  # Added apps
from django.db import models

from ..core.bidirectional_mapper import BidirectionalTypeMapper, MappingError  # Import the new mapper
from ..core.context import ModelContext  # Assuming this exists and is correct

# Core imports
from ..core.factories import (
    BaseFieldFactory,
    BaseModelFactory,
    ConversionCarrier,
    FieldConversionResult,
)
from ..core.imports import ImportHandler  # Import handler
from ..core.relationships import RelationshipConversionAccessor  # Assuming this exists

# Remove old TypeMapper import
# from ..django.mapping import TypeMapper, TypeMappingDefinition
from ..django.utils.naming import sanitize_related_name  # Import naming utils
from ..django.utils.serialization import (
    FieldSerializer,
    RawCode,
    generate_field_definition_string,
)

# Dataclass utils? (If needed, TBD)
# from .utils import ...

# Define DataclassType alias if not globally defined (e.g., in core.defs)
# For now, define locally:
DataclassType = TypeVar("DataclassType")  # Note: Pydantic factory uses Type[BaseModel]

logger = logging.getLogger(__name__)

# --- Dataclass Specific Factories ---


class DataclassFieldFactory(BaseFieldFactory[dataclasses.Field]):
    """Creates Django model fields from dataclass fields."""

    relationship_accessor: RelationshipConversionAccessor  # Changed Optional to required
    bidirectional_mapper: BidirectionalTypeMapper  # Added mapper

    def __init__(
        self, relationship_accessor: RelationshipConversionAccessor, bidirectional_mapper: BidirectionalTypeMapper
    ):
        """Initializes with dependencies."""
        self.relationship_accessor = relationship_accessor
        self.bidirectional_mapper = bidirectional_mapper
        # No super().__init__() needed if BaseFieldFactory.__init__ is empty or handles this

    def create_field(
        self, field_info: dataclasses.Field, model_name: str, carrier: ConversionCarrier[DataclassType]
    ) -> FieldConversionResult:
        """
        Convert a dataclasses.Field to a Django field instance.
        Uses BidirectionalTypeMapper and local instantiation.
        Relies on field_info.metadata['django'] for specific overrides.
        Adds required imports to the result.
        """
        field_name = field_info.name
        original_field_type = field_info.type
        metadata = field_info.metadata or {}  # Ensure metadata is a dict
        django_meta_options = metadata.get("django", {})

        # --- Resolve Forward Reference String if necessary --- #
        type_to_map = original_field_type
        result = FieldConversionResult(field_info=field_info, field_name=field_name)

        if isinstance(original_field_type, str):
            logger.debug(
                f"Field '{field_name}' has string type '{original_field_type}'. Attempting resolution via RelationshipAccessor."
            )
            # Assume string type is a model name known to the accessor
            # Use the newly added method:
            resolved_source_model = self.relationship_accessor.get_source_model_by_name(original_field_type)
            if resolved_source_model:
                logger.debug(f"Resolved string '{original_field_type}' to type {resolved_source_model}")
                type_to_map = resolved_source_model
            else:
                # Critical Error: If it's a string but not in accessor, mapping will fail.
                logger.error(
                    f"Field '{field_name}' type is string '{original_field_type}' but was not found in RelationshipAccessor. Cannot map."
                )
                result.error_str = f"Unresolved forward reference or unknown model name: {original_field_type}"
                result.context_field = field_info
                return result
        # --- End Forward Reference Resolution ---

        logger.debug(
            f"Processing dataclass field {model_name}.{field_name}: Type={original_field_type}, Metadata={metadata}"
        )

        try:
            # --- Use BidirectionalTypeMapper --- #
            try:
                # Pass field_info.type, but no Pydantic FieldInfo equivalent for metadata
                # The mapper primarily relies on the type itself.
                django_field_class, constructor_kwargs = self.bidirectional_mapper.get_django_mapping(
                    python_type=type_to_map,
                    field_info=None,  # Pass None for field_info
                )
                # Add import for the Django field class itself using the result's helper
                result.add_import_for_obj(django_field_class)

            except MappingError as e:
                logger.error(f"Mapping error for '{model_name}.{field_name}' (type: {type_to_map}): {e}")
                result.error_str = str(e)
                result.context_field = field_info
                return result
            except Exception as e:
                logger.error(
                    f"Unexpected error getting Django mapping for '{model_name}.{field_name}': {e}", exc_info=True
                )
                result.error_str = f"Unexpected mapping error: {e}"
                result.context_field = field_info
                return result

            # --- Merge Dataclass Metadata Overrides --- #
            # Apply explicit options from metadata *after* getting defaults from mapper
            constructor_kwargs.update(django_meta_options)

            # --- Apply Dataclass Defaults --- #
            # This logic now handles both `default` and `default_factory`.
            if "default" not in constructor_kwargs:
                default_value = field_info.default
                if default_value is dataclasses.MISSING and field_info.default_factory is not dataclasses.MISSING:
                    default_value = field_info.default_factory

                if default_value is not dataclasses.MISSING:
                    if callable(default_value):
                        try:
                            # Handle regular importable functions
                            if (
                                hasattr(default_value, "__module__")
                                and hasattr(default_value, "__name__")
                                and not default_value.__name__ == "<lambda>"
                            ):
                                module_name = default_value.__module__
                                func_name = default_value.__name__
                                # Avoid importing builtins
                                if module_name != "builtins":
                                    result.add_import(module_name, func_name)
                                constructor_kwargs["default"] = func_name
                            # Handle lambdas
                            else:
                                source = inspect.getsource(default_value).strip()
                                # Remove comma if it's trailing in a lambda definition in a list/dict
                                if source.endswith(","):
                                    source = source[:-1]
                                constructor_kwargs["default"] = RawCode(source)
                        except (TypeError, OSError) as e:
                            logger.warning(
                                f"Could not introspect callable default for '{model_name}.{field_name}': {e}. "
                                "Falling back to `None`."
                            )
                            constructor_kwargs["default"] = None
                            constructor_kwargs["null"] = True
                            constructor_kwargs["blank"] = True
                    elif not isinstance(default_value, (list, dict, set)):
                        constructor_kwargs["default"] = default_value
                    else:
                        logger.warning(
                            f"Field {model_name}.{field_name} has mutable default {default_value}. Skipping Django default."
                        )

            # --- Handle Relationships Specifically (Adjust Kwargs) --- #
            is_relationship = issubclass(
                django_field_class, (models.ForeignKey, models.OneToOneField, models.ManyToManyField)
            )

            if is_relationship:
                if "to" not in constructor_kwargs:
                    result.error_str = f"Mapper failed to determine 'to' for relationship field '{field_name}'."
                    logger.error(result.error_str)
                    result.context_field = field_info
                    return result

                # Sanitize and ensure unique related_name
                user_related_name = django_meta_options.get("related_name")  # Check override from metadata
                target_django_model_str = constructor_kwargs["to"]

                target_model_cls = None
                target_model_cls_name_only = target_django_model_str
                try:
                    app_label, model_cls_name = target_django_model_str.split(".")
                    target_model_cls = apps.get_model(app_label, model_cls_name)
                    target_model_cls_name_only = model_cls_name
                    # Add import for the target model using result helper
                    result.add_import_for_obj(target_model_cls)
                except Exception:
                    logger.warning(
                        f"Could not get target model class for '{target_django_model_str}' when generating related_name for '{field_name}'. Using model name string."
                    )
                    target_model_cls_name_only = target_django_model_str.split(".")[-1]

                related_name_base = (
                    user_related_name
                    if user_related_name
                    # Use carrier.source_model.__name__ for default related name base
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
                logger.debug(f"[REL] Dataclass Field '{field_name}': Assigning related_name='{unique_related_name}'")

                # Re-confirm on_delete (mapper sets default based on Optional, but metadata might override)
                # Need to check optionality of the original type here
                origin = get_origin(original_field_type)
                args = get_args(original_field_type)
                is_optional = origin is Union and type(None) in args

                if (
                    django_field_class in (models.ForeignKey, models.OneToOneField)
                    and "on_delete" not in constructor_kwargs  # Only set if not specified in metadata
                ):
                    constructor_kwargs["on_delete"] = models.SET_NULL if is_optional else models.CASCADE
                    # Add import using result helper
                    result.add_import("django.db.models", "SET_NULL" if is_optional else "CASCADE")
                elif django_field_class == models.ManyToManyField:
                    constructor_kwargs.pop("on_delete", None)
                    constructor_kwargs.pop("null", None)  # M2M cannot be null
                    if "blank" not in constructor_kwargs:  # Default M2M to blank=True if not set
                        constructor_kwargs["blank"] = True

            # --- Perform Instantiation Locally --- #
            try:
                logger.debug(
                    f"Instantiating {django_field_class.__name__} for dataclass field '{field_name}' with kwargs: {constructor_kwargs}"
                )
                result.django_field = django_field_class(**constructor_kwargs)
                result.field_kwargs = constructor_kwargs  # Store final kwargs
            except Exception as e:
                error_msg = f"Failed to instantiate Django field '{field_name}' (type: {django_field_class.__name__}) with kwargs {constructor_kwargs}: {e}"
                logger.error(error_msg, exc_info=True)
                result.error_str = error_msg
                result.context_field = field_info
                return result

            # --- Generate Field Definition String --- #
            result.field_definition_str = self._generate_field_def_string(result, carrier.meta_app_label)

            return result  # Success

        except Exception as e:
            # Catch-all for unexpected errors during conversion
            error_msg = f"Unexpected error converting dataclass field '{model_name}.{field_name}': {e}"
            logger.error(error_msg, exc_info=True)
            result.error_str = error_msg
            result.context_field = field_info
            return result

    def _generate_field_def_string(self, result: FieldConversionResult, app_label: str) -> str:
        """Generates the field definition string safely."""
        if not result.django_field:
            return "# Field generation failed"
        try:
            # Use stored final kwargs if available
            if result.field_kwargs:
                # Pass the result's required_imports to the serialization function
                return generate_field_definition_string(
                    type(result.django_field),
                    result.field_kwargs,
                    app_label,
                )
            else:
                # Fallback: Basic serialization if final kwargs weren't stored for some reason
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

    # --- Removed _handle_relationship_kwargs (logic merged into create_field) --- #


class DataclassModelFactory(BaseModelFactory[DataclassType, dataclasses.Field]):
    """Dynamically creates Django model classes from dataclasses."""

    # Cache specific to Dataclass models
    _converted_models: dict[str, ConversionCarrier[DataclassType]] = {}

    relationship_accessor: RelationshipConversionAccessor  # Changed Optional to required
    import_handler: ImportHandler  # Added import handler

    def __init__(
        self,
        field_factory: DataclassFieldFactory,
        relationship_accessor: RelationshipConversionAccessor,  # Now required
        import_handler: Optional[ImportHandler] = None,  # Accept optionally
    ):
        """Initialize with field factory, relationship accessor, and import handler."""
        self.relationship_accessor = relationship_accessor
        self.import_handler = import_handler or ImportHandler()
        # Call super init
        super().__init__(field_factory)

    def make_django_model(self, carrier: ConversionCarrier[DataclassType]) -> None:
        """
        Orchestrates the Django model creation process.
        Subclasses implement _process_source_fields and _build_model_context.
        Handles caching.
        Passes import handler down.
        """
        # --- Pass import handler via carrier --- (or could add to factory state)
        # Need to set import handler on carrier if passed during init
        # NOTE: BaseModelFactory.make_django_model does this now.
        # carrier.import_handler = self.import_handler
        super().make_django_model(carrier)
        # Register relationship after successful model creation (moved from original)
        if carrier.source_model and carrier.django_model:
            logger.debug(
                f"Mapping relationship in accessor: {carrier.source_model.__name__} -> {carrier.django_model.__name__}"
            )
            self.relationship_accessor.map_relationship(
                source_model=carrier.source_model, django_model=carrier.django_model
            )
        # Cache result (moved from original)
        model_key = carrier.model_key()
        if carrier.django_model and not carrier.existing_model:
            self._converted_models[model_key] = carrier

    def _process_source_fields(self, carrier: ConversionCarrier[DataclassType]):
        """Iterate through source dataclass fields, resolve types, create Django fields, and store results."""
        source_model = carrier.source_model
        if not source_model:
            logger.error(
                f"Cannot process fields: source model missing in carrier for {getattr(carrier, 'target_model_name', '?')}"
            )  # Safely access target_model_name
            carrier.invalid_fields.append(("_source_model", "Source model missing."))  # Use invalid_fields
            return

        # --- Add check: Ensure source_model is a type ---
        if not isinstance(source_model, type):
            error_msg = f"Cannot process fields: expected source_model to be a type, but got {type(source_model)} ({source_model!r}). Problem likely upstream in model discovery/ordering."
            logger.error(error_msg)
            carrier.invalid_fields.append(("_source_model", error_msg))
            return
        # --- End Add check ---

        # --- Use dataclasses.fields for introspection ---
        try:
            # Resolve type hints first to handle forward references (strings)
            # Need globals and potentially locals from the source model's module
            source_module = sys.modules.get(source_model.__module__)
            globalns = getattr(source_module, "__dict__", None)
            # Revert: Use only globals, assuming types are resolvable in module scope

            # resolved_types = get_type_hints(source_model, globalns=globalns, localns=localns)
            # logger.debug(f"Resolved types for {source_model.__name__} using {globalns=}, {localns=}: {resolved_types}")
            # Use updated call without potentially incorrect locals:
            resolved_types = get_type_hints(source_model, globalns=globalns, localns=None)
            logger.debug(f"Resolved types for {source_model.__name__} using module globals: {resolved_types}")

            dataclass_fields = dataclasses.fields(source_model)
        except (TypeError, NameError) as e:  # Catch errors during type hint resolution or fields() call
            error_msg = f"Could not introspect fields or resolve types for {source_model.__name__}: {e}"
            logger.error(error_msg, exc_info=True)
            carrier.invalid_fields.append(("_introspection", error_msg))  # Use invalid_fields
            return

        # Use field definitions directly from carrier
        # field_definitions: dict[str, str] = {}
        # context_field_definitions: dict[str, str] = {} # Dataclasses likely won't use this

        for field_info in dataclass_fields:
            field_name = field_info.name

            # Normalize output Django field identifier
            try:
                from ..core.utils.naming import sanitize_field_identifier

                normalized_field_name = sanitize_field_identifier(field_name)
            except Exception:
                normalized_field_name = field_name

            # Get the *resolved* type for this field
            resolved_type = resolved_types.get(field_name)
            if resolved_type is None:
                logger.warning(
                    f"Could not resolve type hint for field '{field_name}' in {source_model.__name__}. Using original: {field_info.type!r}"
                )
                # Fallback to original, which might be a string
                resolved_type = field_info.type

            # --- Prepare field type for create_field --- #
            type_for_create_field = resolved_type  # Start with the result from get_type_hints

            # If the original type annotation was a string (ForwardRef)
            # and get_type_hints failed to resolve it (resolved_type is None or still the string),
            # try to find the resolved type from the dict populated earlier.
            # This specifically handles nested dataclasses defined in local scopes like fixtures.
            if isinstance(field_info.type, str):
                explicitly_resolved = resolved_types.get(field_name)
                if explicitly_resolved and not isinstance(explicitly_resolved, str):
                    logger.debug(
                        f"Using explicitly resolved type {explicitly_resolved!r} for forward ref '{field_info.type}'"
                    )
                    type_for_create_field = explicitly_resolved
                elif resolved_type is field_info.type:  # Check if resolved_type is still the unresolved string
                    logger.error(
                        f"Type hint for '{field_name}' is string '{field_info.type}' but was not resolved by get_type_hints. Skipping field."
                    )
                    carrier.invalid_fields.append(
                        (field_name, f"Could not resolve forward reference: {field_info.type}")
                    )
                    continue  # Skip this field

            # Temporarily modify a copy of field_info or pass type directly if possible.
            # Modifying field_info directly is simpler for now.
            original_type_attr = field_info.type
            try:
                field_info.type = type_for_create_field  # Use the determined type
                logger.debug(f"Calling create_field for '{field_name}' with type: {field_info.type!r}")

                field_result = self.field_factory.create_field(
                    field_info=field_info, model_name=source_model.__name__, carrier=carrier
                )
            finally:
                # Restore original type attribute
                field_info.type = original_type_attr
                logger.debug("Restored original field_info.type attribute")

            # Process the result (errors, definitions)
            if field_result.error_str:
                carrier.invalid_fields.append((normalized_field_name, field_result.error_str))
            else:
                # Store the definition string if available
                if field_result.field_definition_str:
                    carrier.django_field_definitions[normalized_field_name] = field_result.field_definition_str
                else:
                    logger.warning(
                        f"Field '{normalized_field_name}' processing yielded no error and no definition string."
                    )

                # Store the actual field instance in the correct carrier dict
                if field_result.django_field:
                    if isinstance(
                        field_result.django_field, (models.ForeignKey, models.OneToOneField, models.ManyToManyField)
                    ):
                        carrier.relationship_fields[normalized_field_name] = field_result.django_field
                    else:
                        carrier.django_fields[normalized_field_name] = field_result.django_field
                elif field_result.context_field:
                    # Handle context fields if needed (currently seems unused based on logs)
                    carrier.context_fields[normalized_field_name] = field_result.context_field

                # Merge imports from result into the factory's import handler
                if field_result.required_imports:
                    # Use the new add_import method
                    for module, names in field_result.required_imports.items():
                        for name in names:
                            self.import_handler.add_import(module=module, name=name)

        logger.debug(f"Finished processing fields for {source_model.__name__}. Errors: {len(carrier.invalid_fields)}")

    # Actual implementation of the abstract method _build_model_context
    def _build_model_context(self, carrier: ConversionCarrier[DataclassType]):
        """Builds the ModelContext specifically for dataclass source models."""
        if not carrier.source_model or not carrier.django_model:
            logger.debug("Skipping context build: missing source or django model.")
            return

        try:
            # Remove generic type hint if ModelContext is not generic or if causing issues
            # Assuming ModelContext base class handles the source type appropriately
            model_context = ModelContext(django_model=carrier.django_model, source_class=carrier.source_model)
            for field_name, field_info in carrier.context_fields.items():
                if isinstance(field_info, dataclasses.Field):
                    # Calculate necessary info for ModelContext.add_field
                    origin = get_origin(field_info.type)
                    args = get_args(field_info.type)
                    is_optional = origin is Union and type(None) in args
                    field_type_str = repr(field_info.type)  # Use repr for the type string

                    # Call add_field with expected signature
                    model_context.add_field(
                        field_name=field_name,
                        field_type_str=field_type_str,
                        is_optional=is_optional,
                        # Pass original annotation if ModelContext uses it
                        annotation=field_info.type,
                    )
                else:
                    # Log if context field is not the expected type
                    logger.warning(
                        f"Context field '{field_name}' is not a dataclasses.Field ({type(field_info)}), cannot add to ModelContext."
                    )
            carrier.model_context = model_context
            logger.debug(f"Successfully built ModelContext for {carrier.model_key()}")  # Use method call
        except Exception as e:
            logger.error(f"Failed to build ModelContext for {carrier.model_key()}: {e}", exc_info=True)
            carrier.model_context = None

    # --- Methods Inherited from Base Class (No need to redefine) ---
    # _handle_field_collisions
    # _create_django_meta
    # _assemble_django_model_class


# Helper function (similar to pydantic factory)
def create_dataclass_factory(
    relationship_accessor: RelationshipConversionAccessor, bidirectional_mapper: BidirectionalTypeMapper
) -> DataclassModelFactory:
    """Helper function to create a DataclassModelFactory with dependencies."""
    field_factory = DataclassFieldFactory(relationship_accessor, bidirectional_mapper)
    # Create ImportHandler instance here for the factory
    import_handler = ImportHandler()
    return DataclassModelFactory(
        field_factory=field_factory,
        relationship_accessor=relationship_accessor,
        import_handler=import_handler,
    )
