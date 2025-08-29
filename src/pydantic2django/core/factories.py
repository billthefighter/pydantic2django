import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, Optional, TypeVar, cast

from django.db import models

# Use absolute imports from within the package
from pydantic2django.core.context import ModelContext  # Assuming ModelContext is here

# Correct import path for ImportHandler:

logger = logging.getLogger(__name__)

# --- Generic Type Variables ---
SourceModelType = TypeVar("SourceModelType")  # e.g., Type[BaseModel] or Type[DataclassType]
SourceFieldType = TypeVar("SourceFieldType")  # e.g., FieldInfo or dataclasses.Field
# DjangoModelType = TypeVar("DjangoModelType", bound=models.Model) # Removed, not needed directly in carrier

# --- Data Structures ---


@dataclass
class ConversionCarrier(Generic[SourceModelType]):
    """
    Carrier class for converting a source model (Pydantic/Dataclass) to a Django model.
    Holds configuration and accumulates results during the conversion process.
    Generalized from the original DjangoModelFactoryCarrier.
    """

    source_model: type[SourceModelType]
    meta_app_label: str
    base_django_model: type[models.Model]  # Base Django model to inherit from
    existing_model: Optional[type[models.Model]] = None  # For updating existing models
    class_name_prefix: str = "Django"  # Prefix for generated Django model name
    strict: bool = False  # Strict mode for field collisions
    used_related_names_per_target: dict[str, set[str]] = field(default_factory=dict)
    django_field_definitions: dict[str, str] = field(default_factory=dict)  # Added field defs

    # --- Result fields (populated during conversion) ---
    django_fields: dict[str, models.Field] = field(default_factory=dict)
    relationship_fields: dict[str, models.Field] = field(default_factory=dict)
    context_fields: dict[str, Any] = field(default_factory=dict)  # Store original source field info
    context_data: dict[str, Any] = field(default_factory=dict)
    # Stores (original_field_name, union_details_dict) for multi-FK unions
    pending_multi_fk_unions: list[tuple[str, dict]] = field(default_factory=list)
    invalid_fields: list[tuple[str, str]] = field(default_factory=list)
    django_meta_class: Optional[type] = None
    django_model: Optional[type[models.Model]] = None  # Changed from DjangoModelType
    model_context: Optional[ModelContext] = None
    # Removed import_handler from carrier

    def model_key(self) -> str:
        """Generate a unique key for the source model."""
        module = getattr(self.source_model, "__module__", "?")
        name = getattr(self.source_model, "__name__", "UnknownModel")
        return f"{module}.{name}"

    def __str__(self):
        source_name = getattr(self.source_model, "__name__", "UnknownSource")
        django_name = getattr(self.django_model, "__name__", "None") if self.django_model else "None"
        return f"{source_name} -> {django_name}"


@dataclass
class FieldConversionResult:
    """Data structure holding the result of attempting to convert a single source field."""

    field_info: Any  # Original source field info (FieldInfo, dataclasses.Field)
    field_name: str
    # type_mapping_definition: Optional[TypeMappingDefinition] = None # Keep mapping info internal?
    field_kwargs: dict[str, Any] = field(default_factory=dict)
    django_field: Optional[models.Field] = None
    context_field: Optional[Any] = None  # Holds original field_info if handled by context
    error_str: Optional[str] = None
    field_definition_str: Optional[str] = None  # Added field definition string
    # Added required_imports dictionary
    required_imports: dict[str, list[str]] = field(default_factory=dict)
    # Store the raw kwargs returned by the mapper
    raw_mapper_kwargs: dict[str, Any] = field(default_factory=dict)

    def add_import(self, module: str, name: str):
        """Helper to add an import to this result."""
        if not module or module == "builtins":
            return
        current_names = self.required_imports.setdefault(module, [])
        if name not in current_names:
            current_names.append(name)

    def add_import_for_obj(self, obj: Any):
        """Helper to add an import for a given object (class, function, etc.)."""
        if hasattr(obj, "__module__") and hasattr(obj, "__name__"):
            module = obj.__module__
            name = obj.__name__
            self.add_import(module, name)
        else:
            logger.warning(f"Could not determine import for object: {obj!r}")

    def __str__(self):
        status = "Success" if self.django_field else ("Context" if self.context_field else f"Error: {self.error_str}")
        field_type = type(self.django_field).__name__ if self.django_field else "N/A"
        return f"FieldConversionResult(field={self.field_name}, status={status}, django_type={field_type})"


# --- Abstract Base Classes ---


class BaseFieldFactory(ABC, Generic[SourceFieldType]):
    """Abstract base class for field factories."""

    def __init__(self, *args, **kwargs):
        # Allow subclasses to accept necessary dependencies (e.g., relationship accessors)
        pass

    @abstractmethod
    def create_field(
        self, field_info: SourceFieldType, model_name: str, carrier: ConversionCarrier
    ) -> FieldConversionResult:
        """
        Convert a source field type into a Django Field.

        Args:
            field_info: The field information object from the source (Pydantic/Dataclass).
            model_name: The name of the source model containing the field.
            carrier: The conversion carrier for context (e.g., app_label, relationships).

        Returns:
            A FieldConversionResult containing the generated Django field or context/error info.
        """
        pass


class BaseModelFactory(ABC, Generic[SourceModelType, SourceFieldType]):
    """Abstract base class for model factories."""

    field_factory: BaseFieldFactory[SourceFieldType]

    def __init__(self, field_factory: BaseFieldFactory[SourceFieldType], *args, **kwargs):
        """
        Initializes the model factory with a compatible field factory.
        Allows subclasses to accept additional dependencies.
        """
        self.field_factory = field_factory

    @abstractmethod
    def _process_source_fields(self, carrier: ConversionCarrier[SourceModelType]):
        """Abstract method for subclasses to implement field processing."""
        pass

    # Common logic moved from subclasses
    def _handle_field_collisions(self, carrier: ConversionCarrier[SourceModelType]):
        """Check for field name collisions with the base Django model."""
        base_model = carrier.base_django_model
        if not base_model or not hasattr(base_model, "_meta"):
            return

        try:
            base_fields = base_model._meta.get_fields(include_parents=True, include_hidden=False)
            base_field_names = {f.name for f in base_fields if not f.name.endswith("+")}
        except Exception as e:
            logger.warning(f"Could not get fields from base model {base_model.__name__} for collision check: {e}")
            return

        all_new_fields = set(carrier.django_fields.keys()) | set(carrier.relationship_fields.keys())
        collision_fields = all_new_fields & base_field_names

        if collision_fields:
            source_name = getattr(carrier.source_model, "__name__", "?")
            msg = f"Field collision detected between {source_name} and base model {base_model.__name__}: {collision_fields}."
            if carrier.strict:
                logger.error(msg + " Raising error due to strict=True.")
                raise ValueError(msg + " Use strict=False or rename fields.")
            else:
                logger.warning(msg + " Removing colliding fields from generated model (strict=False).")
                for field_name in collision_fields:
                    carrier.django_fields.pop(field_name, None)
                    carrier.relationship_fields.pop(field_name, None)

    def _create_django_meta(self, carrier: ConversionCarrier[SourceModelType]):
        """Create the Meta class for the generated Django model."""
        source_name = getattr(carrier.source_model, "__name__", "UnknownSourceModel")
        source_model_name_cleaned = source_name.replace("_", " ")
        meta_attrs = {
            "app_label": carrier.meta_app_label,
            "db_table": f"{carrier.meta_app_label}_{source_name.lower()}",
            # Keep dynamic models abstract so Django does not register them in the
            # global app registry. This avoids conflicts with concrete generated models
            # imported later (e.g., during makemigrations in tests).
            "abstract": True,
            "managed": True,
            "verbose_name": source_model_name_cleaned,
            "verbose_name_plural": source_model_name_cleaned + "s",
            "ordering": ["pk"],
        }

        # Create Meta, not inheriting from base Meta to ensure abstract stays True
        logger.debug("Creating new Meta class for dynamic model (abstract=True)")
        carrier.django_meta_class = type("Meta", (), meta_attrs)

    def _assemble_django_model_class(self, carrier: ConversionCarrier[SourceModelType]):
        """Assemble the final Django model class using type()."""
        source_name = getattr(carrier.source_model, "__name__", "UnknownSourceModel")
        source_module = getattr(carrier.source_model, "__module__", None)

        model_attrs: dict[str, Any] = {
            **carrier.django_fields,
            **carrier.relationship_fields,
            # Set __module__ for where the model appears to live
            "__module__": source_module or f"{carrier.meta_app_label}.models",
        }
        if carrier.django_meta_class:
            model_attrs["Meta"] = carrier.django_meta_class

        # Add a reference back to the source model (generic attribute name)
        model_attrs["_pydantic2django_source"] = carrier.source_model

        bases = (carrier.base_django_model,) if carrier.base_django_model else (models.Model,)

        # Even if no fields were generated (e.g., collisions with base removed them),
        # we still assemble the model class so that later phases (e.g., relationship finalization)
        # can inject fields and meta indexes onto the carrier.
        if not carrier.django_fields and not carrier.relationship_fields:
            logger.info(
                f"No Django fields generated for {source_name}, assembling bare model class to allow later injections."
            )

        model_name = f"{carrier.class_name_prefix}{source_name}"
        logger.debug(f"Assembling model class '{model_name}' with bases {bases} and attrs: {list(model_attrs.keys())}")

        try:
            # Use type() to dynamically create the class
            carrier.django_model = cast(type[models.Model], type(model_name, bases, model_attrs))
            logger.info(f"Successfully assembled Django model class: {model_name}")
        except Exception as e:
            logger.error(f"Failed to assemble Django model class {model_name}: {e}", exc_info=True)
            carrier.invalid_fields.append(("_assembly", f"Failed to create model type: {e}"))
            carrier.django_model = None

    @abstractmethod
    def _build_model_context(self, carrier: ConversionCarrier[SourceModelType]):
        """Abstract method for subclasses to build the specific ModelContext."""
        pass

    # Main orchestration method
    def make_django_model(self, carrier: ConversionCarrier[SourceModelType]) -> None:
        """
        Orchestrates the Django model creation process.
        Subclasses implement _process_source_fields and _build_model_context.
        Handles caching.
        """
        model_key = carrier.model_key
        logger.debug(f"Attempting to create Django model for {model_key}")

        # TODO: Cache handling needs refinement - how to access subclass cache?
        # For now, skipping cache check in base class.
        # if model_key in self._converted_models and not carrier.existing_model:
        #     # ... update carrier from cache ...
        #     return

        # Reset results on carrier
        carrier.django_fields = {}
        carrier.relationship_fields = {}
        carrier.context_fields = {}
        carrier.invalid_fields = []
        carrier.django_meta_class = None
        carrier.django_model = None
        carrier.model_context = None
        carrier.django_field_definitions = {}  # Reset definitions dict

        # Core Steps
        self._process_source_fields(carrier)
        self._handle_field_collisions(carrier)
        self._create_django_meta(carrier)
        self._assemble_django_model_class(carrier)

        # Build context only if model assembly succeeded
        if carrier.django_model:
            self._build_model_context(carrier)

        # TODO: Cache result (needs access to subclass cache)
        # if carrier.django_model and not carrier.existing_model:
        # self._converted_models[model_key] = replace(carrier) # Or direct assign
        # elif not carrier.django_model:
        # logger.error(...) # Log failure
