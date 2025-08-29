import dataclasses
import logging
from collections.abc import Callable
from typing import Optional

from django.apps import apps as django_apps
from django.core.exceptions import AppRegistryNotReady
from django.db import models

# Core imports
from pydantic2django.core.base_generator import BaseStaticGenerator
from pydantic2django.core.bidirectional_mapper import BidirectionalTypeMapper
from pydantic2django.core.factories import ConversionCarrier
from pydantic2django.core.relationships import RelationshipConversionAccessor
from pydantic2django.django.timescale.heuristics import TimescaleRole  # type: ignore

# Base Django model will be provided via _get_default_base_model_class
# from pydantic2django.factory import DjangoModelFactoryCarrier # Old carrier, use ConversionCarrier
# Dataclass specific imports
from .discovery import DataclassDiscovery, DataclassType
from .factory import DataclassFieldFactory, DataclassModelFactory  # Corrected filename: factory (singular)

logger = logging.getLogger(__name__)

# Define the specific FieldInfo type for dataclasses (already defined in original)
DataclassFieldInfo = dataclasses.Field


class DataclassDjangoModelGenerator(
    BaseStaticGenerator[DataclassType, DataclassFieldInfo]  # Inherit from BaseStaticGenerator
):
    """Generates Django models.py file content from Python dataclasses."""

    def __init__(
        self,
        output_path: str,
        app_label: str,
        filter_function: Optional[Callable[[DataclassType], bool]],
        verbose: bool,
        # Accept specific discovery and factories, or create defaults
        packages: list[str] | None = None,
        discovery_instance: Optional[DataclassDiscovery] = None,
        model_factory_instance: Optional[DataclassModelFactory] = None,
        field_factory_instance: Optional[DataclassFieldFactory] = None,  # Add field factory param
        relationship_accessor: Optional[RelationshipConversionAccessor] = None,  # Accept accessor
        module_mappings: Optional[dict[str, str]] = None,
    ):
        # 1. Initialize Dataclass-specific discovery
        self.dataclass_discovery_instance = discovery_instance or DataclassDiscovery()

        # 2. Initialize Dataclass-specific factories
        # Dataclass factories might not need RelationshipAccessor, check their definitions
        # Assuming they don't for now.
        # --- Correction: They DO need them now ---
        # Use provided accessor or create a new one
        self.relationship_accessor = relationship_accessor or RelationshipConversionAccessor()
        # Create mapper using the (potentially provided) accessor
        self.bidirectional_mapper = BidirectionalTypeMapper(relationship_accessor=self.relationship_accessor)

        self.dataclass_field_factory = field_factory_instance or DataclassFieldFactory(
            relationship_accessor=self.relationship_accessor,
            bidirectional_mapper=self.bidirectional_mapper,
        )
        self.dataclass_model_factory = model_factory_instance or DataclassModelFactory(
            field_factory=self.dataclass_field_factory,
            relationship_accessor=self.relationship_accessor,  # Pass only accessor
        )

        # 3. Call the base class __init__
        super().__init__(
            output_path=output_path,
            packages=packages,
            app_label=app_label,
            filter_function=filter_function,
            verbose=verbose,
            discovery_instance=self.dataclass_discovery_instance,
            model_factory_instance=self.dataclass_model_factory,
            module_mappings=module_mappings,
            base_model_class=self._get_default_base_model_class(),
        )
        logger.info("DataclassDjangoModelGenerator initialized using BaseStaticGenerator.")
        # Timescale classification results cached per run (name -> role)
        self._timescale_roles: dict[str, TimescaleRole] = {}

    # --- Implement abstract methods from BaseStaticGenerator ---

    def _get_source_model_name(self, carrier: ConversionCarrier[DataclassType]) -> str:
        """Get the name of the original dataclass from the carrier."""
        # Use carrier.source_model (consistent with Base class)
        if carrier.source_model:
            return carrier.source_model.__name__
        # Fallback if source model somehow missing
        # Check if carrier has pydantic_model attribute as a legacy fallback?
        legacy_model = getattr(carrier, "pydantic_model", None)  # Safely check old attribute
        if legacy_model:
            return legacy_model.__name__
        return "UnknownDataclass"

    def _add_source_model_import(self, carrier: ConversionCarrier[DataclassType]):
        """Add the necessary import for the original dataclass."""
        # Use carrier.source_model
        model_to_import = carrier.source_model
        if not model_to_import:
            # Legacy fallback check
            model_to_import = getattr(carrier, "pydantic_model", None)

        if model_to_import:
            # Use add_pydantic_model_import for consistency? Or add_context_field_type_import?
            # Let's assume add_context_field_type_import handles dataclasses too.
            # A dedicated add_dataclass_import or add_general_import would be clearer.
            self.import_handler.add_context_field_type_import(model_to_import)
        else:
            logger.warning("Cannot add source model import: source model missing in carrier.")

    def _prepare_template_context(self, unique_model_definitions, django_model_names, imports) -> dict:
        """Prepare the context specific to dataclasses for the main models_file.py.j2 template."""
        # Base context items are passed in.
        # Add Dataclass-specific items.
        base_context = {
            "model_definitions": unique_model_definitions,  # Already joined by base class
            "django_model_names": django_model_names,  # Already list of quoted names
            # Pass the structured imports dict
            "imports": imports,
            # --- Dataclass Specific ---
            "generation_source_type": "dataclass",  # Flag for template logic
            # --- Keep compatibility if templates expect these --- (review templates later)
            # "django_imports": sorted(imports.get("django", [])), # Provided by imports dict
            # "pydantic_imports": sorted(imports.get("pydantic", [])), # Likely empty for dataclass
            # "general_imports": sorted(imports.get("general", [])),
            # "context_imports": sorted(imports.get("context", [])),
            # Add other dataclass specific flags/lists if needed by the template
            "context_definitions": [],  # Dataclasses don't have separate context classes? Assume empty.
            "context_class_names": [],
            "model_has_context": {},  # Assume no context model mapping needed
        }
        # Common items added by base class generate_models_file after this call.
        return base_context

    def _get_models_in_processing_order(self) -> list[DataclassType]:
        """Return dataclasses in dependency order using the discovery instance."""
        # Add assertion for type checker clarity
        assert isinstance(
            self.discovery_instance, DataclassDiscovery
        ), "Discovery instance must be DataclassDiscovery for this generator"
        # Dependencies analyzed by base class discover_models call
        return self.discovery_instance.get_models_in_registration_order()

    def _get_model_definition_extra_context(self, carrier: ConversionCarrier[DataclassType]) -> dict:
        """Provide extra context specific to dataclasses for model_definition.py.j2."""
        # Removed problematic metadata access from original
        # Add flags for template conditional logic
        return {
            "is_dataclass_source": True,
            "is_pydantic_source": False,
            "has_context": False,  # Dataclasses likely don't generate separate context fields/classes
            # Pass the field definitions dictionary from the carrier
            "field_definitions": carrier.django_field_definitions,
            # Add other specific details if needed, ensuring they access carrier correctly
            # Example: "source_model_module": carrier.source_model.__module__ if carrier.source_model else ""
        }

    # Choose Timescale base per model (lazy roles computation)
    def setup_django_model(self, source_model: DataclassType) -> ConversionCarrier | None:  # type: ignore[override]
        try:
            from pydantic2django.django.timescale.bases import DataclassTimescaleBase
            from pydantic2django.django.timescale.heuristics import (
                classify_dataclass_types,
                should_use_timescale_base,
            )
        except Exception:
            classify_dataclass_types = None  # type: ignore
            should_use_timescale_base = None  # type: ignore
            DataclassTimescaleBase = None  # type: ignore

        # Compute roles lazily if not present
        if not getattr(self, "_timescale_roles", None):
            roles: dict[str, TimescaleRole] = {}
            try:
                models_to_score = []
                try:
                    models_to_score = self._get_models_in_processing_order() or []
                except Exception:
                    pass
                if not models_to_score:
                    models_to_score = [source_model]
                if classify_dataclass_types:
                    roles = classify_dataclass_types(models_to_score)
            except Exception:
                roles = {}
            self._timescale_roles = roles

        # Select base class
        base_cls: type[models.Model] = self.base_model_class
        try:
            name = source_model.__name__
            if should_use_timescale_base and DataclassTimescaleBase:
                if should_use_timescale_base(name, self._timescale_roles):  # type: ignore[arg-type]
                    base_cls = DataclassTimescaleBase
        except Exception:
            pass

        prev_base = self.base_model_class
        self.base_model_class = base_cls
        try:
            carrier = super().setup_django_model(source_model)
        finally:
            self.base_model_class = prev_base

        if carrier is not None:
            carrier.context_data["_timescale_roles"] = getattr(self, "_timescale_roles", {})
        return carrier

    def _get_default_base_model_class(self) -> type[models.Model]:
        """Return the default Django base model for Dataclass conversion."""
        if not django_apps.ready:
            raise AppRegistryNotReady(
                "Django apps are not loaded. Call django.setup() or run within a configured Django context before "
                "instantiating DataclassDjangoModelGenerator."
            )
        try:
            from typed2django.django.models import Dataclass2DjangoBaseClass as _Base

            return _Base
        except Exception as exc:  # pragma: no cover - defensive
            raise ImportError(
                "typed2django.django.models.Dataclass2DjangoBaseClass is required for Dataclass generation."
            ) from exc

    # --- Potentially Override generate_models_file if needed ---
    # For dataclasses, the base generate_models_file might be sufficient as there's no
    # separate context class generation step like in Pydantic.
    # If specific logic is needed (e.g., different handling of empty models), override it.
    # For now, assume base class implementation is okay.

    # def generate_models_file(self) -> str:
    #     """ Override if Dataclass generation needs specific steps. """
    #     # 1. Call super().generate_models_file() to get base content
    #     # content = super().generate_models_file()
    #     # 2. Modify content if needed
    #     # return modified_content
    #     # OR: Reimplement the loop with dataclass specific logic
    #     pass # Using base class version

    # --- Remove methods now implemented in BaseStaticGenerator ---
    # generate(self) -> str: ...
    # _write_models_file(self, content: str) -> None: ...
    # discover_models(self) -> None: ...
    # setup_django_model(self, source_model: DataclassType) -> Optional[ConversionCarrier[DataclassType]]: ...
    # generate_model_definition(self, carrier: ConversionCarrier[DataclassType]) -> str: ...
    # _deduplicate_definitions(self, definitions: list[str]) -> list[str]: ...
    # _clean_generic_type(self, name: str) -> str: ...


# No old methods to remove as they were already replaced in the previous step
# This assumes the original file was already partially refactored or clean.
