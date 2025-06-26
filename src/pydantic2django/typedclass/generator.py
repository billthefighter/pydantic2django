import logging
from collections.abc import Callable
from typing import Optional

from django.db import models

from pydantic2django.core.base_generator import BaseStaticGenerator
from pydantic2django.core.factories import ConversionCarrier
from pydantic2django.core.relationships import RelationshipConversionAccessor

# Import the base Django model for generated classes
from pydantic2django.django.models import TypedClass2DjangoBaseClass  # Assuming this will be created

from .discovery import TypedClassDiscovery, TypedClassType
from .factory import TypedClassFieldFactory, TypedClassFieldInfo, TypedClassModelFactory

logger = logging.getLogger(__name__)


class TypedClassDjangoModelGenerator(
    BaseStaticGenerator[TypedClassType, TypedClassFieldInfo]  # Use TypedClass types
):
    """Generates Django models.py file content from generic Python classes."""

    def __init__(
        self,
        output_path: str,
        app_label: str,
        filter_function: Optional[Callable[[TypedClassType], bool]],
        verbose: bool,
        packages: list[str] | None = None,
        discovery_instance: Optional[TypedClassDiscovery] = None,
        model_factory_instance: Optional[TypedClassModelFactory] = None,
        field_factory_instance: Optional[TypedClassFieldFactory] = None,
        relationship_accessor: Optional[RelationshipConversionAccessor] = None,
        module_mappings: Optional[dict[str, str]] = None,
        base_model_class: type[models.Model] = TypedClass2DjangoBaseClass,
        # Add reckless_mode: bool = False if we pass it down to factories
    ):
        # 1. Initialize TypedClass-specific discovery
        self.typedclass_discovery_instance = discovery_instance or TypedClassDiscovery()

        # 2. Initialize TypedClass-specific factories
        self.relationship_accessor = relationship_accessor or RelationshipConversionAccessor()
        # self.bidirectional_mapper = BidirectionalTypeMapper(relationship_accessor=self.relationship_accessor) # If needed

        self.typedclass_field_factory = field_factory_instance or TypedClassFieldFactory(
            relationship_accessor=self.relationship_accessor,
            # Pass reckless_mode here if implemented
        )
        self.typedclass_model_factory = model_factory_instance or TypedClassModelFactory(
            field_factory=self.typedclass_field_factory,
            relationship_accessor=self.relationship_accessor,
            # Pass reckless_mode here if implemented
        )

        # 3. Call the base class __init__
        super().__init__(
            output_path=output_path,
            packages=packages,
            app_label=app_label,
            filter_function=filter_function,
            verbose=verbose,
            discovery_instance=self.typedclass_discovery_instance,
            model_factory_instance=self.typedclass_model_factory,
            module_mappings=module_mappings,
            base_model_class=base_model_class,
        )
        logger.info("TypedClassDjangoModelGenerator initialized.")

    # --- Implement abstract methods from BaseStaticGenerator ---

    def _get_source_model_name(self, carrier: ConversionCarrier[TypedClassType]) -> str:
        """Get the name of the original generic class from the carrier."""
        if carrier.source_model:
            return carrier.source_model.__name__
        return "UnknownTypedClass"

    def _add_source_model_import(self, carrier: ConversionCarrier[TypedClassType]):
        """Add the necessary import for the original generic class."""
        model_to_import = carrier.source_model
        if model_to_import:
            # For generic classes, these are just standard Python class imports
            self.import_handler.add_general_import(model_to_import.__module__, model_to_import.__name__)
        else:
            logger.warning("Cannot add source model import: source model missing in carrier.")

    def _prepare_template_context(self, unique_model_definitions, django_model_names, imports) -> dict:
        """Prepare the context specific to generic classes for the main models_file.py.j2 template."""
        base_context = {
            "model_definitions": unique_model_definitions,
            "django_model_names": django_model_names,
            "imports": imports,
            "generation_source_type": "typedclass",  # Flag for template logic
            # Ensure compatibility with existing templates or adapt templates
            "context_definitions": [],
            "context_class_names": [],
            "model_has_context": {},
        }
        return base_context

    def _get_models_in_processing_order(self) -> list[TypedClassType]:
        """Return generic classes in dependency order using the discovery instance."""
        assert isinstance(
            self.discovery_instance, TypedClassDiscovery
        ), "Discovery instance must be TypedClassDiscovery for this generator"
        return self.discovery_instance.get_models_in_registration_order()

    def _get_model_definition_extra_context(self, carrier: ConversionCarrier[TypedClassType]) -> dict:
        """Provide extra context specific to generic classes for model_definition.py.j2."""
        return {
            "is_typedclass_source": True,  # New flag for typedclass
            "is_dataclass_source": False,
            "is_pydantic_source": False,
            "has_context": False,
            "field_definitions": carrier.django_field_definitions,
            # Add custom methods if reckless mode generated any serialization helpers
            # "custom_methods": carrier.custom_methods
        }

    # Most other methods like generate_models_file, generate, _write_models_file,
    # discover_models, setup_django_model, generate_model_definition should be
    # inherited from BaseStaticGenerator and work correctly if the factories and
    # discovery mechanisms are properly implemented for TypedClassType.

    # We might need a TypedClass2DjangoBaseClass in pydantic2django.django.models
    # similar to Dataclass2DjangoBaseClass if we want common base functionality.
    # For now, models.Model or a user-provided base can be used.
