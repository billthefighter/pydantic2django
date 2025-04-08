import logging
from collections.abc import Callable
from typing import Optional

from django.db import models
from pydantic import BaseModel

# Import other Pydantic-specific needs
from ....context_storage import ContextClassGenerator

# Import the original factories needed by PydanticModelFactory
from ....factory import DjangoFieldFactory as OriginalDjangoFieldFactory
from ....factory import RelationshipConversionAccessor
from ....models import Pydantic2DjangoBaseClass  # Default base

# Import base classes and specific components
from ..base import BaseStaticGenerator
from .discovery import PydanticDiscovery
from .factories import PydanticModelFactory  # Keep for structure, even if delegating

logger = logging.getLogger(__name__)


class StaticPydanticModelGenerator(
    BaseStaticGenerator[BaseModel, "FieldInfo"]
):  # TModel=BaseModel, TFieldInfo=FieldInfo
    """
    Generates Django models and their context classes from Pydantic models.
    Inherits common logic from BaseStaticGenerator.
    """

    def __init__(
        self,
        output_path: str = "generated_models.py",  # Keep original default
        packages: Optional[list[str]] = None,
        app_label: str = "django_app",  # Keep original default
        filter_function: Optional[Callable[[type[BaseModel]], bool]] = None,
        verbose: bool = False,
        discovery_module: Optional[PydanticDiscovery] = None,
        module_mappings: Optional[dict[str, str]] = None,
        base_model_class: type[models.Model] = Pydantic2DjangoBaseClass,
    ):
        # Initialize Pydantic-specific discovery and factories
        self.discovery_instance = discovery_module or PydanticDiscovery()

        # Initialize RelationshipAccessor needed by factories/generator logic
        # Note: discovery needs to run before relationship accessor can be fully effective
        # We initialize it here, but it relies on discovery.dependencies being populated later.
        self.relationship_accessor = RelationshipConversionAccessor(
            dependencies=self.discovery_instance.dependencies  # Pass the dict ref
        )

        # Use the *original* field factory directly, configured with relationships
        self.field_factory_instance = OriginalDjangoFieldFactory(available_relationships=self.relationship_accessor)

        # Use the PydanticModelFactory wrapper (which delegates to original)
        self.model_factory_instance = PydanticModelFactory(
            field_factory=self.field_factory_instance, relationship_accessor=self.relationship_accessor
        )

        # Initialize the base generator
        super().__init__(
            output_path=output_path,
            packages=packages or ["pydantic_models"],  # Keep original default package
            app_label=app_label,
            filter_function=filter_function,
            verbose=verbose,
            discovery_instance=self.discovery_instance,
            model_factory_instance=self.model_factory_instance,
            module_mappings=module_mappings,
            base_model_class=base_model_class,
        )

        # Pydantic-specific Jinja setup or context generator
        self.context_generator = ContextClassGenerator(jinja_env=self.jinja_env)

        # Track context-specific info during generation
        self.context_definitions: list[str] = []
        self.model_has_context: dict[str, bool] = {}
        self.context_class_names: list[str] = []
        self.seen_context_classes: set[str] = set()

    # --- Implement Abstract Methods from Base ---

    def _get_source_model_name(self, carrier) -> str:
        """Get the name of the original Pydantic model."""
        return carrier.pydantic_model.__name__ if carrier.pydantic_model else "Unknown"

    def _add_source_model_import(self, carrier):
        """Add import for the original Pydantic model."""
        if carrier.pydantic_model:
            self.import_handler.add_pydantic_model_import(carrier.pydantic_model)

    def _get_models_in_processing_order(self) -> list[type[BaseModel]]:
        """Return models in Pydantic dependency order."""
        # Discovery must have run first
        if not self.discovery.filtered_models:
            logger.warning("No models discovered, cannot determine processing order.")
            return []
        # Ensure dependencies are analyzed if not already done
        if not self.discovery.dependencies:
            self.discovery.analyze_dependencies()  # Assuming discovery has this method
        return self.discovery.get_models_in_registration_order()  # Assuming discovery has this method

    def _prepare_template_context(self, unique_model_definitions, django_model_names, imports) -> dict:
        """Prepare the Pydantic-specific context for the main models_file.py.j2 template."""
        # Pydantic generator needs context definitions etc.
        # These are populated during the overridden generate_models_file process
        return {
            "model_definitions": unique_model_definitions,
            "django_model_names": django_model_names,  # For __all__
            # --- Pydantic Specific ---
            "context_definitions": self.context_definitions,
            "all_models": [
                f"'{name}'" for name in self.model_has_context.keys()
            ],  # Original Pydantic names for mapping? Check template usage.
            "context_class_names": self.context_class_names,  # For __all__
            "model_has_context": self.model_has_context,
            # --- Imports ---
            # Assuming ImportHandler separates pydantic/context imports
            "pydantic_imports": sorted(
                imports.get("pydantic", []) + imports.get("general", [])
            ),  # Combine pydantic and general
            "context_imports": sorted(imports.get("context", [])),
        }

    def _get_model_definition_extra_context(self, carrier) -> dict:
        """Provide Pydantic-specific context for model_definition.py.j2."""
        context_fields_info = []
        context_class_name = ""
        has_context = False

        if carrier.model_context and carrier.model_context.context_fields:
            has_context = True
            model_name = self._clean_generic_type(carrier.django_model.__name__)
            context_class_name = f"{model_name}Context"
            for field_name, field_context in carrier.model_context.context_fields.items():
                # Use TypeHandler utility for readable name
                type_name = TypeHandler.get_readable_type_name(field_context.field_type)
                context_fields_info.append((field_name, type_name))

        return {
            "context_class_name": context_class_name,
            "context_fields": context_fields_info,
            "is_pydantic_source": True,
            "is_dataclass_source": False,
            # Add any other Pydantic-specific flags needed by the template
        }

    # --- Override generate_models_file to handle Pydantic context class generation ---

    def generate_models_file(self) -> str:
        """
        Generates the complete models.py file content, including Pydantic context classes.
        Overrides the base method to add context class handling.
        """
        self.discover_models()  # Populates discovery.filtered_models and discovery.dependencies
        models_to_process = self._get_models_in_processing_order()

        # Reset context tracking lists for this run
        self.context_definitions = []
        self.model_has_context = {}
        self.context_class_names = []
        self.seen_context_classes = set()

        model_definitions = []
        django_model_names = []  # Track generated Django model names for __all__
        context_only_models = []  # Track models with only context fields

        # Setup Django models first (populates self.carriers)
        for source_model in models_to_process:
            self.setup_django_model(source_model)  # Calls factory, populates carrier.model_context etc.

        # Generate definitions and handle context classes
        for carrier in self.carriers:
            model_name = self._get_source_model_name(carrier)  # Pydantic model name

            if carrier.django_model:
                try:
                    # Check if it only had context fields (original logic)
                    if not carrier.django_model._meta.fields:  # Simple check if fields list is empty/only pk
                        has_concrete_fields = any(not f.primary_key for f in carrier.django_model._meta.fields)
                        has_m2m = (
                            hasattr(carrier.django_model._meta, "many_to_many")
                            and carrier.django_model._meta.many_to_many
                        )
                        if not has_concrete_fields and not has_m2m:
                            # Check if it *does* have context fields
                            if carrier.model_context and carrier.model_context.context_fields:
                                context_only_models.append(model_name)
                                logger.info(
                                    f"Skipping Django model definition for {model_name} - only has context fields."
                                )
                                # Still process context class below if needed
                            else:
                                logger.warning(
                                    f"Model {model_name} resulted in an empty Django model with no context fields. Skipping."
                                )
                                continue  # Skip entirely if no Django fields AND no context fields

                    # Generate Django model definition only if it has fields
                    if carrier.django_model._meta.fields or (
                        hasattr(carrier.django_model._meta, "many_to_many") and carrier.django_model._meta.many_to_many
                    ):
                        model_def = self.generate_model_definition(carrier)  # Uses base method + extra context
                        model_definitions.append(model_def)
                        django_model_name = self._clean_generic_type(carrier.django_model.__name__)
                        django_model_names.append(f"'{django_model_name}'")

                    # --- Context Class Handling (Pydantic Specific) ---
                    has_context = False
                    if carrier.model_context and carrier.model_context.context_fields:
                        has_context = True
                        # Generate context class definition using the context_generator
                        context_def = self.context_generator.generate_context_class(carrier.model_context)

                        # Add context class name to __all__ list and check for duplicates
                        django_model_name_cleaned = self._clean_generic_type(carrier.django_model.__name__)
                        context_class_name = f"{django_model_name_cleaned}Context"

                        if context_class_name not in self.seen_context_classes:
                            self.context_definitions.append(context_def)
                            self.context_class_names.append(f"'{context_class_name}'")
                            self.seen_context_classes.add(context_class_name)

                        # Process context fields for imports (handled by ContextClassGenerator or add here)
                        for _, field_context in carrier.model_context.context_fields.items():
                            self.import_handler.add_context_field_type_import(field_context.field_type)
                            if field_context.is_optional:
                                self.import_handler.add_extra_import("Optional", "typing")
                            if field_context.is_list:
                                self.import_handler.add_extra_import("List", "typing")

                    self.model_has_context[model_name] = has_context
                    # --- End Context Class Handling ---

                    # Add import for the original source model (Pydantic model)
                    self._add_source_model_import(carrier)

                except Exception as e:
                    logger.error(f"Error generating definition or context for {model_name}: {e}", exc_info=True)

            elif carrier.pydantic_model:
                logger.warning(f"Skipping definition for {model_name} as Django model creation failed.")

        # Log summary of skipped models
        if context_only_models:
            logger.info(
                f"Skipped Django definitions for {len(context_only_models)} models with only context fields: {', '.join(context_only_models)}"
            )

        # Deduplicate definitions (Django models + Context classes separately?)
        # Base class handles Django model definition deduplication.
        unique_model_definitions = self._deduplicate_definitions(model_definitions)
        # Context classes are already deduplicated by name during generation.

        imports = self.import_handler.deduplicate_imports()

        # Prepare context using subclass method (_prepare_template_context)
        template_context = self._prepare_template_context(unique_model_definitions, django_model_names, imports)

        # Add common context items from base class method
        template_context.update(
            {
                "generation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "base_model_module": self.base_model_class.__module__,
                "base_model_name": self.base_model_class.__name__,
                "extra_type_imports": sorted(self.import_handler.extra_type_imports),
                # Add Pydantic specific flag for template if needed
                "generation_source_type": "pydantic",
            }
        )

        template = self.jinja_env.get_template("models_file.py.j2")
        return template.render(**template_context)
