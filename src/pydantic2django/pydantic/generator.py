import datetime
import logging
from collections.abc import Callable
from typing import Optional, cast

from django.apps import apps as django_apps
from django.core.exceptions import AppRegistryNotReady
from django.db import models
from pydantic import BaseModel
from pydantic.fields import FieldInfo

# Import other Pydantic-specific needs
from pydantic2django.core.base_generator import BaseStaticGenerator

# Import the new mapper
from pydantic2django.core.bidirectional_mapper import BidirectionalTypeMapper
from pydantic2django.core.context import ContextClassGenerator

# Core imports - Adjusted path for ConversionCarrier
from pydantic2django.core.factories import ConversionCarrier
from pydantic2django.core.relationships import RelationshipConversionAccessor
from pydantic2django.core.typing import TypeHandler
from pydantic2django.django.timescale.heuristics import TimescaleRole  # type: ignore

# TypeMapper for field generation hints
# Import base classes and specific components
from pydantic2django.pydantic.discovery import PydanticDiscovery
from pydantic2django.pydantic.factory import create_pydantic_factory

# Import the original factories needed by PydanticModelFactory
# from pydantic2django.factory import DjangoFieldFactory as OriginalDjangoFieldFactory
# Note: default base class will be resolved via _get_default_base_model_class

logger = logging.getLogger(__name__)


class StaticPydanticModelGenerator(
    BaseStaticGenerator[type[BaseModel], FieldInfo]
):  # TModel=type[BaseModel], TFieldInfo=FieldInfo
    """
    Generates Django models and their context classes from Pydantic models.
    Inherits common logic from BaseStaticGenerator.
    """

    def __init__(
        self,
        output_path: str = "generated_models.py",  # Keep original default
        packages: Optional[list[str]] = None,
        app_label: str = "django_app",  # Keep original default
        filter_function: Callable[..., bool] | None = None,
        verbose: bool = False,
        discovery_module: Optional[PydanticDiscovery] = None,
        module_mappings: Optional[dict[str, str]] = None,
        # Pydantic specific factories can be passed or constructed here
        # NOTE: Injecting factory instances is less preferred now due to mapper dependency
        # field_factory_instance: Optional[PydanticFieldFactory] = None,
        # model_factory_instance: Optional[PydanticModelFactory] = None,
        # Inject mapper instead?
        bidirectional_mapper_instance: Optional[BidirectionalTypeMapper] = None,
    ):
        # 1. Initialize Pydantic-specific discovery
        # Use provided instance or create a default one
        self.pydantic_discovery_instance = discovery_module or PydanticDiscovery()

        # 2. Initialize RelationshipAccessor (needed by factories and mapper)
        self.relationship_accessor = RelationshipConversionAccessor()

        # 3. Initialize BidirectionalTypeMapper (pass relationship accessor)
        self.bidirectional_mapper = bidirectional_mapper_instance or BidirectionalTypeMapper(
            relationship_accessor=self.relationship_accessor
        )

        # 4. Initialize Pydantic-specific factories (pass mapper and accessor)
        # Remove dependency on passed-in factory instances, create them here
        self.pydantic_model_factory = create_pydantic_factory(
            relationship_accessor=self.relationship_accessor, bidirectional_mapper=self.bidirectional_mapper
        )

        # 5. Call the base class __init__ with all required arguments
        super().__init__(
            output_path=output_path,
            packages=packages or ["pydantic_models"],  # Default Pydantic package
            app_label=app_label,
            filter_function=filter_function,
            verbose=verbose,
            discovery_instance=self.pydantic_discovery_instance,  # Pass the specific discovery instance
            model_factory_instance=self.pydantic_model_factory,  # Pass the newly created model factory
            module_mappings=module_mappings,
            base_model_class=self._get_default_base_model_class(),
            # Jinja setup is handled by base class
        )

        # 6. Pydantic-specific Jinja setup or context generator
        # Context generator needs the jinja_env from the base class
        self.context_generator = ContextClassGenerator(jinja_env=self.jinja_env)

        # 7. Track context-specific info during generation (reset in generate_models_file)
        self.context_definitions: list[str] = []
        self.model_has_context: dict[str, bool] = {}
        self.context_class_names: list[str] = []
        self.seen_context_classes: set[str] = set()
        # Timescale classification results cached per run (name -> role)
        self._timescale_roles: dict[str, TimescaleRole] = {}

    # --- Implement Abstract Methods from Base ---

    def _get_source_model_name(self, carrier: ConversionCarrier[type[BaseModel]]) -> str:
        """Get the name of the original Pydantic model."""
        # Ensure source_model is not None before accessing __name__
        return carrier.source_model.__name__ if carrier.source_model else "UnknownPydanticModel"

    def _add_source_model_import(self, carrier: ConversionCarrier[type[BaseModel]]):
        """Add import for the original Pydantic model."""
        if carrier.source_model:
            # Use the correct method from ImportHandler
            self.import_handler.add_pydantic_model_import(carrier.source_model)
        else:
            logger.warning("Cannot add source model import: source_model is missing from carrier.")

    def _get_models_in_processing_order(self) -> list[type[BaseModel]]:
        """Return models in Pydantic dependency order."""
        # Discovery must have run first (called by base generate_models_file -> discover_models)
        # Cast the discovery_instance from the base class to the specific Pydantic type
        discovery = cast(PydanticDiscovery, self.discovery_instance)
        if not discovery.filtered_models:
            logger.warning("No models discovered or passed filter, cannot determine processing order.")
            return []
        # Ensure dependencies are analyzed if not already done (base class should handle this)
        # if not discovery.dependencies:
        #     discovery.analyze_dependencies() # Base class analyze_dependencies called in discover_models
        return discovery.get_models_in_registration_order()

    def _prepare_template_context(self, unique_model_definitions, django_model_names, imports) -> dict:
        """Prepare the Pydantic-specific context for the main models_file.py.j2 template."""
        # Base context items (model_definitions, django_model_names, imports) are passed in.
        # Add Pydantic-specific items gathered during generate_models_file override.
        base_context = {
            "model_definitions": unique_model_definitions,
            "django_model_names": django_model_names,  # For __all__
            # --- Imports (already structured by base class import_handler) ---
            "django_imports": sorted(imports.get("django", [])),
            "pydantic_imports": sorted(imports.get("pydantic", [])),  # Check if import handler categorizes these
            "general_imports": sorted(imports.get("general", [])),
            "context_imports": sorted(imports.get("context", [])),  # Check if import handler categorizes these
            # It might be simpler to rely on the structured imports dict directly in the template
            "imports": imports,  # Pass the whole structured dict
            # --- Pydantic Specific ---
            "context_definitions": self.context_definitions,  # Populated in generate_models_file override
            "all_models": [  # This seems redundant if django_model_names covers __all__
                f"'{name}'"
                for name in django_model_names  # Use Django names for __all__ consistency?
            ],
            "context_class_names": self.context_class_names,  # Populated in generate_models_file override
            "model_has_context": self.model_has_context,  # Populated in generate_models_file override
            "generation_source_type": "pydantic",  # Flag for template logic
        }
        # Note: Common items like timestamp, base_model info, extra_type_imports
        # are added by the base class generate_models_file method after calling this.
        return base_context

    def _get_model_definition_extra_context(self, carrier: ConversionCarrier[type[BaseModel]]) -> dict:
        """Provide Pydantic-specific context for model_definition.py.j2."""
        context_fields_info = []
        context_class_name = ""
        has_context_for_this_model = False  # Track if this specific model has context

        if carrier.model_context and carrier.model_context.context_fields:
            has_context_for_this_model = True
            django_model_name = (
                self._clean_generic_type(carrier.django_model.__name__) if carrier.django_model else "UnknownModel"
            )
            context_class_name = f"{django_model_name}Context"

            for field_name, field_context_info in carrier.model_context.context_fields.items():
                field_type_attr = getattr(field_context_info, "field_type", None) or getattr(
                    field_context_info, "annotation", None
                )

                if field_type_attr:
                    type_name = TypeHandler.format_type_string(field_type_attr)
                    # Add imports for context field types via import_handler
                    # Use the correct method which handles nested types and typing imports
                    self.import_handler.add_context_field_type_import(field_type_attr)

                    # Remove explicit add_extra_import calls, handled by add_context_field_type_import
                    # if getattr(field_context_info, 'is_optional', False):
                    #      self.import_handler.add_extra_import("Optional", "typing")
                    # if getattr(field_context_info, 'is_list', False):
                    #      self.import_handler.add_extra_import("List", "typing")
                else:
                    type_name = "Any"  # Fallback
                    logger.warning(
                        f"Could not determine context type annotation for field '{field_name}' in {django_model_name}"
                    )

                context_fields_info.append((field_name, type_name))

        return {
            "context_class_name": context_class_name,
            "context_fields": context_fields_info,
            "is_pydantic_source": True,
            "is_dataclass_source": False,
            "has_context": has_context_for_this_model,
            "field_definitions": carrier.django_field_definitions,
        }

    def _get_default_base_model_class(self) -> type[models.Model]:
        """Return the default Django base model for Pydantic conversion.

        Raises a clear ImportError if the base cannot be imported.
        """
        if not django_apps.ready:
            raise AppRegistryNotReady(
                "Django apps are not loaded. Call django.setup() or run within a configured Django context before "
                "instantiating StaticPydanticModelGenerator."
            )
        try:
            from typed2django.django.models import Pydantic2DjangoBaseClass as _Base

            return _Base
        except Exception as exc:  # pragma: no cover - defensive path
            raise ImportError(
                "typed2django.django.models.Pydantic2DjangoBaseClass is required for Pydantic generation."
            ) from exc

    # --- Override generate_models_file to handle Pydantic context class generation ---

    def generate_models_file(self) -> str:
        """
        Generates the complete models.py file content, including Pydantic context classes.
        Overrides the base method to add context class handling during the generation loop.
        """
        # 1. Base discovery and model ordering
        self.discover_models()  # Calls base discovery and dependency analysis
        models_to_process = self._get_models_in_processing_order()  # Uses overridden method

        # 2. Reset state for this run (imports handled by base reset)
        self.carriers = []
        # Manually reset ImportHandler state instead of calling non-existent reset()
        self.import_handler.extra_type_imports.clear()
        self.import_handler.pydantic_imports.clear()
        self.import_handler.context_class_imports.clear()
        self.import_handler.imported_names.clear()
        self.import_handler.processed_field_types.clear()

        # Re-add base model import after clearing
        # Note: add_pydantic_model_import might not be the right method here if base_model_class isn't Pydantic
        # Need a more general import method on ImportHandler or handle it differently.
        # For now, let's assume a general import is needed or handled by template.
        # self.import_handler.add_import(self.base_model_class.__module__, self.base_model_class.__name__)
        # Let's add it back using _add_type_import, although it's protected.
        # A public add_general_import(module, name) on ImportHandler would be better.
        try:
            # This is a workaround - ideally ImportHandler would have a public method
            self.import_handler._add_type_import(self.base_model_class)
        except Exception as e:
            logger.warning(f"Could not add base model import via _add_type_import: {e}")

        # Reset Pydantic-specific tracking lists
        self.context_definitions = []
        self.model_has_context = {}  # Map of Pydantic model name -> bool
        self.context_class_names = []  # For __all__
        self.seen_context_classes = set()  # For deduplication of definitions

        # --- State tracking within the loop ---
        model_definitions = []  # Store generated Django model definition strings
        django_model_names = []  # Store generated Django model names for __all__
        context_only_models = []  # Track Pydantic models yielding only context

        # 3. Setup Django models (populates self.carriers via base method calling factory)
        for source_model in models_to_process:
            self.setup_django_model(source_model)  # Uses base setup_django_model

        # 4. Generate definitions (Django models AND Pydantic Context classes)
        for carrier in self.carriers:
            model_name = self._get_source_model_name(carrier)  # Pydantic model name

            try:
                django_model_def = ""
                django_model_name_cleaned = ""

                # --- A. Generate Django Model Definition (if applicable) ---
                if carrier.django_model:
                    # Check fields using safe getattr for many_to_many
                    has_concrete_fields = any(not f.primary_key for f in carrier.django_model._meta.fields)
                    # Use getattr for safety
                    m2m_fields = getattr(carrier.django_model._meta, "many_to_many", [])
                    has_m2m = bool(m2m_fields)
                    has_fields = bool(carrier.django_model._meta.fields)

                    if has_concrete_fields or has_m2m or (not has_concrete_fields and not has_m2m and has_fields):
                        django_model_def = self.generate_model_definition(carrier)
                        if django_model_def:
                            model_definitions.append(django_model_def)
                            django_model_name_cleaned = self._clean_generic_type(carrier.django_model.__name__)
                            django_model_names.append(f"'{django_model_name_cleaned}'")
                        else:
                            logger.warning(f"Base generate_model_definition returned empty for {model_name}, skipping.")
                    else:
                        # Model exists but seems empty (no concrete fields/M2M)
                        # Check if it *does* have context fields
                        if carrier.model_context and carrier.model_context.context_fields:
                            context_only_models.append(model_name)
                            logger.info(f"Skipping Django model definition for {model_name} - only has context fields.")
                        else:
                            logger.warning(
                                f"Model {model_name} resulted in an empty Django model with no context fields. Skipping definition."
                            )
                            # Continue to next carrier if no Django model AND no context
                            if not (carrier.model_context and carrier.model_context.context_fields):
                                continue

                # --- B. Generate Context Class Definition (Pydantic Specific) ---
                has_context = False
                if carrier.model_context and carrier.model_context.context_fields:
                    has_context = True
                    # Generate context class definition string using the context_generator
                    # This also handles adding necessary imports for context fields via TypeHandler/ImportHandler calls within it
                    context_def = self.context_generator.generate_context_class(carrier.model_context)

                    # Determine context class name (needs Django model name)
                    # Use the cleaned name if available, otherwise construct from Pydantic name?
                    base_name_for_context = django_model_name_cleaned if django_model_name_cleaned else model_name
                    context_class_name = f"{base_name_for_context}Context"

                    # Add context class definition if not seen before
                    if context_class_name not in self.seen_context_classes:
                        self.context_definitions.append(context_def)
                        self.context_class_names.append(f"'{context_class_name}'")
                        self.seen_context_classes.add(context_class_name)

                    # Add imports for context fields (should be handled by context_generator now)
                    # self.import_handler.add_context_field_imports(carrier.model_context) # Example hypothetical method

                # --- C. Update Tracking and Add Source Import ---
                self.model_has_context[model_name] = has_context

                # Add import for the original source model (Pydantic model)
                self._add_source_model_import(carrier)

            except Exception as e:
                logger.error(f"Error processing carrier for source model {model_name}: {e}", exc_info=True)

        # 5. Log Summary
        if context_only_models:
            logger.info(
                f"Skipped Django definitions for {len(context_only_models)} models with only context fields: {', '.join(context_only_models)}"
            )

        # 6. Deduplicate Definitions (Django models only, context defs deduplicated by name during loop)
        unique_model_definitions = self._deduplicate_definitions(model_definitions)  # Use base method

        # 7. Get Imports (handled by base import_handler)
        imports = self.import_handler.deduplicate_imports()

        # 8. Prepare Template Context (using overridden Pydantic-specific method)
        template_context = self._prepare_template_context(unique_model_definitions, django_model_names, imports)

        # 9. Add Common Context Items (handled by base class) - Reuse base class logic
        template_context.update(
            {
                "generation_timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "base_model_module": self.base_model_class.__module__,
                "base_model_name": self.base_model_class.__name__,
                "extra_type_imports": sorted(self.import_handler.extra_type_imports),
                # Ensure generation_source_type is set by _prepare_template_context
            }
        )

        # 10. Render the main template
        template = self.jinja_env.get_template("models_file.py.j2")
        return template.render(**template_context)

    # Choose Timescale base per model (lazy roles computation)
    def setup_django_model(self, source_model: type[BaseModel]) -> ConversionCarrier | None:  # type: ignore[override]
        try:
            from pydantic2django.django.timescale.bases import PydanticTimescaleBase
            from pydantic2django.django.timescale.heuristics import (
                classify_pydantic_models,
                should_use_timescale_base,
            )
        except Exception:
            classify_pydantic_models = None  # type: ignore
            should_use_timescale_base = None  # type: ignore
            PydanticTimescaleBase = None  # type: ignore

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
                if classify_pydantic_models:
                    roles = classify_pydantic_models(models_to_score)
            except Exception:
                roles = {}
            self._timescale_roles = roles

        # Select base class
        base_cls: type[models.Model] = self.base_model_class
        try:
            name = source_model.__name__
            if should_use_timescale_base and PydanticTimescaleBase:
                if should_use_timescale_base(name, self._timescale_roles):  # type: ignore[arg-type]
                    base_cls = PydanticTimescaleBase
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

    # --- Remove methods now implemented in BaseStaticGenerator ---
    # generate(self) -> str: ...
    # _write_models_file(self, content: str) -> None: ...
    # discover_models(self) -> None: ...
    # setup_django_model(self, source_model: Type[BaseModel]) -> Optional[ConversionCarrier[Type[BaseModel]]]: ...
    # generate_model_definition(self, carrier: ConversionCarrier[Type[BaseModel]]) -> str: ... # Base handles this now, uses _get_model_definition_extra_context
    # _deduplicate_definitions(self, definitions: list[str]) -> list[str]: ...
    # _clean_generic_type(self, name: str) -> str: ...
