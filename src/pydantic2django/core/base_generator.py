import logging
import os
import pathlib
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from typing import Generic, TypeVar, cast

import jinja2
from django.db import models

from ..django.utils.serialization import generate_field_definition_string  # Import needed function
from ..pydantic.factory import PydanticModelFactory  # Import for casting
from .discovery import BaseDiscovery  # Renamed to avoid clash
from .factories import BaseModelFactory, ConversionCarrier
from .imports import ImportHandler
from .typing import TypeHandler

logger = logging.getLogger(__name__)

# Define Generic Types for the base class
SourceModelType = TypeVar("SourceModelType")  # Type of the source model (e.g., Type[BaseModel] or DataclassType)
FieldInfoType = TypeVar("FieldInfoType")  # Type of the field info (e.g., FieldInfo or dataclasses.Field)


class BaseStaticGenerator(ABC, Generic[SourceModelType, FieldInfoType]):
    """
    Abstract base class for generating static Django models from source models (like Pydantic or Dataclasses).
    """

    def __init__(
        self,
        output_path: str,
        app_label: str,
        filter_function: Callable[[type[SourceModelType]], bool] | None,
        verbose: bool,
        discovery_instance: BaseDiscovery[SourceModelType],
        model_factory_instance: BaseModelFactory[SourceModelType, FieldInfoType],
        module_mappings: dict[str, str] | None,
        base_model_class: type[models.Model],
        packages: list[str] | None = None,
        class_name_prefix: str = "Django",
    ):
        """
        Initialize the base generator.

        Args:
            output_path: Path to output the generated models.py file.
            packages: List of packages to scan for source models.
            app_label: Django app label to use for the models.
            filter_function: Optional function to filter which source models to include.
            verbose: Print verbose output.
            discovery_instance: An instance of a BaseDiscovery subclass.
            model_factory_instance: An instance of a BaseModelFactory subclass.
            module_mappings: Optional mapping of modules to remap imports.
            base_model_class: The base Django model class to inherit from.
            class_name_prefix: Prefix for generated Django model class names.
        """
        self.output_path = output_path
        self.packages = packages
        self.app_label = app_label
        self.filter_function = filter_function
        self.verbose = verbose
        self.discovery_instance = discovery_instance
        self.model_factory_instance = model_factory_instance
        # Base model class must be provided explicitly by subclass at call site.
        self.base_model_class = base_model_class
        self.class_name_prefix = class_name_prefix
        self.carriers: list[ConversionCarrier[SourceModelType]] = []  # Stores results from model factory

        self.import_handler = ImportHandler(module_mappings=module_mappings)

        # Initialize Jinja2 environment
        # Look for templates in the django/templates subdirectory
        # package_templates_dir = os.path.join(os.path.dirname(__file__), "..", "templates") # Old path
        package_templates_dir = os.path.join(os.path.dirname(__file__), "..", "django", "templates")  # Corrected path

        # If templates don't exist in the package, use the ones relative to the execution?
        # This might need adjustment based on packaging/distribution.
        # For now, assume templates are relative to the package structure.
        if not os.path.exists(package_templates_dir):
            # Fallback or raise error might be needed
            package_templates_dir = os.path.join(pathlib.Path(__file__).parent.parent.absolute(), "templates")
            if not os.path.exists(package_templates_dir):
                logger.warning(
                    f"Templates directory not found at expected location: {package_templates_dir}. Jinja might fail."
                )

        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(package_templates_dir),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Register common custom filters
        self.jinja_env.filters["format_type_string"] = TypeHandler.format_type_string
        # Provide an escaping filter for embedding strings safely in generated Python code
        from ..core.utils.strings import sanitize_string as _escape_py_str  # Local import to avoid cycles

        self.jinja_env.filters["escape_py_str"] = _escape_py_str
        # Add more common filters if needed

        # Add base model import
        self.import_handler._add_type_import(self.base_model_class)

    # --- Abstract Methods to be Implemented by Subclasses ---

    @abstractmethod
    def _get_source_model_name(self, carrier: ConversionCarrier[SourceModelType]) -> str:
        """Get the name of the original source model from the carrier."""
        pass

    @abstractmethod
    def _add_source_model_import(self, carrier: ConversionCarrier[SourceModelType]):
        """Add the necessary import for the original source model."""
        pass

    @abstractmethod
    def _prepare_template_context(
        self, unique_model_definitions: list[str], django_model_names: list[str], imports: dict
    ) -> dict:
        """Prepare the subclass-specific context for the main models_file.py.j2 template."""
        pass

    @abstractmethod
    def _get_models_in_processing_order(self) -> list[SourceModelType]:
        """Return source models in the correct processing (dependency) order."""
        pass

    @abstractmethod
    def _get_model_definition_extra_context(self, carrier: ConversionCarrier[SourceModelType]) -> dict:
        """Provide extra context specific to the source type for model_definition.py.j2."""
        pass

    @abstractmethod
    def _get_default_base_model_class(self) -> type[models.Model]:
        """Return the required base Django model class for this generator.

        Subclasses implement this so they can easily and explicitly resolve
        the correct base and pass it into the base initializer.
        """
        pass

    # --- Common Methods ---

    def generate(self) -> str:
        """
        Main entry point: Generate and write the models file.

        Returns:
            The path to the generated models file.
        """
        try:
            content = self.generate_models_file()
            self._write_models_file(content)
            logger.info(f"Successfully generated models file at {self.output_path}")
            return self.output_path
        except Exception as e:
            logger.exception(f"Error generating models file: {e}", exc_info=True)  # Use exc_info for traceback
            raise

    def _write_models_file(self, content: str) -> None:
        """Write the generated content to the output file."""
        if self.verbose:
            logger.info(f"Writing models to {self.output_path}")

        output_dir = os.path.dirname(self.output_path)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                if self.verbose:
                    logger.info(f"Created output directory: {output_dir}")
            except OSError as e:
                logger.error(f"Failed to create output directory {output_dir}: {e}")
                raise  # Re-raise after logging

        try:
            with open(self.output_path, "w", encoding="utf-8") as f:  # Specify encoding
                f.write(content)
            if self.verbose:
                logger.info(f"Successfully wrote models to {self.output_path}")
        except OSError as e:
            logger.error(f"Failed to write to output file {self.output_path}: {e}")
            raise  # Re-raise after logging

    def discover_models(self) -> None:
        """Discover source models using the configured discovery instance."""
        if self.verbose:
            logger.info(f"Discovering models from packages: {self.packages}")

        # Corrected call matching BaseDiscovery signature
        self.discovery_instance.discover_models(
            self.packages or [],  # Pass empty list if None
            app_label=self.app_label,
            user_filters=self.filter_function,  # Keep as is for now
        )

        # Analyze dependencies after discovery
        self.discovery_instance.analyze_dependencies()

        if self.verbose:
            logger.info(f"Discovered {len(self.discovery_instance.filtered_models)} models after filtering.")
            if self.discovery_instance.filtered_models:
                for name in self.discovery_instance.filtered_models.keys():
                    logger.info(f"  - {name}")
            else:
                logger.info("  (No models found or passed filter)")
            logger.info("Dependency analysis complete.")

    def setup_django_model(self, source_model: SourceModelType) -> ConversionCarrier[SourceModelType] | None:
        """
        Uses the model factory to create a Django model representation from a source model.

        Args:
            source_model: The source model instance (e.g., Pydantic class, Dataclass).

        Returns:
            A ConversionCarrier containing the results, or None if creation failed.
        """
        source_model_name = getattr(source_model, "__name__", str(source_model))
        if self.verbose:
            logger.info(f"Setting up Django model for {source_model_name}")

        # Instantiate the carrier here
        carrier = ConversionCarrier(
            source_model=cast(type[SourceModelType], source_model),
            meta_app_label=self.app_label,
            base_django_model=self.base_model_class,
            class_name_prefix=self.class_name_prefix,
            # Add other defaults/configs if needed, e.g., strict mode
            strict=False,  # Example default
        )

        try:
            # Use the factory to process the source model and populate the carrier
            self.model_factory_instance.make_django_model(carrier)  # Pass carrier to factory

            if carrier.django_model:
                self.carriers.append(carrier)
                if self.verbose:
                    logger.info(f"Successfully processed {source_model_name} -> {carrier.django_model.__name__}")
                return carrier
            else:
                # Log if model creation resulted in None (e.g., only context fields)
                # Check carrier.context_fields or carrier.invalid_fields for details
                if carrier.context_fields and not carrier.django_fields and not carrier.relationship_fields:
                    logger.info(f"Skipped Django model class for {source_model_name} - only context fields found.")
                elif carrier.invalid_fields:
                    logger.warning(
                        f"Skipped Django model class for {source_model_name} due to invalid fields: {carrier.invalid_fields}"
                    )
                else:
                    logger.warning(f"Django model was not generated for {source_model_name} for unknown reasons.")
                return None  # Return None if no Django model was created

        except Exception as e:
            logger.error(f"Error processing {source_model_name} with factory: {e}", exc_info=True)
            return None

    def generate_model_definition(self, carrier: ConversionCarrier[SourceModelType]) -> str:
        """
        Generates a string definition for a single Django model using a template.

        Args:
            carrier: The ConversionCarrier containing the generated Django model and context.

        Returns:
            The string representation of the Django model definition.
        """
        if not carrier.django_model:
            # It's possible a carrier exists only for context, handle gracefully.
            source_name = self._get_source_model_name(carrier)
            if carrier.model_context and carrier.model_context.context_fields:
                logger.info(f"Skipping Django model definition for {source_name} (likely context-only).")
                return ""
            else:
                logger.warning(
                    f"Cannot generate model definition for {source_name}: django_model is missing in carrier."
                )
                return ""

        django_model_name = self._clean_generic_type(carrier.django_model.__name__)
        source_model_name = self._get_source_model_name(carrier)  # Get original name via abstract method

        # --- Prepare Fields ---
        fields_info = []
        # Combine regular and relationship fields from the carrier
        all_django_fields = {**carrier.django_fields, **carrier.relationship_fields}

        for field_name, field_object in all_django_fields.items():
            # The field_object is already an instantiated Django field
            # Add (name, object) tuple directly for the template
            fields_info.append((field_name, field_object))

        # --- Prepare Meta ---
        meta_options = {}
        if hasattr(carrier.django_model, "_meta"):
            model_meta = carrier.django_model._meta
            meta_options = {
                "db_table": getattr(model_meta, "db_table", f"{self.app_label}_{django_model_name.lower()}"),
                "app_label": self.app_label,
                "verbose_name": getattr(model_meta, "verbose_name", django_model_name),
                "verbose_name_plural": getattr(model_meta, "verbose_name_plural", f"{django_model_name}s"),
                # Add other meta options if needed
            }

        # --- Prepare Base Class Info ---
        base_model_name = self.base_model_class.__name__
        if carrier.django_model.__bases__ and carrier.django_model.__bases__[0] != models.Model:
            # Use the immediate parent if it's not the absolute base 'models.Model'
            # Assumes single inheritance for the generated model besides the ultimate base
            parent_class = carrier.django_model.__bases__[0]
            # Check if the parent is our intended base_model_class or something else
            # This logic might need refinement depending on how complex the inheritance gets
            if issubclass(parent_class, models.Model) and parent_class != models.Model:
                base_model_name = parent_class.__name__
                # Add import for the parent if it's not the configured base_model_class
                if parent_class != self.base_model_class:
                    self.import_handler._add_type_import(parent_class)

        # --- Get Subclass Specific Context ---
        extra_context = self._get_model_definition_extra_context(carrier)

        # --- Process Pending Multi-FK Unions and add to definitions dict ---
        multi_fk_field_names = []  # Keep track for validation hint
        validation_needed = False
        if carrier.pending_multi_fk_unions:
            validation_needed = True
            for original_field_name, union_details in carrier.pending_multi_fk_unions:
                pydantic_models = union_details.get("models", [])
                for pydantic_model in pydantic_models:
                    # Construct field name (e.g., original_name_relatedmodel)
                    fk_field_name = f"{original_field_name}_{pydantic_model.__name__.lower()}"
                    multi_fk_field_names.append(fk_field_name)
                    # Get corresponding Django model
                    pydantic_factory = cast(PydanticModelFactory, self.model_factory_instance)
                    django_model_rel = pydantic_factory.relationship_accessor.get_django_model_for_pydantic(
                        pydantic_model
                    )
                    if not django_model_rel:
                        logger.error(
                            f"Could not find Django model for Pydantic model {pydantic_model.__name__} referenced in multi-FK union for {original_field_name}. Skipping FK field."
                        )
                        continue
                    # Use string for model ref in kwargs
                    target_model_str = f"'{django_model_rel._meta.app_label}.{django_model_rel.__name__}'"
                    # Add import for the related Django model
                    self.import_handler._add_type_import(django_model_rel)

                    # Define FK kwargs (always null=True, blank=True)
                    # Use strings for values that need to be represented in code
                    fk_kwargs = {
                        "to": target_model_str,
                        "on_delete": "models.SET_NULL",  # Use string for template
                        "null": True,
                        "blank": True,
                        # Generate related_name to avoid clashes
                        "related_name": f"'{carrier.django_model.__name__.lower()}_{fk_field_name}_set'",  # Ensure related_name is quoted string
                    }
                    # Generate the definition string
                    fk_def_string = generate_field_definition_string(models.ForeignKey, fk_kwargs, self.app_label)
                    # Add to the main definitions dictionary
                    carrier.django_field_definitions[fk_field_name] = fk_def_string

        # --- Prepare Final Context --- #
        # Ensure the context uses the potentially updated definitions dict from the carrier
        # Subclass _get_model_definition_extra_context should already provide this
        # via `field_definitions=carrier.django_field_definitions`
        template_context = {
            "model_name": django_model_name,
            "pydantic_model_name": source_model_name,
            "base_model_name": base_model_name,
            "is_timescale_model": bool(str(base_model_name).endswith("TimescaleBase")),
            "meta": meta_options,
            "app_label": self.app_label,
            "multi_fk_field_names": multi_fk_field_names,  # Pass names for validation hint
            "validation_needed": validation_needed,  # Signal if validation needed
            # Include extra context from subclass (should include field_definitions)
            **extra_context,
        }

        # --- Render Template --- #
        template = self.jinja_env.get_template("model_definition.py.j2")
        definition_str = template.render(**template_context)

        # Add import for the original source model
        self._add_source_model_import(carrier)

        return definition_str

    def _deduplicate_definitions(self, definitions: list[str]) -> list[str]:
        """Remove duplicate model definitions based on class name."""
        unique_definitions = []
        seen_class_names = set()
        for definition in definitions:
            # Basic regex to find 'class ClassName(' - might need adjustment for complex cases
            match = re.search(r"^\s*class\s+(\w+)\(", definition, re.MULTILINE)
            if match:
                class_name = match.group(1)
                if class_name not in seen_class_names:
                    unique_definitions.append(definition)
                    seen_class_names.add(class_name)
                # else: logger.debug(f"Skipping duplicate definition for class: {class_name}")
            else:
                # If no class definition found (e.g., comments, imports), keep it? Or discard?
                # For now, keep non-class definitions assuming they might be needed context/comments.
                unique_definitions.append(definition)
                logger.warning("Could not extract class name from definition block for deduplication.")

        return unique_definitions

    def _clean_generic_type(self, name: str) -> str:
        """Remove generic parameters like [T] or <T> from a type name."""
        # Handles Class[Param] or Class<Param>
        cleaned_name = re.sub(r"[\[<].*?[\]>]", "", name)
        # Also handle cases like 'ModelName.T' if typevars are used this way
        cleaned_name = cleaned_name.split(".")[-1]
        return cleaned_name

    def generate_models_file(self) -> str:
        """
        Generates the complete content for the models.py file.
        This method orchestrates discovery, model setup, definition generation,
        import collection, and template rendering.
        Subclasses might override this to add specific steps (like context class generation).
        """
        self.discover_models()  # Populates discovery instance
        models_to_process = self._get_models_in_processing_order()  # Abstract method

        # Reset state for this run
        self.carriers = []
        self.import_handler.extra_type_imports.clear()
        self.import_handler.pydantic_imports.clear()
        self.import_handler.context_class_imports.clear()
        self.import_handler.imported_names.clear()
        self.import_handler.processed_field_types.clear()

        # Re-add base model import after clearing
        self.import_handler._add_type_import(self.base_model_class)

        model_definitions = []
        django_model_names = []  # For __all__

        # Setup Django models first (populates self.carriers)
        for source_model in models_to_process:
            self.setup_django_model(source_model)  # Calls factory, populates carrier

        # Generate definitions from carriers
        for carrier in self.carriers:
            # Generate Django model definition if model exists
            if carrier.django_model:
                try:
                    model_def = self.generate_model_definition(carrier)  # Uses template
                    if model_def:  # Only add if definition was generated
                        model_definitions.append(model_def)
                        django_model_name = self._clean_generic_type(carrier.django_model.__name__)
                        django_model_names.append(f"'{django_model_name}'")
                except Exception as e:
                    source_name = self._get_source_model_name(carrier)
                    logger.error(f"Error generating definition for source model {source_name}: {e}", exc_info=True)

            # Subclasses might add context class generation here by overriding this method
            # or by generate_model_definition adding context-related imports.

        # Deduplicate definitions
        unique_model_definitions = self._deduplicate_definitions(model_definitions)

        # Deduplicate imports gathered during the process
        imports = self.import_handler.deduplicate_imports()

        # Prepare context using subclass method (_prepare_template_context)
        template_context = self._prepare_template_context(unique_model_definitions, django_model_names, imports)

        # Add common context items
        template_context.update(
            {
                "generation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "base_model_module": self.base_model_class.__module__,
                "base_model_name": self.base_model_class.__name__,
                "extra_type_imports": sorted(self.import_handler.extra_type_imports),
                # Add other common items as needed
            }
        )

        # Render the main template
        template = self.jinja_env.get_template("models_file.py.j2")
        return template.render(**template_context)
