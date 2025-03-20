import logging
import os
import pathlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, TypeVar

import jinja2
from django.db import models
from pydantic import BaseModel

from pydantic2django.context_storage import ContextClassGenerator, ModelContext, TypeHandler

# Import the base class for Django models
from pydantic2django.discovery import ModelDiscovery
from pydantic2django.factory import DjangoFieldFactory, DjangoModelFactory, DjangoModelFactoryCarrier
from pydantic2django.field_utils import FieldSerializer
from pydantic2django.relationships import RelationshipConversionAccessor

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("pydantic2django.generator")

T = TypeVar("T", bound=BaseModel)


@dataclass
class StaticGenerationComponents:
    """
    Components for the static generation of Django models.
    """

    django_model: str
    context_class: str


class StaticDjangoModelGenerator:
    """
    Generates Django models and their context classes from Pydantic models.
    """

    def __init__(
        self,
        output_path: str = "generated_models.py",
        packages: Optional[list[str]] = None,
        app_label: str = "django_app",
        filter_function: Optional[Callable[[type[BaseModel]], bool]] = None,
        verbose: bool = False,
        discovery_module: Optional[ModelDiscovery] = None,
    ):
        """
        Initialize the generator.

        Args:
            output_path: Path to output the generated models.py file
            packages: Packages to scan for Pydantic models
            app_label: Django app label to use for the models
            filter_function: Optional function to filter which models to include
            verbose: Print verbose output
            discovery_module: Optional ModelDiscovery instance to use
        """
        self.output_path = output_path
        self.packages = packages or ["pydantic_models"]
        self.app_label = app_label
        self.filter_function = filter_function
        self.verbose = verbose
        self.discovery = discovery_module or ModelDiscovery()
        self.relationship_accessor: RelationshipConversionAccessor = RelationshipConversionAccessor(
            dependencies=self.discovery.dependencies
        )
        self.field_factory = DjangoFieldFactory(available_relationships=self.relationship_accessor)
        self.django_model_factory = DjangoModelFactory(field_factory=self.field_factory)
        self.carriers: list[DjangoModelFactoryCarrier] = []

        # Track imports for models
        self.extra_type_imports: set[str] = set()
        self.pydantic_imports: set[str] = set()
        self.context_class_imports: set[str] = set()

        # Initialize Jinja2 environment
        # First look for templates in the package directory
        package_templates_dir = os.path.join(os.path.dirname(__file__), "templates")

        # If templates don't exist in the package, use the ones from the current directory
        if not os.path.exists(package_templates_dir):
            package_templates_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "templates")

        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(package_templates_dir),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def generate(self) -> str:
        """
        Main entry point for the generator.

        Generate and write the models file.

        Returns:
            The path to the generated models file
        """
        try:
            self.write_models_file()
            logger.info(f"Successfully generated models file at {self.output_path}")
            return self.output_path
        except Exception as e:
            logger.error(f"Error generating models file: {e}")
            raise

    def write_models_file(self) -> None:
        """
        Write the generated models to the output file.
        """
        if self.verbose:
            logger.info(f"Writing models to {self.output_path}")

        # Generate the file content
        content = self.generate_models_file()

        # Create directory if it doesn't exist
        output_dir = os.path.dirname(self.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Write the file
        with open(self.output_path, "w") as f:
            f.write(content)

        if self.verbose:
            logger.info(f"Successfully wrote models to {self.output_path}")

    def generate_models_file(self) -> str:
        """
        Generate the complete models.py file content.

        Returns:
            String content of the models.py file
        """
        # Discover and set up models
        self.discover_models()
        self.discovery.analyze_dependencies(self.app_label)

        # Clear import collections
        self.extra_type_imports = set()
        self.pydantic_imports = set()
        self.context_class_imports = set()

        # Tracking dictionaries to prevent duplicate imports
        pydantic_imported_names = {}
        context_field_imported_names = {}

        # Track all imported types to avoid duplicates across different import sections
        all_imported_types = set()

        # Get registration order
        models_in_registration_order = self.discovery.get_models_in_registration_order()

        # Generate model definitions and context classes in dependency order
        model_definitions = []
        context_definitions = []
        model_names = []
        django_model_names = []  # Track actual Django model names for __all__
        model_has_context = {}  # Track which models have context classes
        django_model_to_pydantic = {}  # Map Django model names to Pydantic model names
        context_class_names = []  # List to store context class names for __all__

        # Track models that only have context fields for logging
        context_only_models = []

        # Process models in registration order
        for pydantic_model in models_in_registration_order:
            try:
                carrier = self.setup_django_model(pydantic_model)
                if carrier:
                    # Skip models that have no Django fields (only context fields)
                    if not carrier.django_model:
                        context_only_models.append(pydantic_model.__name__)
                        logger.info(
                            f"Skipping model {pydantic_model.__name__} with only context fields, no Django fields"
                        )
                        continue

                    model_def, context_def = self.generate_definitions_from_carrier(carrier)
                    model_definitions.append(model_def)
                    model_name = pydantic_model.__name__
                    model_names.append(f"'{model_name}'")

                    # Add the Django model name (with its prefix) for __all__
                    if carrier.django_model:
                        django_model_name = carrier.django_model.__name__
                        # Clean up any parametrized generic types in model names for __all__
                        django_model_name = self._clean_generic_type(django_model_name)
                        django_model_names.append(f"'{django_model_name}'")
                        # Map Django model name to Pydantic model name
                        django_model_to_pydantic[django_model_name] = model_name

                    # Add import for the Pydantic model (avoid duplicates)
                    module_path = pydantic_model.__module__
                    # Clean up any parametrized generic types in model names for imports
                    cleaned_model_name = self._clean_generic_type(model_name)

                    if (
                        cleaned_model_name not in pydantic_imported_names
                        and cleaned_model_name not in all_imported_types
                    ):
                        self.pydantic_imports.add(f"from {module_path} import {cleaned_model_name}")
                        pydantic_imported_names[cleaned_model_name] = module_path
                        all_imported_types.add(cleaned_model_name)

                    # Check if this model has a non-empty context class
                    has_context = bool(context_def.strip())
                    model_has_context[model_name] = has_context

                    if has_context and carrier.model_context is not None:
                        context_definitions.append(context_def)
                        # Add context class name to __all__ list
                        if carrier.django_model:
                            context_class_name = f"{carrier.django_model.__name__}Context"
                            context_class_names.append(f"'{context_class_name}'")

                        # Track any special type imports needed for context fields
                        for field_context in carrier.model_context.context_fields:
                            field_type_str = str(field_context.field_type)

                            # Add import for context field type if it's a class
                            try:
                                if hasattr(field_context.field_type, "__module__") and hasattr(
                                    field_context.field_type, "__name__"
                                ):
                                    type_module = field_context.field_type.__module__
                                    type_name = field_context.field_type.__name__
                                    if not type_module.startswith("typing") and type_name not in [
                                        "str",
                                        "int",
                                        "float",
                                        "bool",
                                        "dict",
                                        "list",
                                    ]:
                                        # Clean up any parametrized generic types
                                        if "[" in type_name or "<" in type_name:
                                            type_name = re.sub(r"\[.*\]", "", type_name)

                                        # Avoid duplicate context field imports by checking if it's already imported
                                        # as a Pydantic model or other context field type
                                        if (
                                            type_name not in context_field_imported_names
                                            and type_name not in pydantic_imported_names
                                            and type_name not in all_imported_types
                                        ):
                                            self.context_class_imports.add(f"from {type_module} import {type_name}")
                                            context_field_imported_names[type_name] = type_module
                                            all_imported_types.add(type_name)
                            except (AttributeError, TypeError):
                                pass

                            # Use TypeHandler to analyze the field type and get required imports
                            field_imports = TypeHandler.get_required_imports(field_type_str)

                            # Add typing imports
                            for typing_import in field_imports["typing"]:
                                self.extra_type_imports.add(typing_import)

                            # Add custom type imports if they're not already imported
                            for custom_type in field_imports["custom"]:
                                # Skip if it's already in our imports
                                if (
                                    custom_type in context_field_imported_names
                                    or custom_type in pydantic_imported_names
                                    or custom_type in all_imported_types
                                ):
                                    continue

                                # Add to extra_type_imports if it can't be imported from a module
                                self.extra_type_imports.add(custom_type)

                            # Also handle is_optional and is_list flags
                            if field_context.is_optional:
                                self.extra_type_imports.add("Optional")
                            if field_context.is_list:
                                self.extra_type_imports.add("List")
                else:
                    logger.warning(f"Skipping model {pydantic_model.__name__} due to errors")

            except Exception as e:
                logger.error(f"Error generating model definition for {pydantic_model.__name__}: {e}")
                raise

        # Log summary of skipped models with only context fields
        if context_only_models:
            logger.info(
                f"Skipped {len(context_only_models)} models with only context fields: {', '.join(context_only_models)}"
            )

        # De-duplicate model definitions (remove repeated models like DjangoBaseGraph)
        unique_model_definitions = []
        seen_models = set()

        for model_def in model_definitions:
            match = re.search(r"class (\w+)\(", model_def)
            if match:
                model_class_name = match.group(1)
                if model_class_name not in seen_models:
                    seen_models.add(model_class_name)
                    unique_model_definitions.append(model_def)
            else:
                unique_model_definitions.append(model_def)

        # De-duplicate imports by combining them
        pydantic_and_context_imports = self._deduplicate_imports(self.pydantic_imports, self.context_class_imports)

        # Render the models file template
        template = self.jinja_env.get_template("models_file.py.j2")
        return template.render(
            generation_timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            context_definitions=context_definitions,
            model_definitions=unique_model_definitions,
            all_models=model_names,
            django_model_names=django_model_names,  # Pass actual Django model names
            context_class_names=context_class_names,  # Pass context class names
            model_has_context=model_has_context,
            extra_type_imports=sorted(self.extra_type_imports),
            pydantic_imports=sorted(pydantic_and_context_imports["pydantic"]),
            context_class_imports=sorted(pydantic_and_context_imports["context"]),
        )

    def _deduplicate_imports(self, pydantic_imports: set, context_imports: set) -> dict:
        """
        De-duplicate imports between Pydantic models and context field types.

        Args:
            pydantic_imports: Set of Pydantic import statements
            context_imports: Set of context field import statements

        Returns:
            Dict with de-duplicated import sets
        """
        # Extract class names and modules from import statements
        pydantic_classes = {}
        context_classes = {}

        for import_stmt in pydantic_imports:
            if import_stmt.startswith("from ") and " import " in import_stmt:
                module, classes = import_stmt.split(" import ")
                module = module.replace("from ", "")
                for cls in classes.split(", "):
                    # Clean up any parameterized generic types in class names
                    cls = self._clean_generic_type(cls)
                    pydantic_classes[cls] = module

        for import_stmt in context_imports:
            if import_stmt.startswith("from ") and " import " in import_stmt:
                module, classes = import_stmt.split(" import ")
                module = module.replace("from ", "")
                for cls in classes.split(", "):
                    # Clean up any parameterized generic types in class names
                    cls = self._clean_generic_type(cls)
                    # If this class is already imported in pydantic imports, skip it
                    if cls in pydantic_classes:
                        continue
                    context_classes[cls] = module

        # Rebuild import statements
        module_to_classes = {}
        for cls, module in pydantic_classes.items():
            if module not in module_to_classes:
                module_to_classes[module] = []
            module_to_classes[module].append(cls)

        deduplicated_pydantic_imports = set()
        for module, classes in module_to_classes.items():
            deduplicated_pydantic_imports.add(f"from {module} import {', '.join(sorted(classes))}")

        # Same for context imports
        module_to_classes = {}
        for cls, module in context_classes.items():
            if module not in module_to_classes:
                module_to_classes[module] = []
            module_to_classes[module].append(cls)

        deduplicated_context_imports = set()
        for module, classes in module_to_classes.items():
            deduplicated_context_imports.add(f"from {module} import {', '.join(sorted(classes))}")

        return {"pydantic": deduplicated_pydantic_imports, "context": deduplicated_context_imports}

    def discover_models(self) -> None:
        """
        Discover Pydantic models from the specified packages.

        Returns:
            Dict of discovered models
        """
        if self.verbose:
            logger.info(f"Discovering models from packages: {self.packages}")

        # Use the discovery module to find models
        self.discovery.discover_models(
            self.packages,
            app_label=self.app_label,
            filter_function=self.filter_function,
        )

        if self.verbose:
            logger.info(f"Discovered {len(self.discovery.filtered_models)} models")
            for name, model in self.discovery.filtered_models.items():
                logger.info(f"  - {name}: {model}")

    def setup_django_model(self, pydantic_model: type[BaseModel]) -> DjangoModelFactoryCarrier | None:
        """
        Set up a Django model from the discovered Pydantic model.

        For relationships to work, we need to make sure dependencies are processed first.

        Returns:
            Dict of Django models
        """

        logger.info(f"Setting models for {self.app_label}...")

        factory_carrier = DjangoModelFactoryCarrier(
            pydantic_model=pydantic_model,
            meta_app_label=self.app_label,
        )
        try:
            self.django_model_factory.make_django_model(
                carrier=factory_carrier,
            )
            if factory_carrier.django_model:
                # Map relationships
                self.relationship_accessor.map_relationship(
                    pydantic_model=pydantic_model, django_model=factory_carrier.django_model
                )
                logger.info(f"Successfully created Django model: {factory_carrier.django_model}")
                self.carriers.append(factory_carrier)
                return factory_carrier
            else:
                logger.exception(
                    f"Error creating Django model for {pydantic_model.__name__}: {factory_carrier.invalid_fields}"
                )
                return None
        except Exception as e:
            logger.error(f"Error creating Django model for {pydantic_model.__name__}: {e}")
            return None

    def generate_definitions_from_carrier(self, carrier: DjangoModelFactoryCarrier) -> tuple[str, str]:
        if carrier.django_model and carrier.model_context:
            model_def = self.generate_model_definition(carrier)
            # Only generate context class if there are context fields
            if carrier.model_context.context_fields:
                # Create a context generator to capture imports
                context_generator = ContextClassGenerator(jinja_env=self.jinja_env)
                # Generate the context class and capture imports in one step
                context_def = context_generator.generate_context_class(carrier.model_context)
                # Update our imports
                imports = context_generator.get_imports()
                self.extra_type_imports.update(imports["typing"])
                self.context_class_imports.update(imports["context"])
                return model_def, context_def
            else:
                return model_def, ""
        else:
            return "", ""

    def generate_model_definition(self, carrier: DjangoModelFactoryCarrier) -> str:
        """
        Generate a string representation of a Django model.

        Args:
            carrier: The carrier containing the Django model to generate

        Returns:
            String representation of the model
        """
        if carrier.django_model is None:
            logger.warning("Cannot generate model definition for None django_model")
            return ""

        model_name = carrier.django_model.__name__

        # Handle parametrized generic models by extracting the base name
        model_name = self._clean_generic_type(model_name)
        if "<" in carrier.django_model.__name__ or "[" in carrier.django_model.__name__:
            logger.info(f"Processing generic model: {model_name}")

        # Get fields from the model
        fields = []
        if hasattr(carrier.django_model, "_meta") and hasattr(carrier.django_model._meta, "fields"):
            for field in carrier.django_model._meta.fields:
                # Skip the fields from Pydantic2DjangoBaseClass
                if field.name in ["id", "name", "object_type", "object_type_field", "created_at", "updated_at"]:
                    continue

                field_definition = self.generate_field_definition(field)
                # Replace NOT_PROVIDED with null=True
                field_definition = field_definition.replace(
                    "default=<class 'django.db.models.fields.NOT_PROVIDED'>", "null=True"
                )
                # Handle default=True case that causes type errors
                field_definition = field_definition.replace("default=True", "default=True")
                fields.append((field.name, field_definition))

        # Get many-to-many fields safely
        try:
            if hasattr(carrier.django_model, "_meta"):
                meta = carrier.django_model._meta
                # Use safer getattr to avoid linter errors
                many_to_many = getattr(meta, "many_to_many", [])
                if many_to_many:
                    for field in many_to_many:
                        field_definition = self.generate_field_definition(field)
                        # Replace NOT_PROVIDED with null=True
                        field_definition = field_definition.replace(
                            "default=<class 'django.db.models.fields.NOT_PROVIDED'>", "null=True"
                        )
                        fields.append((field.name, field_definition))
        except Exception as e:
            logger.exception(f"Error processing many-to-many fields for {model_name}: {e}")

        # Get module path for the original Pydantic model
        module_path = ""
        try:
            if hasattr(carrier.django_model, "object_type"):
                object_type = getattr(carrier.django_model, "object_type", "")
                if object_type:
                    module_path = object_type.rsplit(".", 1)[0]
        except Exception as e:
            logger.exception(f"Error getting module path for {model_name}: {e}")

        # Prepare meta information
        meta = {
            "db_table": (
                getattr(carrier.django_model._meta, "db_table", None)
                if hasattr(carrier.django_model, "_meta")
                else None
            )
            or f"{self.app_label}_{model_name.lower()}",
            "app_label": self.app_label,
            "verbose_name": (
                getattr(carrier.django_model._meta, "verbose_name", None)
                if hasattr(carrier.django_model, "_meta")
                else None
            )
            or model_name,
            "verbose_name_plural": (
                getattr(carrier.django_model._meta, "verbose_name_plural", None)
                if hasattr(carrier.django_model, "_meta")
                else None
            )
            or f"{model_name}s",
        }

        # Add import for the original Pydantic model
        pydantic_model_name = carrier.pydantic_model.__name__
        # Clean up any generic parameters in the name
        pydantic_model_name = self._clean_generic_type(pydantic_model_name)

        module_path = carrier.pydantic_model.__module__
        self.pydantic_imports.add(f"from {module_path} import {pydantic_model_name}")

        # Extract context fields if they exist
        context_fields = []
        if carrier.model_context and carrier.model_context.context_fields:
            for field_context in carrier.model_context.context_fields:
                # Get a readable type name
                type_name = self._get_readable_type_name(field_context.field_type)
                # Create a tuple of (field_name, type_name)
                context_fields.append((field_context.field_name, type_name))

        # Render the model definition template
        template = self.jinja_env.get_template("model_definition.py.j2")
        return template.render(
            model_name=model_name,
            fields=fields,
            meta=meta,
            module_path=module_path,
            original_name=pydantic_model_name,
            context_fields=context_fields,
        )

    def _get_readable_type_name(self, field_type) -> str:
        """
        Get a readable name for a field type.

        Args:
            field_type: The field type to convert to a readable name

        Returns:
            A string representation of the type
        """
        # Handle common type patterns
        type_str = str(field_type)

        # If it has a __name__ attribute, use it
        if hasattr(field_type, "__name__"):
            type_name = field_type.__name__
        # Otherwise try to extract from string representation
        else:
            # Clean up the type string by removing angle brackets and quotes
            type_name = type_str.replace("<", "").replace(">", "").replace("'", "")
            # Take the last part if it's a module path
            if "." in type_name:
                type_name = type_name.split(".")[-1]

        # Special case for Callable
        if "Callable" in type_str:
            type_name = "Callable"

        # Handle Optional types
        if "Optional" in type_str:
            inner_match = re.search(r"Optional\[(.*?)\]", type_str)
            if inner_match:
                inner_type = inner_match.group(1)
                if "." in inner_type:
                    inner_type = inner_type.split(".")[-1]
                type_name = f"Optional[{inner_type}]"

        # Handle List types
        if "List" in type_str or "list" in type_str:
            inner_match = re.search(r"List\[(.*?)\]", type_str)
            if inner_match:
                inner_type = inner_match.group(1)
                if "." in inner_type:
                    inner_type = inner_type.split(".")[-1]
                type_name = f"List[{inner_type}]"

        return type_name

    def generate_field_definition(self, field: models.Field) -> str:
        """
        Generate a string representation of a Django model field.

        Args:
            field: The Django model field

        Returns:
            String representation of the field
        """
        # Just get the serialized field directly without modifying paths
        return FieldSerializer.serialize_field(field)

    def generate_context_class(self, model_context: ModelContext) -> str:
        """
        Generate a context class for a Django model.

        Args:
            model_context: The ModelContext instance for the model

        Returns:
            String representation of the context class
        """
        # This method is maintained for backward compatibility
        # but now just delegates to the ModelContext class method
        return ModelContext.generate_context_class_code(model_context, jinja_env=self.jinja_env)

    def _clean_generic_type(self, name: str) -> str:
        """
        Clean generic parameters from a type name.

        Args:
            name: The type name to clean

        Returns:
            The cleaned type name without generic parameters
        """
        if "[" in name or "<" in name:
            return re.sub(r"\[.*\]", "", name)
        return name

    def _maybe_add_type_to_imports(self, type_name: str) -> None:
        """
        Add a type to the import list if it's not already present.

        Args:
            type_name: The type name to add to the import list
        """
        if type_name not in self.extra_type_imports:
            self.extra_type_imports.add(type_name)


def main():
    """
    Command-line interface for the StaticDjangoModelGenerator.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Generate static Django models from Pydantic models")
    parser.add_argument("--output", "-o", default="generated_models.py", help="Output file path")
    parser.add_argument(
        "--packages",
        "-p",
        nargs="+",
        required=True,
        help="Packages to scan for Pydantic models",
    )
    parser.add_argument("--app-label", "-a", default="django_app", help="Django app label")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print verbose output")

    args = parser.parse_args()

    generator = StaticDjangoModelGenerator(
        output_path=args.output,
        packages=args.packages,
        app_label=args.app_label,
        verbose=args.verbose,
    )

    generator.generate()


if __name__ == "__main__":
    main()
