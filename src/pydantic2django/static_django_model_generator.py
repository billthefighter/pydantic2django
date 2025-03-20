import logging
import os
import pathlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, TypeVar, get_args

import jinja2
from django.db import models
from pydantic import BaseModel

from pydantic2django.context_storage import ModelContext

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

                            if "Optional[" in field_type_str or field_context.is_optional:
                                self.extra_type_imports.add("Optional")
                            if "List[" in field_type_str or field_context.is_list:
                                self.extra_type_imports.add("List")
                            if "Dict[" in field_type_str:
                                self.extra_type_imports.add("Dict")
                            if "Union[" in field_type_str:
                                self.extra_type_imports.add("Union")
                            if "Type[" in field_type_str:
                                self.extra_type_imports.add("Type")

                            # Add explicit imports for TypeVars or custom types in Type expressions
                            # Look for Type[SomeTypeName] patterns and extract SomeTypeName
                            type_var_match = re.search(r"Type\[([\w.]+)\]", field_type_str)
                            if type_var_match:
                                inner_type = type_var_match.group(1)
                                if "." not in inner_type and not inner_type.startswith(
                                    ("str", "int", "float", "bool", "list", "dict", "Any")
                                ):
                                    # This looks like a TypeVar or other named type, add it to imports
                                    self.extra_type_imports.add(inner_type)

                                    # For TypeVars commonly used in Generic classes, add TypeVar import
                                    if re.match(r"^[A-Z][A-Za-z0-9_]*$", inner_type) and not inner_type.endswith(
                                        ("Type", "Any", "Dict", "List")
                                    ):
                                        self.extra_type_imports.add("TypeVar")

                            # Check for Generic usage
                            if "Generic[" in field_type_str:
                                self.extra_type_imports.add("Generic")
                                # Extract the TypeVars used in Generic
                                generic_match = re.search(r"Generic\[([\w, ]+)\]", field_type_str)
                                if generic_match:
                                    type_vars = generic_match.group(1).split(",")
                                    for tv in type_vars:
                                        tv = tv.strip()
                                        if tv and not tv.startswith(
                                            ("str", "int", "float", "bool", "list", "dict", "Any")
                                        ):
                                            self.extra_type_imports.add(tv)
                                            self.extra_type_imports.add("TypeVar")
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
                context_def = self.generate_context_class(carrier.model_context)
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
            model: The Django model class
            model_context: The ModelContext instance for the model

        Returns:
            String representation of the context class
        """
        if model_context is None or not hasattr(model_context, "django_model") or model_context.django_model is None:
            logger.warning("Cannot generate context class for None model_context or missing django_model")
            return ""

        # Skip generating context class if there are no context fields
        if not model_context.context_fields:
            logger.info(f"Skipping context class for {model_context.django_model.__name__} - no context fields")
            return ""

        template = self.jinja_env.get_template("context_class.py.j2")

        # Prepare field definitions
        field_definitions = []
        if hasattr(model_context, "context_fields"):
            for field_context in model_context.context_fields:
                # Detect Callable types
                type_str = str(field_context.field_type)

                # Check for Callable types
                if "Callable" in type_str or "typing.Callable" in type_str:
                    type_name = "Callable"
                    self.extra_type_imports.add("Callable")
                # Handle typing module references with unclosed brackets
                elif "typing." in type_str and ("[" in type_str and type_str.count("[") > type_str.count("]")):
                    # Extract the base typing construct (Type, Optional, etc)
                    typing_construct = type_str.split("typing.", 1)[1].split("[", 1)[0]

                    # Add the rest of the content, making sure brackets are balanced
                    if "[" in type_str:
                        # Extract everything after the first opening bracket
                        content_part = type_str.split(f"typing.{typing_construct}[", 1)[1]

                        # Clean up tildes and ensure brackets are balanced
                        content_part = re.sub(r"~", "", content_part)

                        # Add any missing closing brackets
                        if content_part.count("[") > content_part.count("]"):
                            content_part += "]" * (content_part.count("[") - content_part.count("]"))
                        # If no brackets at all, add a closing bracket
                        elif "]" not in content_part:
                            content_part += "]"

                        # Process module paths in the content
                        content_part = self._handle_type_with_module_path(content_part)

                        type_name = f"{typing_construct}[{content_part}"
                    else:
                        type_name = typing_construct

                    # Add import for the typing construct
                    self.extra_type_imports.add(typing_construct)
                # Handle typing.Type expressions specifically
                elif "typing.Type[" in type_str:
                    # Extract the content inside the brackets using a more robust approach
                    type_content = type_str.split("typing.Type[", 1)[1]

                    # Find the correct closing bracket by tracking bracket depth
                    bracket_depth = 1
                    content_end = 0
                    for i, char in enumerate(type_content):
                        if char == "[":
                            bracket_depth += 1
                        elif char == "]":
                            bracket_depth -= 1
                            if bracket_depth == 0:
                                content_end = i
                                break

                    # If we didn't find a closing bracket, use all content and fix later
                    if content_end == 0:
                        content_end = len(type_content)
                        if not type_content.endswith("]"):
                            type_content += "]"

                    # Extract the inner type and clean it
                    inner_type = type_content[:content_end]

                    # Clean up the inner type - remove module prefixes, tildes, etc.
                    inner_type = re.sub(r"~", "", inner_type)

                    # Handle module paths in the inner type
                    inner_type = self._handle_type_with_module_path(inner_type)

                    # Check if this is a TypeVar and we need to import it separately
                    if not inner_type.startswith(("str", "int", "float", "bool", "dict", "list", "Any")):
                        # This could be a TypeVar or other custom type - try to add it to imports
                        self._maybe_add_type_to_imports(inner_type)

                        # For TypeVars commonly used in Generic classes, add TypeVar import
                        if re.match(r"^[A-Z][A-Za-z0-9_]*$", inner_type) and not inner_type.endswith(
                            ("Type", "Any", "Dict", "List")
                        ):
                            self.extra_type_imports.add("TypeVar")

                    type_name = f"Type[{inner_type}]"
                    self.extra_type_imports.add("Type")
                # Handle Type[] expressions without typing. prefix
                elif type_str.startswith("Type[") and "[" in type_str:
                    # Extract the content inside the brackets using a more robust approach
                    type_content = type_str.split("Type[", 1)[1]

                    # Find the correct closing bracket by tracking bracket depth
                    bracket_depth = 1
                    content_end = 0
                    for i, char in enumerate(type_content):
                        if char == "[":
                            bracket_depth += 1
                        elif char == "]":
                            bracket_depth -= 1
                            if bracket_depth == 0:
                                content_end = i
                                break

                    # If we couldn't find a closing bracket, use the whole content
                    if content_end == 0:
                        content_end = len(type_content)
                        # Add closing bracket if needed
                        if not type_content.endswith("]"):
                            type_content += "]"

                    # Extract the inner type and clean it
                    inner_type = type_content[:content_end]

                    # Clean up the inner type - remove module prefixes, tildes, etc.
                    inner_type = re.sub(r"~", "", inner_type)

                    # Handle module paths in the inner type
                    inner_type = self._handle_type_with_module_path(inner_type)

                    # Check if this is a TypeVar and we need to import it separately
                    if not inner_type.startswith(("str", "int", "float", "bool", "dict", "list", "Any")):
                        # This could be a TypeVar or other custom type - try to add it to imports
                        self._maybe_add_type_to_imports(inner_type)

                        # For TypeVars commonly used in Generic classes, add TypeVar import
                        if re.match(r"^[A-Z][A-Za-z0-9_]*$", inner_type) and not inner_type.endswith(
                            ("Type", "Any", "Dict", "List")
                        ):
                            self.extra_type_imports.add("TypeVar")

                    type_name = f"Type[{inner_type}]"
                    self.extra_type_imports.add("Type")
                # Handle TypeVar instances with tilde notation (Type[~TypeVarName])
                elif "~" in type_str or "TypeVar" in type_str:
                    # First, try to get the module and base type name if available
                    if hasattr(field_context.field_type, "__module__") and hasattr(
                        field_context.field_type, "__name__"
                    ):
                        module_name = field_context.field_type.__module__
                        type_name = field_context.field_type.__name__
                        # If it's a typing.Type[...], extract just "Type"
                        if module_name == "typing" and type_name == "Type":
                            # Try to get the base bound class of the TypeVar
                            args = get_args(field_context.field_type)
                            if args and len(args) > 0:
                                if hasattr(args[0], "__bound__") and args[0].__bound__:
                                    # Get the bound class of the TypeVar
                                    bound_class = args[0].__bound__
                                    if hasattr(bound_class, "__name__"):
                                        type_name = f"Type[{bound_class.__name__}]"
                                    else:
                                        type_name = "Type"
                                else:
                                    # Use simple Type if we can't extract the bound
                                    type_name = "Type"
                            else:
                                type_name = "Type"
                        else:
                            # Use the type name directly
                            type_name = type_name
                    else:
                        # Clean up the type string, remove tildes and fix unclosed brackets
                        type_name = re.sub(r"~", "", type_str)
                        # Make sure brackets are balanced
                        if "[" in type_name and "]" not in type_name:
                            type_name = type_name + "]"
                        # Fix "Type[Type[X]]" issues
                        if "Type[Type[" in type_name:
                            type_name = re.sub(r"Type\[Type\[([^\]]+)\]\]", r"Type[\1]", type_name)

                    # Make sure we import Type if using it
                    if "Type" in type_name:
                        self.extra_type_imports.add("Type")
                # Handle Optional types better by examining the string representation
                elif "Optional" in type_str and field_context.is_optional:
                    # Extract what's inside Optional[...]
                    inner_match = re.search(r"Optional\[(.*?)\]", type_str)
                    if inner_match:
                        inner_type = inner_match.group(1)
                        # Remove any nested Optional
                        if inner_type.startswith("Optional["):
                            inner_match2 = re.search(r"Optional\[(.*?)\]", inner_type)
                            if inner_match2:
                                inner_type = inner_match2.group(1)
                        type_name = inner_type
                    else:
                        # Just use a simple type name if we can't parse it
                        if hasattr(field_context.field_type, "__name__"):
                            type_name = field_context.field_type.__name__
                        else:
                            type_name = "Any"
                            self.extra_type_imports.add("Any")
                # Default case - use the type's name if available
                elif hasattr(field_context.field_type, "__name__"):
                    type_name = field_context.field_type.__name__
                # Last resort - use string representation
                else:
                    # Start by replacing angle brackets with square brackets
                    type_name = type_str.replace("<", "[").replace(">", "]")

                    # Fix "Optional[Optional]" issues
                    if "Optional[Optional]" in type_name:
                        type_name = type_name.replace("Optional[Optional]", "Optional")

                    # Handle module paths in the type string
                    type_name = self._handle_type_with_module_path(type_name)

                    # Fix any unclosed brackets
                    if type_name.count("[") > type_name.count("]"):
                        type_name += "]" * (type_name.count("[") - type_name.count("]"))

                # Clean up any remaining angle brackets that could cause rendering issues
                type_name = type_name.replace("<", "[").replace(">", "]")
                # Remove any remaining tildes
                type_name = type_name.replace("~", "")

                # Handle any special characters that might cause formatting issues
                type_name = type_name.strip().replace(",", ", ")

                # Check for any type names that should be imported
                # This catches types that weren't handled by the specific cases above
                self._extract_import_types_from_string(type_name)

                # Ensure metadata is a dict and doesn't contain problematic characters
                metadata = {}
                if field_context.additional_metadata:
                    for k, v in field_context.additional_metadata.items():
                        if isinstance(v, str):
                            metadata[k] = v.replace("\n", " ").replace("\r", "")
                        else:
                            metadata[k] = v

                field_def = {
                    "name": field_context.field_name,
                    "type": type_name,
                    "is_optional": field_context.is_optional,
                    "is_list": field_context.is_list,
                    "metadata": metadata,
                }
                field_definitions.append(field_def)

        return template.render(
            model_name=self._clean_generic_type(model_context.django_model.__name__),
            pydantic_class=self._clean_generic_type(model_context.pydantic_class.__name__),
            pydantic_module=model_context.pydantic_class.__module__,
            field_definitions=field_definitions,
        )

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

    def _handle_type_with_module_path(self, type_str: str) -> str:
        """
        Extract just the class name from a fully qualified module path.

        Args:
            type_str: A string containing a module path like 'module.submodule.ClassName'

        Returns:
            Just the class name portion, e.g. 'ClassName'
        """
        # For simple cases, just take the last part after the dot
        if "." in type_str and not ("[" in type_str or "<" in type_str):
            return type_str.split(".")[-1]

        # For complex cases with brackets, we need to be more careful
        if "[" in type_str or "<" in type_str:
            # Replace angle brackets with square brackets for consistency
            type_str = type_str.replace("<", "[").replace(">", "]")

            # Extract the main type and its parameters
            bracket_pos = type_str.find("[")
            if bracket_pos > 0:
                main_type = type_str[:bracket_pos]
                params = type_str[bracket_pos:]

                # Get just the class name from the main type
                if "." in main_type:
                    main_type = main_type.split(".")[-1]

                # For each parameter with module path, recursively extract the class name
                if "." in params:
                    # This is a complex case with nested module paths in parameters
                    # Parse the parameters section, ensuring we respect bracket nesting
                    processed_params = ""
                    current_token = ""
                    bracket_depth = 0

                    for char in params:
                        if char == "[":
                            bracket_depth += 1
                            processed_params += char
                        elif char == "]":
                            bracket_depth -= 1
                            processed_params += char
                        elif char == "," and bracket_depth == 1:
                            # At top level parameter separator, process the token
                            if "." in current_token:
                                processed_params += current_token.split(".")[-1] + ","
                            else:
                                processed_params += current_token + ","
                            current_token = ""
                        elif bracket_depth >= 1:
                            current_token += char
                        else:
                            processed_params += char

                    # Handle the last token if any
                    if current_token and "." in current_token:
                        processed_params = processed_params[:-1] + current_token.split(".")[-1] + "]"

                    return main_type + processed_params

                return main_type + params

        # Fall back to the original string if we couldn't parse it
        return type_str

    def _maybe_add_type_to_imports(self, type_name: str) -> None:
        """
        Add a type to the import list if it's not already present.

        Args:
            type_name: The type name to add to the import list
        """
        if type_name not in self.extra_type_imports:
            self.extra_type_imports.add(type_name)

    def _extract_import_types_from_string(self, type_name: str) -> None:
        """
        Extract and add type names from a string representation.

        Args:
            type_name: The type name to extract from
        """
        # Handle common type patterns
        type_str = str(type_name)

        # Create a clean type name for importing
        # Take the last part if it's a module path
        if "." in type_str:
            clean_type_name = type_str.split(".")[-1]
        else:
            clean_type_name = type_str

        # Remove any generic parameters
        if "[" in clean_type_name:
            clean_type_name = clean_type_name.split("[")[0]

        # Clean up the type name
        clean_type_name = clean_type_name.strip()

        # Special case for Callable
        if "Callable" in type_str:
            clean_type_name = "Callable"

        # Add the extracted type name to the import list if it's not a basic Python type
        if (
            clean_type_name
            and not clean_type_name.startswith(("str", "int", "float", "bool", "dict", "list", "None", "Any"))
            and not clean_type_name.startswith(("Optional", "List", "Dict", "Union", "Tuple"))
        ):
            self._maybe_add_type_to_imports(clean_type_name)


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
