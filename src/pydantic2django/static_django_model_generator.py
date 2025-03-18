import logging
import os
import pathlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, TypeVar

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
        self.relationship_accessor: RelationshipConversionAccessor = RelationshipConversionAccessor()
        self.field_factory = DjangoFieldFactory(available_relationships=self.relationship_accessor)
        self.django_model_factory = DjangoModelFactory(field_factory=self.field_factory)
        self.carriers: list[DjangoModelFactoryCarrier] = []
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

        # Get registration order
        models_in_registration_order = self.discovery.get_models_in_registration_order()

        # Generate model definitions and context classes in dependency order
        model_definitions = []
        context_definitions = []
        model_names = []

        # Process models in registration order
        for pydantic_model in models_in_registration_order:
            try:
                carrier = self.setup_django_model(pydantic_model)
                if carrier:
                    model_def, context_def = self.generate_definitions_from_carrier(carrier)
                    model_definitions.append(model_def)
                    context_definitions.append(context_def)
                    model_names.append(f"'{pydantic_model.__name__}'")
                else:
                    logger.warning(f"Skipping model {pydantic_model.__name__} due to errors")

            except Exception as e:
                logger.error(f"Error generating model definition for {pydantic_model.__name__}: {e}")
        # TODO: Need to go over generate definitions from carrier
        # Prepare imports
        imports = [
            "import uuid",
            "import importlib",
            "from typing import Any, Dict, List, Optional, Union, TypeVar",
            "from dataclasses import dataclass, field",
            "from pydantic2django.context_storage import ModelContext, FieldContext",
        ]

        # Render the models file template
        template = self.jinja_env.get_template("models_file.py.j2")
        return template.render(
            generation_timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            imports=imports,
            context_definitions=context_definitions,
            model_definitions=model_definitions,
            all_models=", ".join(model_names),
        )

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

        # Create Django models for each discovered model
        for model_name, pydantic_model in self.discovery.filtered_models.items():
            logger.info(f"Creating Django model for {model_name}")
            # Define field carrier
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
                    logger.exception(f"Error creating Django model for {model_name}: {factory_carrier.invalid_fields}")
                    return None
            except Exception as e:
                logger.error(f"Error creating Django model for {model_name}: {e}")

    def generate_definitions_from_carrier(self, carrier: DjangoModelFactoryCarrier) -> tuple[str, str]:
        if carrier.django_model and carrier.model_context:
            model_def = self.generate_model_definition(carrier.django_model)
            context_def = self.generate_context_class(carrier.model_context)
            return model_def, context_def
        else:
            return "", ""

    def generate_model_definition(self, model: type[models.Model]) -> str:
        """
        Generate a string representation of a Django model.

        Args:
            model: The Django model class

        Returns:
            String representation of the model
        """
        model_name = model.__name__
        original_name = model_name

        if model_name.startswith("Django"):
            original_name = model_name[6:]  # Remove "Django" prefix

        # Get fields from the model
        fields = []
        for field in model._meta.fields:
            # Skip the fields from Pydantic2DjangoBaseClass
            if field.name in ["id", "name", "object_type", "created_at", "updated_at"]:
                continue

            field_definition = self.generate_field_definition(field)
            fields.append((field.name, field_definition))

        # Get many-to-many fields safely
        try:
            if hasattr(model._meta, "many_to_many"):
                many_to_many = getattr(model._meta, "many_to_many", [])
                for field in many_to_many:
                    field_definition = self.generate_field_definition(field)
                    fields.append((field.name, field_definition))
        except Exception as e:
            logger.exception(f"Error processing many-to-many fields for {model_name}: {e}")

        # Get module path for the original Pydantic model
        module_path = ""
        try:
            if hasattr(model, "object_type"):
                object_type = getattr(model, "object_type", "")
                if object_type:
                    module_path = object_type.rsplit(".", 1)[0]
        except Exception as e:
            logger.exception(f"Error getting module path for {model_name}: {e}")

        # Prepare meta information
        meta = {
            "db_table": model._meta.db_table or f"{self.app_label}_{model_name.lower()}",
            "app_label": self.app_label,
            "verbose_name": model._meta.verbose_name or model_name,
            "verbose_name_plural": model._meta.verbose_name_plural or f"{model_name}s",
        }

        # Render the model definition template
        template = self.jinja_env.get_template("model_definition.py.j2")
        return template.render(
            model_name=model_name,
            fields=fields,
            meta=meta,
            module_path=module_path,
            original_name=original_name,
        )

    def generate_field_definition(self, field: models.Field) -> str:
        """
        Generate a string representation of a Django model field.

        Args:
            field: The Django model field

        Returns:
            String representation of the field
        """
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
        template = self.jinja_env.get_template("context_class.py.j2")

        # Prepare field definitions
        field_definitions = []
        for field_context in model_context.context_fields:
            field_def = {
                "name": field_context.field_name,
                "type": field_context.field_type.__name__,
                "is_optional": field_context.is_optional,
                "is_list": field_context.is_list,
                "metadata": field_context.additional_metadata,
            }
            field_definitions.append(field_def)

        return template.render(
            model_name=model_context.django_model.__name__,
            pydantic_class=model_context.pydantic_class.__name__,
            pydantic_module=model_context.pydantic_class.__module__,
            field_definitions=field_definitions,
        )


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
