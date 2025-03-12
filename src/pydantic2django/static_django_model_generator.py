import logging
import os
import pathlib
from collections.abc import Callable
from datetime import datetime
from typing import Optional, TypeVar

import jinja2
from django.db import models
from pydantic import BaseModel

# Import the base class for Django models
from pydantic2django.discovery import (
    discover_models,
    get_discovered_models,
    setup_dynamic_models,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("pydantic2django.generator")

T = TypeVar("T", bound=BaseModel)


class StaticDjangoModelGenerator:
    """
    Generates Django models from Pydantic models.
    """

    def __init__(
        self,
        output_path: str = "generated_models.py",
        packages: Optional[list[str]] = None,
        app_label: str = "django_app",
        filter_function: Optional[Callable[[str, type[BaseModel]], bool]] = None,
        verbose: bool = False,
    ):
        """
        Initialize the generator.

        Args:
            output_path: Path to output the generated models.py file
            packages: Packages to scan for Pydantic models
            app_label: Django app label to use for the models
            filter_function: Optional function to filter which models to include
            verbose: Print verbose output
        """
        self.output_path = output_path
        self.packages = packages or ["pydantic_models"]
        self.app_label = app_label
        self.filter_function = filter_function
        self.verbose = verbose

        # Initialize Jinja2 environment
        # First look for templates in the package directory
        package_templates_dir = os.path.join(os.path.dirname(__file__), "templates")

        # If templates don't exist in the package, use the ones from the current directory
        if not os.path.exists(package_templates_dir):
            package_templates_dir = os.path.join(
                pathlib.Path(__file__).parent.absolute(), "templates"
            )

        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(package_templates_dir),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def discover_models(self) -> dict[str, type[BaseModel]]:
        """
        Discover Pydantic models from the specified packages.

        Returns:
            Dict of discovered models
        """
        if self.verbose:
            logger.info(f"Discovering models from packages: {self.packages}")

        # Use the discovery module to find models
        discover_models(
            self.packages,
            app_label=self.app_label,
            filter_function=self.filter_function,
        )

        # Get the discovered models
        discovered_models = get_discovered_models()

        if self.verbose:
            logger.info(f"Discovered {len(discovered_models)} models")
            for name, model in discovered_models.items():
                logger.info(f"  - {name}: {model}")

        return discovered_models

    def setup_django_models(self) -> dict[str, type[models.Model]]:
        """
        Set up Django models from the discovered Pydantic models.

        Returns:
            Dict of Django models
        """
        if self.verbose:
            logger.info("Setting up Django models...")

        # Use the discovery module to set up Django models
        django_models = setup_dynamic_models(app_label=self.app_label)

        if self.verbose:
            logger.info(f"Set up {len(django_models)} Django models")
            for name, model in django_models.items():
                logger.info(f"  - {name}: {model}")

        return django_models

    def generate_field_definition(self, field: models.Field) -> str:
        """
        Generate a string representation of a Django model field.

        Args:
            field: The Django model field

        Returns:
            String representation of the field
        """
        field_type = type(field).__name__

        # Start with the field type
        definition = f"models.{field_type}("

        # Add field parameters
        params = []

        # Common field parameters
        if hasattr(field, "verbose_name") and field.verbose_name:
            params.append(f"verbose_name='{field.verbose_name}'")

        if hasattr(field, "help_text") and field.help_text:
            params.append(f"help_text='{field.help_text}'")

        if hasattr(field, "null") and field.null:
            params.append(f"null={field.null}")

        if hasattr(field, "blank") and field.blank:
            params.append(f"blank={field.blank}")

        if hasattr(field, "default") and field.default != models.NOT_PROVIDED:
            if isinstance(field.default, str):
                params.append(f"default='{field.default}'")
            else:
                params.append(f"default={field.default}")

        # Field-specific parameters
        if isinstance(field, models.CharField) and hasattr(field, "max_length"):
            params.append(f"max_length={field.max_length}")

        if isinstance(field, models.DecimalField):
            if hasattr(field, "max_digits"):
                params.append(f"max_digits={field.max_digits}")
            if hasattr(field, "decimal_places"):
                params.append(f"decimal_places={field.decimal_places}")

        # Handle relationship fields
        if isinstance(field, models.ForeignKey):
            # Get the related model name safely
            try:
                related_model = field.related_model
                if isinstance(related_model, type):
                    related_model_name = related_model.__name__
                else:
                    related_model_name = str(related_model)
                params.append(f"to='{related_model_name}'")
            except Exception:
                # Fallback to a string representation
                params.append(
                    "to='self'"
                )  # Default to self-reference if we can't determine

            # Get the on_delete behavior safely
            try:
                on_delete = getattr(field, "on_delete", None)
                if on_delete and hasattr(on_delete, "__name__"):
                    params.append(f"on_delete=models.{on_delete.__name__}")
                else:
                    params.append("on_delete=models.CASCADE")  # Default to CASCADE
            except Exception:
                params.append("on_delete=models.CASCADE")  # Default to CASCADE

        if isinstance(field, models.ManyToManyField):
            # Get the related model name safely
            try:
                related_model = field.related_model
                if isinstance(related_model, type):
                    related_model_name = related_model.__name__
                else:
                    related_model_name = str(related_model)
                params.append(f"to='{related_model_name}'")
            except Exception:
                raise ValueError(f"Related model not found for {field}")
        # Join parameters and close the definition
        definition += ", ".join(params) + ")"

        return definition

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
            logger.exception(
                f"Error processing many-to-many fields for {model_name}: {e}"
            )

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
            "db_table": model._meta.db_table
            or f"{self.app_label}_{model_name.lower()}",
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

    def generate_models_file(self) -> str:
        """
        Generate the complete models.py file content.

        Returns:
            String content of the models.py file
        """
        # Discover and set up models
        self.discover_models()
        django_models = self.setup_django_models()

        # Generate model definitions
        model_definitions = []
        model_names = []

        for name, model in django_models.items():
            try:
                model_def = self.generate_model_definition(model)
                model_definitions.append(model_def)
                model_names.append(f"'{model.__name__}'")
            except Exception as e:
                logger.error(f"Error generating model definition for {name}: {e}")

        # Prepare imports
        imports = [
            "import uuid",
            "import importlib",
            "from typing import Any, Dict, List, Optional, Union",
        ]

        # Render the models file template
        template = self.jinja_env.get_template("models_file.py.j2")
        return template.render(
            generation_timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            imports=imports,
            model_definitions=model_definitions,
            all_models=", ".join(model_names),
        )

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

    def generate(self) -> str:
        """
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


def main():
    """
    Command-line interface for the StaticDjangoModelGenerator.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate static Django models from Pydantic models"
    )
    parser.add_argument(
        "--output", "-o", default="generated_models.py", help="Output file path"
    )
    parser.add_argument(
        "--packages",
        "-p",
        nargs="+",
        required=True,
        help="Packages to scan for Pydantic models",
    )
    parser.add_argument(
        "--app-label", "-a", default="django_app", help="Django app label"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print verbose output"
    )

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
