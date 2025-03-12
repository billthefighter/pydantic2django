import logging
import os
import pathlib
from abc import ABC
from collections.abc import Callable
from typing import Optional, TypeVar, cast

import jinja2
from django.db import models
from pydantic import BaseModel

from pydantic2django.discovery import (
    discover_models,
    get_discovered_models,
    get_django_models,
    setup_dynamic_models,
)
from pydantic2django.field_utils import (
    ForeignKeyField,
    ManyToManyField,
    RelationshipField,
    RelationshipFieldHandler,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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
        filter_function: Optional[Callable[[type[BaseModel]], bool]] = None,
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
        self.filter_function = filter_function or (lambda x: True)  # Default to include all models
        self.verbose = verbose

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

    def _should_convert_to_django_model(self, model_class: type[BaseModel]) -> bool:
        """
        Determine if a Pydantic model should be converted to a Django model.

        Args:
            model_class: The Pydantic model class to check

        Returns:
            True if the model should be converted, False otherwise
        """
        # Skip models that inherit from ABC
        if ABC in model_class.__mro__:
            if self.verbose:
                print(f"Skipping {model_class.__name__} because it inherits from ABC")
            return False

        return True

    def generate(self) -> Optional[str]:
        """
        Generate the models.py file content and write it to the output path.

        Returns:
            Path to the generated file or None if no models were generated
        """
        if self.verbose:
            print(f"Discovering models from packages: {', '.join(self.packages)}")

        # Discover models with the filter function
        adapted_filter = self._adapt_filter_function(self.filter_function)
        discover_models(
            self.packages,
            app_label=self.app_label,
            filter_function=adapted_filter,
        )

        # Get the discovered models (these should be only the ones that passed the filter)
        discovered_models = get_discovered_models()

        if self.verbose:
            print(f"Discovered {len(discovered_models)} models that match the filter:")
            for model_name in discovered_models.keys():
                print(f"  - {model_name}")

        # Setup dynamic models
        setup_dynamic_models()
        django_models = get_django_models()

        # Double-check that we only have models that passed our filter
        filtered_django_models = {}
        for model_name, django_model in django_models.items():
            # Extract the original Pydantic model name (remove the "Django" prefix)
            original_name = model_name[6:] if model_name.startswith("Django") else model_name

            # Check if the original model was in our discovered models
            if original_name in discovered_models or model_name in discovered_models:
                filtered_django_models[model_name] = django_model
            elif self.verbose:
                print(f"Skipping model {model_name} as it didn't pass the filter")

        if not filtered_django_models:
            error_msg = "No Django models were generated that match the filter. Check your filter function."
            if self.verbose:
                print(error_msg)
            raise ValueError(error_msg)

        if self.verbose:
            print(f"Generating {len(filtered_django_models)} Django models:")
            for model_name in filtered_django_models.keys():
                print(f"  - {model_name}")

        try:
            # Generate the models.py content
            content = self.generate_models_file(discovered_models, filtered_django_models)

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

            # Write to file
            with open(self.output_path, "w") as f:
                f.write(content)

            if self.verbose:
                print(f"Successfully generated models file at {self.output_path}")
                print(f"Generated {len(filtered_django_models)} model definitions")

            return self.output_path
        except Exception as e:
            error_msg = f"Failed to generate models file: {str(e)}"
            logger.error(error_msg)
            if self.verbose:
                print(f"ERROR: {error_msg}")
            raise ValueError(error_msg) from e

    def _adapt_filter_function(
        self, filter_func: Callable[[type[BaseModel]], bool]
    ) -> Callable[[str, type[BaseModel]], bool]:
        """
        Adapt a filter function that takes only a model to one that takes a name and a model.

        Args:
            filter_func: The filter function to adapt

        Returns:
            An adapted filter function that matches the expected signature
        """

        def combined_filter(name: str, model: type[BaseModel]) -> bool:
            # First check if it's an ABC - always skip these
            if ABC in model.__mro__:
                if self.verbose:
                    print(f"Skipping {name} ({model.__module__}.{model.__name__}) because it inherits from ABC")
                return False

            # Then apply the user's filter
            result = filter_func(model)
            if not result and self.verbose:
                print(f"Skipping {name} due to custom filter function")
            return result

        return combined_filter

    def generate_models_file(
        self,
        pydantic_models: dict[str, type[BaseModel]],
        django_models: dict[str, type[models.Model]],
    ) -> str:
        """
        Generate a Python file with Django model definitions.
        Uses Jinja2 templating for cleaner code generation.
        """
        from datetime import datetime

        # Generate module mappings and store as instance attribute
        self._module_mappings = self._generate_module_mappings(pydantic_models)

        # Generate model definitions
        model_definitions = []
        model_names = []
        errors = []

        for model_name, django_model in django_models.items():
            pydantic_model = pydantic_models.get(model_name[6:]) if model_name.startswith("Django") else None
            try:
                model_def = self.generate_model_definition(model_name, django_model, pydantic_model)
                model_definitions.append(model_def)
                model_names.append(f'"{model_name}"')
            except Exception as e:
                error_message = f"# Error generating model {model_name}: {str(e)}"
                logger.error(error_message)
                errors.append((model_name, str(e)))
                # Add a placeholder class with error comment
                model_definitions.append(
                    f"""class {model_name}(Pydantic2DjangoBaseClass):
    # Error generating model: {str(e)}
    pass
"""
                )
                model_names.append(f'"{model_name}"')

        # If we have errors, raise an exception after collecting all errors
        if errors:
            error_summary = "\n".join([f"- {name}: {error}" for name, error in errors])
            error_msg = f"Failed to generate {len(errors)} model(s) out of {len(django_models)}:\n{error_summary}"
            logger.error(error_msg)

            # Decide whether to continue or raise an exception based on a threshold
            if len(errors) / len(django_models) > 0.5:  # If more than 50% of models failed
                raise ValueError(f"Too many model generation errors: {error_msg}")
            else:
                logger.warning("Continuing with partial model generation despite errors")

        # Custom imports for the template
        custom_imports = [
            "from pydantic import BaseModel",
            "from typing import Any, Optional, Dict, List, Union",
        ]

        # Prepare template context
        template_context = {
            "generation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "imports": custom_imports,
            "module_mappings": self._module_mappings,
            "model_definitions": model_definitions,
            "all_models": ", ".join(model_names),
        }

        # Render the template
        try:
            template = self.jinja_env.get_template("models_file.py.j2")
            rendered = template.render(**template_context)

            # Replace PydanticUndefined with a proper implementation
            rendered = rendered.replace(
                "PydanticUndefined = None",
                """# Define a proper undefined type that won't cause issues with Django
class UndefinedType:
    def __repr__(self):
        return "Undefined"

    def __bool__(self):
        return False

PydanticUndefined = UndefinedType()""",
            )

            return rendered
        except Exception as e:
            logger.error(f"Error rendering template for models_file.py.j2: {str(e)}")
            raise ValueError(f"Failed to render models file template: {str(e)}") from e

    def generate_model_definition(
        self,
        model_name: str,
        django_model: type[models.Model],
        pydantic_model: Optional[type[BaseModel]] = None,
    ) -> str:
        """Generate a Django model definition from a Pydantic model."""
        # Get fields from Django model
        fields = []
        
        for field in django_model._meta.fields:
            if field.name not in ["id", "name", "object_type", "created_at", "updated_at"]:
                field_str = self.field_to_string(field)
                fields.append((field.name, field_str))

        # If no fields were found in the Django model, try to get them from the Pydantic model
        if pydantic_model is not None and not fields:
            try:
                from pydantic2django.fields import convert_field, get_model_fields

                # Get fields from Pydantic model
                pydantic_fields = get_model_fields(pydantic_model)
                for field_name, field_info in pydantic_fields.items():
                    django_field = convert_field(field_name, field_info)
                    fields.append((field_name, self.field_to_string(django_field)))
            except ImportError:
                self._log(f"Could not import pydantic2django.fields, skipping field conversion for {model_name}")
            except Exception as e:
                self._log(f"Error converting fields for {model_name}: {e}")

        # Get meta information
        meta = {
            "db_table": django_model._meta.db_table,
            "app_label": django_model._meta.app_label,
            "verbose_name": self._clean_docstring(getattr(django_model._meta, "verbose_name", model_name)),
            "verbose_name_plural": self._clean_docstring(getattr(django_model._meta, "verbose_name_plural", f"{model_name}s")),
        }

        # Get module path from PydanticConfig or from the pydantic_model
        module_path = ""
        if hasattr(django_model, "PydanticConfig") and hasattr(django_model.PydanticConfig, "module_path"):
            module_path = django_model.PydanticConfig.module_path
        elif pydantic_model is not None:
            # Use the module path from the Pydantic model
            module_path = pydantic_model.__module__
            
        # Validate module path - it should not be empty
        if not module_path:
            # If we can't determine the module path, use a placeholder that will raise an error
            # This ensures the issue is visible rather than silently passing through
            module_path = "UNKNOWN_MODULE_PATH"
            self._log(f"Warning: Could not determine module_path for {model_name}. Using placeholder that will raise an error.")

        # Get original name (without Django prefix)
        original_name = model_name
        if model_name.startswith("Django"):
            original_name = model_name[6:]

        # Render the template
        template = self.jinja_env.get_template("model_definition.py.j2")
        return template.render(
            model_name=model_name,
            fields=fields,
            meta=meta,
            module_path=module_path,
            original_name=original_name,
        )

    def _clean_docstring(self, text: str) -> str:
        """Clean a docstring for use in a model definition."""
        if not text:
            return ""
        
        # Take only the first line of the docstring
        first_line = text.split("\n")[0].strip()
        return first_line

    def field_to_string(self, field: models.Field) -> str:
        """Convert a Django field to its string representation."""
        field_type = field.__class__.__name__
        kwargs = {}

        # Handle common field attributes
        if field.verbose_name and field.verbose_name != field.name:
            kwargs["verbose_name"] = f'"{field.verbose_name}"'

        if field.primary_key:
            kwargs["primary_key"] = "True"

        if field.blank:
            kwargs["blank"] = "True"

        if field.null:
            kwargs["null"] = "True"

        # Handle field-specific attributes
        if hasattr(field, "max_length") and field.max_length:
            kwargs["max_length"] = str(field.max_length)

        if hasattr(field, "help_text") and field.help_text:
            kwargs["help_text"] = f'"{field.help_text}"'

        # Handle default values - this is a critical part for primitive types
        if hasattr(field, "default") and field.default != models.NOT_PROVIDED:
            # Skip adding default for UndefinedType
            if hasattr(field.default, "__class__") and getattr(field.default.__class__, "__name__", "") == "UndefinedType":
                # Don't add default for UndefinedType
                pass
            # Handle primitive types properly
            elif isinstance(field.default, bool):
                # Use the actual value without quotes for booleans
                kwargs["default"] = str(field.default)
            elif isinstance(field.default, (int, float)):
                # Use the actual value without quotes for numbers
                kwargs["default"] = str(field.default)
            elif isinstance(field.default, str):
                # Use quotes for strings
                kwargs["default"] = f'"{field.default}"'
            elif field.default is None:
                # Use None for null values
                kwargs["default"] = "None"
            else:
                # For other types, convert to string but don't add quotes
                # This might not work for all types, but it's a reasonable default
                try:
                    kwargs["default"] = str(field.default)
                except Exception:
                    # If we can't convert to string, skip the default
                    pass

        # Handle on_delete for ForeignKey and OneToOneField
        if field_type in ["ForeignKey", "OneToOneField"]:
            kwargs["on_delete"] = "models.CASCADE"  # Default to CASCADE
            if hasattr(field, "remote_field") and hasattr(field.remote_field, "on_delete"):
                on_delete_name = field.remote_field.on_delete.__name__
                kwargs["on_delete"] = f"models.{on_delete_name}"

        # Handle to for ForeignKey, OneToOneField, and ManyToManyField
        if field_type in ["ForeignKey", "OneToOneField", "ManyToManyField"]:
            if hasattr(field, "remote_field") and hasattr(field.remote_field, "model"):
                to_model = field.remote_field.model
                if isinstance(to_model, str):
                    kwargs["to"] = f'"{to_model}"'
                else:
                    to_model_name = to_model.__name__
                    kwargs["to"] = f'"{to_model_name}"'

        # Format kwargs as a string
        kwargs_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
        return f"models.{field_type}({kwargs_str})"

    def _generate_module_mappings(self, pydantic_models: dict[str, type[BaseModel]]) -> dict[str, str]:
        """
        Generate a mapping of model names to their module paths.

        Args:
            pydantic_models: Dict of discovered Pydantic models

        Returns:
            Dict mapping model names to their module paths
        """
        mappings = {}
        for model_name, model_class in pydantic_models.items():
            # Get the original name (without Django prefix)
            original_name = model_name[6:] if model_name.startswith("Django") else model_name

            # Get the module path
            module_path = model_class.__module__

            # Add to mappings
            mappings[original_name] = module_path

            if self.verbose:
                print(f"Mapping {original_name} to {module_path}")

        return mappings
