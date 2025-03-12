import logging
import os
import pathlib
from abc import ABC
from collections.abc import Callable
from typing import Any, Optional, TypeVar, cast

import jinja2
from django.db import models
from pydantic import BaseModel

from pydantic2django.discovery import (
    discover_models,
    get_discovered_models,
    get_django_models,
    setup_dynamic_models,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("pydantic2django.generator")

T = TypeVar("T", bound=BaseModel)


# Define types for relationship fields
class RelationshipField(models.Field):
    """Base class for relationship fields to help with type checking."""

    to: Any
    related_name: Optional[str]


class ForeignKeyField(RelationshipField):
    """Type for ForeignKey fields to help with type checking."""

    on_delete: Any


class ManyToManyField(RelationshipField):
    """Type for ManyToManyField fields to help with type checking."""

    through: Any


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
        """
        Generate a Django model definition from a Django model class.
        Uses Jinja2 templating for cleaner code generation.
        """
        logger.info(f"Generating model definition for {model_name}")

        # Access _meta safely using getattr
        meta = getattr(django_model, "_meta", None)
        if meta is None:
            error_msg = f"Could not access _meta for {model_name}. This is required for model generation."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Get fields
        fields = []
        try:
            # Try to get fields from _meta
            if hasattr(meta, "fields"):
                for field in meta.fields:
                    # Skip the ID field since Pydantic2DjangoBaseClass already provides it
                    if field.name == "id" and not getattr(field, "primary_key", False):
                        continue
                    field_str = self.field_to_string(field)
                    fields.append((field.name, field_str))

                # Also get many-to-many fields
                if hasattr(meta, "many_to_many"):
                    for field in meta.many_to_many:
                        field_str = self.field_to_string(field)
                        fields.append((field.name, field_str))

            # If no fields were found, try to get them from the model's __dict__
            if not fields and hasattr(django_model, "__dict__"):
                for name, attr in django_model.__dict__.items():
                    if isinstance(attr, models.Field):
                        # Skip the ID field since Pydantic2DjangoBaseClass already provides it
                        if name == "id":
                            continue
                        field_str = self.field_to_string(attr)
                        fields.append((name, field_str))
        except Exception as e:
            error_msg = f"Error getting fields for {model_name}: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

        # If we have a Pydantic model, try to get fields from it
        if pydantic_model is not None and not fields:
            try:
                from pydantic2django.fields import convert_field, get_model_fields

                # Get fields from Pydantic model
                pydantic_fields = get_model_fields(pydantic_model)
                for name, field_info in pydantic_fields.items():
                    # Skip special fields and id field
                    if name.startswith("_") or name == "id":
                        continue

                    # Convert field
                    django_field = convert_field(
                        name,
                        field_info,
                        app_label=self.app_label,
                        model_name=model_name,
                    )

                    if django_field is not None:
                        field_str = self.field_to_string(django_field)
                        fields.append((name, field_str))
            except Exception as e:
                error_msg = f"Error converting Pydantic fields for {model_name}: {str(e)}"
                logger.error(error_msg)
                raise ValueError(error_msg) from e

        # If we still have no fields, raise an error
        if not fields:
            error_msg = f"No fields could be extracted for {model_name}. Cannot generate model definition."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Get original name (without Django prefix)
        original_name = model_name[6:] if model_name.startswith("Django") else model_name

        # Get verbose name and verbose name plural
        verbose_name = getattr(meta, "verbose_name", original_name)

        # For verbose_name_plural, don't just append 's' if it's already a sentence
        verbose_name_plural = getattr(meta, "verbose_name_plural", None)
        if verbose_name_plural is None:
            # If verbose_name contains spaces or punctuation, it's likely a sentence
            if any(c in verbose_name for c in " .,;:!?"):
                verbose_name_plural = verbose_name  # Don't append 's' to sentences
            else:
                verbose_name_plural = f"{verbose_name}s"  # Append 's' to simple names

        # Prepare meta data for the template
        meta_data = {
            "db_table": getattr(meta, "db_table", f"{original_name.lower()}"),
            "app_label": self.app_label,
            "verbose_name": verbose_name,
            "verbose_name_plural": verbose_name_plural,
        }

        # Get the module path for the Pydantic class
        module_path = ""
        if pydantic_model is not None:
            module_path = pydantic_model.__module__
        else:
            # Try to find the module path from the module mappings in the class
            module_mappings = getattr(self, "_module_mappings", {})
            if original_name in module_mappings:
                module_path = module_mappings.get(original_name, "")
            else:
                # As a fallback, check if there's a module path in the Meta class
                # First check PydanticConfig
                pydantic_config = getattr(django_model, "PydanticConfig", None)
                if pydantic_config and hasattr(pydantic_config, "module_path"):
                    module_path = pydantic_config.module_path
                # For backward compatibility, also check Meta
                elif hasattr(meta, "pydantic_module_path"):
                    module_path = meta.pydantic_module_path

        # Prepare template context
        template_context = {
            "model_name": model_name,
            "original_name": original_name,
            "fields": fields,
            "meta": meta_data,
            "module_path": module_path,
        }

        # Render the template
        try:
            template = self.jinja_env.get_template("model_definition.py.j2")
            return template.render(**template_context)
        except Exception as e:
            error_msg = f"Error rendering template for {model_name}: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

    def field_to_string(self, field: models.Field) -> str:
        """
        Convert a Django field to a string representation.
        """
        field_class = field.__class__.__name__
        kwargs = {}

        # Handle common field attributes
        if hasattr(field, "null") and field.null:
            kwargs["null"] = True
        else:
            kwargs["null"] = False

        if hasattr(field, "blank") and field.blank:
            kwargs["blank"] = True
        else:
            kwargs["blank"] = False

        # Handle default values - avoid using PydanticUndefined
        if hasattr(field, "default") and field.default != models.NOT_PROVIDED:
            # Skip defaults that would be PydanticUndefined
            if field.default is None:
                kwargs["default"] = None
            elif (
                isinstance(field.default, (int, float, bool, str))
                or field.default.__class__.__name__ != "UndefinedType"
            ):
                if isinstance(field.default, int | float | bool):
                    kwargs["default"] = field.default
                elif isinstance(field.default, str):
                    kwargs["default"] = f'"{field.default}"'
                else:
                    # For complex defaults, use repr
                    kwargs["default"] = repr(field.default)

        # Handle field-specific attributes
        if field_class == "CharField" or field_class == "TextField":
            if hasattr(field, "max_length") and field.max_length is not None:
                kwargs["max_length"] = field.max_length

        # Handle verbose_name
        if hasattr(field, "verbose_name") and field.verbose_name:
            kwargs["verbose_name"] = f'"{field.verbose_name}"'

        # Handle help_text
        if hasattr(field, "help_text") and field.help_text:
            kwargs["help_text"] = f'"{field.help_text}"'

        # Handle relationship fields
        if field_class in ["ForeignKey", "OneToOneField", "ManyToManyField"]:
            logger.info(f"  Handling relationship field: {field_class}")

            # Get the related model - safely handle all possible field structures
            related_model_name = None

            # Try different ways to get the related model
            if hasattr(field, "related_model") and field.related_model is not None:
                if isinstance(field.related_model, str):
                    related_model_name = field.related_model
                else:
                    try:
                        related_model_name = field.related_model.__name__
                    except (AttributeError, TypeError):
                        pass

            # Try remote_field.model if related_model didn't work
            if not related_model_name and hasattr(field, "remote_field") and field.remote_field is not None:
                remote_field = field.remote_field
                if hasattr(remote_field, "model") and remote_field.model is not None:
                    if isinstance(remote_field.model, str):
                        related_model_name = remote_field.model
                    else:
                        try:
                            related_model_name = remote_field.model.__name__
                        except (AttributeError, TypeError):
                            pass

            # Try to_field as a last resort
            if not related_model_name:
                # Cast the field to the appropriate type for type checking
                if field_class == "ForeignKey" or field_class == "OneToOneField":
                    rel_field = cast(ForeignKeyField, field)
                elif field_class == "ManyToManyField":
                    rel_field = cast(ManyToManyField, field)
                else:
                    rel_field = cast(RelationshipField, field)

                # Check if the field has a 'to' attribute
                if hasattr(rel_field, "to") and rel_field.to is not None:
                    if isinstance(rel_field.to, str):
                        related_model_name = rel_field.to
                    else:
                        try:
                            related_model_name = rel_field.to.__name__
                        except (AttributeError, TypeError):
                            pass

            # Set the 'to' parameter if we found a related model
            if related_model_name:
                # Check if it's a fully qualified name (with app_label)
                if "." in related_model_name:
                    kwargs["to"] = f'"{related_model_name}"'
                else:
                    # Assume it's in the same app
                    kwargs["to"] = f'"{self.app_label}.{related_model_name}"'
            else:
                # Default to a placeholder if we couldn't determine the related model
                kwargs["to"] = f'"{self.app_label}.UnknownModel"'

            # Handle on_delete for ForeignKey and OneToOneField
            if field_class in ["ForeignKey", "OneToOneField"]:
                # Default to CASCADE
                on_delete = "CASCADE"

                # Cast the field to ForeignKeyField for type checking
                fk_field = cast(ForeignKeyField, field)

                # Try to get the actual on_delete value
                if hasattr(fk_field, "on_delete") and fk_field.on_delete is not None:
                    if isinstance(fk_field.on_delete, str):
                        on_delete = fk_field.on_delete
                    else:
                        try:
                            on_delete = fk_field.on_delete.__name__
                        except (AttributeError, TypeError):
                            pass

                kwargs["on_delete"] = f"models.{on_delete}"

            # Handle related_name
            rel_field = cast(RelationshipField, field)
            if hasattr(rel_field, "related_name") and rel_field.related_name:
                kwargs["related_name"] = f'"{rel_field.related_name}"'

            # Handle through for ManyToManyField
            if field_class == "ManyToManyField":
                m2m_field = cast(ManyToManyField, field)
                if hasattr(m2m_field, "through") and m2m_field.through:
                    through_name = None
                    if isinstance(m2m_field.through, str):
                        through_name = m2m_field.through
                    else:
                        try:
                            through_name = m2m_field.through.__name__
                        except (AttributeError, TypeError):
                            pass

                    if through_name:
                        kwargs["through"] = f'"{through_name}"'

        # Format kwargs as string
        kwargs_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])

        # Return the field definition
        return f"models.{field_class}({kwargs_str})"

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
