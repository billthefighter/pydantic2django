import os
from abc import ABC
from collections.abc import Callable
from typing import Optional, TypeVar, Dict, List, Any, Union, Type
import inspect
import importlib
import pkgutil
import sys
import pathlib

from django.db import models
from pydantic import BaseModel
import jinja2

# Try to import the real discovery module, but fall back to the mock if it's not available
try:
    from pydantic2django.discovery import (
        discover_models,
        get_discovered_models,
        get_django_models,
        setup_dynamic_models,
    )
except ImportError:
    # Use the mock discovery module if the real one is not available
    from pydantic2django.mock_discovery import (
        discover_models,
        get_discovered_models,
        get_django_models,
        setup_dynamic_models,
    )


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
            if self.verbose:
                print("No Django models were generated that match the filter. Check your filter function.")
            return None

        if self.verbose:
            print(f"Generating {len(filtered_django_models)} Django models:")
            for model_name in filtered_django_models.keys():
                print(f"  - {model_name}")

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
        
        # Generate module mappings
        module_mappings = self._generate_module_mappings(pydantic_models)
        
        # Generate model definitions
        model_definitions = []
        model_names = []
        
        for model_name, django_model in django_models.items():
            pydantic_model = pydantic_models.get(model_name[6:]) if model_name.startswith("Django") else None
            model_def = self.generate_model_definition(model_name, django_model, pydantic_model)
            model_definitions.append(model_def)
            model_names.append(f'"{model_name}"')
        
        # Prepare template context
        template_context = {
            "generation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "imports": [],  # Add any additional imports here if needed
            "module_mappings": module_mappings,
            "model_definitions": model_definitions,
            "all_models": ", ".join(model_names)
        }
        
        # Render the template
        try:
            template = self.jinja_env.get_template("models_file.py.j2")
            return template.render(**template_context)
        except jinja2.exceptions.TemplateNotFound:
            if self.verbose:
                print("Template 'models_file.py.j2' not found. Using fallback template.")
            # Fallback to a simple template if the file doesn't exist
            return self._generate_fallback_models_file(template_context)

    def _generate_fallback_models_file(self, context: dict) -> str:
        """Generate a fallback models file when the template is not found."""
        lines = [
            "# This file was generated by pydantic2django",
            "# Do not edit this file manually",
            f"# Generated on {context['generation_timestamp']}",
            "",
            "from django.db import models",
            "from pydantic2django import Pydantic2DjangoBaseClass",
            "",
            "def get_module_path(object_type: str) -> str:",
            "    # Map object types to their module paths",
            "    module_mappings = {",
        ]
        
        # Add module mappings
        for model_name, module_path in context["module_mappings"].items():
            lines.append(f'        "{model_name}": "{module_path}",')
        
        lines.append("    }")
        lines.append('    return module_mappings.get(object_type, "")')
        lines.append("")
        
        # Add model definitions
        for model_def in context["model_definitions"]:
            lines.append(model_def)
            lines.append("")
        
        # Add __all__
        lines.append(f"__all__ = [{context['all_models']}]")
        
        return "\n".join(lines)

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
        meta = django_model._meta
        
        # Get the original Pydantic model name (remove the "Django" prefix)
        original_name = model_name[6:] if model_name.startswith("Django") else model_name
        
        # Add fields, but skip those that are already in Pydantic2DjangoBaseClass
        base_fields = {"id", "name", "object_type", "data", "created_at", "updated_at"}
        
        fields = []
        for field in meta.fields:
            field_name = field.name
            if field_name in base_fields:
                continue  # Skip fields that are already in Pydantic2DjangoBaseClass
            
            field_def = self.field_to_string(field)
            fields.append({"name": field_name, "definition": field_def})
        
        # Process verbose_name and verbose_name_plural
        verbose_name = str(meta.verbose_name)
        if "\n" in verbose_name:
            # Replace newlines with spaces for single-line string
            verbose_name = verbose_name.replace("\n", " ").strip()
        
        verbose_name_plural = str(meta.verbose_name_plural)
        if "\n" in verbose_name_plural:
            # Replace newlines with spaces for single-line string
            verbose_name_plural = verbose_name_plural.replace("\n", " ").strip()
        
        # Fix pluralization
        if verbose_name_plural.endswith('.s'):
            verbose_name_plural = verbose_name_plural[:-2] + 's'
        elif verbose_name_plural.endswith('s.s'):
            verbose_name_plural = verbose_name_plural[:-2]
        
        # Prepare template context
        template_context = {
            "model_name": model_name[6:] if model_name.startswith("Django") else model_name,
            "fields": fields,
            "meta": {
                "db_table": meta.db_table,
                "app_label": self.app_label,
                "verbose_name": verbose_name,
                "verbose_name_plural": verbose_name_plural
            },
            "original_name": original_name
        }
        
        # Render the template
        try:
            template = self.jinja_env.get_template("model_definition.py.j2")
            return template.render(**template_context)
        except jinja2.exceptions.TemplateNotFound:
            if self.verbose:
                print("Template 'model_definition.py.j2' not found. Using fallback template.")
            # Fallback to a simple template if the file doesn't exist
            return self._generate_fallback_model_definition(template_context)
            
    def _generate_fallback_model_definition(self, context: dict) -> str:
        """Generate a fallback model definition when the template is not found."""
        lines = [
            f"class Django{context['model_name']}(Pydantic2DjangoBaseClass):"
        ]
        
        # Add fields
        for field in context["fields"]:
            lines.append(f"    {field['name']} = {field['definition']}")
        
        # Add Meta class
        lines.append("    class Meta(Pydantic2DjangoBaseClass.Meta):")
        lines.append(f"        db_table = \"{context['meta']['db_table']}\"")
        lines.append(f"        app_label = \"{context['meta']['app_label']}\"")
        lines.append(f"        verbose_name = \"{context['meta']['verbose_name']}\"")
        lines.append(f"        verbose_name_plural = \"{context['meta']['verbose_name_plural']}\"")
        lines.append("        abstract = False")
        
        # Add __init__ method
        lines.append("")
        lines.append("    def __init__(self, *args, **kwargs):")
        lines.append(f"        kwargs.setdefault(\"object_type\", \"{context['original_name']}\")")
        lines.append("        super().__init__(*args, **kwargs)")
        
        # Add _get_module_path method
        lines.append("")
        lines.append("    def _get_module_path(self) -> str:")
        lines.append("        # Use the utility function to get the module path")
        lines.append(f"        return get_module_path(\"{context['original_name']}\")")
        
        return "\n".join(lines)

    def field_to_string(self, field: models.Field) -> str:
        """
        Convert a Django field to a string representation.
        """
        field_type = type(field).__name__
        args = []
        kwargs = {}

        # Handle special field types
        if field_type == "ForeignKey":
            # Handle related_model safely
            if hasattr(field, 'related_model') and field.related_model is not None:
                related_model = field.related_model.__name__
                args.append(f'"{related_model}"')
            else:
                # Fallback to a generic related model name
                args.append('"django_llm.UnknownModel"')
            
            # Handle on_delete safely
            if hasattr(field, 'on_delete') and field.on_delete is not None:
                on_delete_name = getattr(field.on_delete, '__name__', 'CASCADE')
                kwargs["on_delete"] = f"models.{on_delete_name}"
            else:
                kwargs["on_delete"] = "models.CASCADE"  # Default to CASCADE
                
        elif field_type == "ManyToManyField":
            # Handle related_model safely
            if hasattr(field, 'related_model') and field.related_model is not None:
                related_model = field.related_model.__name__
                args.append(f'"{related_model}"')
            else:
                # Fallback to a generic related model name
                args.append('"django_llm.UnknownModel"')

        # Add common field attributes
        if hasattr(field, 'null') and field.null:
            kwargs["null"] = field.null
        if hasattr(field, 'blank') and field.blank:
            kwargs["blank"] = field.blank
        if hasattr(field, "max_length") and field.max_length:
            kwargs["max_length"] = field.max_length
            
        # Handle default values
        if hasattr(field, "default") and field.default != models.NOT_PROVIDED:
            # For BooleanField, we need to handle the default value specially
            if field_type == "BooleanField" and isinstance(field.default, bool):
                kwargs["default"] = str(field.default).lower()  # Convert True/False to 'true'/'false'
            elif field.default is not None:
                # For other types, just use the default value
                if isinstance(field.default, str):
                    kwargs["default"] = f'"{field.default}"'
                else:
                    kwargs["default"] = str(field.default)

        # Add verbose_name if it's not the same as the field name
        if hasattr(field, "verbose_name") and field.verbose_name and field.verbose_name != field.name:
            kwargs["verbose_name"] = f'"{field.verbose_name}"'

        # Add help_text if it exists
        if hasattr(field, "help_text") and field.help_text:
            kwargs["help_text"] = f'"{field.help_text}"'

        # Format args and kwargs
        args_str = ", ".join(args)
        kwargs_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
        if args_str and kwargs_str:
            return f"models.{field_type}({args_str}, {kwargs_str})"
        elif args_str:
            return f"models.{field_type}({args_str})"
        elif kwargs_str:
            return f"models.{field_type}({kwargs_str})"
        else:
            return f"models.{field_type}()"

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
