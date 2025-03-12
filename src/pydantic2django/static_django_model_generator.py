import os
from collections.abc import Callable
from typing import Optional, TypeVar

from django.db import models
from pydantic import BaseModel

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
    Generates a static models.py file from discovered Pydantic models
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
        return lambda name, model: filter_func(model)

    def generate_models_file(
        self,
        pydantic_models: dict[str, type[BaseModel]],
        django_models: dict[str, type[models.Model]],
    ) -> str:
        """Generate the content for the models.py file."""
        imports = [
            "from django.db import models",
            "from django.contrib.postgres.fields import ArrayField",
            "from django.db.models import JSONField",
            "import uuid",
            "import importlib",
            "from datetime import datetime, date, time, timedelta",
            "from decimal import Decimal",
            "from typing import Dict, List, Optional, Any, Union",
            "from pydantic2django.base_django_model import Pydantic2DjangoBaseClass",
            "",
        ]

        model_definitions = []

        # Add all models
        for model_name, django_model in django_models.items():
            try:
                # Get the original Pydantic model
                original_name = model_name[6:] if model_name.startswith("Django") else model_name
                pydantic_model = pydantic_models.get(original_name)

                model_def = self.generate_model_definition(model_name, django_model, pydantic_model)
                model_definitions.append(model_def)
            except Exception as e:
                if self.verbose:
                    print(f"Error generating model {model_name}: {e}")

        # Combine everything
        content = "\n".join(imports) + "\n\n" + "\n\n".join(model_definitions)

        # Add __all__ export
        all_models = [f'"{name}"' for name in django_models.keys()]
        content += f'\n\n__all__ = [{", ".join(all_models)}]\n'

        return content

    def generate_model_definition(
        self,
        model_name: str,
        django_model: type[models.Model],
        pydantic_model: Optional[type[BaseModel]] = None,
    ) -> str:
        """Generate a Django model definition as a string."""
        # Use Pydantic2DjangoBaseClass as the base class instead of models.Model
        lines = [f"class {model_name}(Pydantic2DjangoBaseClass):"]

        # Get field definitions
        meta = getattr(django_model, "_meta", None)
        if not meta:
            raise ValueError(f"Model {model_name} has no _meta attribute")

        # Get the original Pydantic model name (remove the "Django" prefix)
        original_name = model_name[6:] if model_name.startswith("Django") else model_name

        # Add fields, but skip those that are already in Pydantic2DjangoBaseClass
        base_fields = {"id", "name", "object_type", "data", "created_at", "updated_at"}

        for field in meta.fields:
            field_name = field.name
            if field_name in base_fields:
                continue  # Skip fields that are already in Pydantic2DjangoBaseClass

            field_def = self.field_to_string(field)
            lines.append(f"    {field_name} = {field_def}")

        # Add Meta class
        meta_lines = ["    class Meta(Pydantic2DjangoBaseClass.Meta):"]
        meta_lines.append(f'        db_table = "{meta.db_table}"')
        meta_lines.append(f'        app_label = "{self.app_label}"')
        meta_lines.append(f'        verbose_name = "{meta.verbose_name}"')
        meta_lines.append(f'        verbose_name_plural = "{meta.verbose_name_plural}"')
        meta_lines.append("        abstract = False")  # Override the abstract=True from Pydantic2DjangoBaseClass

        lines.extend(meta_lines)

        # Add a default __init__ method that sets object_type
        lines.append("")
        lines.append("    def __init__(self, *args, **kwargs):")
        lines.append(f'        kwargs.setdefault("object_type", "{original_name}")')
        lines.append("        super().__init__(*args, **kwargs)")

        # Add _get_module_path method for to_pydantic
        if pydantic_model:
            module_name = pydantic_model.__module__
            lines.append("")
            lines.append("    def _get_module_path(self) -> str:")
            lines.append(f'        return "{module_name}"')
        else:
            # Add a placeholder implementation that raises an error
            lines.append("")
            lines.append("    def _get_module_path(self) -> str:")
            lines.append('        raise NotImplementedError("Module path not available for this model")')

        return "\n".join(lines)

    def field_to_string(self, field: models.Field) -> str:
        """Convert a Django field to its string representation."""
        field_class = field.__class__.__name__

        # Handle common field types
        if isinstance(field, models.CharField):
            return f"models.CharField(max_length={field.max_length}, null={field.null}, blank={field.blank})"
        elif isinstance(field, models.TextField):
            return f"models.TextField(null={field.null}, blank={field.blank})"
        elif isinstance(field, models.IntegerField):
            return f"models.IntegerField(null={field.null}, blank={field.blank})"
        elif isinstance(field, models.BooleanField):
            default_value = "False" if field.default is False else "True" if field.default is True else "None"
            return f"models.BooleanField(default={default_value}, null={field.null})"
        elif isinstance(field, models.DateTimeField):
            return f"models.DateTimeField(auto_now_add={field.auto_now_add}, auto_now={field.auto_now}, null={field.null}, blank={field.blank})"
        elif isinstance(field, models.ForeignKey):
            # Safely handle related model name
            related_model = "self"
            if hasattr(field, "remote_field") and field.remote_field is not None:
                if hasattr(field.remote_field, "model"):
                    if isinstance(field.remote_field.model, str):
                        related_model = field.remote_field.model
                    elif hasattr(field.remote_field.model, "__name__"):
                        related_model = field.remote_field.model.__name__

            # Safely handle on_delete
            on_delete = "CASCADE"  # Default
            if hasattr(field, "remote_field") and field.remote_field is not None:
                if hasattr(field.remote_field, "on_delete"):
                    on_delete_obj = field.remote_field.on_delete
                    if hasattr(on_delete_obj, "__name__"):
                        on_delete = on_delete_obj.__name__
                    else:
                        on_delete = str(on_delete_obj).split(".")[-1]

            return f'models.ForeignKey("{related_model}", on_delete=models.{on_delete}, null={field.null}, blank={field.blank})'
        elif isinstance(field, models.JSONField):
            return f"models.JSONField(null={field.null}, blank={field.blank})"
        elif isinstance(field, models.UUIDField):
            default = "uuid.uuid4" if str(field.default).endswith("uuid4") else "None"
            return f"models.UUIDField(default={default}, null={field.null}, blank={field.blank})"

        # Default fallback
        params = []
        for key, value in field.__dict__.items():
            if key.startswith("_") or key in (
                "attname",
                "column",
                "concrete",
                "model",
                "name",
                "remote_field",
            ):
                continue
            if isinstance(value, str):
                params.append(f'{key}="{value}"')
            elif value is not None and not callable(value):
                try:
                    # Try to represent the value as a string
                    if isinstance(value, (bool, int, float)):
                        params.append(f"{key}={value}")
                    else:
                        params.append(f'{key}="{value}"')
                except:
                    # If that fails, skip this parameter
                    pass

        return f'models.{field_class}({", ".join(params)})'
