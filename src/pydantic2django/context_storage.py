"""
Context storage system for handling non-serializable fields in Pydantic2Django.

This module provides the core functionality for managing context fields and their
mapping back to Pydantic objects. It handles the storage and retrieval of context
information needed for field reconstruction.
"""
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional, TypeVar, Union

from django.db import models
from pydantic import BaseModel

from pydantic2django.type_handler import TypeHandler

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("pydantic2django.context_storage")

T = TypeVar("T", bound=BaseModel)


@dataclass
class FieldContext:
    """
    Represents context information for a single field.
    """

    field_name: str
    field_type: type[Any]
    is_optional: bool = False
    is_list: bool = False
    additional_metadata: dict[str, Any] = field(default_factory=dict)
    value: Optional[Any] = None
    required_imports: list[str] = field(default_factory=list)


@dataclass
class ModelContext:
    """
    Base class for model context classes.
    Stores context information for a Django model's fields that require special handling
    during conversion back to Pydantic objects.
    """

    django_model: type[models.Model]
    pydantic_class: type[BaseModel]
    context_fields: dict[str, FieldContext] = field(default_factory=dict)

    @property
    def required_context_keys(self) -> set[str]:
        required_fields = {
            field_name for field_name, field_context in self.context_fields.items() if not field_context.is_optional
        }
        return required_fields

    def add_field(self, field_name: str, field_type: type[Any], **kwargs) -> None:
        """
        Add a field to the context storage.

        Args:
            field_name: Name of the field
            field_type: Type of the field
            **kwargs: Additional metadata for the field
        """
        field_context = FieldContext(field_name=field_name, field_type=field_type, **kwargs)

        # Add required imports based on field_type
        if hasattr(field_type, "__module__") and field_type.__module__ != "builtins":
            if field_type.__module__ != "typing":
                # Add the import for this custom type
                field_context.required_imports.append(f"from {field_type.__module__} import {field_type.__name__}")

        self.context_fields[field_name] = field_context

    def validate_context(self, context: dict[str, Any]) -> None:
        """
        Validate that all required context fields are present.

        Args:
            context: The context dictionary to validate

        Raises:
            ValueError: If required context fields are missing
        """

        missing_fields = self.required_context_keys - set(context.keys())
        if missing_fields:
            raise ValueError(f"Missing required context fields: {', '.join(missing_fields)}")

    def get_field_type(self, field_name: str) -> Optional[type[Any]]:
        """
        Get the type of a context field.

        Args:
            field_name: Name of the field

        Returns:
            The field type if it exists in the context, None otherwise
        """
        field_context = self.context_fields.get(field_name)
        return field_context.field_type if field_context else None

    def get_field_by_name(self, field_name: str) -> Optional[FieldContext]:
        """
        Get a field context by name.

        Args:
            field_name: Name of the field to find

        Returns:
            The FieldContext if found, None otherwise
        """
        return self.context_fields.get(field_name)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert context to a dictionary format suitable for to_pydantic().

        Returns:
            Dictionary containing all context values
        """
        return {
            field_name: field_context.value
            for field_name, field_context in self.context_fields.items()
            if field_context.value is not None
        }

    def set_value(self, field_name: str, value: Any) -> None:
        """
        Set the value for a context field.

        Args:
            field_name: Name of the field
            value: Value to set

        Raises:
            ValueError: If the field doesn't exist in the context
        """
        field = self.get_field_by_name(field_name)
        if field is None:
            raise ValueError(f"Field {field_name} not found in context")
        field.value = value

    def get_value(self, field_name: str) -> Optional[Any]:
        """
        Get the value of a context field.

        Args:
            field_name: Name of the field

        Returns:
            The field value if it exists and has been set, None otherwise
        """
        field = self.get_field_by_name(field_name)
        if field is not None:
            return field.value
        return None

    def get_formatted_field_type(self, field_name: str) -> Optional[str]:
        """
        Get a clean, formatted string representation of a field's type.

        Args:
            field_name: Name of the field

        Returns:
            Formatted type string if field exists, None otherwise
        """
        field_context = self.get_field_by_name(field_name)
        if field_context is None:
            return None

        # Use the TypeHandler's centralized method to process the field type
        type_name, _ = TypeHandler.process_field_type(field_context.field_type)
        return type_name

    def get_required_imports(self) -> dict[str, list[str]]:
        """
        Get all required imports for the context class fields.

        Returns:
            Dict with keys 'typing' and 'custom', containing lists of required imports
        """
        imports = {"typing": [], "custom": [], "explicit": []}

        # Process each field
        for _, field_context in self.context_fields.items():
            # Use TypeHandler to get all required imports for this field type
            _, field_imports = TypeHandler.process_field_type(field_context.field_type)
            imports["explicit"].extend(field_imports)

            # Get additional imports from the type string
            field_type_str = str(field_context.field_type)
            type_imports = TypeHandler.get_required_imports(field_type_str)

            # Add to our overall imports
            imports["typing"].extend(type_imports["typing"])
            imports["custom"].extend(type_imports["custom"])
            imports["explicit"].extend(type_imports["explicit"])

            # Get explicit imports from the field
            if field_context.required_imports:
                imports["explicit"].extend(field_context.required_imports)

            # Handle is_optional and is_list flags
            if field_context.is_optional:
                imports["typing"].append("Optional")
            if field_context.is_list:
                imports["typing"].append("List")

        # Deduplicate imports
        imports["typing"] = list(set(imports["typing"]))
        imports["custom"] = list(set(imports["custom"]))
        imports["explicit"] = list(set(imports["explicit"]))

        return imports

    @classmethod
    def generate_context_class_code(cls, model_context: "ModelContext", jinja_env=None) -> str:
        """
        Generate a string representation of the context class.

        Args:
            model_context: The ModelContext to generate a class for
            jinja_env: Optional Jinja2 environment to use for rendering

        Returns:
            String representation of the context class
        """
        # Create a ContextClassGenerator and use it to generate the class
        generator = ContextClassGenerator(jinja_env=jinja_env)
        return generator.generate_context_class(model_context)


class ContextClassGenerator:
    """
    Utility class for generating context class code from ModelContext objects.
    """

    def __init__(self, jinja_env=None):
        """
        Initialize the ContextClassGenerator.

        Args:
            jinja_env: Optional Jinja2 environment to use for template rendering.
                      If not provided, a new environment will be created.
        """
        import logging
        import os
        import pathlib

        import jinja2

        # Configure logging
        self.logger = logging.getLogger("pydantic2django.context_generator")

        # Initialize Jinja2 environment if not provided
        if jinja_env is None:
            # Look for templates in the package directory
            package_templates_dir = os.path.join(os.path.dirname(__file__), "templates")

            # If templates don't exist in the package, use the ones from the current directory
            if not os.path.exists(package_templates_dir):
                package_templates_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "templates")

            self.jinja_env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(package_templates_dir),
                trim_blocks=True,
                lstrip_blocks=True,
            )
            # Register custom filters
            from pydantic2django.type_handler import TypeHandler

            self.jinja_env.filters["clean_field_type_for_template"] = TypeHandler.clean_field_type_for_template
        else:
            self.jinja_env = jinja_env
            # Ensure the filter is registered in the provided env
            if "clean_field_type_for_template" not in self.jinja_env.filters:
                from pydantic2django.type_handler import TypeHandler

                self.jinja_env.filters["clean_field_type_for_template"] = TypeHandler.clean_field_type_for_template

        # Collection for imports that need to be added to the final template
        self.extra_type_imports: set[str] = set()
        self.context_class_imports: set[str] = set()

    def generate_context_class(self, model_context: ModelContext) -> str:
        """
        Generate a context class for a Django model.

        Args:
            model_context: The ModelContext instance for the model

        Returns:
            String representation of the context class
        """
        if model_context is None or not hasattr(model_context, "django_model") or model_context.django_model is None:
            self.logger.warning("Cannot generate context class for None model_context or missing django_model")
            return ""

        # Skip generating context class if there are no context fields
        if not model_context.context_fields:
            self.logger.info(f"Skipping context class for {model_context.django_model.__name__} - no context fields")
            return ""

        template = self.jinja_env.get_template("context_class.py.j2")

        # Prepare field definitions and collect required imports
        field_definitions = []

        # Get required imports and add them to our import sets
        required_imports = model_context.get_required_imports()
        self.extra_type_imports.update(required_imports["typing"])

        # Add explicit typing import if we use typing-specific constructs
        if any(imp in required_imports["typing"] for imp in ["Dict", "List", "Optional", "Union"]):
            self.context_class_imports.add("import typing")

        # Add custom types to imports if they're not already imported
        for custom_type in required_imports["custom"]:
            self._maybe_add_type_to_imports(custom_type)

        # Add explicit imports to the imports
        for import_stmt in required_imports.get("explicit", []):
            if import_stmt.startswith("from ") and " import " in import_stmt:
                self.context_class_imports.add(import_stmt)

        # Handle is_optional and is_list flags
        for field_name, field_context in model_context.context_fields.items():  # noqa: B007
            if field_context.is_optional:
                self.context_class_imports.add("Optional")
            if field_context.is_list:
                self.context_class_imports.add("List")

        # Generate field definitions for the template
        for field_name, field_context in model_context.context_fields.items():
            # Extract the field type information
            field_type = field_context.field_type

            # Initialize type_str to a default value
            type_str = "Any"
            self.context_class_imports.add("from typing import Any")

            # Use a safer approach to handle type annotations
            if isinstance(field_type, type) and not isinstance(field_type, str):
                # For simple type objects like RetryStrategy, use the type directly
                if hasattr(field_type, "__name__"):
                    type_str = field_type.__name__

                    # Add import for the class if needed
                    if hasattr(field_type, "__module__") and field_type.__module__ not in ["builtins", "typing"]:
                        self.context_class_imports.add(f"from {field_type.__module__} import {type_str}")
            elif field_type in [Any, Optional, Callable, list, dict, Union]:
                # For basic typing constructs, use their names
                type_str = field_type.__name__
                self.context_class_imports.add(f"from typing import {type_str}")
            else:
                # For complex types or string representations, use a simplified representation
                # that works in Python code but doesn't try to preserve the full type structure
                if field_context.is_optional:
                    type_str = "Optional[Callable]"
                    self.context_class_imports.add("from typing import Optional, Callable")
                elif "Callable" in str(field_type):
                    type_str = "Callable"
                    self.context_class_imports.add("from typing import Callable")
                elif "type" in str(field_type).lower():
                    type_str = "type"
                else:
                    # Use a simple string for other complex types
                    type_str = "Any"
                    self.context_class_imports.add("from typing import Any")

            # Ensure metadata is a dict and doesn't contain problematic characters
            metadata = {}
            if field_context.additional_metadata:
                for k, v in field_context.additional_metadata.items():
                    if isinstance(v, str):
                        metadata[k] = v.replace("\n", " ").replace("\r", "")
                    else:
                        metadata[k] = v

            field_def = {
                "name": field_name,
                "type": type_str,
                "is_optional": field_context.is_optional,
                "is_list": field_context.is_list,
                "metadata": metadata,
            }
            field_definitions.append(field_def)

        model_name = self._clean_generic_type(model_context.django_model.__name__)
        pydantic_class = self._clean_generic_type(model_context.pydantic_class.__name__)

        return template.render(
            model_name=model_name,
            pydantic_class=pydantic_class,
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
        import re

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

    def get_imports(self) -> dict[str, set[str]]:
        """
        Get the imports collected during context class generation.

        Returns:
            Dictionary containing typing and context-specific imports
        """
        return {"typing": self.extra_type_imports, "context": self.context_class_imports}
