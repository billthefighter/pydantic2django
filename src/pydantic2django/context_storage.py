"""
Context storage system for handling non-serializable fields in Pydantic2Django.

This module provides the core functionality for managing context fields and their
mapping back to Pydantic objects. It handles the storage and retrieval of context
information needed for field reconstruction.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, TypeVar

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
    field_type: str
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

    def add_field(self, field_name: str, field_type: str, **kwargs) -> None:
        """
        Add a field to the context storage.

        Args:
            field_name: Name of the field
            field_type: String representation of the field's type
            **kwargs: Additional metadata for the field
        """
        field_context = FieldContext(field_name=field_name, field_type=field_type, **kwargs)

        # Commenting out import generation based on type object for now,
        # as field_type is now a string. This logic might need revisiting.
        # if hasattr(field_type, "__module__") and field_type.__module__ != "builtins":
        #     if field_type.__module__ != "typing":
        #         # Add the import for this custom type
        #         field_context.required_imports.append(f"from {field_type.__module__} import {field_type.__name__}")

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

    def get_field_type(self, field_name: str) -> Optional[str]:
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

        # Since field_type is now a string, TypeHandler might need adjustment
        # or we handle the string directly here. For now, returning the string.
        # type_name, _ = TypeHandler.process_field_type(field_context.field_type) # Old logic
        return field_context.field_type  # Return the stored string representation

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
            # This part needs revision as field_context.field_type is now a string
            # _, field_imports = TypeHandler.process_field_type(field_context.field_type) # Old logic
            # imports[\"explicit\"].extend(field_imports) # Old logic

            # Get additional imports from the type string
            field_type_str = field_context.field_type  # field_type is already the string
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

    def _simplify_type_string(self, type_str: str) -> str:
        """
        Simplifies a type string by removing module paths.
        Handles standard type strings and those wrapped in <class \'...\'> recursively.

        Args:
            type_str: The full type string (e.g., "typing.Optional[myapp.models.MyModel]",
            "<class \'myapp.models.MyModel\'>")

        Returns:
            The simplified type string (e.g., "Optional[MyModel]", "MyModel")
        """
        import re

        # 1. Simplify <class 'module.path.ClassName'> occurrences to ClassName
        def replacer_class(match):
            full_path = match.group(1)
            return full_path.split(".")[-1]  # Get only the ClassName

        simplified = re.sub(r"<class \'([\w\.]+)\'>", replacer_class, type_str)

        # 2. Simplify remaining module.path.ClassName occurrences to ClassName
        simplified = re.sub(r"\b[\w\.]+\.(\w+)", r"\1", simplified)

        # 3. Explicitly remove __main__ prefix if it remains (common in local execution)
        simplified = simplified.replace("__main__.", "")

        return simplified

    def generate_context_class(self, model_context: ModelContext) -> str:
        """
        Generate the Python code for a context class using a Jinja template.

        Args:
            model_context: The context information for the model.

        Returns:
            The generated Python code as a string.
        """
        template = self.jinja_env.get_template("context_class.py.j2")
        field_definitions = []

        for field_name, field_context in model_context.context_fields.items():
            field_type_str = field_context.field_type  # field_type is now the string representation

            # Use TypeHandler._get_raw_type_string to get the clean, unquoted type string
            # We need to import TypeHandler directly here
            from pydantic2django.type_handler import TypeHandler

            try:
                # Note: _get_raw_type_string expects a type object or string representation.
                # Since field_context.field_type is already the string, we pass it.
                # We may need to adjust _get_raw_type_string if it strictly expects type objects.
                # For now, assuming it handles strings appropriately or needs adjustment there.
                raw_type_str = TypeHandler._get_raw_type_string(field_type_str)
            except Exception as e:
                self.logger.warning(
                    f"Failed to get raw type string for {field_name} ({field_type_str}): {e}. Defaulting to 'Any'."
                )
                raw_type_str = "Any"
                self.extra_type_imports.add("Any")

            # Get the literal representation by using repr() on the raw string
            literal_type_repr = repr(raw_type_str)

            # Simplify the raw string for display/annotation if needed (optional)
            # simplified_raw_type_str = self._simplify_type_string(raw_type_str)

            # Ensure metadata is a dict
            metadata = field_context.additional_metadata or {}

            field_def = {
                "name": field_name,
                "raw_type": raw_type_str,  # Pass the raw string for annotations
                "literal_type": literal_type_repr,  # Pass the literal string for assignments
                "is_optional": field_context.is_optional,
                "is_list": field_context.is_list,
                "metadata": metadata,
            }
            field_definitions.append(field_def)

        # Add 'Callable' to typing imports if it was used in any raw types
        if any("Callable" in fd["raw_type"] for fd in field_definitions):
            self.extra_type_imports.add("Callable")

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
        Returns sets of imports.
        """
        # Imports are already collected in self.extra_type_imports and self.context_class_imports
        # Return them directly as sets
        return {
            "typing": self.extra_type_imports,
            "context": self.context_class_imports,
        }
