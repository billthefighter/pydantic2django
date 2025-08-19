"""
Context storage system for handling non-serializable fields in Pydantic2Django.

This module provides the core functionality for managing context fields and their
mapping back to Pydantic objects. It handles the storage and retrieval of context
information needed for field reconstruction.
"""
import dataclasses  # Import needed for DataclassType check later
import logging
from dataclasses import dataclass, field
from typing import Any, Generic, Optional, TypeVar  # Added Generic

from django.db import models
from pydantic import BaseModel

# Use correct import paths after refactoring
from .typing import TypeHandler  # Core typing utilities

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("pydantic2django.core.context")

# Define generic SourceModelType (could potentially be imported from core.factories)
SourceModelType = TypeVar("SourceModelType")  # Represents BaseModel or Dataclass type


@dataclass
class FieldContext:
    """
    Represents context information for a single field.
    """

    field_name: str
    field_type_str: str  # Renamed from field_type for clarity
    is_optional: bool = False
    is_list: bool = False
    additional_metadata: dict[str, Any] = field(default_factory=dict)
    value: Optional[Any] = None
    # required_imports removed - handled by get_required_imports using TypeHandler


@dataclass
class ModelContext(Generic[SourceModelType]):  # Make ModelContext generic
    """
    Base class for model context classes.
    Stores context information for a Django model's fields that require special handling
    during conversion back to the source object (Pydantic/Dataclass).
    """

    django_model: type[models.Model]
    source_class: type[SourceModelType]  # Changed from pydantic_class
    context_fields: dict[str, FieldContext] = field(default_factory=dict)
    context_data: dict[str, Any] = field(default_factory=dict)

    @property
    def required_context_keys(self) -> set[str]:
        required_fields = {fc.field_name for fc in self.context_fields.values() if not fc.is_optional}
        return required_fields

    def add_field(
        self,
        field_name: str,
        field_type_str: str,
        is_optional: bool = False,
        is_list: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Add a field to the context storage.

        Args:
            field_name: Name of the field.
            field_type_str: String representation of the field's type.
            is_optional: Whether the field is optional.
            is_list: Whether the field is a list.
            **kwargs: Additional metadata for the field.
        """
        # Pass is_optional, is_list explicitly
        field_context = FieldContext(
            field_name=field_name,
            field_type_str=field_type_str,
            is_optional=is_optional,
            is_list=is_list,
            additional_metadata=kwargs,
        )
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

    def get_field_type_str(self, field_name: str) -> Optional[str]:
        """Get the string representation type of a context field."""
        field_context = self.context_fields.get(field_name)
        return field_context.field_type_str if field_context else None

    def get_field_by_name(self, field_name: str) -> Optional[FieldContext]:
        """
        Get a field context by name.

        Args:
            field_name: Name of the field to find

        Returns:
            The FieldContext if found, None otherwise
        """
        return self.context_fields.get(field_name)

    def to_conversion_dict(self) -> dict[str, Any]:
        """Convert context to a dictionary format suitable for conversion back to source object."""
        # Renamed from to_dict to be more generic
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

    def get_required_imports(self) -> dict[str, set[str]]:  # Return sets for auto-dedup
        """
        Get all required imports for the context class fields using TypeHandler.
        """
        imports: dict[str, set[str]] = {"typing": set(), "custom": set()}

        # Process each field
        for _, field_context in self.context_fields.items():
            # Use TypeHandler with the stored type string
            type_imports = TypeHandler.get_required_imports(field_context.field_type_str)

            # Add to our overall imports
            imports["typing"].update(type_imports.get("typing", []))
            imports["custom"].update(type_imports.get("datetime", []))  # Example specific types
            imports["custom"].update(type_imports.get("decimal", []))
            imports["custom"].update(type_imports.get("uuid", []))
            # Add any other known modules TypeHandler might return

            # Add Optional/List based on flags
            if field_context.is_optional:
                imports["typing"].add("Optional")
            if field_context.is_list:
                imports["typing"].add("List")

        # Add base source model import
        source_module = getattr(self.source_class, "__module__", None)
        source_name = getattr(self.source_class, "__name__", None)
        if source_module and source_name and source_module != "builtins":
            imports["custom"].add(f"from {source_module} import {source_name}")

        # Add BaseModel or dataclass import
        if isinstance(self.source_class, type) and issubclass(self.source_class, BaseModel):
            imports["custom"].add("from pydantic import BaseModel")
        elif dataclasses.is_dataclass(self.source_class):
            imports["custom"].add("from dataclasses import dataclass")

        # Add Any import if needed
        if any("Any" in fc.field_type_str for fc in self.context_fields.values()):
            imports["typing"].add("Any")

        return imports

    @classmethod
    def generate_context_class_code(cls, model_context: "ModelContext", jinja_env: Any | None = None) -> str:
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

    def __init__(self, jinja_env: Any | None = None) -> None:
        """
        Initialize the ContextClassGenerator.

        Args:
            jinja_env: Optional Jinja2 environment to use for template rendering.
        """
        self.jinja_env = jinja_env
        # Initialize imports needed for the context class generation
        self.imports: dict[str, set[str]] = {"typing": set(), "custom": set()}

    def _load_template(self, template_name: str) -> Any:
        """Load Jinja2 template."""
        if self.jinja_env:
            return self.jinja_env.get_template(template_name)
        else:
            # Fallback to basic string formatting if Jinja2 is not available
            # Note: This is a simplified fallback and might not handle complex templates
            # Load template content from file or define as string here
            # Example using basic string formatting:
            # template_content = "... {model_name} ... {field_definitions} ..."
            # return template_content
            raise ImportError("Jinja2 environment not provided for template loading.")

    def _simplify_type_string(self, type_str: str) -> str:
        """
        Simplifies complex type strings for cleaner code generation.
        Removes module paths like 'typing.' or full paths for common types.
        """
        # Basic simplification: remove typing module path
        simplified = type_str.replace("typing.", "")

        # Use TypeHandler to potentially clean further if needed
        # simplified = TypeHandler.clean_type_string(simplified)

        # Regex to remove full paths for nested standard types like list, dict, etc.
        # Define common standard types that might appear with full paths
        standard_types = ["list", "dict", "tuple", "set", "Optional", "Union"]

        def replacer_class(match):
            full_path = match.group(0)
            # Extract the class name after the last dot
            class_name = full_path.split(".")[-1]
            # Check if the extracted class name is a standard type we want to simplify
            if class_name in standard_types:
                # If it is, return just the class name
                return class_name
            else:
                # Otherwise, keep the full path (or handle custom imports)
                # For now, keeping full path for non-standard types
                # self._maybe_add_type_to_imports(full_path) # Add import for custom type
                return full_path

        # Pattern to find qualified names (e.g., some.module.ClassName)
        # This needs careful crafting to avoid unintended replacements
        # Example: r'\b([a-zA-Z_][\w\.]*\.)?([A-Z][a-zA-Z0-9_]*)\b' might be too broad
        # Focusing on paths likely coming from TypeHandler.get_required_imports might be safer
        # For now, rely on basic replace and potential TypeHandler cleaning

        return simplified

    def generate_context_class(self, model_context: ModelContext) -> str:
        """
        Generates the Python code string for a context dataclass.
        """
        template = self._load_template("context_class.py.j2")
        self.imports = model_context.get_required_imports()  # Get imports first

        field_definitions = []
        for field_name, field_context in model_context.context_fields.items():
            field_type_str = field_context.field_type_str  # field_type is now the string representation

            # Use TypeHandler._get_raw_type_string to get the clean, unquoted type string
            # --- Corrected import path for TypeHandler ---
            from .typing import TypeHandler

            clean_type = TypeHandler._get_raw_type_string(field_type_str)

            # Simplify the type string for display
            simplified_type = self._simplify_type_string(clean_type)

            # Add necessary imports based on the simplified type
            # (Assuming _simplify_type_string and get_required_imports handle this)

            # Format default value if present
            default_repr = repr(field_context.value) if field_context.value is not None else "None"

            field_def = f"    {field_name}: {simplified_type} = field(default={default_repr})"
            field_definitions.append(field_def)

        # Prepare imports for the template
        typing_imports_str = ", ".join(sorted(self.imports["typing"]))
        custom_imports_list = sorted(self.imports["custom"])  # Keep as list of strings

        model_name = self._clean_generic_type(model_context.django_model.__name__)
        source_class_name = self._clean_generic_type(model_context.source_class.__name__)

        return template.render(
            model_name=model_name,
            # Use source_class_name instead of pydantic_class
            source_class_name=source_class_name,
            source_module=model_context.source_class.__module__,
            field_definitions="\n".join(field_definitions),
            typing_imports=typing_imports_str,
            custom_imports=custom_imports_list,
        )

    def _clean_generic_type(self, name: str) -> str:
        """Remove generic parameters like [T] from class names."""
        return name.split("[")[0]

    # Removed _maybe_add_type_to_imports and get_imports - Handled by model_context.get_required_imports


# Remove internal uses of pydantic2django.type_handler if any remain
# (Check _simplify_type_string and others)
