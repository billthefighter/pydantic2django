import logging
import re
from typing import Any

from pydantic2django.type_handler import TypeHandler

# Configure logging
logger = logging.getLogger("pydantic2django.import_handler")


class ImportHandler:
    """
    Handles import statements for generated Django models and their context classes.
    Tracks and deduplicates imports from multiple sources while ensuring all necessary
    dependencies are included.
    """

    def __init__(self):
        """Initialize empty collections for different types of imports."""
        # Track imports by category
        self.extra_type_imports: set[str] = set()  # For typing and other utility imports
        self.pydantic_imports: set[str] = set()  # For Pydantic model imports
        self.context_class_imports: set[str] = set()  # For context class and field type imports

        # For tracking imported names to avoid duplicates
        self.imported_names: dict[str, str] = {}  # Maps type name to its module

        # For tracking field type dependencies we've already processed
        self.processed_field_types: set[str] = set()

        logger.info("ImportHandler initialized")

    def add_pydantic_model_import(self, model_class: type) -> None:
        """
        Add an import statement for a Pydantic model.

        Args:
            model_class: The Pydantic model class to import
        """
        if not hasattr(model_class, "__module__") or not hasattr(model_class, "__name__"):
            logger.warning(f"Cannot add import for {model_class}: missing __module__ or __name__")
            return

        module_path = model_class.__module__
        model_name = self._clean_generic_type(model_class.__name__)

        logger.debug(f"Processing Pydantic model import: {model_name} from {module_path}")

        # Skip if already imported
        if model_name in self.imported_names:
            logger.debug(f"Skipping already imported model: {model_name}")
            return

        import_statement = f"from {module_path} import {model_name}"
        logger.info(f"Adding Pydantic import: {import_statement}")
        self.pydantic_imports.add(import_statement)
        self.imported_names[model_name] = module_path

    def add_context_field_type_import(self, field_type: Any) -> None:
        """
        Add imports for a context field type.

        Args:
            field_type: The field type to add imports for
        """
        try:
            logger.debug(f"Adding imports for context field type: {field_type}")

            # Handle TypeVar imports specially
            if isinstance(field_type, type) and "TypeVar" in str(field_type):
                logger.debug(f"Processing a TypeVar: {field_type}")
                type_str = str(field_type)
                if "type[" in type_str.lower():
                    type_match = re.search(r"type\[(.*?)\]", type_str, re.IGNORECASE)
                    if type_match:
                        inner_type = type_match.group(1)
                        logger.debug(f"Found TypeVar in type[], inner type: {inner_type}")

                        # Add TypeVar import
                        self.extra_type_imports.add("TypeVar")
                        self.extra_type_imports.add("Type")

                        # If the inner type is a TypeVar, add it to context imports
                        if "." not in inner_type:
                            self.context_class_imports.add(f"{inner_type} = TypeVar('{inner_type}')")

                        # Return as we've already handled this special case
                        return

            # Special handling for built-in type class
            if field_type is type:
                self.extra_type_imports.add("Type")
                return

            # Handle class objects with __module__ and __name__
            if hasattr(field_type, "__module__") and hasattr(field_type, "__name__"):
                module_name = field_type.__module__
                class_name = field_type.__name__

                # Skip builtins
                if module_name == "builtins":
                    return

                # Handle __main__ special case - these classes are defined in the example script
                if module_name == "__main__":
                    # Fix the import path for classes defined in __main__
                    # Assuming these are from examples.simple_model_conversion_example
                    self.pydantic_imports.add(f"from examples.simple_model_conversion_example import {class_name}")
                    return

                # Handle type objects from actual modules (not string representations)
                if not str(field_type).startswith("<class '"):
                    # Add a proper import
                    self.context_class_imports.add(f"from {module_name} import {class_name}")
                    return

            # Get required imports from type string
            type_str = str(field_type)

            # Clean up the string representation if it's a class object
            if type_str.startswith("<class '"):
                # Extract the actual class name from the string
                match = re.search(r"<class '([^']+)'>", type_str)
                if match:
                    class_path = match.group(1)

                    # Handle builtin types
                    if class_path in ("type", "builtins.type"):
                        self.extra_type_imports.add("Type")
                        return

                    # Handle module.class format
                    if "." in class_path:
                        module_path, class_name = class_path.rsplit(".", 1)

                        # Skip builtins
                        if module_path == "builtins":
                            return

                        # Handle __main__ special case
                        if module_path == "__main__":
                            # Fix __main__ imports to use examples.simple_model_conversion_example
                            self.pydantic_imports.add(
                                f"from examples.simple_model_conversion_example import {class_name}"
                            )
                        else:
                            # Add a proper import for other modules
                            self.context_class_imports.add(f"from {module_path} import {class_name}")
                        return
                    else:
                        # Just a class name without module
                        self.extra_type_imports.add(class_path)
                        return

            # Handle complex typing types like Callable, List, etc.
            required_imports = TypeHandler.get_required_imports(type_str)

            # Add typing imports
            for typing_import in required_imports["typing"]:
                self.extra_type_imports.add(typing_import)

            # Add explicit imports
            for explicit_import in required_imports.get("explicit", []):
                # Only add if it's not already in context imports
                if not any(explicit_import in import_str for import_str in self.context_class_imports):
                    self.context_class_imports.add(explicit_import)

            # Process custom types
            for _custom_type in required_imports["custom"]:
                # Try to extract the module from the custom type
                if hasattr(field_type, "__module__") and field_type.__module__ not in ["builtins", "typing"]:
                    module = field_type.__module__
                    if hasattr(field_type, "__name__"):
                        name = field_type.__name__

                        # Handle __main__ special case
                        if module == "__main__":
                            self.pydantic_imports.add(f"from examples.simple_model_conversion_example import {name}")
                        else:
                            self.context_class_imports.add(f"from {module} import {name}")
        except Exception as e:
            logger.error(f"Error adding imports for context field type {field_type}: {e}")

    def _add_type_import(self, field_type: Any) -> None:
        """
        Add an import for a single type object if it has module and name attributes.

        Args:
            field_type: The type to import
        """
        try:
            if hasattr(field_type, "__module__") and hasattr(field_type, "__name__"):
                type_module = field_type.__module__
                type_name = field_type.__name__

                logger.debug(f"Examining type: {type_name} from module {type_module}")

                # Skip built-in types and typing module types
                if (
                    type_module.startswith("typing")
                    or type_module == "builtins"
                    or type_name in ["str", "int", "float", "bool", "dict", "list"]
                ):
                    logger.debug(f"Skipping built-in or typing type: {type_name}")
                    return

                # Clean up any parametrized generic types
                clean_type_name = self._clean_generic_type(type_name)

                # Skip if already imported
                if clean_type_name in self.imported_names:
                    logger.debug(f"Skipping already imported type: {clean_type_name}")
                    return

                # Add to context class imports
                import_statement = f"from {type_module} import {clean_type_name}"
                logger.info(f"Adding context class import: {import_statement}")
                self.context_class_imports.add(import_statement)
                self.imported_names[clean_type_name] = type_module
        except (AttributeError, TypeError) as e:
            logger.warning(f"Error processing type import for {field_type}: {e}")

    def _process_nested_types(self, field_type: Any) -> None:
        """
        Recursively process nested types in generics, unions, etc.

        Args:
            field_type: The type that might contain nested types
        """
        # Handle __args__ for generic types, unions, etc.
        if hasattr(field_type, "__args__"):
            logger.debug(f"Processing nested types for {field_type}")
            for arg_type in field_type.__args__:
                logger.debug(f"Found nested type argument: {arg_type}")
                # Recursively process each argument type
                self.add_context_field_type_import(arg_type)

        # Handle __origin__ for generic types (like List, Dict, etc.)
        if hasattr(field_type, "__origin__"):
            logger.debug(f"Processing origin type for {field_type}: {field_type.__origin__}")
            self.add_context_field_type_import(field_type.__origin__)

    def _add_typing_imports(self, field_type_str: str) -> None:
        """
        Add required typing imports based on the string representation of the field type.

        Args:
            field_type_str: String representation of the field type
        """
        # Check for common typing constructs
        if "List[" in field_type_str or "list[" in field_type_str:
            logger.debug(f"Adding List import from {field_type_str}")
            self.extra_type_imports.add("List")

        if "Dict[" in field_type_str or "dict[" in field_type_str:
            logger.debug(f"Adding Dict import from {field_type_str}")
            self.extra_type_imports.add("Dict")

        if "Tuple[" in field_type_str or "tuple[" in field_type_str:
            logger.debug(f"Adding Tuple import from {field_type_str}")
            self.extra_type_imports.add("Tuple")

        if "Optional[" in field_type_str or "Union[" in field_type_str or "None" in field_type_str:
            logger.debug(f"Adding Optional import from {field_type_str}")
            self.extra_type_imports.add("Optional")

        if "Union[" in field_type_str:
            logger.debug(f"Adding Union import from {field_type_str}")
            self.extra_type_imports.add("Union")

        if "Callable[" in field_type_str:
            logger.debug(f"Adding Callable import from {field_type_str}")
            self.extra_type_imports.add("Callable")

        if "Any" in field_type_str:
            logger.debug(f"Adding Any import from {field_type_str}")
            self.extra_type_imports.add("Any")

        # Extract custom types from the field type string
        self._extract_custom_types_from_string(field_type_str)

    def _extract_custom_types_from_string(self, field_type_str: str) -> None:
        """
        Extract custom type names from a string representation of a field type.

        Args:
            field_type_str: String representation of the field type
        """
        # Extract potential type names from the string
        # This regex looks for capitalized words that might be type names
        type_names = re.findall(r"[A-Z][a-zA-Z0-9]*", field_type_str)

        logger.debug(f"Extracted potential type names from string {field_type_str}: {type_names}")

        for type_name in type_names:
            # Skip common type names that are already handled
            if type_name in ["List", "Dict", "Optional", "Union", "Tuple", "Callable", "Any"]:
                logger.debug(f"Skipping common typing name: {type_name}")
                continue

            # Skip if already in imported names
            if type_name in self.imported_names:
                logger.debug(f"Skipping already imported name: {type_name}")
                continue

            # Log potential custom type
            logger.info(f"Adding potential custom type to extra_type_imports: {type_name}")

            # Add to extra type imports - these are types that we couldn't resolve to a module
            # They'll need to be imported elsewhere or we might generate an error
            self.extra_type_imports.add(type_name)

    def get_required_imports(self, field_type_str: str) -> dict[str, list[str]]:
        """
        Get typing and custom type imports required for a field type.

        Args:
            field_type_str: String representation of a field type

        Returns:
            Dictionary with "typing" and "custom" import lists
        """
        logger.debug(f"Getting required imports for: {field_type_str}")
        self._add_typing_imports(field_type_str)

        # Get custom types (non-typing types)
        custom_types = [
            name
            for name in self.extra_type_imports
            if name not in ["List", "Dict", "Tuple", "Set", "Optional", "Union", "Any", "Callable"]
        ]

        logger.debug(f"Found custom types: {custom_types}")

        # Return the latest state of imports
        return {
            "typing": list(self.extra_type_imports),
            "custom": custom_types,
        }

    def deduplicate_imports(self) -> dict[str, set[str]]:
        """
        De-duplicate imports between Pydantic models and context field types.

        Returns:
            Dict with de-duplicated import sets
        """
        logger.info("Deduplicating imports")
        logger.debug(f"Current pydantic imports: {self.pydantic_imports}")
        logger.debug(f"Current context imports: {self.context_class_imports}")

        # Extract class names and modules from import statements
        pydantic_classes = {}
        context_classes = {}

        for import_stmt in self.pydantic_imports:
            if import_stmt.startswith("from ") and " import " in import_stmt:
                module, classes = import_stmt.split(" import ")
                module = module.replace("from ", "")
                for cls in classes.split(", "):
                    # Clean up any parameterized generic types in class names
                    cls = self._clean_generic_type(cls)
                    pydantic_classes[cls] = module

        for import_stmt in self.context_class_imports:
            if import_stmt.startswith("from ") and " import " in import_stmt:
                module, classes = import_stmt.split(" import ")
                module = module.replace("from ", "")
                for cls in classes.split(", "):
                    # Clean up any parameterized generic types in class names
                    cls = self._clean_generic_type(cls)
                    # If this class is already imported in pydantic imports, skip it
                    if cls in pydantic_classes:
                        logger.debug(f"Skipping duplicate context import for {cls}, already in pydantic imports")
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

        logger.info(f"Final pydantic imports: {deduplicated_pydantic_imports}")
        logger.info(f"Final context imports: {deduplicated_context_imports}")

        return {"pydantic": deduplicated_pydantic_imports, "context": deduplicated_context_imports}

    def _clean_generic_type(self, name: str) -> str:
        """
        Clean generic parameters from a type name.

        Args:
            name: The type name to clean

        Returns:
            The cleaned type name without generic parameters
        """
        if "[" in name or "<" in name:
            cleaned = re.sub(r"\[.*\]", "", name)
            logger.debug(f"Cleaned generic type {name} to {cleaned}")
            return cleaned
        return name
