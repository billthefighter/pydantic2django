import logging
import re
from typing import Any, Optional

# Configure logging
logger = logging.getLogger("pydantic2django.import_handler")


class ImportHandler:
    """
    Handles import statements for generated Django models and their context classes.
    Tracks and deduplicates imports from multiple sources while ensuring all necessary
    dependencies are included.
    """

    def __init__(self, module_mappings: Optional[dict[str, str]] = None):
        """
        Initialize empty collections for different types of imports.

        Args:
            module_mappings: Optional mapping of modules to remap (e.g. {"__main__": "my_app.models"})
        """
        # Track imports by category
        self.extra_type_imports: set[str] = set()  # For typing and other utility imports
        self.pydantic_imports: set[str] = set()  # For Pydantic model imports
        self.context_class_imports: set[str] = set()  # For context class and field type imports

        # For tracking imported names to avoid duplicates
        self.imported_names: dict[str, str] = {}  # Maps type name to its module

        # For tracking field type dependencies we've already processed
        self.processed_field_types: set[str] = set()

        # Module mappings to remap imports (e.g. "__main__" -> "my_app.models")
        self.module_mappings = module_mappings or {}

        logger.info("ImportHandler initialized")
        if self.module_mappings:
            logger.info(f"Using module mappings: {self.module_mappings}")

    def add_import(self, module: str, name: str):
        """Adds a single import based on module and name strings."""
        if not module or module == "builtins":
            return

        # Apply module mappings
        if module in self.module_mappings:
            module = self.module_mappings[module]

        # Clean name (e.g., remove generics for import statement)
        clean_name = self._clean_generic_type(name)

        # Check if already imported
        if name in self.imported_names:
            # Could verify module matches, but usually name is unique enough
            logger.debug(f"Skipping already imported name: {name} (from module {module})")
            return
        if clean_name != name and clean_name in self.imported_names:
            logger.debug(f"Skipping already imported clean name: {clean_name} (from module {module})")
            return

        # Determine category
        # Simplistic: If module is known Pydantic, Django, or common stdlib -> context
        # Otherwise, if it's 'typing' -> extra_type
        # TODO: Refine categorization if needed (e.g., dedicated django_imports set)
        import_statement = f"from {module} import {clean_name}"
        if module == "typing":
            self.extra_type_imports.add(clean_name)  # Add only name to typing imports set
            logger.debug(f"Adding typing import: {clean_name}")
        # elif module.startswith("django."):
        # Add to a dedicated django set if we create one
        #    self.context_class_imports.add(import_statement)
        #    logger.info(f"Adding Django import: {import_statement}")
        else:
            # Default to context imports for non-typing
            self.context_class_imports.add(import_statement)
            logger.info(f"Adding context class import: {import_statement}")

        # Mark as imported
        self.imported_names[name] = module
        if clean_name != name:
            self.imported_names[clean_name] = module

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

        # Apply module mappings if needed
        if module_path in self.module_mappings:
            actual_module = self.module_mappings[module_path]
            logger.debug(f"Remapping module import: {module_path} -> {actual_module}")
            module_path = actual_module

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
        Add an import statement for a context field type with recursive dependency detection.

        Args:
            field_type: The field type to import
        """
        # Skip if we've already processed this field type
        field_type_str = str(field_type)
        if field_type_str in self.processed_field_types:
            logger.debug(f"Skipping already processed field type: {field_type_str}")
            return

        logger.info(f"Processing context field type: {field_type_str}")
        self.processed_field_types.add(field_type_str)

        # Try to add direct import for the field type if it's a class
        self._add_type_import(field_type)

        # Handle nested types in generics, unions, etc.
        self._process_nested_types(field_type)

        # Add typing imports based on the field type string
        self._add_typing_imports(field_type_str)

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

                # Apply module mappings if needed
                if type_module in self.module_mappings:
                    actual_module = self.module_mappings[type_module]
                    logger.debug(f"Remapping module import: {type_module} -> {actual_module}")
                    type_module = actual_module

                logger.debug(f"Examining type: {type_name} from module {type_module}")

                # Skip built-in types and typing module types
                if (
                    type_module.startswith("typing")
                    or type_module == "builtins"
                    or type_name in ["str", "int", "float", "bool", "dict", "list"]
                ):
                    logger.debug(f"Skipping built-in or typing type: {type_name}")
                    return

                # Skip TypeVar definitions to avoid conflicts
                if type_name == "T" or type_name == "TypeVar":
                    logger.debug(f"Skipping TypeVar definition: {type_name} - will be defined locally")
                    return

                # Clean up any parametrized generic types for the import statement
                clean_type_name = self._clean_generic_type(type_name)

                # Use the original type_name (potentially with generics) for the imported_names check
                if type_name in self.imported_names:
                    logger.debug(f"Skipping already imported type: {type_name}")
                    return

                # Add to context class imports *before* marking as imported
                # Use the clean name for the import statement itself
                import_statement = f"from {type_module} import {clean_type_name}"
                logger.info(f"Adding context class import: {import_statement}")
                self.context_class_imports.add(import_statement)

                # Add the original type name to imported_names to prevent re-processing
                self.imported_names[type_name] = type_module
                # Also add the cleaned name in case it's encountered separately
                if clean_type_name != type_name:
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

        # Handle special case for TypeVar imports
        typevars = set()

        for import_stmt in self.pydantic_imports:
            if import_stmt.startswith("from ") and " import " in import_stmt:
                module, classes = import_stmt.split(" import ")
                module = module.replace("from ", "")

                # Skip __main__ and rewrite to real module paths if possible
                if module == "__main__":
                    logger.warning(f"Skipping __main__ import: {import_stmt} - these won't work when imported")
                    continue

                for cls in classes.split(", "):
                    # Check if it's a TypeVar to handle duplicate definitions
                    if cls == "T" or cls == "TypeVar":
                        typevars.add(cls)
                        continue

                    # Clean up any parameterized generic types in class names
                    cls = self._clean_generic_type(cls)
                    pydantic_classes[cls] = module

        for import_stmt in self.context_class_imports:
            if import_stmt.startswith("from ") and " import " in import_stmt:
                module, classes = import_stmt.split(" import ")
                module = module.replace("from ", "")

                # Skip __main__ imports or rewrite to real module paths if possible
                if module == "__main__":
                    logger.warning(f"Skipping __main__ import: {import_stmt} - these won't work when imported")
                    continue

                for cls in classes.split(", "):
                    # Check if it's a TypeVar to handle duplicate definitions
                    if cls == "T" or cls == "TypeVar":
                        typevars.add(cls)
                        continue

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

        # Log any TypeVar names we're skipping
        if typevars:
            logger.info(f"Skipping TypeVar imports: {typevars} - these will be defined locally")

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
