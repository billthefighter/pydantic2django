import inspect
import logging
import re
from collections.abc import Callable
from dataclasses import is_dataclass
from typing import Any, TypeVar, Union, get_args, get_origin

from pydantic import BaseModel

# Configure logger
logger = logging.getLogger("pydantic2django.core.typing")


# Add a function to configure logging
def configure_core_typing_logging(
    level=logging.WARNING, format_str="%Y-%m-%d %H:%M:%S - %(name)s - %(levelname)s - %(message)s"
):
    """
    Configure the logging for core typing module.

    Args:
        level: The logging level (e.g., logging.DEBUG, logging.INFO)
        format_str: The format string for log messages
    """
    handler = logging.StreamHandler()
    formatter = logging.Formatter(format_str)
    handler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False  # Prevent duplicate logging

    logger.debug("Core typing logging configured")


# Simplified TypeHandler focusing on processing and import generation


class TypeHandler:
    PATTERNS = {
        "angle_bracket_class": re.compile(r"<class '([^']+)'>"),
    }

    @staticmethod
    def _add_import(imports: dict[str, list[str]], module: str, name: str):
        """Safely add an import to the dictionary."""
        if not module or module == "builtins":
            return
        # Avoid adding the module itself if name matches module (e.g., import datetime)
        # if name == module.split('.')[-1]:
        #     name = module # This logic might be too simplistic, revert for now
        current_names = imports.setdefault(module, [])
        if name not in current_names:
            current_names.append(name)

    @staticmethod
    def _merge_imports(dict1: dict, dict2: dict) -> dict:
        """Merge two import dictionaries."""
        merged = dict1.copy()
        for module, names in dict2.items():
            current_names = merged.setdefault(module, [])
            for name in names:
                if name not in current_names:
                    current_names.append(name)
        # Sort names within each module for consistency
        for module in merged:
            merged[module].sort()
        return merged

    @staticmethod
    def get_class_name(type_obj: Any) -> str:
        """Extract a simple, usable class name from a type object."""
        origin = get_origin(type_obj)
        args = get_args(type_obj)

        # Check for Optional[T] specifically first (Union[T, NoneType])
        if origin is Union and len(args) == 2 and type(None) in args:
            return "Optional"

        if origin:
            # Now check for other origins
            if origin is Union:  # Handles Union[A, B, ...]
                return "Union"
            if origin is list:
                return "List"  # Use capital L consistently
            if origin is dict:
                return "Dict"  # Use capital D consistently
            if origin is tuple:
                return "Tuple"  # Use capital T consistently
            if origin is set:
                return "Set"  # Use capital S consistently
            if origin is Callable:
                return "Callable"
            if origin is type:
                return "Type"
            # Fallback for other generic types
            return getattr(origin, "__name__", str(origin))

        # Handle non-generic types
        if hasattr(type_obj, "__name__"):
            return type_obj.__name__

        type_str = str(type_obj)
        match = TypeHandler.PATTERNS["angle_bracket_class"].match(type_str)
        if match:
            return match.group(1).split(".")[-1]

        return str(type_obj)

    @staticmethod
    def get_required_imports(type_obj: Any) -> dict[str, list[str]]:
        """Determine necessary imports by traversing a type object."""
        imports: dict[str, list[str]] = {}
        processed_types = set()

        # Define modules for known Pydantic types that might need explicit import
        pydantic_module_map = {
            "EmailStr": "pydantic",
            "IPvAnyAddress": "pydantic",
            "Json": "pydantic",
            "BaseModel": "pydantic",
            # Add others if needed (e.g., SecretStr, UrlStr)
        }

        def _traverse(current_type: Any):
            nonlocal imports
            try:
                type_repr = repr(current_type)
                if type_repr in processed_types:
                    return
                processed_types.add(type_repr)
            except TypeError:
                # Handle unhashable types if necessary, e.g., log a warning
                pass

            origin = get_origin(current_type)
            args = get_args(current_type)

            if origin:
                # Handle Generic Alias (List, Dict, Union, Optional, Callable, Type)
                module_name = getattr(origin, "__module__", None)
                type_name = getattr(origin, "__name__", None)

                # Add import for the origin type itself if it's from typing
                if module_name == "typing":
                    if origin is Union:
                        # Optional is Union[T, None], handled below
                        is_optional = len(args) == 2 and type(None) in args
                        if is_optional:
                            TypeHandler._add_import(imports, "typing", "Optional")
                        else:
                            TypeHandler._add_import(imports, "typing", "Union")
                    # Add common typing constructs like List, Dict, Callable, Type, Any
                    elif type_name and type_name not in ("NoneType", "Generic"):
                        # Ensure capitalization matches actual typing names (List, Dict, etc.)
                        # Convert common lowercase names from get_origin to capitalized versions
                        capitalized_name = (
                            type_name.capitalize() if type_name in ["list", "dict", "tuple", "set"] else type_name
                        )
                        if capitalized_name in [
                            "List",
                            "Dict",
                            "Tuple",
                            "Set",
                            "Callable",
                            "Type",
                            "Any",
                            "Optional",
                            "Union",
                        ]:
                            TypeHandler._add_import(imports, "typing", capitalized_name)

                # Traverse arguments regardless of origin's module
                for arg in args:
                    if arg is not type(None):  # Skip NoneType in Optional/Union
                        if isinstance(arg, TypeVar):
                            # Handle TypeVar by traversing its constraints/bound
                            constraints = getattr(arg, "__constraints__", ())
                            bound = getattr(arg, "__bound__", None)
                            if bound:
                                _traverse(bound)
                            for constraint in constraints:
                                _traverse(constraint)
                        else:
                            _traverse(arg)  # Recursively traverse arguments
            # Handle Base Types or Classes (int, str, MyClass, etc.)
            elif isinstance(current_type, type):
                module_name = getattr(current_type, "__module__", "")
                type_name = getattr(current_type, "__name__", "")

                if not type_name or module_name == "builtins":
                    pass  # Skip builtins or types without names
                elif module_name == "typing" and type_name not in ("NoneType", "Generic"):
                    # Catch Any, etc. used directly
                    TypeHandler._add_import(imports, "typing", type_name)
                # Check for dataclasses and Pydantic models specifically
                elif is_dataclass(current_type) or (
                    inspect.isclass(current_type) and issubclass(current_type, BaseModel)
                ):
                    actual_module = inspect.getmodule(current_type)
                    if actual_module and actual_module.__name__ != "__main__":
                        TypeHandler._add_import(imports, actual_module.__name__, type_name)
                    # Add specific imports if needed (e.g., dataclasses.dataclass, pydantic.BaseModel)
                    if is_dataclass(current_type):
                        TypeHandler._add_import(imports, "dataclasses", "dataclass")
                    # No need to add BaseModel here usually, handled by pydantic_module_map or direct usage
                elif module_name:
                    # Handle known standard library modules explicitly
                    known_stdlib = {"datetime", "decimal", "uuid", "pathlib"}
                    if module_name in known_stdlib:
                        TypeHandler._add_import(imports, module_name, type_name)
                    # Handle known Pydantic types explicitly (redundant with BaseModel check?)
                    elif type_name in pydantic_module_map:
                        TypeHandler._add_import(imports, pydantic_module_map[type_name], type_name)
                    # Assume other types defined in modules need importing
                    elif module_name != "__main__":  # Avoid importing from main script context
                        TypeHandler._add_import(imports, module_name, type_name)

            elif current_type is Any:
                TypeHandler._add_import(imports, "typing", "Any")
            elif isinstance(current_type, TypeVar):
                # Handle TypeVar used directly
                constraints = getattr(current_type, "__constraints__", ())
                bound = getattr(current_type, "__bound__", None)
                if bound:
                    _traverse(bound)
                for c in constraints:
                    _traverse(c)
            # Consider adding ForwardRef handling if needed:
            # elif isinstance(current_type, typing.ForwardRef):
            #     # Potentially add logic to resolve/import forward refs
            #     pass

        _traverse(type_obj)

        # Clean up imports (unique, sorted)
        final_imports = {}
        for module, names in imports.items():
            unique_names = sorted(list(set(names)))
            if unique_names:
                final_imports[module] = unique_names
        return final_imports

    @staticmethod
    def process_field_type(field_type: Any) -> dict[str, Any]:
        """Process a field type to get name, flags, and imports."""
        logger.debug(f"Processing field type: {field_type}")
        is_optional = False
        is_list = False
        base_type_obj = field_type

        origin = get_origin(base_type_obj)
        args = get_args(base_type_obj)
        if origin is Union and type(None) in args and len(args) == 2:
            is_optional = True
            base_type_obj = next(arg for arg in args if arg is not type(None))
            logger.debug(f"Unwrapped Optional. Inner type: {base_type_obj}")
            origin = get_origin(base_type_obj)
            args = get_args(base_type_obj)

        if origin is list or origin is list:
            is_list = True
            if args:
                base_type_obj = args[0]
                logger.debug(f"Unwrapped List. Inner type: {base_type_obj}")
            else:
                base_type_obj = Any
                logger.debug("Unwrapped List without args, assuming Any.")

        all_imports = TypeHandler.get_required_imports(field_type)
        type_name = TypeHandler.get_class_name(base_type_obj)
        logger.debug(f"Base type name: {type_name}")

        result = {
            "type": type_name,
            "is_optional": is_optional,
            "is_list": is_list,
            "imports": all_imports,
        }
        logger.debug(f"Processed type result: {result}")
        return result

    @staticmethod
    def format_type_string(type_obj: Any) -> str:
        """Return a string representation suitable for generated code."""
        # This version reconstructs the type string from processed info
        processed = TypeHandler.process_field_type(type_obj)
        base_name = processed["type"]
        # Attempt to get a more precise name if it's a known complex type
        if base_name in ["Union", "Callable", "Type", "Dict"]:
            raw_repr = TypeHandler._get_raw_type_string(type_obj)
            base_name = raw_repr.replace("typing.", "")
            match = TypeHandler.PATTERNS["angle_bracket_class"].match(base_name)
            if match:
                base_name = match.group(1).split(".")[-1]

        if processed["is_list"]:
            # Avoid double wrapping if base_name is already List[...]
            if not base_name.startswith("List["):
                base_name = f"List[{base_name}]"
        if processed["is_optional"]:
            # Avoid double wrapping if base_name is already Optional[...]
            if not base_name.startswith("Optional["):
                base_name = f"Optional[{base_name}]"
        return base_name

    @staticmethod
    def _get_raw_type_string(type_obj: Any) -> str:
        module = getattr(type_obj, "__module__", "")
        if module == "typing":
            return repr(type_obj).replace("typing.", "")
        # Use name for classes/dataclasses
        if hasattr(type_obj, "__name__") and isinstance(type_obj, type):
            return type_obj.__name__
        # Fallback to str
        return str(type_obj)
