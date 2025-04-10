import inspect
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import is_dataclass
from typing import Any, Literal, TypeVar, Union, get_args, get_origin

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
        """Process a field type to get name, flags, imports, and contained dataclasses."""
        logger.debug(f"Received field_type: {field_type!r}, type: {type(field_type)}")
        logger.debug(f"Processing field type: {field_type!r}")
        is_optional = False
        is_list = False
        imports = {}
        contained_dataclasses: set[type] = set()
        simplified_type = field_type  # Start with the original

        # Helper function (remains the same)
        def _is_potential_dataclass(t: Any) -> bool:
            return inspect.isclass(t) and is_dataclass(t)

        def _find_contained_dataclasses(current_type: Any):
            origin = get_origin(current_type)
            args = get_args(current_type)
            if origin:
                for arg in args:
                    if arg is not type(None):
                        _find_contained_dataclasses(arg)
            elif _is_potential_dataclass(current_type):
                contained_dataclasses.add(current_type)

        _find_contained_dataclasses(field_type)
        if contained_dataclasses:
            logger.debug(f"  Found potential contained dataclasses: {[dc.__name__ for dc in contained_dataclasses]}")

        # --- Simplification Loop ---
        # Repeatedly unwrap until we hit a base type or Any
        processed = True
        while processed:
            processed = False
            origin = get_origin(simplified_type)
            args = get_args(simplified_type)

            # 1. Unwrap Optional[T] (Union[T, NoneType])
            if origin is Union and len(args) == 2 and type(None) in args:
                is_optional = True  # Flag it
                simplified_type = next(arg for arg in args if arg is not type(None))
                logger.debug(f"  Unwrapped Optional, current type: {simplified_type!r}")
                processed = True
                continue  # Restart loop with unwrapped type

            # 2. Unwrap List[T] or Sequence[T]
            if origin in (list, Sequence):
                is_list = True  # Flag it
                if args:
                    simplified_type = args[0]
                    logger.debug(f"  Unwrapped List/Sequence, current element type: {simplified_type!r}")
                else:
                    simplified_type = Any  # List without args -> List[Any]
                    logger.debug("  Unwrapped List/Sequence without args, assuming Any")
                processed = True
                continue  # Restart loop with unwrapped element type

            # 3. Unwrap Literal[...]
            if origin is Literal:
                if args:
                    simplified_type = type(args[0])  # Use type of the *value*
                    logger.debug(f"  Unwrapped Literal, current type: {simplified_type!r}")
                else:
                    simplified_type = Any  # Literal without args?
                    logger.debug("  Unwrapped Literal without args? Assuming Any")
                processed = True
                continue  # Restart loop with unwrapped type

        # --- Post-Loop Handling ---
        # At this point, simplified_type should be the base type (int, str, datetime, Any, etc.)
        # or a complex type we don't simplify further (like a raw Union or a specific class)
        base_type_obj = simplified_type

        # Final check: If it's still a complex Union, default to Any for mapping
        origin = get_origin(base_type_obj)
        if origin is Union:  # Handles Union[A, B] etc.
            logger.debug(f"  Final type is complex Union {base_type_obj!r}, defaulting base object to Any for mapping.")
            base_type_obj = Any

        # --- Result Assembly ---
        imports = TypeHandler.get_required_imports(field_type)  # Imports based on original
        type_string = TypeHandler.format_type_string(field_type)  # Formatting based on original

        result = {
            "type_str": type_string,
            "type_obj": base_type_obj,  # THIS is the crucial simplified type object
            "is_optional": is_optional,
            "is_list": is_list,
            "imports": imports,
            "contained_dataclasses": contained_dataclasses,
        }
        logger.debug(f"  Processed type result: {result!r}")  # Use !r for clearer debug
        return result

    @staticmethod
    def format_type_string(type_obj: Any) -> str:
        """Return a string representation suitable for generated code."""
        # --- Simplified version to break recursion ---
        # Get the raw string representation first
        raw_repr = TypeHandler._get_raw_type_string(type_obj)

        # Basic cleanup for common typing constructs
        base_name = raw_repr.replace("typing.", "")

        # Attempt to refine based on origin/args if needed (optional)
        origin = get_origin(type_obj)
        args = get_args(type_obj)

        if origin is Union and len(args) == 2 and type(None) in args:
            # Handle Optional[T]
            inner_type_str = TypeHandler.format_type_string(next(arg for arg in args if arg is not type(None)))
            return f"Optional[{inner_type_str}]"
        elif origin in (list, Sequence):
            # Handle List[T] / Sequence[T]
            if args:
                inner_type_str = TypeHandler.format_type_string(args[0])
                return f"List[{inner_type_str}]"  # Prefer List for generated code
            else:
                return "List[Any]"
        elif origin is Union:  # Non-optional Union
            inner_types = [TypeHandler.format_type_string(arg) for arg in args]
            return f"Union[{', '.join(inner_types)}]"
        elif origin is Literal:
            inner_values = [repr(arg) for arg in args]
            return f"Literal[{', '.join(inner_values)}]"
        # Add other origins like Dict, Tuple, Callable if needed

        # Fallback to the cleaned raw representation
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
