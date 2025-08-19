import inspect
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import is_dataclass
from functools import reduce
from types import UnionType
from typing import Annotated, Any, Literal, TypeVar, Union, get_args, get_origin

from pydantic import BaseModel

# Configure logger
logger = logging.getLogger("pydantic2django.core.typing")


# Add a function to configure logging
def configure_core_typing_logging(
    level: int = logging.WARNING,
    format_str: str = "%Y-%m-%d %H:%M:%S - %(name)s - %(levelname)s - %(message)s",
) -> None:
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
        if origin in (Union, UnionType) and len(args) == 2 and type(None) in args:
            return "Optional"

        if origin:
            # Now check for other origins
            if origin in (Union, UnionType):  # Handles Union[A, B, ...]
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
                origin_module = getattr(origin, "__module__", "")
                origin_name = getattr(origin, "__name__", "")

                # Determine the canonical name used in 'typing' imports (e.g., List, Dict, Callable)
                typing_name = None
                if origin is list:
                    typing_name = "List"
                elif origin is dict:
                    typing_name = "Dict"
                elif origin is tuple:
                    typing_name = "Tuple"
                elif origin is set:
                    typing_name = "Set"
                elif origin in (Union, UnionType):  # Handle types.UnionType for Python 3.10+
                    # We don't need to add Union or Optional imports anymore with | syntax
                    typing_name = None
                elif origin is type:
                    typing_name = "Type"
                # Check both typing.Callable and collections.abc.Callable
                elif origin_module == "typing" and origin_name == "Callable":
                    typing_name = "Callable"
                elif origin_module == "collections.abc" and origin_name == "Callable":
                    typing_name = "Callable"
                # Add more specific checks if needed (e.g., Sequence, Mapping)

                # Add import if we identified a standard typing construct
                if typing_name:
                    TypeHandler._add_import(imports, "typing", typing_name)

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
            unique_names = sorted(set(names))
            if unique_names:
                final_imports[module] = unique_names
        return final_imports

    @staticmethod
    def process_field_type(field_type: Any) -> dict[str, Any]:
        """Process a field type to get name, flags, imports, and contained dataclasses."""
        logger.debug(f"[TypeHandler] Processing type: {field_type!r}")
        is_optional = False
        is_list = False
        metadata: tuple[Any, ...] | None = None  # Initialize metadata with type hint
        imports = set()
        contained_dataclasses = set()
        current_type = field_type  # Keep track of the potentially unwrapped type

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
            origin = get_origin(current_type)
            args = get_args(current_type)

            # 0. Unwrap Annotated[T, ...]
            # Check if the origin exists and has the name 'Annotated'
            # This check is more robust than `origin is Annotated` across Python versions
            if origin is Annotated:
                if args:
                    core_type = args[0]
                    metadata = args[1:]
                    current_type = core_type
                    logger.debug(f"  Unwrapped Annotated, current type: {current_type!r}, metadata: {metadata!r}")
                    processed = True
                    continue  # Restart loop with unwrapped type
                else:
                    logger.warning("  Found Annotated without arguments? Treating as Any.")
                    current_type = Any
                    processed = True
                    continue

            # 1. Unwrap Optional[T] (Union[T, NoneType])
            if origin in (Union, UnionType) and type(None) in args:
                is_optional = True  # Flag it
                # Rebuild the Union without NoneType
                non_none_args = tuple(arg for arg in args if arg is not type(None))
                if len(non_none_args) == 1:
                    current_type = non_none_args[0]  # Simplify Union[T, None] to T
                elif len(non_none_args) > 1:
                    # Use UnionType to rebuild
                    current_type = reduce(lambda x, y: x | y, non_none_args)
                else:  # pragma: no cover
                    # Should not happen if NoneType was in args
                    current_type = Any
                logger.debug(f"  Unwrapped Union with None, current type: {current_type!r}")
                processed = True
                continue  # Restart loop with the non-optional type

            # 2. Unwrap List[T] or Sequence[T]
            if origin in (list, Sequence):
                is_list = True  # Flag it
                if args:
                    current_type = args[0]
                    logger.debug(f"  Unwrapped List/Sequence, current element type: {current_type!r}")
                else:
                    current_type = Any  # List without args -> List[Any]
                    logger.debug("  Unwrapped List/Sequence without args, assuming Any")
                processed = True
                continue  # Restart loop with unwrapped element type

            # 3. Unwrap Literal[...]
            if origin is Literal:
                # Keep the Literal origin, but simplify args if possible?
                # No, the mapper needs the original Literal to extract choices.
                # Just log and break the loop for Literal.
                logger.debug("  Hit Literal origin, stopping simplification loop.")
                break  # Stop simplification here, keep Literal type

        # --- Post-Loop Handling ---
        # At this point, current_type should be the base type (int, str, datetime, Any, etc.)
        # or a complex type we don't simplify further (like a raw Union or a specific class)
        base_type_obj = current_type

        # --- FIX: If the original type was a list, ensure base_type_obj reflects the *List* --- #
        # The simplification loop above sets current_type to the *inner* type of the list.
        # We need the actual List type for the mapper logic.
        if is_list:
            # Determine the simplified inner type from the end of the loop
            simplified_inner_type = base_type_obj

            # Check if the original type involved Optional wrapping the list
            # A simple check: was is_optional also flagged?
            if is_optional:
                # Reconstruct Optional[List[SimplifiedInner]]
                reconstructed_type = list[simplified_inner_type] | None
                logger.debug(
                    f"  Original was Optional[List-like]. Reconstructing List[...] | None "
                    f"around simplified inner type {simplified_inner_type!r} -> {reconstructed_type!r}"
                )
            else:
                # Reconstruct List[SimplifiedInner]
                reconstructed_type = list[simplified_inner_type]
                logger.debug(
                    f"  Original was List-like (non-optional). Reconstructing List[...] "
                    f"around simplified inner type {simplified_inner_type!r} -> {reconstructed_type!r}"
                )

            # Check against original type structure (might be more robust but complex?)
            # original_origin = get_origin(field_type)
            # if original_origin is Optional and get_origin(get_args(field_type)[0]) in (list, Sequence):
            #     # Handle Optional[List[...]] structure
            # elif original_origin in (list, Sequence):
            #     # Handle List[...] structure
            # else:
            #     # Handle complex cases like Annotated[Optional[List[...]]]

            base_type_obj = reconstructed_type

        # --- End FIX --- #

        # Add check for Callable simplification
        origin = get_origin(base_type_obj)
        if origin is Callable or (
            hasattr(base_type_obj, "__module__")
            and base_type_obj.__module__ == "collections.abc"
            and base_type_obj.__name__ == "Callable"
        ):
            logger.debug(
                f"  Final type is complex Callable {base_type_obj!r}, simplifying base object to Callable origin."
            )
            base_type_obj = Callable

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
            "metadata": metadata,
        }
        logger.debug(f"[TypeHandler] Processed result: {result!r}")
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

        if origin in (Union, UnionType) and len(args) == 2 and type(None) in args:
            # Handle Optional[T]
            inner_type_str = TypeHandler.format_type_string(next(arg for arg in args if arg is not type(None)))
            return f"{inner_type_str} | None"
        elif origin in (list, Sequence):
            # Handle List[T] / Sequence[T]
            if args:
                inner_type_str = TypeHandler.format_type_string(args[0])
                return f"List[{inner_type_str}]"  # Prefer List for generated code
            else:
                return "List[Any]"
        elif origin is dict:
            if args and len(args) == 2:
                key_type_str = TypeHandler.format_type_string(args[0])
                value_type_str = TypeHandler.format_type_string(args[1])
                return f"Dict[{key_type_str}, {value_type_str}]"
            else:
                return "dict"
        elif origin is Callable:
            if args:
                # For Callable[[A, B], R], args is ([A, B], R) in Py3.9+
                # For Callable[A, R], args is (A, R)
                # For Callable[[], R], args is ([], R)
                param_part = args[0]
                return_part = args[-1]

                if param_part is ...:
                    param_str = "..."
                elif isinstance(param_part, list):
                    param_types = [TypeHandler.format_type_string(p) for p in param_part]
                    param_str = f'[{", ".join(param_types)}]'
                else:  # Single argument
                    param_str = f"[{TypeHandler.format_type_string(param_part)}]"

                return_type_str = TypeHandler.format_type_string(return_part)
                return f"Callable[{param_str}, {return_type_str}]"
            else:
                return "Callable"
        elif origin in (Union, UnionType):  # Non-optional Union
            inner_types = [TypeHandler.format_type_string(arg) for arg in args]
            return " | ".join(inner_types)
        elif origin is Literal:
            inner_values = [repr(arg) for arg in args]
            return f"Literal[{', '.join(inner_values)}]"
        # Add other origins like Dict, Tuple, Callable if needed

        # Fallback to the cleaned raw representation
        return base_name.replace("collections.abc.", "")

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
