from typing import Any, Union

from django.utils.functional import Promise  # Keep for sanitize_string


def sanitize_string(value: Union[str, Promise, Any]) -> str:
    """
    Sanitize a string value, handling Django's lazy translation Promises.

    Ensures the output is a clean string, escaping potential issues for templates or code generation.

    Args:
        value: The value to sanitize (can be str, Promise, or other types)

    Returns:
        A sanitized string representation.
    """
    processed_value: str
    # Handle lazy translation objects first
    if isinstance(value, Promise):
        try:
            # Force the promise to evaluate to a string
            processed_value = str(value)
        except Exception:
            # Fallback if evaluation fails
            processed_value = ""  # Or some default placeholder
            # Consider logging this error
            # logger.error(f"Could not resolve Django Promise: {e}")
    elif isinstance(value, str):
        processed_value = value
    else:
        # Handle Any other type by converting to string
        try:
            processed_value = str(value)
        except Exception:
            # If conversion to string fails, return a default empty string
            # logger.warning(f"Could not convert value of type {type(value)} to string in sanitize_string")
            processed_value = ""

    # Basic sanitization: escape backslashes and quotes, and normalize newlines
    # This is crucial for safely embedding the string in generated Python code
    sanitized = processed_value.replace("\\", "\\\\")  # Escape backslashes first
    sanitized = sanitized.replace("\n", "\\n").replace("\r", "")  # Normalize newlines
    sanitized = sanitized.replace("'", "\\'")  # Escape single quotes
    sanitized = sanitized.replace('"', '\\"')  # Escape double quotes for double-quoted strings

    return sanitized


def balanced(s: str) -> bool:
    """
    Check if a string containing brackets, parentheses, or braces is balanced.

    Args:
        s: The string to check.

    Returns:
        True if balanced, False otherwise.
    """
    bracket_map = {")": "(", "]": "[", "}": "{"}
    opening_brackets = set(bracket_map.values())
    stack = []

    for char in s:
        if char in opening_brackets:
            stack.append(char)
        elif char in bracket_map:
            if not stack or stack[-1] != bracket_map[char]:
                return False
            stack.pop()

    return not stack  # Stack should be empty if balanced
