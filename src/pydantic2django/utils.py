"""
Utility functions for pydantic2django.
"""
import re


def normalize_model_name(name: str) -> str:
    """
    Normalize a model name by removing generic type parameters and ensuring proper Django model naming.

    Args:
        name: The model name to normalize

    Returns:
        Normalized model name
    """
    # Remove generic type parameters
    name = re.sub(r"\[.*?\]", "", name)

    # Ensure Django prefix
    if not name.startswith("Django"):
        name = f"Django{name}"

    return name
