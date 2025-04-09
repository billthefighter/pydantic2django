"""
Tests for the TypeHandler.get_class_name method.

This file tests the hypothesis that TypeHandler.get_class_name is truncating
complex types like Optional[Callable[[ChainContext, Any], dict[str, Any]]]
to just "Optional" when used in the context class template.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import pytest

from pydantic2django.core.typing import TypeHandler


class ChainContext:
    """Sample class for chain context."""

    pass


class LLMResponse:
    """Sample class for LLM response."""

    pass


class RetryStrategy:
    """Sample retry strategy class."""

    pass


def test_get_class_name_with_simple_types():
    """Test get_class_name with simple types."""
    assert TypeHandler.get_class_name(str) == "str"
    assert TypeHandler.get_class_name(int) == "int"
    assert TypeHandler.get_class_name(bool) == "bool"
    assert TypeHandler.get_class_name(list) == "list"
    assert TypeHandler.get_class_name(dict) == "dict"


def test_get_class_name_with_custom_classes():
    """Test get_class_name with custom classes."""
    assert TypeHandler.get_class_name(ChainContext) == "ChainContext"
    assert TypeHandler.get_class_name(LLMResponse) == "LLMResponse"
    assert TypeHandler.get_class_name(RetryStrategy) == "RetryStrategy"


def test_get_class_name_with_complex_types():
    """
    Test get_class_name with complex types - note this method intentionally
    extracts only the base class name and discards parameters.
    """
    # get_class_name intentionally extracts only the base type name
    # This behavior is by design for its original purpose, but not
    # suitable for preserving complex type information in templates
    assert TypeHandler.get_class_name(Optional[int]) == "Optional"
    assert TypeHandler.get_class_name(List[str]) == "List"
    assert TypeHandler.get_class_name(Dict[str, Any]) == "Dict"

    # This critical case shows the behavior that was causing the bug
    complex_type = Optional[Callable[[ChainContext, Any], Dict[str, Any]]]
    result = TypeHandler.get_class_name(complex_type)
    assert result == "Optional"


def test_process_field_type_vs_get_class_name():
    """
    Test the difference between process_field_type and get_class_name.

    This demonstrates that process_field_type correctly preserves type information
    while get_class_name truncates it.
    """
    complex_type = Optional[Callable[[ChainContext, Any], Dict[str, Any]]]

    # process_field_type preserves complex type information (used in ModelContext.get_formatted_field_type)
    type_str, _ = TypeHandler.process_field_type(complex_type)
    assert "Optional" in type_str
    assert "Callable" in type_str

    # get_class_name truncates to just the base type (used in clean_field_type_for_template)
    class_name = TypeHandler.get_class_name(complex_type)
    assert class_name == "Optional"
    assert "Callable" not in class_name


def test_clean_field_type_for_template(type_str, expected):
    """
    Test the clean_field_type_for_template method with various inputs.
    """
    # Use the likely replacement method
    assert TypeHandler.clean_type_string(type_str) == expected
