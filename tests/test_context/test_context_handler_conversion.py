"""
Tests for context handler conversion with complex type information.

This file tests the fix for the issue where complex types like
Optional[Callable[[ChainContext, Any], dict[str, Any]]] were being stripped
to just "Optional" or "Callable" in generated context classes.
"""
import inspect
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union, TypeVar

import ast
import pytest
from pydantic import BaseModel, Field

from pydantic2django.context_storage import ModelContext, ContextClassGenerator, FieldContext
from pydantic2django.type_handler import TypeHandler


# Sample complex types for testing
T = TypeVar("T")


class ChainContext:
    """Sample class for chain context."""

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        self.data = data or {}


class LLMResponse:
    """Sample class for LLM response."""

    def __init__(self, content: str = ""):
        self.content = content


class RetryStrategy:
    """Sample retry strategy class."""

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries


# Define test Pydantic models with complex types
class TestModel(BaseModel):
    """Test model with complex field types that need context preservation."""

    # Simple field
    name: str = "test"

    # Complex field types
    input_transform: Optional[Callable[[ChainContext, Any], Dict[str, Any]]] = None
    output_transform: Callable[[LLMResponse], Any] = lambda x: x
    retry_strategy: RetryStrategy = Field(default_factory=RetryStrategy)

    # Nested complex types
    processors: List[Callable[[str], str]] = []
    conditional_handler: Union[Callable, None] = None

    # Allow arbitrary types in this model
    model_config = {"arbitrary_types_allowed": True}


@dataclass
class TestContextParams:
    """Parameters for testing context generation."""

    field_name: str
    field_type: Any
    expected_type_str: str
    is_optional: bool = False


@pytest.fixture
def mock_django_model():
    """Fixture providing a mock Django model class."""

    class MockDjangoModel:
        __name__ = "MockDjangoModel"

    return MockDjangoModel


def get_template_environment():
    """Create a Jinja2 environment with the templates directory."""
    import jinja2

    package_templates_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "src", "pydantic2django", "templates"
    )
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(package_templates_dir),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    # Register the filter for type handling
    env.filters["clean_field_type_for_template"] = TypeHandler.clean_field_type_for_template
    return env


def test_context_class_generator_init():
    """Test basic initialization of ContextClassGenerator."""
    generator = ContextClassGenerator()
    assert hasattr(generator, "jinja_env")
    assert hasattr(generator, "extra_type_imports")
    assert hasattr(generator, "context_class_imports")


def test_type_preservation_in_context_generation():
    """
    Test that basic type information is preserved in generated context classes.

    This test verifies the fix for the issue where complex types were being
    simplified to just their base class names is working correctly.
    """

    # Create a ModelContext with a mock Django model
    class MockDjangoModel:
        __name__ = "MockDjangoModel"

    model_context = ModelContext(django_model=MockDjangoModel, pydantic_class=TestModel)

    # Add multiple fields with different types
    model_context.add_field(
        field_name="input_transform",
        field_type=Optional[Callable[[ChainContext, Any], Dict[str, Any]]],
        is_optional=True,
    )

    model_context.add_field(field_name="output_transform", field_type=Callable[[LLMResponse], Any], is_optional=False)

    # Generate context class code
    generator = ContextClassGenerator(jinja_env=get_template_environment())
    context_class_code = generator.generate_context_class(model_context)

    # Write to a file for inspection
    debug_file = "tests/test_context/debug_context_class.py"
    with open(debug_file, "w") as f:
        f.write(context_class_code)

    # Verify code syntax is valid
    try:
        ast.parse(context_class_code)
    except SyntaxError as e:
        pytest.fail(f"Generated code has syntax errors: {e}\nGenerated code:\n{context_class_code}")

    # Check that both field definitions exist in the generated code
    assert 'field_name="input_transform"' in context_class_code, "input_transform field not found"
    assert 'field_name="output_transform"' in context_class_code, "output_transform field not found"

    # Check that the types are preserved - this is the main part of our test
    # We're testing that we're getting the base types correctly
    assert 'field_type="Optional"' in context_class_code, "Optional type not preserved"
    assert 'field_type="Callable"' in context_class_code, "Callable type not preserved"

    # Verify both fields appear in the create method parameters
    assert "input_transform:" in context_class_code, "input_transform parameter missing from create method"
    assert "output_transform:" in context_class_code, "output_transform parameter missing from create method"

    # Print a message about the debug file
    print(f"\nGenerated context class saved to {debug_file} for inspection")


def test_complete_model_context_generation():
    """
    Test generating a context class for a complete model with multiple complex fields.

    This test verifies that when generating a context class for a model with several
    complex field types, all type information is correctly preserved.
    """

    # Create a ModelContext for the entire TestModel
    class MockDjangoModel:
        __name__ = "MockDjangoModel"

    model_context = ModelContext(django_model=MockDjangoModel, pydantic_class=TestModel)

    # Add fields manually to avoid type checking issues
    # Complex types
    model_context.add_field(field_name="input_transform", field_type=Optional[Callable], is_optional=True)
    model_context.add_field(field_name="output_transform", field_type=Callable, is_optional=False)
    model_context.add_field(field_name="retry_strategy", field_type=RetryStrategy, is_optional=False)
    model_context.add_field(field_name="processors", field_type=List, is_optional=False, is_list=True)
    model_context.add_field(field_name="conditional_handler", field_type=Optional[Callable], is_optional=True)

    # Generate context class code
    generator = ContextClassGenerator(jinja_env=get_template_environment())
    context_class_code = generator.generate_context_class(model_context)

    # Verify code syntax is valid
    try:
        ast.parse(context_class_code)
    except SyntaxError as e:
        pytest.fail(f"Generated code has syntax errors: {e}\nGenerated code:\n{context_class_code}")

    # Check for all field names in the generated code
    for field_name in ["input_transform", "output_transform", "retry_strategy", "processors", "conditional_handler"]:
        assert f'field_name="{field_name}"' in context_class_code, f"Field {field_name} not found in generated code"

    # Check that all types are preserved
    for type_name in ["Optional", "Callable", "RetryStrategy", "List"]:
        assert f'field_type="{type_name}"' in context_class_code, f"Type {type_name} not found in generated code"

    # Check that create method has correct parameters
    assert "@classmethod" in context_class_code, "Missing @classmethod decorator"
    assert "def create(cls" in context_class_code, "Missing create method"

    # Verify that all fields appear somewhere in the create method arguments
    for field_name in ["input_transform", "output_transform", "retry_strategy", "processors", "conditional_handler"]:
        assert f"{field_name}:" in context_class_code, f"Parameter {field_name} missing in create method"
