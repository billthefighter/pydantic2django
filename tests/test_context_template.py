"""
Tests for the context_class.py.j2 template to ensure it generates valid Python code.

These tests verify:
1. Template syntax is correct
2. Generated Python code is well-formatted and valid
3. Different field type patterns are properly handled
4. Template produces code that can be executed without syntax errors
"""
import ast
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

import jinja2
import pytest
from pydantic import BaseModel

from pydantic2django.context_storage import FieldContext, ModelContext
from pydantic2django.type_handler import TypeHandler


# Create mock class objects for testing class reference handling
class MockChainNode:
    __module__ = "llmaestro.chains.chains"
    __name__ = "ChainNode"


class MockConversationNode:
    __module__ = "llmaestro.core.conversations"
    __name__ = "ConversationNode"


class MockChainContext:
    __module__ = "llmaestro.chains.chains"
    __name__ = "ChainContext"


def get_template_environment():
    """Create a Jinja2 environment with the templates directory."""
    package_templates_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "src", "pydantic2django", "templates"
    )
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(package_templates_dir),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    # Register the filter for type handling
    env.filters["clean_field_type_for_template"] = TypeHandler.clean_field_type_for_template
    return env


@dataclass
class MockFieldDefinition:
    """Mock field definition for template testing."""

    name: str
    type: str
    is_optional: bool = False
    is_list: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class MockModel:
    """Mock Django model for template testing."""

    __name__ = "TestModel"


class MockPydanticClass:
    """Mock Pydantic class for template testing."""

    __name__ = "TestPydantic"
    __module__ = "test.module"


@pytest.mark.parametrize(
    "field_definitions",
    [
        # Test case with no fields
        [],
        # Test case with simple fields
        [
            MockFieldDefinition(name="field1", type="str"),
            MockFieldDefinition(name="field2", type="int"),
        ],
        # Test case with complex fields
        [
            MockFieldDefinition(name="field1", type="str"),
            MockFieldDefinition(name="complex_field", type="Complex[Type]"),
            MockFieldDefinition(name="callable_field", type="Callable[[str], int]"),
        ],
        # Test case with optional fields
        [
            MockFieldDefinition(name="required_field", type="str", is_optional=False),
            MockFieldDefinition(name="optional_field", type="Optional[str]", is_optional=True),
        ],
        # Test case with list fields
        [
            MockFieldDefinition(name="list_field", type="List[str]", is_list=True),
            MockFieldDefinition(name="optional_list", type="Optional[List[int]]", is_optional=True, is_list=True),
        ],
        # Test case with complex nested types
        [
            MockFieldDefinition(
                name="nested_field", type="Dict[str, List[Optional[Complex]]]", is_optional=False, is_list=False
            ),
        ],
        # Test case with metadata
        [
            MockFieldDefinition(name="meta_field", type="str", metadata={"description": "A field with metadata"}),
        ],
        # Test case with problematic field names and types
        [
            MockFieldDefinition(name="field_with,comma", type="Type,With,Commas"),
            MockFieldDefinition(name="field_with_spaces", type="Type With Spaces"),
        ],
    ],
    ids=[
        "no_fields",
        "simple_fields",
        "complex_fields",
        "optional_fields",
        "list_fields",
        "nested_types",
        "with_metadata",
        "problematic_names",
    ],
)
def test_context_class_template_output(field_definitions):
    """Test that the context_class.py.j2 template produces valid Python code."""
    # Get the template
    jinja_env = get_template_environment()
    template = jinja_env.get_template("context_class.py.j2")

    # Render the template
    rendered = template.render(
        model_name="TestModel",
        pydantic_class="TestPydantic",
        pydantic_module="test.module",
        field_definitions=field_definitions,
    )

    # Write to temporary file for debugging
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
        f.write(rendered.encode())
        temp_filename = f.name

    try:
        # Verify the output is valid Python code by parsing it
        try:
            ast.parse(rendered)
        except SyntaxError as e:
            pytest.fail(f"Generated code has syntax errors: {e}\nGenerated code:\n{rendered}")

        # Specifically check the create method parameters format
        if field_definitions:
            # The closing parenthesis should be on the right line
            create_method_lines = [
                line for line in rendered.split("\n") if "@classmethod" in line or "def create" in line
            ]

            # Make sure closing parenthesis is properly placed
            create_signature = next((line for line in rendered.split("\n") if "def create" in line), "")
            assert (
                ") ->" in create_signature or "):" in create_signature
            ), f"Closing parenthesis missing from create method signature: {create_signature}"

            # Check docstring format in the create method
            docstring_lines = []
            in_create_method = False
            for line in rendered.split("\n"):
                if "def create" in line:
                    in_create_method = True
                if in_create_method and '"""' in line:
                    docstring_lines.append(line)
                if len(docstring_lines) > 0 and '"""' in line and docstring_lines[0] != line:
                    break

            # Check that Args section is properly formatted
            if field_definitions:
                args_section = False
                args_indentation = None

                for i, line in enumerate(rendered.split("\n")):
                    if "Args:" in line and in_create_method:
                        args_section = True
                        args_indentation = len(line) - len(line.lstrip())
                        continue

                    if args_section and "Returns:" in line:
                        args_section = False
                        continue

                    if args_section and line.strip() and i + 1 < len(rendered.split("\n")):
                        # Check indentation of field descriptions
                        next_line = rendered.split("\n")[i + 1]
                        next_indentation = len(next_line) - len(next_line.lstrip()) if next_line.strip() else 0

                        if next_indentation > 0 and args_indentation is not None:
                            assert (
                                next_indentation >= args_indentation
                            ), f"Field description not properly indented: {line}\n{next_line}"
    finally:
        # Clean up the temporary file
        os.unlink(temp_filename)


@pytest.mark.parametrize(
    "field_config",
    [
        # Test with comma in type name
        {
            "name": "complex_callable",
            "type": "Callable[[str, int], Dict[str, Any]]",
            "is_optional": False,
        },
        # Test with very complex nested types
        {
            "name": "nested_complex",
            "type": "Dict[str, List[Optional[Tuple[int, str, Dict[str, Any]]]]]",
            "is_optional": True,
        },
        # Test with special characters
        {
            "name": "special_chars",
            "type": "Type<with>Angle[and]Square{Brackets}",
            "is_optional": False,
        },
    ],
    ids=[
        "callable_with_comma",
        "complex_nested_type",
        "special_chars",
    ],
)
def test_complex_type_handling(field_config):
    """Test handling of particularly complex or problematic type names."""
    # Get the template
    jinja_env = get_template_environment()
    template = jinja_env.get_template("context_class.py.j2")

    # Create the mock field definition
    field_def = MockFieldDefinition(
        name=field_config["name"],
        type=field_config["type"],
        is_optional=field_config["is_optional"],
    )

    # Render the template
    rendered = template.render(
        model_name="ComplexTypeModel",
        pydantic_class="ComplexTypeModel",
        pydantic_module="test.module",
        field_definitions=[field_def],
    )

    # Write to temporary file for debugging
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
        f.write(rendered.encode())
        temp_filename = f.name

    try:
        # Verify the output is valid Python code
        try:
            ast.parse(rendered)
        except SyntaxError as e:
            pytest.fail(f"Generated code has syntax errors: {e}\nGenerated code:\n{rendered}")
    finally:
        # Clean up the temporary file
        os.unlink(temp_filename)


def test_execute_generated_code():
    """Test that the generated code has valid syntax that could be executed."""
    # Get the template
    jinja_env = get_template_environment()
    template = jinja_env.get_template("context_class.py.j2")

    # Create sample field definitions
    field_definitions = [
        MockFieldDefinition(name="simple_field", type="str"),
        MockFieldDefinition(name="optional_field", type="Optional[str]", is_optional=True),
        MockFieldDefinition(name="list_field", type="List[int]", is_list=True),
        MockFieldDefinition(name="complex_field", type="Dict[str, Any]", metadata={"description": "A complex field"}),
    ]

    # Render the template
    rendered = template.render(
        model_name="TestModel",
        pydantic_class="TestPydantic",
        pydantic_module="test_module",
        field_definitions=field_definitions,
    )

    # Write to temporary file for debugging and inspection
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
        # Add import statements and mock classes
        module_content = """
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

# Mock classes for syntax validation
class ModelContext:
    pass

class FieldContext:
    def __init__(self, field_name, field_type, is_optional=False, is_list=False, additional_metadata=None):
        self.field_name = field_name
        self.field_type = field_type
        self.is_optional = is_optional
        self.is_list = is_list
        self.additional_metadata = additional_metadata or {}
        self.value = None

# Mock module for import
class test_module:
    class TestPydantic:
        pass

"""
        f.write(module_content.encode())
        f.write(rendered.encode())
        temp_filename = f.name

    try:
        # Verify the output has valid Python syntax
        with open(temp_filename, "r") as f:
            file_content = f.read()

        try:
            # This will raise SyntaxError if there are syntax issues
            ast.parse(file_content)
        except SyntaxError as e:
            pytest.fail(f"Generated code has syntax errors: {e}\nGenerated code:\n{rendered}")

        # Check key structural elements
        assert "@dataclass" in file_content
        assert "class TestModelContext(ModelContext):" in file_content
        assert "def __post_init__(self):" in file_content
        assert "@classmethod" in file_content
        assert "def create(cls," in file_content
        assert "def to_dict(self)" in file_content

        # Verify field handling
        for field_name in ["simple_field", "optional_field", "list_field", "complex_field"]:
            assert f'"{field_name}"' in file_content, f"Field {field_name} not found in generated code"
            assert "for field_name, field_context in self.context_fields.items()" in file_content

    finally:
        # Clean up
        os.unlink(temp_filename)


@pytest.mark.parametrize(
    "field_config",
    [
        # Test for class object representation handling (the main issue we found)
        {
            "name": "source",
            "type": MockChainNode(),
            "is_optional": False,
        },
        # Test for class object with full module path
        {
            "name": "target",
            "type": MockConversationNode(),
            "is_optional": False,
        },
        # Test for class object references as string with angle brackets
        {
            "name": "context",
            "type": "<class 'llmaestro.chains.chains.ChainContext'>",
            "is_optional": False,
        },
        # Test for typing complex type with dotted notation
        {
            "name": "nodes",
            "type": "typing.Dict[str, llmaestro.chains.chains.ChainNode]",
            "is_optional": False,
        },
    ],
    ids=["class_object", "module_class_object", "angle_bracket_class_string", "dotted_type_notation"],
)
def test_class_reference_type_handling(field_config):
    """Test that the template properly handles class object references."""
    # Get the template
    jinja_env = get_template_environment()
    template = jinja_env.get_template("context_class.py.j2")

    # Create field definition with the problematic type
    field_def = {
        "name": field_config["name"],
        "type": field_config["type"],
        "is_optional": field_config["is_optional"],
        "is_list": False,
        "metadata": {},
    }

    # Render the template
    rendered = template.render(
        model_name="ClassRefModel",
        pydantic_class="ClassRefModel",
        pydantic_module="test.module",
        field_definitions=[field_def],
    )

    # Write to temporary file for debugging
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
        f.write(rendered.encode())
        temp_filename = f.name

    try:
        # Verify the output is valid Python code
        try:
            ast.parse(rendered)
        except SyntaxError as e:
            pytest.fail(f"Generated code has syntax errors: {e}\nGenerated code:\n{rendered}")

        # Check for proper string quoting of type in field_type parameter
        if isinstance(field_config["type"], str) and field_config["type"].startswith("<class '"):
            # For angle bracket notation, should be converted to just the class name
            class_name = field_config["type"].split(".")[-1].rstrip("'>")
            assert f'field_type="{class_name}"' in rendered
        elif hasattr(field_config["type"], "__name__"):
            # For class objects, should use the class name
            assert f'field_type="{field_config["type"].__name__}"' in rendered
        else:
            # For regular string types, should be quoted properly
            assert f'field_type="{field_config["type"]}"' in rendered

    finally:
        # Clean up the temporary file
        os.unlink(temp_filename)
