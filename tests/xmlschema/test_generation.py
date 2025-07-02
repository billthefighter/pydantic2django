"""
Functional tests for Django model generation from XML Schemas.
"""

import pytest
import re
from pathlib import Path

from pydantic2django.xmlschema.generator import XmlSchemaDjangoModelGenerator

# Helper to check generated code
def assert_field_definition_xml(
    code: str,
    field_name: str,
    expected_type: str,
    expected_kwargs: dict[str, str] | None = None,
    absent_kwargs: list[str] | None = None,
    model_name: str = "",
):
    """Asserts that a field definition exists with the correct type and key kwargs,
    and optionally asserts that specific kwargs are absent."""
    pattern_str = f"^\\s*{field_name}\\s*=\\s*models\\.{expected_type}\\((.*)\\)"
    pattern = re.compile(pattern_str, re.MULTILINE)
    match = pattern.search(code)

    assert (
        match
    ), f"Field '{field_name}: models.{expected_type}' not found in {model_name}'s generated code.\\nCode:\\n{code}"

    kwargs_str = match.group(1)

    if expected_kwargs:
        for key, expected_value_str in expected_kwargs.items():
            # Using a flexible pattern to match different quoting and spacing
            pattern = re.compile(rf"{re.escape(key)}\s*=\s*(?:{re.escape(expected_value_str)}|'{re.escape(expected_value_str)}'|\"{re.escape(expected_value_str)}\")")
            assert pattern.search(
                kwargs_str
            ), f"Expected kwarg '{key} = {expected_value_str}' not found for field '{field_name}' in {model_name}. Found kwargs: '{kwargs_str}'"

    if absent_kwargs:
        for key in absent_kwargs:
            # Check if the key exists as a kwarg (key=...)
            pattern = re.compile(rf"\\b{re.escape(key)}\\s*=")
            assert not pattern.search(
                kwargs_str
            ), f"Expected kwarg '{key}' to be absent for field '{field_name}' in {model_name}, but found it. Found kwargs: '{kwargs_str}'"


@pytest.fixture(scope="module")
def simple_xsd_path() -> Path:
    """Provides the path to the simple test XSD file."""
    return Path(__file__).parent / "fixtures" / "simple_schema.xsd"


def test_generate_simple_model_from_xsd(simple_xsd_path, tmp_path):
    """
    Verify generation of a simple Django model from a basic XSD.
    """
    output_file = tmp_path / "models.py"
    app_label = "test_app"

    generator = XmlSchemaDjangoModelGenerator(
        schema_files=[str(simple_xsd_path)],
        output_path=str(output_file),
        app_label=app_label,
    )
    generator.generate()

    assert output_file.exists()
    generated_code = output_file.read_text()

    print(f"\\n--- Generated Code from simple_schema.xsd ---")
    print(generated_code)
    print("-------------------------------------------")

    # Check for model class definition
    assert "class BookType(Xml2DjangoBaseClass):" in generated_code

    # Check for individual fields
    assert_field_definition_xml(generated_code, "id", "CharField", {"max_length": "255"}, model_name="BookType")
    assert_field_definition_xml(generated_code, "title", "CharField", {}, model_name="BookType")
    assert_field_definition_xml(generated_code, "published_date", "DateField", {}, model_name="BookType")
    assert_field_definition_xml(generated_code, "pages", "IntegerField", {}, model_name="BookType")
    assert_field_definition_xml(generated_code, "in_print", "BooleanField", {}, model_name="BookType")
    assert_field_definition_xml(generated_code, "price", "DecimalField", {"null": "True", "blank": "True"}, model_name="BookType")

    # Check for Meta class
    assert "class Meta:" in generated_code
    assert f"app_label = '{app_label}'" in generated_code
