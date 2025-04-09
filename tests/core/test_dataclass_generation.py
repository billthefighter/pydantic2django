"""
Functional tests for Django model generation from Python dataclasses.
"""

import pytest
from dataclasses import dataclass
from django.db import models

# Import the correct generator for dataclasses
from pydantic2django.dataclass.generator import DataclassDjangoModelGenerator

# Import the basic dataclass fixture using a fully qualified path
from tests.fixtures.fixtures import basic_dataclass


# Re-use or adapt the helper from the pydantic test
def contains_field(code: str, field_name: str, field_type: str) -> bool:
    """Checks if the generated code string contains a specific field definition."""
    # Simple string check - adjust if dataclass generation differs significantly
    return f"{field_name} = models.{field_type}(" in code


def test_generate_basic_dataclass_model(basic_dataclass):
    """Verify generation of a simple Django model from basic_dataclass."""
    # Instantiate the dataclass generator with minimal required args for testing
    generator = DataclassDjangoModelGenerator(
        output_path="dummy_output.py",
        packages=["dummy_package"],  # Needs a package to look in
        app_label="tests",  # Match the expected Meta app_label
        filter_function=None,  # Provide None explicitly if not used
        verbose=False,  # Provide False explicitly
    )
    model_name = basic_dataclass.__name__

    # Use the same approach: setup carrier, then generate definition
    carrier = generator.setup_django_model(basic_dataclass)

    # Check if carrier creation was successful
    if not carrier or not carrier.django_model:
        pytest.fail(f"Failed to setup Django model carrier for {model_name}")

    generated_code = generator.generate_model_definition(carrier)

    print(f"\n--- Generated Code for {model_name} ---")
    print(f"Generated Code:\n{generated_code}")
    print("--------------------------------------")

    # Asserts need to match expected Django field types for basic dataclass fields
    assert f"class Django{model_name}(Pydantic2DjangoBaseClass):" in generated_code
    assert contains_field(generated_code, "string_field", "CharField")
    assert contains_field(generated_code, "int_field", "IntegerField")
    assert contains_field(generated_code, "float_field", "FloatField")
    assert contains_field(generated_code, "bool_field", "BooleanField")

    # Check for Meta class (adjust app_label as needed)
    assert "class Meta:" in generated_code
    assert "app_label = 'tests'" in generated_code  # Assuming 'tests' app


# Add more tests here using other dataclass fixtures...
