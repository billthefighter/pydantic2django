"""
Functional tests for Django model generation from Pydantic models.
"""

import pytest
from pydantic import BaseModel
from django.db import models
import re  # Import re for helper function

# Assuming the generator lives here - adjust if needed
# from pydantic2django.pydantic.generator import PydanticDjangoModelGenerator # Original attempt
# from pydantic2django.pydantic.generator import ModelGenerator # Trying alternative name
# from pydantic2django.pydantic.generator import PydanticModelGenerator # Trying PydanticModelGenerator
from pydantic2django.pydantic.generator import StaticPydanticModelGenerator  # Correct name from file
from tests.fixtures.fixtures import basic_pydantic_model  # Fully qualified import
from tests.fixtures.fixtures import (
    basic_pydantic_model,
    datetime_pydantic_model,
    optional_fields_model,
    constrained_fields_model,
    relationship_models,  # Import the dict containing related models
)


# Helper to check generated code (avoids direct exec)
def contains_field(code: str, field_name: str, field_type: str) -> bool:
    """Checks if the generated code string contains a specific field definition."""
    # Simple string check - might need refinement for complex cases (e.g., kwargs)
    return f"{field_name} = models.{field_type}(" in code


# Enhanced helper specific to this file to check field type and specific kwargs
def assert_field_definition_pydantic(
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
            pattern = re.compile(rf"\b{re.escape(key)}\s*=\s*{re.escape(expected_value_str)}(?:\s*,|\s*\)|\Z)")
            assert pattern.search(
                kwargs_str
            ), f"Expected kwarg pattern '{key} = {expected_value_str}' not found for field '{field_name}' in {model_name}. Found kwargs: '{kwargs_str}'"

    if absent_kwargs:
        for key in absent_kwargs:
            # Check if the key exists as a kwarg (key=...)
            pattern = re.compile(rf"\b{re.escape(key)}\s*=")
            assert not pattern.search(
                kwargs_str
            ), f"Expected kwarg '{key}' to be absent for field '{field_name}' in {model_name}, but found it. Found kwargs: '{kwargs_str}'"


def test_generate_basic_pydantic_model(basic_pydantic_model):
    """Verify generation of a simple Django model from basic_pydantic_model."""
    # generator = PydanticDjangoModelGenerator() # Assuming default instantiation - Original attempt
    # generator = ModelGenerator() # Trying alternative name
    # generator = PydanticModelGenerator() # Trying PydanticModelGenerator
    # Instantiate with minimal required args for testing
    generator = StaticPydanticModelGenerator(
        output_path="dummy_output.py",
        packages=["dummy_package"],  # Needs a package to look in, even if unused for this test
        app_label="tests",  # Match the expected Meta app_label
    )
    model_name = basic_pydantic_model.__name__

    # The base generator class handles the full file generation.
    # To test a single model's code, we use the generator's setup_django_model
    # to create the carrier, and then generate_model_definition for that carrier.
    carrier = generator.setup_django_model(basic_pydantic_model)

    # Check if carrier creation was successful before proceeding
    if not carrier or not carrier.django_model:
        pytest.fail(f"Failed to setup Django model carrier for {model_name}")

    generated_code = generator.generate_model_definition(carrier)

    print(f"\n--- Generated Code for {model_name} ---")
    # print(generated_code) # Original print
    print(f"Generated Code:\n{generated_code}")  # Debug print
    print("--------------------------------------")

    assert f"class Django{model_name}(Pydantic2DjangoBaseClass):" in generated_code
    # Use the new helper for more robust checks
    assert_field_definition_pydantic(generated_code, "string_field", "TextField", model_name=f"Django{model_name}")
    assert_field_definition_pydantic(generated_code, "int_field", "IntegerField", model_name=f"Django{model_name}")
    assert_field_definition_pydantic(generated_code, "float_field", "FloatField", model_name=f"Django{model_name}")
    assert_field_definition_pydantic(generated_code, "bool_field", "BooleanField", model_name=f"Django{model_name}")
    assert_field_definition_pydantic(generated_code, "decimal_field", "DecimalField", model_name=f"Django{model_name}")
    assert_field_definition_pydantic(generated_code, "email_field", "EmailField", model_name=f"Django{model_name}")

    # Check for Meta class (adjust app_label as needed)
    assert "class Meta:" in generated_code
    assert "app_label = 'tests'" in generated_code  # Assuming 'tests' app


# Add more tests here using other pydantic fixtures...


def test_generate_datetime_pydantic_model(datetime_pydantic_model):
    """Verify generation from datetime_pydantic_model."""
    generator = StaticPydanticModelGenerator(output_path="dummy_output.py", packages=["dummy"], app_label="tests")
    model_name = datetime_pydantic_model.__name__
    carrier = generator.setup_django_model(datetime_pydantic_model)
    if not carrier or not carrier.django_model:
        pytest.fail(f"Failed to setup carrier for {model_name}")

    generated_code = generator.generate_model_definition(carrier)

    print(f"\n--- Generated Code for {model_name} ---")
    # print(generated_code) # Original print
    print(f"Generated Code:\n{generated_code}")  # Debug print
    print("--------------------------------------")

    assert f"class Django{model_name}(Pydantic2DjangoBaseClass):" in generated_code
    assert_field_definition_pydantic(
        generated_code, "datetime_field", "DateTimeField", model_name=f"Django{model_name}"
    )
    assert_field_definition_pydantic(generated_code, "date_field", "DateField", model_name=f"Django{model_name}")
    assert_field_definition_pydantic(generated_code, "time_field", "TimeField", model_name=f"Django{model_name}")
    assert_field_definition_pydantic(
        generated_code, "duration_field", "DurationField", model_name=f"Django{model_name}"
    )
    assert "class Meta:" in generated_code
    assert "app_label = 'tests'" in generated_code


def test_generate_optional_fields_model(optional_fields_model):
    """Verify generation from optional_fields_model (check null=True)."""
    generator = StaticPydanticModelGenerator(output_path="dummy_output.py", packages=["dummy"], app_label="tests")
    model_name = optional_fields_model.__name__
    carrier = generator.setup_django_model(optional_fields_model)
    if not carrier or not carrier.django_model:
        pytest.fail(f"Failed to setup carrier for {model_name}")

    generated_code = generator.generate_model_definition(carrier)

    print(f"\n--- Generated Code for {model_name} ---")
    # print(generated_code) # Original print
    print(f"Generated Code:\n{generated_code}")  # Debug print
    print("--------------------------------------")

    assert f"class Django{model_name}(Pydantic2DjangoBaseClass):" in generated_code
    # Required fields: check type and ensure null/blank are absent or False
    assert_field_definition_pydantic(
        generated_code,
        "required_string",
        "TextField",
        expected_kwargs={"null": "False", "blank": "False"},  # Explicitly check they are False
        absent_kwargs=None,
        model_name=f"Django{model_name}",
    )
    assert_field_definition_pydantic(
        generated_code,
        "required_int",
        "IntegerField",
        expected_kwargs={"null": "False", "blank": "False"},  # Explicitly check they are False
        absent_kwargs=None,
        model_name=f"Django{model_name}",
    )
    # Optional fields: check type and ensure null/blank are present and True
    assert_field_definition_pydantic(
        generated_code,
        "optional_string",
        "TextField",  # Was incorrectly expecting JSONField
        expected_kwargs={"null": "True", "blank": "True"},
        absent_kwargs=None,
        model_name=f"Django{model_name}",
    )
    assert_field_definition_pydantic(
        generated_code,
        "optional_int",
        "IntegerField",  # Was incorrectly expecting JSONField
        expected_kwargs={"null": "True", "blank": "True"},
        absent_kwargs=None,
        model_name=f"Django{model_name}",
    )
    assert "class Meta:" in generated_code
    assert "app_label = 'tests'" in generated_code


def test_generate_constrained_fields_model(constrained_fields_model):
    """Verify generation from constrained_fields_model (check constraints)."""
    generator = StaticPydanticModelGenerator(output_path="dummy_output.py", packages=["dummy"], app_label="tests")
    model_name = constrained_fields_model.__name__
    carrier = generator.setup_django_model(constrained_fields_model)
    if not carrier or not carrier.django_model:
        pytest.fail(f"Failed to setup carrier for {model_name}")

    generated_code = generator.generate_model_definition(carrier)

    print(f"\n--- Generated Code for {model_name} ---")
    # print(generated_code) # Original print
    print(f"Generated Code:\n{generated_code}")  # Debug print
    print("--------------------------------------")

    assert f"class Django{model_name}(Pydantic2DjangoBaseClass):" in generated_code
    # Check constraints mapping using the helper
    assert_field_definition_pydantic(
        generated_code, "name", "CharField", {"max_length": "100"}, model_name=f"Django{model_name}"
    )
    assert_field_definition_pydantic(generated_code, "age", "IntegerField", model_name=f"Django{model_name}")
    assert_field_definition_pydantic(
        generated_code,
        "balance",
        "DecimalField",
        {"max_digits": "10", "decimal_places": "2"},
        model_name=f"Django{model_name}",
    )
    assert "class Meta:" in generated_code
    assert "app_label = 'tests'" in generated_code


def test_generate_relationship_models(relationship_models):
    """Verify generation of related models, focusing on the User model."""
    # We need to process all related models for relationships to be setup correctly
    generator = StaticPydanticModelGenerator(output_path="dummy_output.py", packages=["dummy"], app_label="tests")

    # Process dependent models first (Address, Profile, Tag)
    address_carrier = generator.setup_django_model(relationship_models["Address"])
    profile_carrier = generator.setup_django_model(relationship_models["Profile"])
    tag_carrier = generator.setup_django_model(relationship_models["Tag"])

    # Process the main User model
    user_model = relationship_models["User"]
    model_name = user_model.__name__
    user_carrier = generator.setup_django_model(user_model)

    if not user_carrier or not user_carrier.django_model:
        pytest.fail(f"Failed to setup carrier for {model_name}")
    # Ensure dependent models were also processed successfully (basic check)
    assert address_carrier and address_carrier.django_model
    assert profile_carrier and profile_carrier.django_model
    assert tag_carrier and tag_carrier.django_model

    generated_code = generator.generate_model_definition(user_carrier)

    print(f"\n--- Generated Code for {model_name} ---")
    # print(generated_code) # Original print
    print(f"Generated Code:\n{generated_code}")  # Debug print
    print("--------------------------------------")

    assert f"class Django{model_name}(Pydantic2DjangoBaseClass):" in generated_code
    assert_field_definition_pydantic(generated_code, "name", "TextField", model_name=f"Django{model_name}")

    # Check relationship fields using the helper and updated on_delete
    assert_field_definition_pydantic(
        generated_code,
        "address",
        "ForeignKey",
        {"to": "'tests.djangoaddresspydantic'", "on_delete": "models.CASCADE"},  # Changed PROTECT to CASCADE
        model_name=f"Django{model_name}",
    )
    assert_field_definition_pydantic(
        generated_code,
        "profile",
        "ForeignKey",  # Mapped as FK
        {"to": "'tests.djangoprofilepydantic'", "on_delete": "models.CASCADE"},  # Changed PROTECT to CASCADE
        model_name=f"Django{model_name}",
    )
    assert_field_definition_pydantic(
        generated_code,
        "tags",
        "ManyToManyField",
        {"to": "'tests.djangotagpydantic'"},
        model_name=f"Django{model_name}",
    )

    assert "class Meta:" in generated_code
    assert "app_label = 'tests'" in generated_code


def test_pydantic_field_name_normalization(tmp_path):
    """Ensure Pydantic field alias with namespace/punct is normalized to snake_case identifier."""
    from pydantic import BaseModel, Field
    from pydantic2django.pydantic.generator import StaticPydanticModelGenerator

    class WeirdNames(BaseModel):
        xlink_type: str = Field(alias="xlink:type")
        camelCase: int

    gen = StaticPydanticModelGenerator(output_path=str(tmp_path / "out.py"), packages=["dummy"], app_label="tests")
    carrier = gen.setup_django_model(WeirdNames)
    assert carrier and carrier.django_model
    code = gen.generate_model_definition(carrier)
    assert "xlink_type = models.TextField(" in code or "xlink_type = models.CharField(" in code
    assert "camel_case = models.IntegerField(" in code
