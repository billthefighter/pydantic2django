"""
Functional tests for Django model generation from Pydantic models.
"""

import pytest
from pydantic import BaseModel
from django.db import models

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
    assert contains_field(generated_code, "string_field", "TextField")
    assert contains_field(generated_code, "int_field", "IntegerField")
    assert contains_field(generated_code, "float_field", "FloatField")
    assert contains_field(generated_code, "bool_field", "BooleanField")
    assert contains_field(generated_code, "decimal_field", "DecimalField")
    assert contains_field(generated_code, "email_field", "EmailField")

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
    assert contains_field(generated_code, "datetime_field", "DateTimeField")
    assert contains_field(generated_code, "date_field", "DateField")
    assert contains_field(generated_code, "time_field", "TimeField")
    assert contains_field(generated_code, "duration_field", "DurationField")
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
    # Required fields should not have null=True by default
    assert contains_field(generated_code, "required_string", "TextField")
    assert "required_string = models.TextField(" in generated_code and "null=True" not in generated_code
    assert contains_field(generated_code, "required_int", "IntegerField")
    assert "required_int = models.IntegerField(" in generated_code and "null=True" not in generated_code
    # Optional fields should have null=True (or be JSONField)
    assert contains_field(generated_code, "optional_string", "JSONField")  # Fallback for Optional[str]
    assert "optional_string = models.JSONField(" in generated_code  # JSONField handles null
    assert contains_field(generated_code, "optional_int", "JSONField")  # Fallback for Optional[int]
    assert "optional_int = models.JSONField(" in generated_code  # JSONField handles null
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
    # Check constraints mapping
    assert contains_field(generated_code, "name", "CharField")
    assert "max_length=100" in generated_code  # From Pydantic Field max_length
    assert contains_field(generated_code, "age", "IntegerField")
    assert contains_field(generated_code, "balance", "DecimalField")
    assert "max_digits=10" in generated_code
    assert "decimal_places=2" in generated_code
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
    assert contains_field(generated_code, "name", "TextField")

    # Check relationship fields
    # Address (ForeignKey)
    assert contains_field(generated_code, "address", "ForeignKey")
    assert "to='tests.djangoaddresspydantic'" in generated_code  # Expect fully qualified name
    assert "on_delete=models.PROTECT" in generated_code  # Assuming PROTECT is default/sensible

    # Profile (OneToOneField - mapped as ForeignKey by default)
    assert contains_field(generated_code, "profile", "ForeignKey")  # Mapped as FK
    assert "to='tests.djangoprofilepydantic'" in generated_code  # Expect fully qualified name
    assert "on_delete=models.PROTECT" in generated_code

    # Tags (ManyToManyField)
    assert contains_field(generated_code, "tags", "ManyToManyField")
    assert "to='tests.djangotagpydantic'" in generated_code  # Expect fully qualified name

    assert "class Meta:" in generated_code
    assert "app_label = 'tests'" in generated_code
