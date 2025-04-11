"""
Functional tests for Django model generation from Python dataclasses.
"""

import pytest
from dataclasses import dataclass
import re  # Import re for more advanced checks
from django.db import models

# Import the correct generator for dataclasses
from pydantic2django.dataclass.generator import DataclassDjangoModelGenerator

# Import dataclass fixtures using fully qualified paths
from tests.fixtures.fixtures import (  # noqa: F401 Needed for pytest fixtures
    basic_dataclass,
    datetime_dataclass,
    optional_dataclass,
    relationship_dataclasses,  # This provides a tuple (UserDC, AddressDC, ProfileDC, TagDC)
    advanced_types_dataclass,
    StatusEnum,  # Import the enum used in advanced_types_dataclass for choices check
)


# Enhanced helper to check field type and specific kwargs
def assert_field_definition(
    code: str,
    field_name: str,
    expected_type: str,
    expected_kwargs: dict[str, str] | None = None,
    model_name: str = "",  # Add model_name for better error messages
):
    """Asserts that a field definition exists with the correct type and key kwargs."""
    # Regex to find the field definition line more reliably
    # Matches: field_name = models.FieldType(kwarg1=value1, kwarg2=value2, ...)
    # Makes kwargs optional and non-greedy
    pattern_str = f"^\\s*{field_name}\\s*=\\s*models\\.{expected_type}\\((.*?)\\)"
    pattern = re.compile(pattern_str, re.MULTILINE)
    match = pattern.search(code)

    assert (
        match
    ), f"Field '{field_name}: models.{expected_type}' not found in {model_name}'s generated code.\\nCode:\\n{code}"

    if expected_kwargs:
        kwargs_str = match.group(1)  # Get the content within the parentheses
        for key, expected_value_str in expected_kwargs.items():
            # Simpler check: look for `key=expected_value_str` within the found kwargs string.
            # This requires expected_value_str to be formatted exactly as it appears in the code.
            expected_kwarg_pair = f"{key}={expected_value_str}"
            assert (
                expected_kwarg_pair in kwargs_str
            ), f"Expected kwarg pair '{expected_kwarg_pair}' not found for field '{field_name}' in {model_name}. Found kwargs: '{kwargs_str}'"


def test_generate_basic_dataclass_model(basic_dataclass):
    """Verify generation of a simple Django model from basic_dataclass."""
    generator = DataclassDjangoModelGenerator(
        output_path="dummy_output.py",
        packages=["dummy_package"],
        app_label="tests",
        filter_function=None,
        verbose=False,
    )
    model_name = basic_dataclass.__name__
    django_model_name = f"Django{model_name}"

    carrier = generator.setup_django_model(basic_dataclass)
    if not carrier or not carrier.django_model:
        pytest.fail(f"Failed to setup Django model carrier for {model_name}")

    generated_code = generator.generate_model_definition(carrier)

    print(f"\n--- Generated Code for {model_name} ---")
    print(f"Generated Code:\n{generated_code}")
    print("--------------------------------------")

    # --- Assertions --- #
    assert f"class {django_model_name}(Dataclass2DjangoBaseClass):" in generated_code

    expected_fields = [
        ("string_field", "TextField", {}),
        ("int_field", "IntegerField", {}),
        ("float_field", "FloatField", {}),
        ("bool_field", "BooleanField", {}),
    ]

    for name, type_str, kwargs in expected_fields:
        assert_field_definition(generated_code, name, type_str, kwargs, model_name=django_model_name)

    # Check Meta
    assert "class Meta:" in generated_code
    assert "app_label = 'tests'" in generated_code


def test_generate_datetime_dataclass_model(datetime_dataclass):
    """Verify generation for datetime related fields."""
    generator = DataclassDjangoModelGenerator(
        output_path="dummy_output.py",
        packages=["dummy_package"],
        app_label="tests",
        filter_function=None,
        verbose=False,
    )
    model_name = datetime_dataclass.__name__
    django_model_name = f"Django{model_name}"

    carrier = generator.setup_django_model(datetime_dataclass)
    if not carrier or not carrier.django_model:
        pytest.fail(f"Failed to setup carrier for {model_name}")

    generated_code = generator.generate_model_definition(carrier)
    print(f"\n--- Generated Code for {model_name} ---")
    print(f"Generated Code:\n{generated_code}")
    print("--------------------------------------")

    assert f"class {django_model_name}(Dataclass2DjangoBaseClass):" in generated_code

    expected_fields = [
        ("datetime_field", "DateTimeField", {}),
        ("date_field", "DateField", {}),
        ("time_field", "TimeField", {}),
        ("duration_field", "DurationField", {}),
    ]

    for name, type_str, kwargs in expected_fields:
        assert_field_definition(generated_code, name, type_str, kwargs, model_name=django_model_name)


def test_generate_optional_dataclass_model(optional_dataclass):
    """Verify generation for optional fields."""
    generator = DataclassDjangoModelGenerator(
        output_path="dummy_output.py",
        packages=["dummy_package"],
        app_label="tests",
        filter_function=None,
        verbose=False,
    )
    model_name = optional_dataclass.__name__
    django_model_name = f"Django{model_name}"

    carrier = generator.setup_django_model(optional_dataclass)
    if not carrier or not carrier.django_model:
        pytest.fail(f"Failed to setup carrier for {model_name}")

    generated_code = generator.generate_model_definition(carrier)
    print(f"\n--- Generated Code for {model_name} ---")
    print(f"Generated Code:\n{generated_code}")
    print("--------------------------------------")

    assert f"class {django_model_name}(Dataclass2DjangoBaseClass):" in generated_code

    expected_fields = [
        ("required_string", "TextField", {}),
        ("required_int", "IntegerField", {}),
        ("optional_string", "TextField", {"null": "True", "blank": "True"}),
        ("optional_int", "IntegerField", {"null": "True", "blank": "True"}),
    ]

    for name, type_str, kwargs in expected_fields:
        assert_field_definition(generated_code, name, type_str, kwargs, model_name=django_model_name)


def test_generate_relationship_dataclass_model(relationship_dataclasses):
    """Verify generation for nested dataclasses simulating relationships."""
    UserDC, AddressDC, ProfileDC, TagDC = relationship_dataclasses

    generator = DataclassDjangoModelGenerator(
        output_path="dummy_output.py",
        packages=["dummy_package"],
        app_label="tests",
        filter_function=None,
        verbose=False,
    )

    # --- Setup Phase --- #
    # Generate definitions for all related dataclasses first
    # We need the RelationshipAccessor within the generator to be populated
    address_carrier = generator.setup_django_model(AddressDC)
    profile_carrier = generator.setup_django_model(ProfileDC)
    tag_carrier = generator.setup_django_model(TagDC)

    # Generate the main UserDC model (depends on the others)
    user_carrier = generator.setup_django_model(UserDC)

    if not all([address_carrier, profile_carrier, tag_carrier, user_carrier]):
        pytest.fail("Failed to setup carriers for relationship dataclasses")
    assert address_carrier and profile_carrier and tag_carrier and user_carrier

    if not all([c.django_model for c in [address_carrier, profile_carrier, tag_carrier, user_carrier]]):
        pytest.fail("Failed to generate Django models for relationship dataclasses")

    # --- Test UserDC Generation --- #
    user_model_name = UserDC.__name__
    django_user_model_name = f"Django{user_model_name}"
    assert user_carrier  # Needed for type checker
    generated_code = generator.generate_model_definition(user_carrier)

    print(f"\n--- Generated Code for {user_model_name} ---")
    print(f"Generated Code:\n{generated_code}")
    print("--------------------------------------")

    assert f"class {django_user_model_name}(Dataclass2DjangoBaseClass):" in generated_code

    expected_fields = [
        ("name", "TextField", {}),
        ("address", "ForeignKey", {"to": "'tests.DjangoAddressDC'", "on_delete": "models.CASCADE"}),
        # The mapper currently defaults nested models to FK. Test this assumption.
        # TODO: Update test if O2O detection is added to DataclassFieldFactory/Mapper
        ("profile", "ForeignKey", {"to": "'tests.DjangoProfileDC'", "on_delete": "models.CASCADE"}),
        # ("profile", "OneToOneField", {"to": "'tests.DjangoProfileDC'", "on_delete": "models.CASCADE"}), # Ideal O2O mapping
        ("tags", "ManyToManyField", {"to": "'tests.DjangoTagDC'"}),  # M2M doesn't typically have on_delete
    ]

    for name, type_str, kwargs in expected_fields:
        assert_field_definition(generated_code, name, type_str, kwargs, model_name=django_user_model_name)


def test_generate_advanced_types_dataclass_model(advanced_types_dataclass):
    """Verify generation for Decimal, UUID, and Enum fields."""
    generator = DataclassDjangoModelGenerator(
        output_path="dummy_output.py",
        packages=["dummy_package"],
        app_label="tests",
        filter_function=None,
        verbose=False,
    )
    model_name = advanced_types_dataclass.__name__
    django_model_name = f"Django{model_name}"

    carrier = generator.setup_django_model(advanced_types_dataclass)
    if not carrier or not carrier.django_model:
        pytest.fail(f"Failed to setup carrier for {model_name}")

    generated_code = generator.generate_model_definition(carrier)
    print(f"\n--- Generated Code for {model_name} ---")
    print(f"Generated Code:\n{generated_code}")
    print("--------------------------------------")

    assert f"class {django_model_name}(Dataclass2DjangoBaseClass):" in generated_code

    expected_fields = [
        ("decimal_field", "DecimalField", {"max_digits": "10", "decimal_places": "2"}),  # Check defaults from mapper
        ("uuid_field", "UUIDField", {}),  # Basic check
        ("enum_field", "CharField", {"max_length": "9", "choices": "StatusEnum.choices"}),  # Enum maps to CharField
    ]

    for name, type_str, kwargs in expected_fields:
        assert_field_definition(generated_code, name, type_str, kwargs, model_name=django_model_name)


# TODO: Add tests for metadata_dataclass if metadata['django'] overrides are implemented
# TODO: Add tests for nested_dataclass (potentially maps to JSONField?)
