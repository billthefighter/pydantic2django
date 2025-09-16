"""
Functional tests for Django model generation from Python dataclasses.
"""

import pytest
from dataclasses import dataclass
import dataclasses
import re  # Import re for more advanced checks
from django.db import models

# Import the correct generator for dataclasses
from pydantic2django.dataclass.generator import DataclassDjangoModelGenerator

# Import RelationshipAccessor needed for the relationship test
from pydantic2django.core.relationships import RelationshipConversionAccessor

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
    # Makes kwargs optional and greedy
    pattern_str = f"^\\s*{field_name}\\s*=\\s*models\\.{expected_type}\\((.*)\\)"
    pattern = re.compile(pattern_str, re.MULTILINE)
    match = pattern.search(code)

    assert (
        match
    ), f"Field '{field_name}: models.{expected_type}' not found in {model_name}'s generated code.\\nCode:\\n{code}"

    if expected_kwargs:
        kwargs_str = match.group(1)  # Get the content within the parentheses
        for key, expected_value_str in expected_kwargs.items():
            # More robust check: Use regex to find `key = value`, allowing whitespace variations.
            # Escape the key and value strings for regex usage.
            # Look for word boundary before key, whitespace, equals, whitespace, escaped value,
            # followed by (whitespace and comma) OR (whitespace and closing parenthesis) OR (end of string).
            pattern = re.compile(rf"\b{re.escape(key)}\s*=\s*{re.escape(expected_value_str)}(?:\s*,|\s*\)|\Z)")
            assert pattern.search(
                kwargs_str
            ), f"Expected kwarg pattern '{key} = {expected_value_str}' not found for field '{field_name}' in {model_name}. Found kwargs: '{kwargs_str}'"


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
    # Unpack models (AddressDC, ProfileDC, TagDC are now module level)
    UserDC = relationship_dataclasses["UserDC"]
    AddressDC = relationship_dataclasses["AddressDC"]
    ProfileDC = relationship_dataclasses["ProfileDC"]
    TagDC = relationship_dataclasses["TagDC"]

    # --- Pre-populate Relationship Accessor --- #
    # We need to tell the generator about the expected Django mappings
    # Note: We don't have the actual Django types yet, so we might need
    # a way to map source types to expected *names* if accessor requires types.
    # For now, assume the accessor can handle this, or we adapt it later.
    rel_accessor = RelationshipConversionAccessor()
    # Ideally: rel_accessor.map_relationship(AddressDC, DjangoAddressDC) etc.
    # Workaround: Let's try adding source models first, maybe factory uses this?
    # This assumes the factory might populate the Django side later.
    # rel_accessor.add_dataclass_model(AddressDC)
    # rel_accessor.add_dataclass_model(ProfileDC)
    # rel_accessor.add_dataclass_model(TagDC)
    # rel_accessor.add_dataclass_model(UserDC)
    # REMOVED: Map source dataclasses to their expected Django model NAMES
    # rel_accessor.map_relationship(AddressDC, "tests.DjangoAddressDC")
    # rel_accessor.map_relationship(ProfileDC, "tests.DjangoProfileDC")
    # rel_accessor.map_relationship(TagDC, "tests.DjangoTagDC")
    # rel_accessor.map_relationship(UserDC, "tests.DjangoUserDC")

    # --- End Pre-populate ---

    generator = DataclassDjangoModelGenerator(
        output_path="dummy_output.py",
        packages=["dummy_package"],
        app_label="tests",
        filter_function=None,
        verbose=False,
        relationship_accessor=rel_accessor,  # Pass the populated accessor
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
        ("user_name", "TextField", {}),
        ("address", "ForeignKey", {"to": "'tests.djangoaddressdc'", "on_delete": "models.CASCADE"}),
        # The mapper currently defaults nested models to FK. Test this assumption.
        # TODO: Update test if O2O detection is added to DataclassFieldFactory/Mapper
        ("profile", "ForeignKey", {"to": "'tests.djangoprofiledc'", "on_delete": "models.CASCADE"}),
        # ("profile", "OneToOneField", {"to": "'tests.DjangoProfileDC'", "on_delete": "models.CASCADE"}), # Ideal O2O mapping
        ("tags", "ManyToManyField", {"to": "'tests.djangotagdc'"}),  # Adjusted 'to' case
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
        (
            "enum_field",
            "CharField",
            {
                "max_length": "9",
                "choices": "[('pending', 'PENDING'), ('completed', 'COMPLETED'), ('failed', 'FAILED')]",
            },
        ),
    ]

    for name, type_str, kwargs in expected_fields:
        assert_field_definition(generated_code, name, type_str, kwargs, model_name=django_model_name)


# --- Test for Callable Default Bug --- #


@dataclass
class ModelWithCallableDefault:
    """A model with a lambda function as a default value for regression testing."""

    field_with_lambda: dict = dataclasses.field(default_factory=lambda: {"key": "value"})
    another_field: dict = dataclasses.field(default_factory=dict)
    # The problematic field from the original issue
    diff_checker: callable = dataclasses.field(default=lambda a, b: a == b)


def test_callable_default_is_serialized_to_none():
    """
    Verify that a field with a callable `default` is correctly handled by serializing
    the default value to `None` in the generated Django model. This prevents a
    SyntaxError during generation.
    """
    generator = DataclassDjangoModelGenerator(
        output_path="dummy_output.py",
        app_label="tests",
        filter_function=None,
        verbose=False,
    )
    model_name = ModelWithCallableDefault.__name__
    django_model_name = f"Django{model_name}"

    # Generate the model definition
    carrier = generator.setup_django_model(ModelWithCallableDefault)
    if not carrier or not carrier.django_model:
        pytest.fail(f"Failed to setup Django model carrier for {model_name}")

    generated_code = generator.generate_model_definition(carrier)

    # Assert that the problematic field is generated with the lambda source as its default
    assert_field_definition(
        generated_code,
        "diff_checker",
        "JSONField",  # Fallback mapping
        {"default": "lambda a, b: a == b"},
        model_name=django_model_name,
    )

    # Also check that the raw, invalid string is not present
    assert "<function" not in generated_code


# TODO: Add tests for metadata_dataclass if metadata['django'] overrides are implemented
# TODO: Add tests for nested_dataclass (potentially maps to JSONField?)


def test_dataclass_field_name_normalization(tmp_path):
    """Ensure dataclass field names are normalized: namespaces and camelCase handled."""
    from dataclasses import dataclass
    from pydantic2django.dataclass.generator import DataclassDjangoModelGenerator

    @dataclass
    class WeirdDC:
        xlink_type: str  # simulate source "xlink:type" by naming target style; generator should not break
        camelCase: int

    gen = DataclassDjangoModelGenerator(output_path=str(tmp_path / "out.py"), packages=["dummy"], app_label="tests", filter_function=None, verbose=False)
    carrier = gen.setup_django_model(WeirdDC)
    assert carrier and carrier.django_model
    code = gen.generate_model_definition(carrier)
    # xlink_type should stay xlink_type, camelCase normalized to camel_case
    assert "xlink_type = models.TextField(" in code or "xlink_type = models.CharField(" in code
    assert "camel_case = models.IntegerField(" in code
