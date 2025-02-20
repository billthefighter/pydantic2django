"""Tests for field conversion functionality."""
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Union, get_origin
from uuid import UUID

import pytest
from django.db import models
from pydantic import BaseModel, EmailStr, Field

from pydantic2django.fields import (
    FieldConverter,
    FieldTypeManager,
    TypeResolver,
    get_django_field,
    get_field_kwargs,
    get_relationship_field,
    is_pydantic_model,
    resolve_field_type,
    sanitize_related_name,
    handle_id_field,
)

# Test data for field type resolution
FIELD_TYPE_TEST_CASES = [
    (str, (models.CharField, False)),
    (int, (models.IntegerField, False)),
    (float, (models.FloatField, False)),
    (bool, (models.BooleanField, False)),
    (datetime, (models.DateTimeField, False)),
    (date, (models.DateField, False)),
    (time, (models.TimeField, False)),
    (timedelta, (models.DurationField, False)),
    (Decimal, (models.DecimalField, False)),
    (UUID, (models.UUIDField, False)),
    (EmailStr, (models.EmailField, False)),
    (dict, (models.JSONField, False)),
    (list, (models.JSONField, False)),
    (set, (models.JSONField, False)),
    (Any, (models.JSONField, False)),
]

# Test data for field name sanitization
SANITIZE_NAME_TEST_CASES = [
    ("normal_name", "normal_name"),
    ("CamelCase", "camelcase"),
    ("with spaces", "with_spaces"),
    ("with-hyphens", "with_hyphens"),
    ("with@special#chars", "with_special_chars"),
    ("", "related"),
    ("_leading_underscore", "_leading_underscore"),
    ("123numeric", "_123numeric"),
    ("a" * 100, f"{'a' * 54}_{'a' * 8}"),  # Test length truncation
]


@pytest.fixture
def basic_pydantic_model():
    """Create a basic Pydantic model for testing."""

    class BasicModel(BaseModel):
        string_field: str
        int_field: int
        float_field: float
        bool_field: bool

    return BasicModel


@pytest.fixture
def constrained_fields_model():
    """Create a Pydantic model with constrained fields for testing."""

    class ConstrainedModel(BaseModel):
        name: str = Field(
            max_length=100, title="Full Name", description="Full name of the user"
        )
        balance: Decimal = Field(max_digits=10, decimal_places=2, gt=0.01)

    return ConstrainedModel


@pytest.fixture
def relationship_models():
    """Create Pydantic models with relationships for testing."""

    class Profile(BaseModel):
        bio: str

    class Address(BaseModel):
        street: str
        city: str

    class Tag(BaseModel):
        name: str

    class User(BaseModel):
        profile: Profile = Field(json_schema_extra={"one_to_one": True})
        address: Address
        tags: List[Tag]

    return {"User": User, "Profile": Profile, "Address": Address, "Tag": Tag}


@pytest.fixture
def optional_fields_model():
    """Create a Pydantic model with optional fields for testing."""

    class OptionalModel(BaseModel):
        optional_string: Optional[str]
        optional_int: Optional[int]

    return OptionalModel


@pytest.fixture
def id_fields_model():
    """Create a Pydantic model with various ID fields for testing."""

    class ModelWithIds(BaseModel):
        id: str = Field(description="Custom string ID")
        ID: int = Field(description="Custom integer ID")
        user_id: str = Field(description="Regular field with id in name")
        normal_field: str = Field(description="Normal field without id")

    return ModelWithIds


@pytest.mark.parametrize("input_type,expected", FIELD_TYPE_TEST_CASES)
def test_type_resolver_basic(input_type, expected):
    """Test basic field type resolution using TypeResolver."""
    resolver = TypeResolver()
    result = resolver.resolve_type(input_type)
    assert result == expected


def test_field_type_manager():
    """Test FieldTypeManager functionality."""
    manager = FieldTypeManager()

    # Test basic type resolution
    field_class = manager.get_field_type(str)
    assert field_class == models.CharField

    # Test max length resolution
    assert manager.get_default_max_length("error_message", models.CharField) == 1000
    assert manager.get_default_max_length("user_email", models.EmailField) == 254

    # Test custom type registration
    class CustomType:
        pass

    manager.register_field_type(CustomType, models.TextField)
    assert manager.get_field_type(CustomType) == models.TextField


def test_field_converter(basic_pydantic_model):
    """Test FieldConverter functionality."""
    converter = FieldConverter()

    # Test basic field conversion
    string_field = converter.convert_field(
        "string_field", basic_pydantic_model.model_fields["string_field"]
    )
    assert isinstance(string_field, models.CharField)

    # Test type inference
    int_field = converter.convert_field(
        "int_field", basic_pydantic_model.model_fields["int_field"]
    )
    assert isinstance(int_field, models.IntegerField)


def test_field_converter_optional_fields(optional_fields_model):
    """Test FieldConverter with optional fields."""
    converter = FieldConverter()

    optional_string = converter.convert_field(
        "optional_string", optional_fields_model.model_fields["optional_string"]
    )
    assert isinstance(optional_string, models.CharField)
    assert optional_string.null
    assert optional_string.blank


def test_field_converter_relationships(relationship_models):
    """Test FieldConverter with relationship fields."""
    converter = FieldConverter()
    User = relationship_models["User"]

    # Test OneToOne relationship
    profile_field = converter.convert_field("profile", User.model_fields["profile"])
    assert isinstance(profile_field, models.OneToOneField)

    # Test ManyToMany relationship
    tags_field = converter.convert_field("tags", User.model_fields["tags"])
    assert isinstance(tags_field, models.ManyToManyField)


@pytest.mark.parametrize("name,expected", SANITIZE_NAME_TEST_CASES)
def test_sanitize_related_name(name, expected):
    """Test related name sanitization."""
    result = sanitize_related_name(name)
    assert result == expected


def test_sanitize_related_name_with_model_and_field():
    """Test related name sanitization with model and field names."""
    result = sanitize_related_name("test", model_name="MyModel", field_name="my_field")
    assert result == "mymodel_my_field_test"


def test_field_kwargs_with_constraints(constrained_fields_model):
    """Test field kwargs generation with constraints."""
    converter = FieldConverter()

    field_info = constrained_fields_model.model_fields["name"]
    print("\nName field info structure:")
    print(f"Field info type: {type(field_info)}")
    print(f"Field info dir: {dir(field_info)}")
    print(f"Field info metadata: {getattr(field_info, 'metadata', None)}")
    print(
        f"Field info json_schema_extra: {getattr(field_info, 'json_schema_extra', None)}"
    )

    name_field = converter.convert_field("name", field_info)
    assert name_field.max_length == 100
    assert name_field.verbose_name == "Full Name"
    assert name_field.help_text == "Full name of the user"

    balance_field_info = constrained_fields_model.model_fields["balance"]
    print("\nBalance field info structure:")
    print(f"Field info type: {type(balance_field_info)}")
    print(f"Field info dir: {dir(balance_field_info)}")
    print(f"Field info metadata: {getattr(balance_field_info, 'metadata', None)}")
    print(
        f"Field info json_schema_extra: {getattr(balance_field_info, 'json_schema_extra', None)}"
    )

    # Debug Gt constraint
    for constraint in balance_field_info.metadata:
        if type(constraint).__name__ == "Gt":
            print("\nGt constraint structure:")
            print(f"Constraint type: {type(constraint)}")
            print(f"Constraint dir: {dir(constraint)}")
            print(f"Constraint gt value: {constraint.gt}")

    balance_field = converter.convert_field("balance", balance_field_info)
    assert isinstance(balance_field, models.DecimalField)
    assert balance_field.max_digits == 10
    assert balance_field.decimal_places == 2

    # Check that we have a MinValueValidator with limit_value > 0
    validators = getattr(balance_field, "validators", [])
    print("\nValidators:")
    for validator in validators:
        print(f"Validator type: {type(validator)}")
        print(f"Validator dir: {dir(validator)}")
        if hasattr(validator, "limit_value"):
            print(f"Validator limit_value: {validator.limit_value}")

    assert any(hasattr(v, "limit_value") and v.limit_value > 0 for v in validators)


@pytest.mark.parametrize(
    "field_type",
    [
        List[str],
        Set[int],
        Dict[str, Any],
        Optional[str],
        Union[str, int],
    ],
)
def test_type_resolver_complex_types(field_type):
    """Test TypeResolver with complex types."""
    resolver = TypeResolver()
    field_class, is_collection = resolver.resolve_type(field_type)

    if get_origin(field_type) in (list, set, dict):
        assert is_collection
    else:
        assert not is_collection

    if field_type in (Optional[str], Union[str, int]):
        assert isinstance(field_class(), models.JSONField)
    elif get_origin(field_type) in (list, set, dict):
        assert isinstance(field_class(), models.JSONField)


def test_type_resolver_edge_cases():
    """Test TypeResolver edge cases."""
    resolver = TypeResolver()

    # Test unknown type
    class UnknownType:
        pass

    field_class, is_collection = resolver.resolve_type(UnknownType)
    assert isinstance(field_class(), models.JSONField)
    assert not is_collection

    # Test custom type registration
    resolver.register_field_type(UnknownType, models.TextField)
    field_class, is_collection = resolver.resolve_type(UnknownType)
    assert isinstance(field_class(), models.TextField)


def test_field_type_manager_pattern_matching():
    """Test FieldTypeManager pattern matching for max lengths."""
    manager = FieldTypeManager()

    # Test exact pattern matches
    assert manager.get_default_max_length("error_message", models.CharField) == 1000
    assert manager.get_default_max_length("api_key_field", models.CharField) == 512

    # Test substring matches
    assert manager.get_default_max_length("user_api_key", models.CharField) == 512
    assert (
        manager.get_default_max_length("system_prompt_text", models.CharField) == 4000
    )

    # Test default fallback
    assert manager.get_default_max_length("no_pattern_match", models.CharField) == 255


def test_id_field_handling(id_fields_model):
    """Test handling of ID field name conflicts."""
    converter = FieldConverter()

    # Test lowercase 'id' field
    id_field = converter.convert_field("id", id_fields_model.model_fields["id"])
    assert isinstance(id_field, models.CharField)
    assert id_field.db_column == "id"
    assert id_field.verbose_name and "custom_id" in id_field.verbose_name.lower()

    # Test uppercase 'ID' field
    ID_field = converter.convert_field("ID", id_fields_model.model_fields["ID"])
    assert isinstance(ID_field, models.IntegerField)
    assert ID_field.db_column == "ID"
    assert ID_field.verbose_name and "custom_id" in ID_field.verbose_name.lower()

    # Test field with 'id' as part of name (should not be modified)
    user_id_field = converter.convert_field(
        "user_id", id_fields_model.model_fields["user_id"]
    )
    assert isinstance(user_id_field, models.CharField)
    assert not hasattr(user_id_field, "db_column")
    assert not hasattr(user_id_field, "verbose_name") or (
        user_id_field.verbose_name
        and "custom_id" not in user_id_field.verbose_name.lower()
    )

    # Test normal field (should not be modified)
    normal_field = converter.convert_field(
        "normal_field", id_fields_model.model_fields["normal_field"]
    )
    assert isinstance(normal_field, models.CharField)
    assert not hasattr(normal_field, "db_column")
    assert not hasattr(normal_field, "verbose_name") or (
        normal_field.verbose_name
        and "custom_id" not in normal_field.verbose_name.lower()
    )


def test_handle_id_field_function():
    """Test the handle_id_field function directly."""
    from pydantic.fields import FieldInfo

    # Test lowercase 'id'
    field_name, kwargs = handle_id_field("id", FieldInfo(annotation=str))
    assert field_name == "custom_id"
    assert kwargs["db_column"] == "id"
    assert "Custom id" in kwargs["verbose_name"]

    # Test uppercase 'ID'
    field_name, kwargs = handle_id_field("ID", FieldInfo(annotation=int))
    assert field_name == "custom_id"
    assert kwargs["db_column"] == "ID"
    assert "Custom ID" in kwargs["verbose_name"]

    # Test non-id field
    field_name, kwargs = handle_id_field("name", FieldInfo(annotation=str))
    assert field_name == "name"
    assert not kwargs
