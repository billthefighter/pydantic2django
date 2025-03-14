"""Tests for field conversion functionality."""
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, TypeVar, Union, get_origin, get_args
from uuid import UUID
from pathlib import Path

import pytest
from django.core.validators import MinValueValidator
from django.db import models
from pydantic import BaseModel, EmailStr, Field
from pydantic.fields import FieldInfo

from pydantic2django.fields import (
    FieldConverter,
    convert_field,
    handle_id_field,
)

from pydantic2django.utils import normalize_model_name
from pydantic2django.field_type_resolver import is_pydantic_model
from pydantic2django.field_utils import FieldAttributeHandler
from pydantic2django.field_type_mapping import TypeMapper

# Test data for field type resolution
FIELD_TYPE_TEST_CASES = [
    (str, models.CharField),
    (int, models.IntegerField),
    (float, models.FloatField),
    (bool, models.BooleanField),
    (datetime, models.DateTimeField),
    (date, models.DateField),
    (time, models.TimeField),
    (timedelta, models.DurationField),
    (Decimal, models.DecimalField),
    (UUID, models.UUIDField),
    (List[str], models.JSONField),
    (Dict[str, Any], models.JSONField),
    (Set[int], models.JSONField),
    (Optional[str], models.CharField),
    (Union[str, int], models.JSONField),
    (EmailStr, models.EmailField),
    (Path, models.FilePathField),
]

# Test cases for sanitize_related_name
SANITIZE_NAME_TEST_CASES = [
    ("camelCase", "camel_case"),
    ("PascalCase", "pascal_case"),
    ("snake_case", "snake_case"),
    ("kebab-case", "kebab_case"),
    ("with spaces", "with_spaces"),
    ("with.dots", "with_dots"),
    ("with/slashes", "with_slashes"),
    ("with\\backslashes", "with_backslashes"),
    ("with+plus", "with_plus"),
    ("with@at", "with_at"),
    ("with#hash", "with_hash"),
    ("with$dollar", "with_dollar"),
    ("with%percent", "with_percent"),
    ("with^caret", "with_caret"),
    ("with&ampersand", "with_ampersand"),
    ("with*asterisk", "with_asterisk"),
    ("with(parentheses)", "with_parentheses"),
    ("with[brackets]", "with_brackets"),
    ("with{braces}", "with_braces"),
    ("with<angles>", "with_angles"),
    ("with|pipe", "with_pipe"),
    ("with~tilde", "with_tilde"),
    ("with`backtick", "with_backtick"),
    ("with'quote", "with_quote"),
    ('with"doublequote', "with_doublequote"),
    ("with!exclamation", "with_exclamation"),
    ("with?question", "with_question"),
    ("with=equals", "with_equals"),
    ("with:colon", "with_colon"),
    ("with;semicolon", "with_semicolon"),
    ("with,comma", "with_comma"),
]


@pytest.fixture
def basic_pydantic_model():
    """Fixture for a basic Pydantic model with simple field types."""

    class BasicModel(BaseModel):
        string_field: str
        int_field: int
        float_field: float
        bool_field: bool

    return BasicModel


@pytest.fixture
def constrained_fields_model():
    """Fixture for a Pydantic model with constrained fields."""

    class ConstrainedModel(BaseModel):
        name: str = Field(
            max_length=100, title="Full Name", description="Full name of the user"
        )
        balance: Decimal = Field(max_digits=10, decimal_places=2, gt=0.01)

    return ConstrainedModel


@pytest.fixture
def relationship_models():
    """Fixture for Pydantic models with relationships."""

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

    return User, Profile, Address, Tag


@pytest.fixture
def optional_fields_model():
    """Fixture for a Pydantic model with optional fields."""

    class OptionalModel(BaseModel):
        optional_string: Optional[str]
        optional_int: Optional[int]

    return OptionalModel


@pytest.fixture
def id_fields_model():
    """Fixture for a Pydantic model with ID fields."""

    class ModelWithIds(BaseModel):
        id: str = Field(description="Custom string ID")
        ID: int = Field(description="Custom integer ID")
        user_id: str = Field(description="Regular field with id in name")
        normal_field: str = Field(description="Normal field without id")

    return ModelWithIds


@pytest.fixture
def custom_class_model():
    """Fixture for a Pydantic model with custom class fields."""

    class CustomFieldsModel(BaseModel):
        with_to_dict: CustomClass
        with_str: CustomClassWithStr
        basic: CustomClassBasic
        optional_custom: Optional[CustomClass] = None

        model_config = {"arbitrary_types_allowed": True}

    return CustomFieldsModel


@pytest.mark.parametrize("input_type,expected", FIELD_TYPE_TEST_CASES)
def test_type_resolver_basic(input_type, expected):
    """Test basic type resolution for simple Python types."""
    # Use TypeMapper instead of TypeResolver
    django_field = TypeMapper.get_django_field_for_type(input_type)
    assert django_field == expected


def test_field_type_manager():
    """Test the field type manager's pattern matching capabilities."""
    # Use TypeMapper instead of FieldTypeManager
    # Test basic types
    assert TypeMapper.get_django_field_for_type(str) == models.CharField
    assert TypeMapper.get_django_field_for_type(int) == models.IntegerField
    assert TypeMapper.get_django_field_for_type(float) == models.FloatField
    assert TypeMapper.get_django_field_for_type(bool) == models.BooleanField

    # Test custom type handling
    class CustomType:
        pass

    # Should default to JSONField for unknown types when strict=False
    assert TypeMapper.python_to_django_field(CustomType) == models.JSONField

    # Should raise an exception for unknown types when strict=True
    with pytest.raises(TypeMapper.UnsupportedTypeError):
        TypeMapper.get_django_field_for_type(CustomType, strict=True)


def test_field_converter(basic_pydantic_model):
    """Test the field converter with a basic Pydantic model."""
    converter = FieldConverter()

    # Test string field conversion
    string_field_info = basic_pydantic_model.model_fields["string_field"]
    string_django_field = converter.convert_field("string_field", string_field_info)
    assert isinstance(string_django_field, models.CharField)

    # Test int field conversion
    int_field_info = basic_pydantic_model.model_fields["int_field"]
    int_django_field = converter.convert_field("int_field", int_field_info)
    assert isinstance(int_django_field, models.IntegerField)

    # Test float field conversion
    float_field_info = basic_pydantic_model.model_fields["float_field"]
    float_django_field = converter.convert_field("float_field", float_field_info)
    assert isinstance(float_django_field, models.FloatField)

    # Test bool field conversion
    bool_field_info = basic_pydantic_model.model_fields["bool_field"]
    bool_django_field = converter.convert_field("bool_field", bool_field_info)
    assert isinstance(bool_django_field, models.BooleanField)


def test_field_converter_optional_fields(optional_fields_model):
    """Test the field converter with optional fields."""
    converter = FieldConverter()

    # Test optional string field
    optional_string_info = optional_fields_model.model_fields["optional_string"]
    optional_string_field = converter.convert_field(
        "optional_string", optional_string_info
    )
    assert isinstance(optional_string_field, models.CharField)
    assert optional_string_field.null

    # Test optional int field
    optional_int_info = optional_fields_model.model_fields["optional_int"]
    optional_int_field = converter.convert_field("optional_int", optional_int_info)
    assert isinstance(optional_int_field, models.IntegerField)
    assert optional_int_field.null


def test_field_converter_relationships(relationship_models):
    """Test the field converter with relationship fields."""
    User, Profile, Address, Tag = relationship_models
    converter = FieldConverter()

    # Test one-to-one relationship
    profile_field_info = User.model_fields["profile"]
    profile_django_field = converter.convert_field("profile", profile_field_info)
    assert isinstance(profile_django_field, models.OneToOneField)

    # Test foreign key relationship
    address_field_info = User.model_fields["address"]
    address_django_field = converter.convert_field("address", address_field_info)
    assert isinstance(address_django_field, models.ForeignKey)

    # Test many-to-many relationship
    tags_field_info = User.model_fields["tags"]
    tags_django_field = converter.convert_field("tags", tags_field_info)
    assert isinstance(tags_django_field, models.ManyToManyField)


@pytest.mark.parametrize("name,expected", SANITIZE_NAME_TEST_CASES)
def test_sanitize_related_name(name, expected):
    """Test sanitizing related names for Django models."""
    assert sanitize_related_name(name) == expected


def test_sanitize_related_name_with_model_and_field():
    """Test sanitizing related names with model and field names."""
    result = sanitize_related_name("userAddress", "User", "address")
    assert result == "user_address"


def test_field_kwargs_with_constraints(constrained_fields_model):
    """Test that field constraints are properly converted to Django field kwargs."""
    converter = FieldConverter()

    # Test name field with max_length
    name_field_info = constrained_fields_model.model_fields["name"]
    name_django_field = converter.convert_field("name", name_field_info)
    assert isinstance(name_django_field, models.CharField)
    assert name_django_field.max_length == 100
    assert name_django_field.verbose_name == "Full Name"
    assert name_django_field.help_text == "Full name of the user"

    # Test balance field with decimal constraints
    balance_field_info = constrained_fields_model.model_fields["balance"]
    balance_django_field = converter.convert_field("balance", balance_field_info)
    assert isinstance(balance_django_field, models.DecimalField)
    assert balance_django_field.max_digits == 10
    assert balance_django_field.decimal_places == 2

    # Check that validators were added for gt constraint
    assert len(balance_django_field.validators) > 0
    validator_values = [
        v.limit_value
        for v in balance_django_field.validators
        if isinstance(v, MinValueValidator)
    ]
    assert Decimal("0.01") in validator_values


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
    """Test type resolution for complex Python types."""
    # Use TypeMapper instead of TypeResolver
    origin = get_origin(field_type)
    args = get_args(field_type)

    # For Optional types (Union[T, None])
    if origin is Union and type(None) in args:
        # If it's Optional[str], we should get CharField
        if str in args:
            assert TypeMapper.get_django_field_for_type(field_type) == models.CharField
        # If it's Optional[int], we should get IntegerField
        elif int in args:
            assert (
                TypeMapper.get_django_field_for_type(field_type) == models.IntegerField
            )
    # For other collection types, we should get JSONField
    elif origin in (list, set, dict) or origin is Union:
        assert TypeMapper.get_django_field_for_type(field_type) == models.JSONField


def test_type_resolver_edge_cases():
    """Test type resolution for edge cases."""
    # Use TypeMapper instead of TypeResolver
    # None should default to JSONField
    assert TypeMapper.python_to_django_field(None) == models.JSONField

    # Custom class should default to JSONField
    class UnknownType:
        pass

    assert TypeMapper.python_to_django_field(UnknownType) == models.JSONField

    # Any should default to JSONField
    assert TypeMapper.python_to_django_field(Any) == models.JSONField

    # TypeVar should default to JSONField
    T = TypeVar("T")
    assert TypeMapper.python_to_django_field(T) == models.JSONField


def test_field_type_manager_pattern_matching():
    """Test pattern matching in the field type manager."""
    # Use TypeMapper instead of FieldTypeManager
    # Test that EmailStr maps to EmailField
    assert TypeMapper.get_django_field_for_type(EmailStr) == models.EmailField

    # Test that UUID maps to UUIDField
    assert TypeMapper.get_django_field_for_type(UUID) == models.UUIDField

    # Test that Path maps to FilePathField
    assert TypeMapper.get_django_field_for_type(Path) == models.FilePathField

    # Test that datetime maps to DateTimeField
    assert TypeMapper.get_django_field_for_type(datetime) == models.DateTimeField

    # Test that date maps to DateField
    assert TypeMapper.get_django_field_for_type(date) == models.DateField

    # Test that time maps to TimeField
    assert TypeMapper.get_django_field_for_type(time) == models.TimeField


def test_id_field_handling_function():
    """Test the handle_id_field function's behavior."""
    # We'll test the behavior through the FieldConverter instead
    # since it's difficult to create valid FieldInfo objects directly
    converter = FieldConverter()

    # Create a simple model with ID fields for testing
    class TestModel(BaseModel):
        id: str = Field(description="Custom string ID")
        ID: int = Field(description="Custom integer ID")
        user_id: str = Field(description="Regular field with id in name")
        normal_field: str = Field(description="Normal field without id")

    # Test string ID field
    id_field = converter.convert_field("id", TestModel.model_fields["id"])
    assert isinstance(id_field, models.CharField)
    assert id_field.primary_key

    # Test integer ID field
    ID_field = converter.convert_field("ID", TestModel.model_fields["ID"])
    assert isinstance(ID_field, models.IntegerField)
    assert ID_field.primary_key

    # Test field with "id" in the name but not an actual ID field
    user_id_field = converter.convert_field(
        "user_id", TestModel.model_fields["user_id"]
    )
    assert isinstance(user_id_field, models.CharField)
    assert not hasattr(user_id_field, "primary_key") or not user_id_field.primary_key

    # Test normal field
    normal_field = converter.convert_field(
        "normal_field", TestModel.model_fields["normal_field"]
    )
    assert isinstance(normal_field, models.CharField)
    assert not hasattr(normal_field, "primary_key") or not normal_field.primary_key


class CustomClass:
    """A custom class with a to_dict method."""

    def __init__(self, value: str):
        self.value = value

    def to_dict(self) -> dict:
        return {"value": self.value}


class CustomClassWithStr:
    """A custom class with a __str__ method."""

    def __init__(self, value: str):
        self.value = value

    def __str__(self) -> str:
        return self.value


class CustomClassBasic:
    """A basic custom class with no special methods."""

    def __init__(self, value: str):
        self.value = value


def test_custom_class_field_conversion(custom_class_model):
    """Test conversion of custom class fields."""
    converter = FieldConverter()

    # Test custom class with to_dict method
    with_to_dict_info = custom_class_model.model_fields["with_to_dict"]
    with_to_dict_field = converter.convert_field("with_to_dict", with_to_dict_info)
    assert isinstance(with_to_dict_field, models.JSONField)

    # Test custom class with __str__ method
    with_str_info = custom_class_model.model_fields["with_str"]
    with_str_field = converter.convert_field("with_str", with_str_info)
    assert isinstance(with_str_field, models.CharField)

    # Test basic custom class
    basic_info = custom_class_model.model_fields["basic"]
    basic_field = converter.convert_field("basic", basic_info)
    assert isinstance(basic_field, models.JSONField)

    # Test optional custom class
    optional_custom_info = custom_class_model.model_fields["optional_custom"]
    optional_custom_field = converter.convert_field(
        "optional_custom", optional_custom_info
    )
    assert isinstance(optional_custom_field, models.JSONField)
    assert optional_custom_field.null
