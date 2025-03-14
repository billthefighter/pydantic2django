"""Tests for TypeMapper and TypeMappingDefinition classes."""
import datetime
from dataclasses import dataclass
from datetime import date, time, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

import pytest
from django.db import models
from pydantic import BaseModel, EmailStr, IPvAnyAddress, Json

from pydantic2django.field_type_mapping import TypeMapper, TypeMappingDefinition


@dataclass
class BasicTypeTestParams:
    """Test parameters for basic Python types."""

    python_type: type
    django_field: type[models.Field]
    sample_value: Any
    expected_attributes: Optional[dict[str, Any]] = None

    def __post_init__(self):
        if self.expected_attributes is None:
            self.expected_attributes = {}


@dataclass
class RelationshipTestParams:
    """Test parameters for relationship types."""

    python_type: type
    django_field: type[models.Field]
    relationship_type: str
    sample_model: Optional[type[BaseModel]] = None
    expected_attributes: Optional[dict[str, Any]] = None

    def __post_init__(self):
        if self.expected_attributes is None:
            self.expected_attributes = {}
        if self.sample_model is None:
            # Create a sample Pydantic model for testing
            class SampleModel(BaseModel):
                id: int

            self.sample_model = SampleModel


class SampleEnum(Enum):
    """Sample enum for testing."""

    OPTION_A = "A"
    OPTION_B = "B"


# Test cases for basic Python types
BASIC_TYPE_TEST_CASES = [
    pytest.param(
        BasicTypeTestParams(python_type=str, django_field=models.TextField, sample_value="test string"),
        id="string-to-text-field",
    ),
    pytest.param(
        BasicTypeTestParams(python_type=int, django_field=models.IntegerField, sample_value=42), id="integer-field"
    ),
    pytest.param(
        BasicTypeTestParams(python_type=float, django_field=models.FloatField, sample_value=3.14), id="float-field"
    ),
    pytest.param(
        BasicTypeTestParams(python_type=bool, django_field=models.BooleanField, sample_value=True), id="boolean-field"
    ),
    pytest.param(
        BasicTypeTestParams(
            python_type=datetime.datetime, django_field=models.DateTimeField, sample_value=datetime.datetime.now()
        ),
        id="datetime-field",
    ),
    pytest.param(
        BasicTypeTestParams(python_type=date, django_field=models.DateField, sample_value=date.today()), id="date-field"
    ),
    pytest.param(
        BasicTypeTestParams(python_type=time, django_field=models.TimeField, sample_value=time(12, 0)), id="time-field"
    ),
    pytest.param(
        BasicTypeTestParams(python_type=timedelta, django_field=models.DurationField, sample_value=timedelta(days=1)),
        id="duration-field",
    ),
    pytest.param(
        BasicTypeTestParams(python_type=Decimal, django_field=models.DecimalField, sample_value=Decimal("10.99")),
        id="decimal-field",
    ),
    pytest.param(
        BasicTypeTestParams(
            python_type=UUID, django_field=models.UUIDField, sample_value=UUID("12345678-1234-5678-1234-567812345678")
        ),
        id="uuid-field",
    ),
    pytest.param(
        BasicTypeTestParams(
            python_type=EmailStr,
            django_field=models.EmailField,
            sample_value="test@example.com",
            expected_attributes={"max_length": 254},
        ),
        id="email-field",
    ),
    pytest.param(
        BasicTypeTestParams(python_type=bytes, django_field=models.BinaryField, sample_value=b"binary data"),
        id="binary-field",
    ),
]

# Test cases for collection types
COLLECTION_TYPE_TEST_CASES = [
    pytest.param(
        BasicTypeTestParams(python_type=dict, django_field=models.JSONField, sample_value={"key": "value"}),
        id="dict-to-json-field",
    ),
    pytest.param(
        BasicTypeTestParams(python_type=list, django_field=models.JSONField, sample_value=["item1", "item2"]),
        id="list-to-json-field",
    ),
    pytest.param(
        BasicTypeTestParams(python_type=set, django_field=models.JSONField, sample_value={"item1", "item2"}),
        id="set-to-json-field",
    ),
]

# Test cases for special types
SPECIAL_TYPE_TEST_CASES = [
    pytest.param(
        BasicTypeTestParams(python_type=Path, django_field=models.FilePathField, sample_value=Path("/path/to/file")),
        id="path-to-filepath-field",
    ),
    pytest.param(
        BasicTypeTestParams(python_type=Enum, django_field=models.CharField, sample_value=SampleEnum.OPTION_A),
        id="enum-to-char-field",
    ),
    pytest.param(
        BasicTypeTestParams(
            python_type=IPvAnyAddress, django_field=models.GenericIPAddressField, sample_value="192.168.1.1"
        ),
        id="ip-address-field",
    ),
    pytest.param(
        BasicTypeTestParams(python_type=Json, django_field=models.JSONField, sample_value={"key": "value"}),
        id="json-type-field",
    ),
]

# Test cases for relationship types
RELATIONSHIP_TEST_CASES = [
    pytest.param(
        RelationshipTestParams(
            python_type=BaseModel,
            django_field=models.ForeignKey,
            relationship_type="foreign_key",
            expected_attributes={"on_delete": models.CASCADE},
        ),
        id="foreign-key-relationship",
    ),
    pytest.param(
        RelationshipTestParams(
            python_type=list[BaseModel],
            django_field=models.ManyToManyField,
            relationship_type="many_to_many",
        ),
        id="list-many-to-many-relationship",
    ),
    pytest.param(
        RelationshipTestParams(
            python_type=dict[str, BaseModel],
            django_field=models.ManyToManyField,
            relationship_type="many_to_many",
        ),
        id="dict-many-to-many-relationship",
    ),
]


@pytest.mark.parametrize("params", BASIC_TYPE_TEST_CASES)
def test_basic_type_mappings(params: BasicTypeTestParams):
    """Test mappings for basic Python types."""
    mapping = TypeMapper.get_mapping_for_type(params.python_type)
    assert mapping is not None, f"No mapping found for {params.python_type}"
    assert mapping.django_field == params.django_field
    assert mapping.matches_type(params.python_type)

    # Test field attributes
    field_attrs = TypeMapper.get_field_attributes(params.python_type)
    if params.expected_attributes is not None:
        for key, value in params.expected_attributes.items():
            assert field_attrs.get(key) == value


@pytest.mark.parametrize("params", COLLECTION_TYPE_TEST_CASES)
def test_collection_type_mappings(params: BasicTypeTestParams):
    """Test mappings for collection types."""
    mapping = TypeMapper.get_mapping_for_type(params.python_type)
    assert mapping is not None, f"No mapping found for {params.python_type}"
    assert mapping.django_field == params.django_field
    assert mapping.matches_type(params.python_type)


@pytest.mark.parametrize("params", SPECIAL_TYPE_TEST_CASES)
def test_special_type_mappings(params: BasicTypeTestParams):
    """Test mappings for special types."""
    mapping = TypeMapper.get_mapping_for_type(params.python_type)
    assert mapping is not None, f"No mapping found for {params.python_type}"
    assert mapping.django_field == params.django_field
    assert mapping.matches_type(params.python_type)


@pytest.mark.parametrize("params", RELATIONSHIP_TEST_CASES)
def test_relationship_type_mappings(params: RelationshipTestParams):
    """Test mappings for relationship types."""
    mapping = TypeMapper.get_mapping_for_type(params.python_type)
    assert mapping is not None, f"No mapping found for {params.python_type}"
    assert mapping.django_field == params.django_field
    assert mapping.is_relationship is True
    assert mapping.relationship_type == params.relationship_type
    assert mapping.matches_type(params.python_type)

    # Test with a concrete model class
    if params.sample_model is not None:

        class ConcreteModel(params.sample_model):
            pass

        if params.relationship_type == "many_to_many":
            if isinstance(params.python_type, type):
                # Test list-based many-to-many
                mapping = TypeMapper.get_mapping_for_type(List[ConcreteModel])
            else:
                # Test dict-based many-to-many
                mapping = TypeMapper.get_mapping_for_type(Dict[str, ConcreteModel])
        else:
            mapping = TypeMapper.get_mapping_for_type(ConcreteModel)

        assert mapping is not None
        assert mapping.django_field == params.django_field
        assert mapping.relationship_type == params.relationship_type

    # Test field attributes
    field_attrs = TypeMapper.get_field_attributes(params.python_type)
    if params.expected_attributes is not None:
        for key, value in params.expected_attributes.items():
            assert field_attrs.get(key) == value


def test_optional_type_handling():
    """Test handling of Optional types."""
    # Test Optional[str]
    mapping = TypeMapper.get_mapping_for_type(Optional[str])
    assert mapping is not None
    assert mapping.django_field == models.TextField

    # Test field attributes for Optional types
    attrs = TypeMapper.get_field_attributes(Optional[str])
    assert attrs.get("null") is True
    assert attrs.get("blank") is True


def test_unsupported_type_handling():
    """Test handling of unsupported types."""

    class UnsupportedType:
        pass

    # Test with strict=True
    with pytest.raises(TypeMapper.UnsupportedTypeError):
        TypeMapper.get_django_field_for_type(UnsupportedType, strict=True)

    # Test with strict=False
    field_type = TypeMapper.get_django_field_for_type(UnsupportedType, strict=False)
    assert field_type == models.JSONField


def test_type_mapping_definition_methods():
    """Test TypeMappingDefinition class methods."""
    # Test char_field classmethod
    char_mapping = TypeMappingDefinition.char_field(str, max_length=100)
    assert char_mapping.django_field == models.CharField
    assert char_mapping.max_length == 100

    # Test text_field classmethod
    text_mapping = TypeMappingDefinition.text_field(str)
    assert text_mapping.django_field == models.TextField

    # Test json_field classmethod
    json_mapping = TypeMappingDefinition.json_field(dict)
    assert json_mapping.django_field == models.JSONField

    # Test email_field classmethod
    email_mapping = TypeMappingDefinition.email_field()
    assert email_mapping.django_field == models.EmailField
    assert email_mapping.max_length == 254

    # Test relationship field classmethods
    fk_mapping = TypeMappingDefinition.foreign_key(BaseModel)
    assert fk_mapping.django_field == models.ForeignKey
    assert fk_mapping.is_relationship is True
    assert fk_mapping.relationship_type == "foreign_key"
    assert fk_mapping.on_delete == models.CASCADE

    m2m_mapping = TypeMappingDefinition.many_to_many(BaseModel)
    assert m2m_mapping.django_field == models.ManyToManyField
    assert m2m_mapping.is_relationship is True
    assert m2m_mapping.relationship_type == "many_to_many"

    o2o_mapping = TypeMappingDefinition.one_to_one(BaseModel)
    assert o2o_mapping.django_field == models.OneToOneField
    assert o2o_mapping.is_relationship is True
    assert o2o_mapping.relationship_type == "one_to_one"
    assert o2o_mapping.on_delete == models.CASCADE
