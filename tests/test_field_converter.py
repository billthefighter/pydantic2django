"""Tests for field_converter module."""
import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import List, Optional
from uuid import UUID

import pytest
from django.db import models
from pydantic import BaseModel, EmailStr, IPvAnyAddress, Json
from pydantic.fields import FieldInfo

from pydantic2django.field_converter import convert_field


class SampleEnum(Enum):
    """Sample enum for testing."""

    OPTION_A = "A"
    OPTION_B = "B"


class SampleModel(BaseModel):
    """Sample model for testing relationships."""

    id: int


def test_handle_id_field():
    """Test handling of ID fields."""
    # Test integer ID
    field_info = FieldInfo(annotation=int)
    field = convert_field("id", field_info)
    assert isinstance(field, models.AutoField)
    assert field.primary_key is True

    # Test string ID
    field_info = FieldInfo(annotation=str)
    field = convert_field("id", field_info)
    assert isinstance(field, models.CharField)
    assert field.primary_key is True
    assert field.max_length == 255


def test_basic_type_conversion():
    """Test conversion of basic Python types."""
    test_cases = [
        (str, models.TextField),
        (int, models.IntegerField),
        (float, models.FloatField),
        (bool, models.BooleanField),
        (datetime.datetime, models.DateTimeField),
        (datetime.date, models.DateField),
        (datetime.time, models.TimeField),
        (datetime.timedelta, models.DurationField),
        (Decimal, models.DecimalField),
        (UUID, models.UUIDField),
        (EmailStr, models.EmailField),
        (bytes, models.BinaryField),
    ]

    for python_type, django_field in test_cases:
        field_info = FieldInfo(annotation=python_type)
        field = convert_field("test", field_info)
        assert isinstance(field, django_field)


def test_collection_type_conversion():
    """Test conversion of collection types."""
    test_cases = [
        (dict, models.JSONField),
        (list, models.JSONField),
        (set, models.JSONField),
    ]

    for python_type, django_field in test_cases:
        field_info = FieldInfo(annotation=python_type)
        field = convert_field("test", field_info)
        assert isinstance(field, django_field)


def test_special_type_conversion():
    """Test conversion of special types."""
    test_cases = [
        (Path, models.FilePathField),
        (SampleEnum, models.CharField),
        (IPvAnyAddress, models.GenericIPAddressField),
        (Json, models.JSONField),
    ]

    for python_type, django_field in test_cases:
        field_info = FieldInfo(annotation=python_type)
        field = convert_field("test", field_info)
        assert isinstance(field, django_field)


def test_relationship_type_conversion():
    """Test conversion of relationship types."""
    # Test ForeignKey
    field_info = FieldInfo(annotation=SampleModel)
    field = convert_field("test", field_info)
    assert isinstance(field, models.ForeignKey)
    assert field.remote_field.on_delete == models.CASCADE

    # Test ManyToManyField with List
    field_info = FieldInfo(annotation=List[SampleModel])
    field = convert_field("test", field_info)
    assert isinstance(field, models.ManyToManyField)


def test_optional_type_handling():
    """Test handling of Optional types."""
    # Test Optional[str]
    field_info = FieldInfo(annotation=Optional[str])
    field = convert_field("test", field_info)
    assert isinstance(field, models.TextField)
    assert field.null is True
    assert field.blank is True


def test_skip_relationships():
    """Test skipping relationship fields."""
    # Test ForeignKey with skip_relationships=True
    field_info = FieldInfo(annotation=SampleModel)
    field = convert_field("test", field_info, skip_relationships=True)
    assert field is None

    # Test ManyToManyField with skip_relationships=True
    field_info = FieldInfo(annotation=List[SampleModel])
    field = convert_field("test", field_info, skip_relationships=True)
    assert field is None


def test_unsupported_type_handling():
    """Test handling of unsupported types."""

    class UnsupportedType:
        pass

    field_info = FieldInfo(annotation=UnsupportedType)
    with pytest.raises(ValueError):
        convert_field("test", field_info)
