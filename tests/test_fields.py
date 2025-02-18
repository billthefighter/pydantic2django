"""
Tests for field mapping functionality.
"""
from django.db import models
from pydantic import BaseModel

from pydantic2django import make_django_model
from .fixtures import get_model_fields


def test_basic_field_types(basic_pydantic_model):
    """Test mapping of basic field types."""
    DjangoModel = make_django_model(basic_pydantic_model)
    fields = get_model_fields(DjangoModel)

    assert isinstance(fields["string_field"], models.CharField)
    assert isinstance(fields["int_field"], models.IntegerField)
    assert isinstance(fields["float_field"], models.FloatField)
    assert isinstance(fields["bool_field"], models.BooleanField)
    assert isinstance(fields["decimal_field"], models.DecimalField)
    assert isinstance(fields["email_field"], models.EmailField)


def test_datetime_field_types(datetime_pydantic_model):
    """Test mapping of datetime-related field types."""
    DjangoModel = make_django_model(datetime_pydantic_model)
    fields = get_model_fields(DjangoModel)

    assert isinstance(fields["datetime_field"], models.DateTimeField)
    assert isinstance(fields["date_field"], models.DateField)
    assert isinstance(fields["time_field"], models.TimeField)
    assert isinstance(fields["duration_field"], models.DurationField)


def test_optional_fields(optional_fields_model):
    """Test mapping of optional fields."""
    DjangoModel = make_django_model(optional_fields_model)
    fields = get_model_fields(DjangoModel)

    # Required fields should not allow null
    assert not fields["required_string"].null
    assert not fields["required_int"].null

    # Optional fields should allow null
    assert fields["optional_string"].null
    assert fields["optional_int"].null


def test_field_constraints(constrained_fields_model):
    """Test that field constraints are properly mapped."""
    DjangoModel = make_django_model(constrained_fields_model)
    fields = get_model_fields(DjangoModel)

    # Check field constraints
    assert fields["name"].max_length == 100
    assert fields["name"].verbose_name == "Full Name"
    assert fields["name"].help_text == "Full name of the user"

    # Check DecimalField constraints
    assert fields["balance"].max_digits == 10
    assert fields["balance"].decimal_places == 2
    assert fields["balance"].verbose_name == "Account Balance"


def test_special_field_types():
    """Test mapping of special field types."""
    from uuid import UUID

    class TestModel(BaseModel):
        id: UUID
        data: bytes

    DjangoModel = make_django_model(TestModel)
    fields = get_model_fields(DjangoModel)

    assert isinstance(fields["id"], models.UUIDField)
    assert isinstance(fields["data"], models.BinaryField)
