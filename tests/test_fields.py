"""
Tests for field mapping functionality.
"""
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Optional
from uuid import UUID

from django.db import models
from pydantic import BaseModel, EmailStr, Field

from pydantic2django import make_django_model


def test_basic_field_types():
    """Test mapping of basic field types."""

    class TestModel(BaseModel):
        string_field: str
        int_field: int
        float_field: float
        bool_field: bool
        decimal_field: Decimal
        email_field: EmailStr

    DjangoModel = make_django_model(TestModel)

    fields = {f.name: type(f) for f in DjangoModel._meta.get_fields()}

    assert fields["string_field"] == models.CharField
    assert fields["int_field"] == models.IntegerField
    assert fields["float_field"] == models.FloatField
    assert fields["bool_field"] == models.BooleanField
    assert fields["decimal_field"] == models.DecimalField
    assert fields["email_field"] == models.EmailField


def test_datetime_field_types():
    """Test mapping of datetime-related field types."""

    class TestModel(BaseModel):
        datetime_field: datetime
        date_field: date
        time_field: time
        duration_field: timedelta

    DjangoModel = make_django_model(TestModel)

    fields = {f.name: type(f) for f in DjangoModel._meta.get_fields()}

    assert fields["datetime_field"] == models.DateTimeField
    assert fields["date_field"] == models.DateField
    assert fields["time_field"] == models.TimeField
    assert fields["duration_field"] == models.DurationField


def test_optional_fields():
    """Test mapping of optional fields."""

    class TestModel(BaseModel):
        required_string: str
        optional_string: Optional[str]
        required_int: int
        optional_int: Optional[int]

    DjangoModel = make_django_model(TestModel)

    fields = {f.name: f for f in DjangoModel._meta.get_fields()}

    # Required fields should not allow null
    assert not fields["required_string"].null
    assert not fields["required_int"].null

    # Optional fields should allow null
    assert fields["optional_string"].null
    assert fields["optional_int"].null


def test_field_constraints():
    """Test that field constraints are properly mapped."""

    class TestModel(BaseModel):
        name: str = Field(title="Full Name", description="Full name of the user", max_length=100)
        age: int = Field(title="Age", description="User's age in years")
        balance: Decimal = Field(
            title="Account Balance", description="Current account balance", max_digits=10, decimal_places=2
        )

    DjangoModel = make_django_model(TestModel)

    fields = {f.name: f for f in DjangoModel._meta.get_fields()}

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

    class TestModel(BaseModel):
        id: UUID
        data: bytes

    DjangoModel = make_django_model(TestModel)

    fields = {f.name: type(f) for f in DjangoModel._meta.get_fields()}

    assert fields["id"] == models.UUIDField
    assert fields["data"] == models.BinaryField
