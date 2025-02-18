"""
Common test fixtures for pydantic2django tests.
"""
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import ClassVar, Optional
from uuid import UUID

import pytest
from django.db import models
from pydantic import BaseModel, EmailStr, Field


@pytest.fixture
def basic_pydantic_model():
    """Fixture providing a basic Pydantic model with common field types."""

    class BasicModel(BaseModel):
        string_field: str
        int_field: int
        float_field: float
        bool_field: bool
        decimal_field: Decimal
        email_field: EmailStr

    return BasicModel


@pytest.fixture
def datetime_pydantic_model():
    """Fixture providing a Pydantic model with datetime-related fields."""

    class DateTimeModel(BaseModel):
        datetime_field: datetime
        date_field: date
        time_field: time
        duration_field: timedelta

    return DateTimeModel


@pytest.fixture
def optional_fields_model():
    """Fixture providing a Pydantic model with optional fields."""

    class OptionalModel(BaseModel):
        required_string: str
        optional_string: Optional[str]
        required_int: int
        optional_int: Optional[int]

    return OptionalModel


@pytest.fixture
def constrained_fields_model():
    """Fixture providing a Pydantic model with field constraints."""

    class ConstrainedModel(BaseModel):
        name: str = Field(
            title="Full Name", description="Full name of the user", max_length=100
        )
        age: int = Field(title="Age", description="User's age in years")
        balance: Decimal = Field(
            title="Account Balance",
            description="Current account balance",
            max_digits=10,
            decimal_places=2,
        )

    return ConstrainedModel


@pytest.fixture
def relationship_models():
    """Fixture providing a set of related Pydantic models."""

    class Address(BaseModel):
        street: str
        city: str
        country: str

    class Profile(BaseModel):
        bio: str
        website: str

    class Tag(BaseModel):
        name: str

    class User(BaseModel):
        name: str
        address: Address
        profile: Profile = Field(one_to_one=True)
        tags: list[Tag]

    return {"Address": Address, "Profile": Profile, "Tag": Tag, "User": User}


@pytest.fixture
def method_model():
    """Fixture providing a Pydantic model with various method types."""

    class MethodModel(BaseModel):
        name: str
        value: int
        CONSTANTS: ClassVar[list[str]] = ["A", "B", "C"]

        def instance_method(self) -> str:
            return f"Instance: {self.name}"

        @property
        def computed_value(self) -> int:
            return self.value * 2

        @classmethod
        def class_method(cls) -> list[str]:
            return cls.CONSTANTS

        @staticmethod
        def static_method(x: int) -> int:
            return x * 2

    return MethodModel


def get_model_fields(django_model):
    """Helper function to get fields from a Django model."""
    return {f.name: f for f in django_model._meta.get_fields()}
