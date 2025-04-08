"""
Pytest configuration for Django tests.
"""
import os
import sys
from pathlib import Path
import logging
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any, Callable, ClassVar, Optional
from uuid import UUID

import django
import pytest
from django.conf import settings
from django.db import models
from pydantic import BaseModel, EmailStr, Field, ConfigDict

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Add src directory to Python path
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Configure Django settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tests.settings")


def pytest_configure():
    """Configure Django for tests."""
    django.setup()


@pytest.fixture(scope="session")
def django_db_setup(django_db_blocker):
    """Configure the test database."""
    settings.DATABASES["default"] = {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    }

    from django.core.management import call_command

    with django_db_blocker.unblock():
        call_command("migrate", "tests", verbosity=0)


@pytest.fixture(autouse=True)
def setup_logging():
    """Set up logging for all tests."""
    logging.basicConfig(level=logging.INFO)
    for logger_name in ["tests", "pydantic2django"]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)


# Fixtures moved from tests/fixtures/fixtures.py


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
        optional_string: str | None = None
        required_int: int
        optional_int: int | None = None

    return OptionalModel


@pytest.fixture
def constrained_fields_model():
    """Fixture providing a Pydantic model with field constraints."""

    class ConstrainedModel(BaseModel):
        name: str = Field(title="Full Name", description="Full name of the user", max_length=100)
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
        profile: Profile = Field(json_schema_extra={"one_to_one": True})
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


@pytest.fixture
def factory_model():
    """Fixture providing a Pydantic model that can create instances of another model."""

    class Product(BaseModel):
        name: str
        price: Decimal
        description: str

    class ProductFactory(BaseModel):
        default_price: Decimal = Decimal("9.99")
        default_description: str = "A great product"

        def create_product(
            self,
            name: str,
            price: Optional[Decimal] = None,
            description: Optional[str] = None,
        ) -> Product:
            return Product(
                name=name,
                price=price or self.default_price,
                description=description or self.default_description,
            )

        def create_simple_product(self, name: str) -> Product:
            """A simpler method that just creates a basic product with a name."""
            return Product(name=name, price=Decimal("0.99"), description="A basic product")

    return ProductFactory


@pytest.fixture
def product_django_model():
    """Fixture providing a Django model for Product."""

    class Product(models.Model):
        name = models.CharField(max_length=100)
        price = models.DecimalField(max_digits=10, decimal_places=2)
        description = models.TextField()

        class Meta:
            app_label = "tests"

    return Product


@pytest.fixture
def user_django_model():
    """Fixture providing Django models for User and related models."""

    class Address(models.Model):
        street = models.CharField(max_length=200)
        city = models.CharField(max_length=100)
        country = models.CharField(max_length=100)

        class Meta:
            app_label = "tests"

    class Profile(models.Model):
        bio = models.TextField()
        website = models.URLField()

        class Meta:
            app_label = "tests"

    class Tag(models.Model):
        name = models.CharField(max_length=50)

        class Meta:
            app_label = "tests"

    class User(models.Model):
        name = models.CharField(max_length=100)
        address = models.ForeignKey(Address, on_delete=models.CASCADE)
        profile = models.OneToOneField(Profile, on_delete=models.CASCADE)
        tags = models.ManyToManyField(Tag)

        class Meta:
            app_label = "tests"

    return User


def get_model_fields(django_model):
    """Helper function to get fields from a Django model."""
    return {f.name: f for f in django_model._meta.get_fields()}


class UnserializableType:
    """A type that can't be serialized to JSON."""

    def __init__(self, value: str):
        self.value = value


class ComplexHandler:
    """A complex handler that can't be serialized."""

    def process(self, data: Any) -> Any:
        return data


class SerializableType(BaseModel):
    """A type that can be serialized to JSON."""

    value: str

    model_config = ConfigDict(json_schema_extra={"examples": [{"value": "example"}]})


@pytest.fixture
def context_pydantic_model():
    """Fixture providing a Pydantic model with both serializable and non-serializable fields."""

    class ContextTestModel(BaseModel):
        """Test model with various field types that may need context."""

        model_config = ConfigDict(arbitrary_types_allowed=True)

        # Regular fields (no context needed)
        name: str
        value: int
        serializable: SerializableType  # Has schema, no context needed

        # Fields needing context
        handler: ComplexHandler  # Arbitrary type, needs context
        processor: Callable[[str], str]  # Function type, needs context
        unserializable: UnserializableType  # Arbitrary type, needs context

    return ContextTestModel


@pytest.fixture
def context_with_data():
    """Fixture providing data for ContextTestModel."""

    def sample_processor(s: str) -> str:
        return s.upper()

    return {
        "name": "test",
        "value": 42,
        "serializable": SerializableType(value="can_serialize"),
        "handler": ComplexHandler(),
        "processor": sample_processor,
        "unserializable": UnserializableType(value="needs_context"),
    }
