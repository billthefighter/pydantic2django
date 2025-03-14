"""
Common test fixtures for pydantic2django tests.
"""
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any, Callable, ClassVar, Optional
from uuid import UUID

import pytest
from django.db import models
from pydantic import BaseModel, EmailStr, Field, ConfigDict


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
            return Product(
                name=name, price=Decimal("0.99"), description="A basic product"
            )

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
    """Fixture providing test data for context testing."""
    return {
        "name": "test",
        "value": 42,
        "serializable": SerializableType(value="can_serialize"),
        "handler": ComplexHandler(),
        "processor": lambda x: x.upper(),
        "unserializable": UnserializableType("needs_context"),
    }


@pytest.fixture
def context_django_model(context_pydantic_model):
    """Fixture providing a Django model generated from the context test Pydantic model."""
    from pydantic2django.core import make_django_model

    django_model, _, _ = make_django_model(
        pydantic_model=context_pydantic_model,
        app_label="tests",
    )
    return django_model


@pytest.fixture
def context_model_context(context_django_model, context_pydantic_model):
    """Fixture providing a ModelContext for the context test model."""
    from pydantic2django.context_storage import create_context_for_model

    return create_context_for_model(
        django_model=context_django_model,
        pydantic_model=context_pydantic_model,
    )


@pytest.fixture
def context_temp_file(context_django_model, context_model_context, tmp_path):
    """Fixture providing a temporary file with the generated context class."""
    from pydantic2django.static_django_model_generator import StaticDjangoModelGenerator
    import os

    # Create generator instance
    generator = StaticDjangoModelGenerator(
        output_path=str(tmp_path / "generated_models.py"),
        packages=["tests"],
        app_label="tests",
    )

    # Generate the context class
    context_def = generator.generate_context_class(
        model=context_django_model,
        model_context=context_model_context,
    )

    # Write to temporary file
    output_file = tmp_path / "context_class.py"
    with open(output_file, "w") as f:
        f.write(context_def)

    return output_file
