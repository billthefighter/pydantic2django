"""
Common test fixtures for pydantic2django tests.
"""
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any, Callable, ClassVar, Optional
from uuid import UUID
from dataclasses import dataclass, field
from enum import Enum

import pytest
from django.db import models
from pydantic import BaseModel, EmailStr, Field, ConfigDict
import uuid
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _

# Import helper classes needed by fixtures (e.g., for context_pydantic_model)
# These are defined in the parent conftest.py
from ..conftest import ComplexHandler, SerializableType, UnserializableType


# --- Module-level Django Model Definitions ---


# Define validator and choices needed by AllFieldsModel first
def custom_validator(value):
    # Example custom validator logic
    if isinstance(value, str) and "invalid" in value:
        raise ValidationError("Value contains 'invalid'.")
    # Add more checks as needed for different types if this validator is reused
    # Example for Decimal:
    # if isinstance(value, Decimal) and value < 0:
    #     raise ValidationError("Decimal value cannot be negative.")


class StatusChoices(models.TextChoices):
    PENDING = "PENDING", "Pending"
    COMPLETED = "COMPLETED", "Completed"


# Define AbstractModel
class AbstractModel(models.Model):
    """Abstract model for testing."""

    name = models.CharField(max_length=100)

    class Meta:
        app_label = "tests"
        abstract = True


# Define RelatedModel
class RelatedModel(models.Model):
    name = models.CharField(max_length=50)

    class Meta:
        app_label = "tests"


# Define ConcreteModel inheriting from AbstractModel
class ConcreteModel(AbstractModel):
    """Concrete model inheriting from AbstractModel."""

    # Add any additional fields specific to ConcreteModel if needed
    pass

    class Meta:
        app_label = "tests"
        abstract = False


# Define AllFieldsModel (renamed from AllFieldsModelBase)
class AllFieldsModel(models.Model):
    """Model demonstrating various Django fields and their common arguments."""

    # --- Basic Fields ---
    auto_field = models.AutoField(primary_key=True, verbose_name="ID")
    boolean_field = models.BooleanField(default=False, help_text="A true/false value")
    null_boolean_field = models.BooleanField(null=True, blank=True)
    char_field = models.CharField(max_length=255)
    char_field_choices = models.CharField(
        max_length=50,  # Adjusted to fit choice values
        choices=StatusChoices.choices,
        default=StatusChoices.PENDING,
        null=True,  # Allow null in DB to bypass IntegrityError in some tests
        # Removed unique=True, db_index=True, validators=[] for simplicity in conversion tests
        help_text="A choice field",
    )
    text_field = models.TextField(blank=True, verbose_name="Detailed Description")
    slug_field = models.SlugField(
        max_length=100, allow_unicode=True, db_index=True, help_text="URL-friendly identifier"
    )
    email_field = models.EmailField(max_length=254, unique=True)
    url_field = models.URLField(max_length=300)
    ip_address_field = models.GenericIPAddressField(protocol="both", unpack_ipv4=False)
    uuid_field = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    # --- Numeric Fields ---
    integer_field = models.IntegerField(default=0)
    big_integer_field = models.BigIntegerField(null=True, blank=True)
    small_integer_field = models.SmallIntegerField()
    positive_integer_field = models.PositiveIntegerField()
    positive_small_integer_field = models.PositiveSmallIntegerField()
    positive_big_integer_field = models.PositiveBigIntegerField()
    float_field = models.FloatField(null=True, blank=True)
    decimal_field = models.DecimalField(max_digits=10, decimal_places=2)  # Removed validator for simplicity
    # --- Date/Time Fields ---
    date_field = models.DateField(auto_now_add=True)
    datetime_field = models.DateTimeField(auto_now=True)
    time_field = models.TimeField(null=True, blank=True)
    duration_field = models.DurationField()
    # --- Binary Fields ---
    binary_field = models.BinaryField(editable=True)  # Make editable for testing assignment
    # --- File Fields ---
    file_field = models.FileField(upload_to="uploads/%Y/%m/%d/", blank=True, null=True)
    image_field = models.ImageField(
        upload_to="images/", height_field="image_height", width_field="image_width", blank=True, null=True
    )
    image_height = models.PositiveIntegerField(null=True, blank=True, editable=False)
    image_width = models.PositiveIntegerField(null=True, blank=True, editable=False)
    # --- Relational Fields ---
    foreign_key_field = models.ForeignKey(
        RelatedModel,
        on_delete=models.SET_NULL,
        related_name="all_fields_models_fk",  # Renamed related_name
        # to_field="name", # Removed to simplify testing, requires name to be unique on RelatedModel
        null=True,
        blank=True,
    )
    one_to_one_field = models.OneToOneField(
        RelatedModel,
        on_delete=models.CASCADE,
        related_name="all_fields_model_o2o",
        null=True,
        blank=True,
    )
    many_to_many_field = models.ManyToManyField(
        RelatedModel,
        related_name="all_fields_models_m2m",
        through="Membership",
        through_fields=("all_fields_model", "related_model"),
        blank=True,
    )
    # --- Other Fields ---
    json_field = models.JSONField(null=True, blank=True, default=dict)  # Added default

    class Meta:
        app_label = "tests"
        db_table = "all_fields_comprehensive"
        ordering = ["-datetime_field", "char_field"]
        verbose_name = "Comprehensive Field Model"
        verbose_name_plural = "Comprehensive Field Models"
        # unique_together = (("char_field", "integer_field"),) # Removed for simplicity
        # index_together = [["char_field", "date_field"]] # Removed for simplicity


# Define Membership linking AllFieldsModel and RelatedModel
class Membership(models.Model):
    all_fields_model = models.ForeignKey(AllFieldsModel, on_delete=models.CASCADE)
    related_model = models.ForeignKey(RelatedModel, on_delete=models.CASCADE)
    date_joined = models.DateField(auto_now_add=True)

    class Meta:
        app_label = "tests"
        unique_together = (("all_fields_model", "related_model"),)


# Define Product model (from product_django_model)
class Product(models.Model):
    name = models.CharField(max_length=100)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    description = models.TextField()

    class Meta:
        app_label = "tests"


# Define User-related models (from user_django_model)
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


# --- Pydantic Fixtures (Remain mostly unchanged) ---


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
        optional_string: Optional[str] = None
        required_int: int
        optional_int: Optional[int] = None

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

    class AddressPydantic(BaseModel):  # Renamed to avoid clash with Django model
        street: str
        city: str
        country: str

    class ProfilePydantic(BaseModel):  # Renamed
        bio: str
        website: str

    class TagPydantic(BaseModel):  # Renamed
        tag_name: str  # Keep original field name from Pydantic definition

    class UserPydantic(BaseModel):  # Renamed
        name: str
        address: AddressPydantic
        profile: ProfilePydantic = Field(json_schema_extra={"one_to_one": True})
        tags: list[TagPydantic]

    return {"Address": AddressPydantic, "Profile": ProfilePydantic, "Tag": TagPydantic, "User": UserPydantic}


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


# --- Start Dataclass Fixtures ---


@pytest.fixture
def basic_dataclass():
    """Fixture providing a basic dataclass with common field types."""

    @dataclass
    class BasicDC:
        string_field: str
        int_field: int
        float_field: float
        bool_field: bool

    return BasicDC


@pytest.fixture
def datetime_dataclass():
    """Fixture providing a dataclass with datetime-related fields."""

    @dataclass
    class DateTimeDC:
        datetime_field: datetime
        date_field: date
        time_field: time
        duration_field: timedelta

    return DateTimeDC


# Define StatusEnum at module level so it can be imported
class StatusEnum(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


@pytest.fixture
def advanced_types_dataclass():
    """Fixture providing a dataclass with advanced types like Decimal, UUID, and Enum."""

    @dataclass
    class AdvancedTypesDC:
        decimal_field: Decimal
        uuid_field: UUID
        enum_field: StatusEnum

    return AdvancedTypesDC


# --- Define Relationship Dataclasses at Module Level for Resolution ---
@dataclass
class AddressDC:
    street: str
    city: str
    country: str


@dataclass
class ProfileDC:
    bio: str
    website: str


@dataclass
class TagDC:
    tag_name: str


# --- End Module Level Relationship Dataclasses ---


@pytest.fixture
def optional_dataclass():
    """Fixture providing a dataclass with optional fields."""

    @dataclass
    class OptionalDC:
        required_string: str
        required_int: int
        optional_string: Optional[str] = None
        optional_int: Optional[int] = None

    return OptionalDC


@pytest.fixture
def nested_dataclass():
    """Fixture providing nested dataclasses."""

    @dataclass
    class InnerDC:
        inner_value: int

    @dataclass
    class OuterDC:
        name: str
        inner: InnerDC
        items: list[int] = field(default_factory=list)

    return {"InnerDC": InnerDC, "OuterDC": OuterDC}


@pytest.fixture
def relationship_dataclasses():
    """Fixture providing a set of related dataclasses mimicking relationships."""

    # Define UserDC locally, referencing the module-level types
    @dataclass
    class UserDC:
        user_name: str
        address: AddressDC  # Simulates ForeignKey (references module-level AddressDC)
        profile: ProfileDC  # Simulates OneToOne (references module-level ProfileDC)
        tags: list[TagDC]  # Simulates ManyToMany (references module-level TagDC)

    # Return the UserDC defined here PLUS the module-level ones
    return {"AddressDC": AddressDC, "ProfileDC": ProfileDC, "TagDC": TagDC, "UserDC": UserDC}


@pytest.fixture
def metadata_dataclass():
    """Fixture providing a dataclass using field metadata."""

    @dataclass
    class MetadataDC:
        item_name: str = field(metadata={"description": "The name of the item"})
        value: int = field(metadata={"units": "meters"})

    return MetadataDC


# --- End Dataclass Fixtures ---

# --- Updated Django Model Fixtures ---


@pytest.fixture
def abstract_model():
    """Fixture providing the AbstractModel Django class."""
    return AbstractModel


@pytest.fixture
def related_model():
    """Fixture providing the RelatedModel Django class."""
    return RelatedModel


@pytest.fixture
def concrete_model():  # Removed abstract_model dependency as it's module level
    """Fixture providing the ConcreteModel Django class."""
    return ConcreteModel


@pytest.fixture
def all_fields_model():  # Removed related_model dependency
    """Fixture providing the AllFieldsModel Django class."""
    return AllFieldsModel


@pytest.fixture
def membership_model():  # Removed all_fields_model and related_model dependencies
    """Fixture providing the Membership Django class (through model)."""
    return Membership


@pytest.fixture
def user_django_model():
    """Fixture providing the User Django class (and implicitly Address, Profile, Tag)."""
    # The fixture still makes sense to return the main 'User' model,
    # but Address, Profile, Tag are now also available at module level if needed directly.
    return User
