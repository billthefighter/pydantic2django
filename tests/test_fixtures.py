"""Tests for ensuring all fixtures can be instantiated and work correctly."""
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Callable

import pytest
from django.db import models
from pydantic import EmailStr

from .fixtures import ComplexHandler, UnserializableType


def test_basic_pydantic_model(basic_pydantic_model):
    """Test that basic_pydantic_model can be instantiated with valid data."""
    model = basic_pydantic_model(
        string_field="test",
        int_field=42,
        float_field=3.14,
        bool_field=True,
        decimal_field=Decimal("10.99"),
        email_field="test@example.com",
    )

    assert model.string_field == "test"
    assert model.int_field == 42
    assert model.float_field == 3.14
    assert model.bool_field is True
    assert model.decimal_field == Decimal("10.99")
    assert model.email_field == "test@example.com"


def test_datetime_pydantic_model(datetime_pydantic_model):
    """Test that datetime_pydantic_model can be instantiated with valid data."""
    test_datetime = datetime(2024, 2, 19, 12, 0)
    test_date = date(2024, 2, 19)
    test_time = time(12, 0)
    test_duration = timedelta(days=1)

    model = datetime_pydantic_model(
        datetime_field=test_datetime,
        date_field=test_date,
        time_field=test_time,
        duration_field=test_duration,
    )

    assert model.datetime_field == test_datetime
    assert model.date_field == test_date
    assert model.time_field == test_time
    assert model.duration_field == test_duration


def test_optional_fields_model(optional_fields_model):
    """Test that optional_fields_model works with both required and optional fields."""
    # Test with all fields
    model_full = optional_fields_model(
        required_string="required",
        optional_string="optional",
        required_int=42,
        optional_int=24,
    )
    assert model_full.required_string == "required"
    assert model_full.optional_string == "optional"
    assert model_full.required_int == 42
    assert model_full.optional_int == 24

    # Test with only required fields
    model_required = optional_fields_model(required_string="required", required_int=42)
    assert model_required.required_string == "required"
    assert model_required.optional_string is None
    assert model_required.required_int == 42
    assert model_required.optional_int is None


def test_constrained_fields_model(constrained_fields_model):
    """Test that constrained_fields_model validates constraints correctly."""
    model = constrained_fields_model(
        name="John Doe", age=30, balance=Decimal("1000.50")
    )

    assert model.name == "John Doe"
    assert model.age == 30
    assert model.balance == Decimal("1000.50")

    # Test max_length constraint
    with pytest.raises(ValueError):
        constrained_fields_model(
            name="x" * 101,  # Exceeds max_length of 100
            age=30,
            balance=Decimal("1000.50"),
        )


def test_relationship_models(relationship_models):
    """Test that relationship_models can be instantiated and nested correctly."""
    Address = relationship_models["Address"]
    Profile = relationship_models["Profile"]
    Tag = relationship_models["Tag"]
    User = relationship_models["User"]

    address = Address(street="123 Main St", city="Anytown", country="USA")
    profile = Profile(bio="Test bio", website="http://example.com")
    tags = [Tag(name="tag1"), Tag(name="tag2")]

    user = User(name="John Doe", address=address, profile=profile, tags=tags)

    assert user.name == "John Doe"
    assert user.address.street == "123 Main St"
    assert user.profile.bio == "Test bio"
    assert len(user.tags) == 2
    assert user.tags[0].name == "tag1"


def test_method_model(method_model):
    """Test that method_model's various method types work correctly."""
    model = method_model(name="test", value=5)

    # Test instance method
    assert model.instance_method() == "Instance: test"

    # Test property
    assert model.computed_value == 10

    # Test class method
    assert model.class_method() == ["A", "B", "C"]
    assert method_model.class_method() == ["A", "B", "C"]

    # Test static method
    assert model.static_method(3) == 6
    assert method_model.static_method(3) == 6


def test_factory_model(factory_model):
    """Test that factory_model can create products using both methods."""
    factory = factory_model()

    # Test create_product with defaults
    product1 = factory.create_product("Test Product")
    assert product1.name == "Test Product"
    assert product1.price == Decimal("9.99")
    assert product1.description == "A great product"

    # Test create_product with custom values
    product2 = factory.create_product(
        name="Custom Product", price=Decimal("19.99"), description="Custom description"
    )
    assert product2.name == "Custom Product"
    assert product2.price == Decimal("19.99")
    assert product2.description == "Custom description"

    # Test create_simple_product
    product3 = factory.create_simple_product("Simple Product")
    assert product3.name == "Simple Product"
    assert product3.price == Decimal("0.99")
    assert product3.description == "A basic product"


def test_context_pydantic_model(context_pydantic_model, context_with_data):
    """Test that context_pydantic_model can be instantiated with valid data."""
    # Create model instance
    model = context_pydantic_model(**context_with_data)

    # Verify regular fields
    assert model.name == "test"
    assert model.value == 42
    assert model.serializable.value == "can_serialize"

    # Verify fields needing context
    assert isinstance(model.handler, ComplexHandler)
    assert callable(model.processor)
    assert isinstance(model.unserializable, UnserializableType)
    assert model.unserializable.value == "needs_context"

    # Test serialization behavior
    model_dict = model.model_dump()
    # Serializable type should convert to string
    assert isinstance(model_dict["serializable"], str)
    # Non-serializable types should raise errors
    with pytest.raises(TypeError):
        model.model_dump_json()


def test_context_django_model(context_django_model):
    """Test that context_django_model was generated correctly."""
    # Check model attributes
    assert hasattr(context_django_model, "name")
    assert hasattr(context_django_model, "value")
    assert hasattr(context_django_model, "serializable")

    # Get all fields
    fields = {f.name: f for f in context_django_model._meta.get_fields()}

    # Regular fields should have appropriate Django field types
    assert isinstance(fields["name"], models.CharField)
    assert isinstance(fields["value"], models.IntegerField)
    # Serializable type should be stored as TextField without is_relationship
    assert isinstance(fields["serializable"], models.TextField)
    assert not getattr(fields["serializable"], "is_relationship", False)

    # Non-serializable fields should be TextField with is_relationship=True
    assert isinstance(fields["handler"], models.TextField)
    assert getattr(fields["handler"], "is_relationship", False) is True

    assert isinstance(fields["processor"], models.TextField)
    assert getattr(fields["processor"], "is_relationship", False) is True

    assert isinstance(fields["unserializable"], models.TextField)
    assert getattr(fields["unserializable"], "is_relationship", False) is True


def test_context_model_context(context_model_context):
    """Test that context_model_context was created correctly."""
    # Check basic attributes
    assert context_model_context.model_name == "DjangoContextTestModel"
    assert context_model_context.pydantic_class.__name__ == "ContextTestModel"

    # Regular and serializable fields should not be in context
    assert "name" not in context_model_context.required_context_keys
    assert "value" not in context_model_context.required_context_keys
    assert "serializable" not in context_model_context.required_context_keys

    # Only non-serializable fields should be tracked
    assert "handler" in context_model_context.required_context_keys
    assert "processor" in context_model_context.required_context_keys
    assert "unserializable" in context_model_context.required_context_keys

    # Check field contexts
    handler_context = context_model_context.context_fields["handler"]
    assert handler_context.field_type == ComplexHandler
    assert not handler_context.is_optional
    assert not handler_context.is_list

    processor_context = context_model_context.context_fields["processor"]
    assert processor_context.field_type == Callable
    assert not processor_context.is_optional
    assert not processor_context.is_list

    unserializable_context = context_model_context.context_fields["unserializable"]
    assert unserializable_context.field_type == UnserializableType
    assert not unserializable_context.is_optional
    assert not unserializable_context.is_list


def test_context_temp_file(context_temp_file):
    """Test that the generated context class file has the expected content."""
    # Read the generated file
    content = context_temp_file.read_text()

    # Check for key components
    assert "class DjangoContextTestModelContext(ModelContext):" in content
    assert 'model_name: str = "DjangoContextTestModel"' in content
    assert "pydantic_class: Type = ContextTestModel" in content

    # Regular fields should not be in context
    assert "name:" not in content
    assert "value:" not in content
    assert "serializable:" not in content

    # Only non-serializable fields should be in context
    assert "handler: ComplexHandler" in content
    assert "processor: Callable" in content
    assert "unserializable: UnserializableType" in content

    # Check for required imports
    assert "from typing import" in content
    assert "from dataclasses import dataclass, field" in content
    assert (
        "from pydantic2django.context_storage import ModelContext, FieldContext"
        in content
    )

    # Check for create method
    assert "@classmethod" in content
    assert "def create(cls" in content
    assert "def to_dict(self)" in content
