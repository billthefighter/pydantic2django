"""Tests for ensuring all fixtures can be instantiated and work correctly."""
from datetime import date, datetime, time, timedelta
from decimal import Decimal

import pytest
from pydantic import EmailStr


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
