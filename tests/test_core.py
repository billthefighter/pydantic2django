"""Tests for core functionality of converting Pydantic models to Django models."""
from decimal import Decimal

import pytest
from django.db import models
from pydantic import BaseModel

from pydantic2django.core import make_django_model


# Define a custom model for inheritance testing
class CustomModel(models.Model):
    custom_field = models.CharField(max_length=100)

    class Meta:
        abstract = True


def test_basic_model_conversion(basic_pydantic_model):
    """Test conversion of a basic Pydantic model with simple field types."""
    django_model, _ = make_django_model(basic_pydantic_model, app_label="testapp")

    # Check model class
    assert issubclass(django_model, models.Model)
    assert django_model.__name__ == "DjangoBasicModel"

    # Check fields
    fields = {f.name: type(f) for f in django_model._meta.fields}
    assert fields["string_field"] == models.CharField
    assert fields["int_field"] == models.IntegerField
    assert fields["float_field"] == models.FloatField
    assert fields["bool_field"] == models.BooleanField
    assert fields["decimal_field"] == models.DecimalField
    assert fields["email_field"] == models.EmailField


def test_datetime_model_conversion(datetime_pydantic_model):
    """Test conversion of a model with datetime-related fields."""
    django_model, _ = make_django_model(datetime_pydantic_model, app_label="testapp")

    fields = {f.name: type(f) for f in django_model._meta.fields}
    assert fields["datetime_field"] == models.DateTimeField
    assert fields["date_field"] == models.DateField
    assert fields["time_field"] == models.TimeField
    assert fields["duration_field"] == models.DurationField


def test_optional_fields_model_conversion(optional_fields_model):
    """Test conversion of a model with optional fields."""
    django_model, _ = make_django_model(
        optional_fields_model, app_label="testapp", ignore_errors=True
    )

    # Check model class
    assert issubclass(django_model, models.Model)
    assert django_model.__name__ == "DjangoOptionalModel"

    # Check fields - only required fields should be present
    fields = {f.name: type(f) for f in django_model._meta.fields}
    assert "required_string" in fields
    assert "required_int" in fields
    assert isinstance(fields["required_string"], models.CharField)
    assert isinstance(fields["required_int"], models.IntegerField)

    # Optional fields should be skipped due to ignore_errors=True
    assert "optional_string" not in fields
    assert "optional_int" not in fields


def test_constrained_fields_model_conversion(constrained_fields_model):
    """Test conversion of a model with field constraints."""
    django_model, _ = make_django_model(constrained_fields_model, app_label="testapp")

    # Check name field constraints
    name_field = django_model._meta.get_field("name")
    assert isinstance(name_field, models.CharField)
    assert name_field.max_length == 100
    assert name_field.verbose_name == "Full Name"
    assert name_field.help_text == "Full name of the user"

    # Check balance field constraints
    balance_field = django_model._meta.get_field("balance")
    assert isinstance(balance_field, models.DecimalField)
    assert balance_field.max_digits == 10
    assert balance_field.decimal_places == 2
    assert balance_field.verbose_name == "Account Balance"
    assert balance_field.help_text == "Current account balance"


def test_relationship_models_conversion(relationship_models):
    """Test conversion of models with relationships."""
    Address = relationship_models["Address"]
    Profile = relationship_models["Profile"]
    Tag = relationship_models["Tag"]
    User = relationship_models["User"]

    # First convert the related models
    django_address, _ = make_django_model(Address, app_label="testapp")
    django_profile, _ = make_django_model(Profile, app_label="testapp")
    django_tag, _ = make_django_model(Tag, app_label="testapp")

    # Then convert the main model
    django_user, _ = make_django_model(User, app_label="testapp")

    # Check relationship fields
    address_field = django_user._meta.get_field("address")
    profile_field = django_user._meta.get_field("profile")
    tags_field = django_user._meta.get_field("tags")

    assert isinstance(address_field, models.ForeignKey)
    assert isinstance(profile_field, models.OneToOneField)
    assert isinstance(tags_field, models.ManyToManyField)

    assert address_field.related_model == django_address
    assert profile_field.related_model == django_profile
    assert tags_field.related_model == django_tag


def test_model_inheritance(basic_pydantic_model):
    """Test Django model inheritance."""

    # Define a custom model for inheritance testing in the test function
    class LocalCustomModel(models.Model):
        custom_field = models.CharField(max_length=100)

        class Meta:
            abstract = True
            app_label = "testapp"

    django_model, _ = make_django_model(
        basic_pydantic_model, base_django_model=LocalCustomModel, app_label="testapp"
    )

    # Check that the model has the custom field from the base model
    fields = {f.name: type(f) for f in django_model._meta.fields}
    assert "custom_field" in fields
    assert isinstance(fields["custom_field"], models.CharField)

    # Check that the model also has the fields from the Pydantic model
    assert "string_field" in fields
    assert "int_field" in fields
    assert isinstance(fields["string_field"], models.CharField)
    assert isinstance(fields["int_field"], models.IntegerField)


def test_model_caching(basic_pydantic_model):
    """Test that converted models are cached and reused."""
    django_model1, _ = make_django_model(basic_pydantic_model, app_label="testapp")
    django_model2, _ = make_django_model(basic_pydantic_model, app_label="testapp")

    # Should return the same class instance
    assert django_model1 is django_model2


def test_factory_model_conversion(factory_model):
    """Test conversion of a model with methods."""
    django_model, _ = make_django_model(factory_model, app_label="testapp")

    # Check that the fields were converted correctly
    fields = {f.name: type(f) for f in django_model._meta.fields}
    assert fields["default_price"] == models.DecimalField
    assert fields["default_description"] == models.CharField

    # Create an instance to verify default values
    instance = django_model(
        default_price=Decimal("9.99"), default_description="A great product"
    )
    assert instance.default_price == Decimal("9.99")
    assert instance.default_description == "A great product"


def test_invalid_conversion():
    """Test that invalid conversions raise appropriate errors."""

    class InvalidModel(BaseModel):
        invalid_field: complex  # Using a type that can't be converted to Django
        valid_field: str = "test"  # Add a valid field to ensure the model is processed

    # This should raise a ValueError because complex is not supported
    with pytest.raises(ValueError):
        make_django_model(InvalidModel, app_label="testapp")


def test_missing_app_label(basic_pydantic_model):
    """Test that missing app_label raises an error."""
    with pytest.raises(ValueError, match="app_label must be provided in options"):
        make_django_model(basic_pydantic_model)
