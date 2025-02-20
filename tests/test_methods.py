"""Tests for the methods module functionality."""
import pytest
from django.db import models
from pydantic import BaseModel

from pydantic2django.methods import (
    PydanticModelConversionError,
    convert_pydantic_to_django,
    create_django_model_with_methods,
    get_methods_and_properties,
    is_pydantic_model_type,
)


def test_is_pydantic_model_type(basic_pydantic_model):
    """Test detection of Pydantic model types."""
    assert is_pydantic_model_type(basic_pydantic_model)
    assert is_pydantic_model_type(list[basic_pydantic_model])
    assert is_pydantic_model_type(dict[str, basic_pydantic_model])
    assert not is_pydantic_model_type(str)
    assert not is_pydantic_model_type(list[str])


def test_get_methods_and_properties(method_model):
    """Test extraction of methods and properties from Pydantic model."""
    attrs = get_methods_and_properties(method_model)

    # Check that methods were copied
    assert "instance_method" in attrs
    assert "class_method" in attrs
    assert "static_method" in attrs
    assert "computed_value" in attrs

    # Check that property was properly copied
    assert isinstance(attrs["computed_value"], property)

    # Check that class variables were copied
    assert "CONSTANTS" in attrs


@pytest.mark.django_db
def test_create_django_model_with_methods(method_model):
    """Test creation of Django model with copied methods."""
    django_attrs = {
        "name": models.CharField(max_length=100),
        "value": models.IntegerField(),
    }

    DjangoModel = create_django_model_with_methods(
        "TestModel", method_model, django_attrs
    )

    # Create the table for the model
    from django.db import connection

    # Create the table using raw SQL
    with connection.cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tests_testmodel (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name VARCHAR(100) NOT NULL,
                value INTEGER NOT NULL
            )
        """
        )

    # Test instance creation and method calls
    instance = DjangoModel(name="test", value=10)
    instance.save()

    assert instance.instance_method() == "Instance: test"
    assert instance.computed_value == 20
    assert DjangoModel.class_method() == ["A", "B", "C"]
    assert DjangoModel.static_method(5) == 10


@pytest.mark.django_db
def test_convert_pydantic_to_django(factory_model, product_django_model):
    """Test conversion of Pydantic model instances to Django models."""
    factory = factory_model()
    pydantic_product = factory.create_product("Test Product")

    django_product = convert_pydantic_to_django(
        pydantic_product, app_label="tests", return_pydantic_model=False
    )

    assert isinstance(django_product, product_django_model)
    assert django_product.name == "Test Product"
    assert django_product.price == pydantic_product.price


@pytest.mark.parametrize("return_pydantic_model", [True, False])
def test_convert_pydantic_to_django_not_found(
    basic_pydantic_model, return_pydantic_model
):
    """Test conversion behavior when Django model is not found."""
    instance = basic_pydantic_model(
        string_field="test",
        int_field=1,
        float_field=1.0,
        bool_field=True,
        decimal_field="10.00",
        email_field="test@example.com",
    )

    if return_pydantic_model:
        result = convert_pydantic_to_django(
            instance, app_label="nonexistent", return_pydantic_model=True
        )
        assert isinstance(result, basic_pydantic_model)
    else:
        with pytest.raises(PydanticModelConversionError):
            convert_pydantic_to_django(
                instance, app_label="nonexistent", return_pydantic_model=False
            )


@pytest.mark.django_db
def test_convert_pydantic_to_django_collections(relationship_models, user_django_model):
    """Test conversion of collections containing Pydantic models."""
    User = relationship_models["User"]
    Address = relationship_models["Address"]
    Profile = relationship_models["Profile"]
    Tag = relationship_models["Tag"]

    user = User(
        name="Test User",
        address=Address(street="123 Main St", city="Test City", country="Test Country"),
        profile=Profile(bio="Test Bio", website="http://test.com"),
        tags=[Tag(name="tag1"), Tag(name="tag2")],
    )

    django_user = convert_pydantic_to_django(
        user, app_label="tests", return_pydantic_model=False
    )

    assert isinstance(django_user, user_django_model)
    assert django_user.name == "Test User"
    assert len(django_user.tags.all()) == 2
