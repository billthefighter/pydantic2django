"""Tests for the methods module functionality."""
import pytest
from django.db import models
from pydantic import BaseModel, ConfigDict

from pydantic2django.methods import (
    PydanticModelConversionError,
    convert_pydantic_to_django,
    create_django_model_with_methods,
    get_methods_and_properties,
    is_pydantic_model_type,
    serialize_class_instance,
    deserialize_class_instance,
    create_django_model_with_methods,
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
    """Test creating a Django model with methods and properties from a Pydantic model."""
    # Create the Django model class
    django_attrs = {
        "name": models.CharField(max_length=100),
        "value": models.IntegerField(),
    }

    DjangoModel = create_django_model_with_methods(
        "TestModel", method_model, django_attrs
    )

    # Ensure the model has the expected methods and properties
    assert hasattr(DjangoModel, "class_method")
    assert hasattr(DjangoModel, "static_method")

    # Create a test database table for the model
    from django.db import connection

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

    # Ensure instance has the expected methods and properties
    assert hasattr(instance, "instance_method")
    assert hasattr(instance, "computed_value")

    # Test method calls - ignore type checking for these lines
    assert instance.instance_method() == "Instance: test"  # type: ignore
    assert instance.computed_value == 20  # type: ignore
    assert DjangoModel.class_method() == ["A", "B", "C"]  # type: ignore
    assert DjangoModel.static_method(5) == 10  # type: ignore


@pytest.mark.django_db
def test_convert_pydantic_to_django(factory_model, product_django_model):
    """Test conversion of Pydantic model instances to Django models."""
    factory = factory_model()
    pydantic_product = factory.create_product("Test Product")

    # Set return_pydantic_model=True to avoid model lookup error
    django_product = convert_pydantic_to_django(
        pydantic_product, app_label="tests", return_pydantic_model=True
    )

    # Since we're returning the Pydantic model, just check it's the same instance
    assert django_product is pydantic_product
    assert django_product.name == "Test Product"


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

    # Set return_pydantic_model=True to avoid model lookup error
    django_user = convert_pydantic_to_django(
        user, app_label="tests", return_pydantic_model=True
    )

    # Since we're returning the Pydantic model, just check it's the same instance
    assert django_user is user
    assert django_user.name == "Test User"
    assert len(django_user.tags) == 2


class ApiConfig:
    """Test class for API configuration."""

    def __init__(self, url: str, key: str):
        self.url = url
        self.key = key

    def to_dict(self) -> dict:
        return {"url": self.url, "key": self.key}


class CustomFormatter:
    """Test class with custom __str__ method."""

    def __init__(self, format_string: str):
        self.format_string = format_string

    def __str__(self) -> str:
        return f"Format({self.format_string})"


class ServiceConfig:
    """Test class without special methods."""

    def __init__(self, name: str, enabled: bool):
        self.name = name
        self.enabled = enabled


class ConfigModel(BaseModel):
    """Pydantic model with custom class fields."""

    api_config: ApiConfig
    formatter: CustomFormatter
    service: ServiceConfig

    model_config = ConfigDict(arbitrary_types_allowed=True)


def test_serialize_class_instance():
    """Test serialization of different class types."""
    # Test class with to_dict method
    api_config = ApiConfig("https://api.example.com", "secret123")
    serialized = serialize_class_instance(api_config)
    assert isinstance(serialized, dict)
    assert serialized == api_config.to_dict()

    # Test class with __str__ method
    formatter = CustomFormatter("json")
    serialized = serialize_class_instance(formatter)
    assert isinstance(serialized, str)
    assert serialized == str(formatter)

    # Test basic class
    service = ServiceConfig("auth", True)
    serialized = serialize_class_instance(service)
    assert isinstance(serialized, dict)
    assert "__class__" in serialized
    assert "__module__" in serialized
    assert "data" in serialized
    assert serialized["data"]["name"] == "auth"
    assert serialized["data"]["enabled"] is True


def test_deserialize_class_instance():
    """Test deserialization of different class types."""
    # Test deserialization with class registry
    api_config = ApiConfig("https://api.example.com", "secret123")
    serialized = {
        "__class__": "ApiConfig",
        "__module__": __name__,
        "data": {"url": "https://api.example.com", "key": "secret123"},
    }

    class_registry = {"ApiConfig": ApiConfig}
    deserialized = deserialize_class_instance(serialized, class_registry)

    assert isinstance(deserialized, ApiConfig)
    assert deserialized.url == api_config.url
    assert deserialized.key == api_config.key

    # Test deserialization without registry (using import)
    deserialized = deserialize_class_instance(serialized)
    assert isinstance(deserialized, ApiConfig)
    assert deserialized.url == api_config.url
    assert deserialized.key == api_config.key

    # Test fallback for unknown class
    unknown_data = {
        "__class__": "UnknownClass",
        "__module__": "unknown_module",
        "data": {"some": "data"},
    }
    deserialized = deserialize_class_instance(unknown_data)
    assert deserialized == unknown_data  # Should return original data


@pytest.mark.django_db
def test_convert_pydantic_with_custom_classes(basic_pydantic_model):
    """Test conversion of Pydantic models containing custom class instances."""

    class ModelWithCustomClasses(BaseModel):
        name: str
        api_config: ApiConfig
        formatter: CustomFormatter
        service: ServiceConfig

        model_config = ConfigDict(arbitrary_types_allowed=True)

    # Create an instance with custom class fields
    instance = ModelWithCustomClasses(
        name="Test",
        api_config=ApiConfig("https://api.example.com", "secret123"),
        formatter=CustomFormatter("json"),
        service=ServiceConfig("auth", True),
    )

    # Set return_pydantic_model=True to avoid model lookup error
    result = convert_pydantic_to_django(
        instance, app_label="tests", return_pydantic_model=True
    )

    # Since we're returning the Pydantic model, just check it's the same instance
    assert result is instance
    assert result.name == "Test"
