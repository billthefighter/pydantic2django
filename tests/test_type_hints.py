"""Tests for type hinting and IDE completion functionality."""
from typing import Type, cast
import pytest
import logging
from django.db import models, connection
from django.apps import apps
from django.db.migrations.executor import MigrationExecutor
from pydantic import BaseModel, ConfigDict

from pydantic2django import DjangoModelFactory, discovery
from pydantic2django.types import DjangoBaseModel
from pydantic2django.registry import normalize_model_name

logger = logging.getLogger(__name__)


class Configuration:
    """Example non-Pydantic class for testing."""

    def __init__(self, api_key: str, timeout: int):
        self.api_key = api_key
        self.timeout = timeout

    def to_dict(self) -> dict:
        return {"api_key": self.api_key, "timeout": self.timeout}

    def __str__(self) -> str:
        return f"Config(api_key={self.api_key}, timeout={self.timeout})"


class UserModelWithConfig(BaseModel):
    """Pydantic model with a non-Pydantic class field."""

    name: str
    age: int
    config: Configuration

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_display_name(self) -> str:
        return f"{self.name} ({self.age})"

    def get_config_timeout(self) -> int:
        return self.config.timeout


class UserModel(BaseModel):
    """Base Pydantic model for testing."""

    name: str
    age: int

    def get_display_name(self) -> str:
        return f"{self.name} ({self.age})"

    def calculate_birth_year(self, current_year: int) -> int:
        return current_year - self.age

    @property
    def is_adult(self) -> bool:
        return self.age >= 18

    @classmethod
    def create_default(cls) -> "UserModel":
        return cls(name="Default", age=25)


@pytest.fixture
def typed_pydantic_model() -> Type[UserModel]:
    """Fixture providing a Pydantic model with type-annotated methods."""
    return UserModel


@pytest.fixture
def typed_model_with_config() -> Type[UserModelWithConfig]:
    """Fixture providing a Pydantic model with a non-Pydantic class field."""
    return UserModelWithConfig


def test_factory_type_preservation(typed_pydantic_model: Type[UserModel]):
    """Test that the factory preserves type information."""
    # Register the model first
    registry = discovery.get_registry()
    registry.discovered_models["UserModel"] = typed_pydantic_model
    django_models = discovery.setup_dynamic_models(app_label="tests")

    UserDjango, _ = DjangoModelFactory[typed_pydantic_model].create_model(
        typed_pydantic_model, app_label="tests"
    )

    # Verify the model is properly typed
    assert issubclass(UserDjango, DjangoBaseModel)
    assert UserDjango._pydantic_model == typed_pydantic_model

    # Create an instance and verify method types
    user = UserDjango(name="Test", age=30)

    # Method return types should match Pydantic model
    display_name = user.get_display_name()
    assert isinstance(display_name, str)

    birth_year = user.calculate_birth_year(2024)
    assert isinstance(birth_year, int)

    # Property types should be preserved
    assert isinstance(user.is_adult, bool)


def test_discovery_type_preservation(typed_pydantic_model: Type[UserModel]):
    """Test that discovery mechanism preserves type information."""
    # Register the model
    registry = discovery.get_registry()
    registry.discovered_models["UserModel"] = typed_pydantic_model

    # Set up models
    django_models = discovery.setup_dynamic_models(app_label="tests")

    # Get model with type hints
    UserDjango = discovery.get_django_model(typed_pydantic_model)

    # Verify type information is preserved
    user = UserDjango(name="Test", age=30)

    # These assertions verify both runtime behavior and type checking
    display_name: str = user.get_display_name()  # Type checker should accept this
    birth_year: int = user.calculate_birth_year(2024)  # Type checker should accept this
    is_adult: bool = user.is_adult  # Type checker should accept this


@pytest.mark.django_db(transaction=True)
def test_conversion_type_preservation(typed_pydantic_model: Type[UserModel], db):
    """Test type preservation during model conversion."""
    # Register the model first
    registry = discovery.get_registry()
    model_name = normalize_model_name(typed_pydantic_model.__name__)
    logger.info(f"Registering model {typed_pydantic_model.__name__} as {model_name}")
    registry.discovered_models[model_name] = typed_pydantic_model

    logger.info("Registry contents before setup:")
    for name, model in registry.discovered_models.items():
        logger.info(f"  - {name}: {model}")

    django_models = discovery.setup_dynamic_models(app_label="tests")

    logger.info("Django models after setup:")
    for name, model in django_models.items():
        logger.info(f"  - {name}: {model}")

    logger.info("Registered models in Django apps:")
    try:
        app_config = apps.get_app_config("tests")
        if app_config:
            for model in app_config.get_models():
                logger.info(f"  - {model._meta.model_name}: {model}")
    except Exception as e:
        logger.error(f"Error getting app models: {e}")

    UserDjango = discovery.get_django_model(typed_pydantic_model)
    logger.info(f"Got Django model: {UserDjango}")

    # Disable foreign key checks before schema changes
    with connection.constraint_checks_disabled():
        # Create the table for our dynamic model
        with connection.schema_editor() as schema_editor:
            schema_editor.create_model(UserDjango)
            logger.info(f"Created table for model: {UserDjango._meta.db_table}")

        # Create Pydantic instance
        pydantic_user = typed_pydantic_model(name="Test", age=30)

        # Convert to Django
        django_user = UserDjango.from_pydantic(pydantic_user)
        assert isinstance(django_user, UserDjango)

        # Convert back to Pydantic
        converted_user = django_user.to_pydantic()
        assert isinstance(converted_user, typed_pydantic_model)

        # Verify method types are preserved in both directions
        assert django_user.get_display_name() == pydantic_user.get_display_name()
        assert django_user.is_adult == pydantic_user.is_adult

        # Clean up - drop the table
        with connection.schema_editor() as schema_editor:
            schema_editor.delete_model(UserDjango)


def test_type_checking_errors(typed_pydantic_model: Type[UserModel]):
    """
    Test that type errors are caught by static type checkers.

    Note: These tests are meant to fail type checking but pass runtime.
    They help verify that mypy/pyright are correctly checking types.
    """

    class WrongModel(BaseModel):
        different_field: str

    def requires_user_model(model: Type[DjangoBaseModel[UserModel]]):
        pass

    # Register the model first
    registry = discovery.get_registry()
    registry.discovered_models["UserModel"] = typed_pydantic_model
    django_models = discovery.setup_dynamic_models(app_label="tests")

    UserDjango, _ = DjangoModelFactory[typed_pydantic_model].create_model(
        typed_pydantic_model, app_label="tests"
    )

    # The following lines should raise type errors in your IDE/type checker
    # but will pass at runtime. Uncomment to test type checking.

    # These should fail type checking:
    # requires_user_model(DjangoModelFactory[WrongModel])  # Wrong model type
    # user = UserDjango(name=42)  # Wrong type for name
    # user = UserDjango(invalid_field="test")  # Non-existent field
    # birth_year: str = user.calculate_birth_year(2024)  # Wrong return type annotation


@pytest.mark.mypy_testing
def test_mypy_compatibility(typed_pydantic_model: Type[UserModel]):
    """
    Test that mypy can properly type check our models.

    Note: This test is meant to be run with mypy and may not actually execute.
    It serves as a typing specification test.
    """
    # Register the model first
    registry = discovery.get_registry()
    registry.discovered_models["UserModel"] = typed_pydantic_model
    django_models = discovery.setup_dynamic_models(app_label="tests")

    UserDjango = discovery.get_django_model(typed_pydantic_model)

    def process_user(user: DjangoBaseModel[UserModel]) -> str:
        return user.get_display_name()

    user = UserDjango(name="Test", age=30)
    result: str = process_user(user)  # Should type check correctly

    # The following should raise type errors in mypy:
    # result: int = process_user(user)  # Wrong return type annotation
    # process_user("not a user")  # Wrong argument type


def test_non_pydantic_class_type_preservation(
    typed_model_with_config: Type[UserModelWithConfig],
):
    """Test that type information is preserved for non-Pydantic class fields."""
    # Register the model first
    registry = discovery.get_registry()
    registry.discovered_models["UserModelWithConfig"] = typed_model_with_config
    django_models = discovery.setup_dynamic_models(app_label="tests")

    UserDjango = discovery.get_django_model(typed_model_with_config)

    # Create test data
    config = Configuration(api_key="test123", timeout=30)
    user = UserDjango(name="Test", age=30, config=config.to_dict())

    # Verify the config field is stored as JSON
    assert isinstance(user._meta.get_field("config"), models.JSONField)

    # Verify the stored data matches the original
    assert user.config == config.to_dict()

    # Type checker should accept these
    name: str = user.get_display_name()
    assert isinstance(name, str)

    # The config field should be accessible as a dict
    assert isinstance(user.config, dict)
    assert user.config["timeout"] == 30
    assert user.config["api_key"] == "test123"
