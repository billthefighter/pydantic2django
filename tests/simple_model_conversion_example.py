from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Callable, Any, Generic, Dict, TypeVar
from uuid import uuid4
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("model_conversion")

# Type variable for model classes
T = TypeVar("T")


class BasePrompt(BaseModel):
    prompt: str


class ChainContext(BaseModel):
    context: dict[str, Any]


class LLMResponse(BaseModel):
    response: str


class RetryStrategy(BaseModel):
    max_retries: int = 3
    delay: int = 1


class ChainStep(BaseModel, Generic[T]):
    """Represents a single step in a chain."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    prompt: BasePrompt
    input_transform: Optional[Callable[[ChainContext, Any], Dict[str, Any]]] = None
    output_transform: Optional[Callable[[LLMResponse], T]] = None
    retry_strategy: RetryStrategy = Field(default_factory=RetryStrategy)

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)


def is_persistent_model(obj: Any) -> bool:
    """
    Filter function that returns True if the object is a ChainStep or RetryStrategy class.

    Args:
        obj: The object to check

    Returns:
        bool: True if obj is ChainStep or RetryStrategy, False otherwise
    """
    return isclass(obj) and (obj is ChainStep or obj is RetryStrategy)


# Configure Django settings before importing any Django-related modules
import os
import django
from django.conf import settings

# Configure Django settings if not already configured
if not settings.configured:
    settings.configure(
        INSTALLED_APPS=[
            "django.contrib.contenttypes",
            "django.contrib.auth",
        ],
        DATABASES={
            "default": {
                "ENGINE": "django.db.backends.sqlite3",
                "NAME": ":memory:",
            }
        },
        DEFAULT_AUTO_FIELD="django.db.models.BigAutoField",
    )
    django.setup()

# Now import other modules that depend on Django
from pydantic2django.static_django_model_generator import StaticDjangoModelGenerator
from inspect import isclass
from tests.mock_discovery import MockDiscovery, register_model, get_discovered_models


def generate_models():
    """
    Generate Django models from Pydantic models using the mock discovery system.
    """
    logger.info("Starting model generation process")

    # Register models with the mock discovery
    logger.debug(f"Registering ChainStep and RetryStrategy models")
    register_model("ChainStep", ChainStep, has_context=True)
    register_model("RetryStrategy", RetryStrategy, has_context=False)

    # Get and log the registered models
    discovered = get_discovered_models()
    logger.info(f"Registered models: {list(discovered.keys())}")

    # Use a mock discovery instance instead of the real generator
    logger.debug("Creating MockDiscovery instance")
    discovery = MockDiscovery()

    logger.debug("Calling discover_models")
    discovery.discover_models(package_names=[], app_label="django_llm")

    # Check what models were discovered
    logger.info(f"Discovery models after discover_models: {list(discovery.discovered_models.keys())}")

    # Setup the models
    logger.debug("Setting up dynamic models")
    django_models = discovery.setup_dynamic_models(app_label="django_llm")

    logger.info(f"Generated {len(django_models)} Django models: {list(django_models.keys())}")

    # Make sure output directory exists
    output_path = "tests/django_llm/models/models.py"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.debug(f"Created output directory for {output_path}")

    # Use the generator with our mock discovery
    logger.debug("Creating StaticDjangoModelGenerator with discovery_module")
    gen = StaticDjangoModelGenerator(
        output_path=output_path,
        packages=[],
        app_label="django_llm",
        discovery_module=discovery,
    )

    # Check discovery state before generation
    logger.debug(f"Discovery module has models: {discovery.discovered_models}")
    logger.debug(f"Discovery module has django models: {discovery.django_models}")
    logger.debug(f"Discovery module has model_has_context: {discovery.get_model_has_context()}")

    logger.debug("Calling generate() method")
    gen.generate()

    logger.info("Model generation complete")

    # Check if the output file exists and has content
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            content = f.read()
            logger.info(f"Output file size: {len(content)} bytes")
            # Check if any models were actually generated
            if "__all__ = []" in content:
                logger.warning("No models were added to __all__ list in the output file")


if __name__ == "__main__":
    generate_models()

    def test_manual_model_creation():
        """Test manual model creation without importing the generated models."""
        from pydantic2django.base_django_model import Pydantic2DjangoBaseClass
        from pydantic2django.context_storage import ModelContext, FieldContext
        from dataclasses import dataclass, field
        from typing import Type, Optional
        from django.db import models

        logger.info("Testing manual model creation")

        # Create a simple RetryStrategy model manually
        class ManualDjangoRetryStrategy(Pydantic2DjangoBaseClass[RetryStrategy]):
            max_retries = models.IntegerField(verbose_name="max retries", default=3)
            delay = models.IntegerField(verbose_name="delay", default=1)
            pydantic_data = models.TextField(default="{}")

            class Meta(Pydantic2DjangoBaseClass.Meta):
                app_label = "test_app"

            @classmethod
            def from_pydantic(cls, pydantic_instance, **kwargs):
                """Custom from_pydantic implementation that handles the conversion manually."""
                django_instance = cls()
                django_instance.max_retries = pydantic_instance.max_retries
                django_instance.delay = pydantic_instance.delay
                # Save JSON representation for to_pydantic
                django_instance.pydantic_data = pydantic_instance.model_dump_json()
                return django_instance

            def to_pydantic(self, **kwargs):
                """Custom to_pydantic implementation."""
                # In Pydantic v2, we use model_validate_json instead of parse_raw_as
                return RetryStrategy.model_validate_json(self.pydantic_data)

        # Create test objects
        retry = RetryStrategy(max_retries=5, delay=2)
        logger.info(f"Created Pydantic RetryStrategy: {retry}")

        # Test RetryStrategy conversion - simpler case without context
        django_retry = ManualDjangoRetryStrategy.from_pydantic(retry)
        logger.info(f"Created Manual Django RetryStrategy: {django_retry}")

        # Convert back to Pydantic
        recovered_retry = django_retry.to_pydantic()
        logger.info(f"Recovered Pydantic RetryStrategy: {recovered_retry}")

        # Verify the conversion worked
        assert (
            recovered_retry.max_retries == retry.max_retries
        ), f"Expected {retry.max_retries}, got {recovered_retry.max_retries}"
        assert recovered_retry.delay == retry.delay, f"Expected {retry.delay}, got {recovered_retry.delay}"
        logger.info("RetryStrategy conversion test passed")

        logger.info("All conversion tests passed!")
        return True

    try:
        # Run the test
        test_manual_model_creation()
        logger.info("All tests passed!")
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)
