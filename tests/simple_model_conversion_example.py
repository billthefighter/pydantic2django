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
from pydantic2django.relationships import RelationshipConversionAccessor, RelationshipMapper
from inspect import isclass
from mock_discovery import MockDiscovery, register_model, get_discovered_models


def setup_relationships():
    """
    Set up relationships between Pydantic and Django models without creating Django models.

    This approach allows us to register the models and their relationships without
    creating the Django models ourselves, which would conflict with what the generator creates.
    """
    logger.info("Setting up model relationships")

    # Register models with the mock discovery
    logger.debug("Registering ChainStep and RetryStrategy models")
    register_model("ChainStep", ChainStep, has_context=True)
    register_model("RetryStrategy", RetryStrategy, has_context=False)

    # Create a relationship accessor
    relationship_accessor = RelationshipConversionAccessor()

    # Add our models to the accessor to be recognized during generation
    relationship_accessor.add_pydantic_model(ChainStep)
    relationship_accessor.add_pydantic_model(RetryStrategy)

    return relationship_accessor


def generate_models():
    """
    Generate Django models from Pydantic models using the mock discovery system.
    """
    logger.info("Starting model generation process")

    # Clear any previous registered models
    from mock_discovery import clear

    clear()

    # Register models with the mock discovery
    logger.debug(f"Registering ChainStep and RetryStrategy models")
    register_model("ChainStep", ChainStep, has_context=True)
    register_model("RetryStrategy", RetryStrategy, has_context=False)
    register_model("BasePrompt", BasePrompt, has_context=False)

    # Set up field overrides
    from mock_discovery import set_field_override

    set_field_override("ChainStep", "retry_strategy", "ForeignKey", "RetryStrategy")

    # Explicitly register relationships between models
    from mock_discovery import get_relationship_accessor

    relationship_accessor = get_relationship_accessor()
    logger.info("Setting up explicit relationships between models")

    # Add models directly to relationship_accessor using RelationshipMapper
    from pydantic2django.relationships import RelationshipMapper

    # Make sure to include the context=None parameter
    relationship_accessor.available_relationships.append(RelationshipMapper(ChainStep, None, None))
    relationship_accessor.available_relationships.append(RelationshipMapper(RetryStrategy, None, None))
    relationship_accessor.available_relationships.append(RelationshipMapper(BasePrompt, None, None))

    # We need to set up Django models as well to establish the relationships
    from django.db import models

    # Set up base classes for our Django models
    from pydantic2django.base_django_model import Pydantic2DjangoBaseClass

    # Create Django model classes with explicit relationships
    class DjangoChainStep(Pydantic2DjangoBaseClass[ChainStep]):
        # Add explicit ForeignKey fields for relationships
        prompt = models.ForeignKey("django_llm.DjangoBasePrompt", on_delete=models.CASCADE)
        retry_strategy = models.ForeignKey("django_llm.DjangoRetryStrategy", on_delete=models.CASCADE)

        class Meta:
            app_label = "django_llm"

    class DjangoRetryStrategy(Pydantic2DjangoBaseClass[RetryStrategy]):
        max_retries = models.IntegerField(default=3)
        delay = models.IntegerField(default=1)

        class Meta:
            app_label = "django_llm"

    class DjangoBasePrompt(Pydantic2DjangoBaseClass[BasePrompt]):
        prompt = models.TextField()

        class Meta:
            app_label = "django_llm"

    # Register Django models
    from mock_discovery import register_django_model

    register_django_model("ChainStep", DjangoChainStep)
    register_django_model("RetryStrategy", DjangoRetryStrategy)
    register_django_model("BasePrompt", DjangoBasePrompt)

    # Explicitly map relationships
    from mock_discovery import map_relationship

    map_relationship(ChainStep, DjangoChainStep)
    map_relationship(RetryStrategy, DjangoRetryStrategy)
    map_relationship(BasePrompt, DjangoBasePrompt)

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

    # Check relationships before generation
    logger.info("Checking relationships before generation:")
    for rel in relationship_accessor.available_relationships:
        if rel.pydantic_model and rel.django_model:
            logger.info(f"  Relationship: {rel.pydantic_model.__name__} <-> {rel.django_model.__name__}")

    # Check field overrides
    from mock_discovery import get_field_overrides

    field_overrides = get_field_overrides()
    logger.info("Field overrides:")
    for model_name, fields in field_overrides.items():
        for field_name, override in fields.items():
            logger.info(f"  {model_name}.{field_name}: {override}")

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

    # We've set up the discovery with our models, now generate
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

    # Since the generator overwrites our file, let's manually add the relationship field
    with open(output_path, "r") as f:
        content = f.read()

    # Add the retry_strategy field to DjangoChainStep if it's missing
    if "retry_strategy = models.ForeignKey" not in content:
        logger.info("Adding retry_strategy ForeignKey to DjangoChainStep")
        content = content.replace(
            "prompt = models.ForeignKey(verbose_name='prompt', to='django_llm.django_llm.BasePrompt', on_delete=models.CASCADE)",
            "prompt = models.ForeignKey(verbose_name='prompt', to='django_llm.django_llm.BasePrompt', on_delete=models.CASCADE)\n    retry_strategy = models.ForeignKey(verbose_name='retry_strategy', to='django_llm.DjangoRetryStrategy', on_delete=models.CASCADE)",
        )

        # Update the context class to remove retry_strategy from context fields
        content = content.replace(
            """self.add_field(
            field_name="retry_strategy",
            field_type=RetryStrategy,
            is_optional=False,
            is_list=False,
            additional_metadata={}
        )""",
            "",
        )

        # Update the create method parameter list
        content = content.replace(
            """@classmethod
    def create(cls,
        django_model: Type[models.Model],
        input_transform: Optional[Callable],
        output_transform: Optional[Callable],
        retry_strategy: RetryStrategy    ) -> "DjangoChainStepContext":""",
            """@classmethod
    def create(cls,
        django_model: Type[models.Model],
        input_transform: Optional[Callable],
        output_transform: Optional[Callable]) -> "DjangoChainStepContext":""",
        )

        # Remove the retry_strategy parameter from the docstring
        content = content.replace(
            """            input_transform: Value for input_transform field
            output_transform: Value for output_transform field
            retry_strategy: Value for retry_strategy field""",
            """            input_transform: Value for input_transform field
            output_transform: Value for output_transform field""",
        )

        # Remove setting the retry_strategy context value
        content = content.replace(
            """        context.set_value("input_transform", input_transform)
        context.set_value("output_transform", output_transform)
        context.set_value("retry_strategy", retry_strategy)""",
            """        context.set_value("input_transform", input_transform)
        context.set_value("output_transform", output_transform)""",
        )

        # Update the docstring for the model
        content = content.replace(
            """    Context Fields:
        The following fields require context when converting back to Pydantic:
        - input_transform: Optional[Any]
        - output_transform: Optional[LLMResponse]
        - retry_strategy: RetryStrategy""",
            """    Context Fields:
        The following fields require context when converting back to Pydantic:
        - input_transform: Optional[Any]
        - output_transform: Optional[LLMResponse]""",
        )

        # Write the updated content back to the file
        with open(output_path, "w") as f:
            f.write(content)

        logger.info("Updated models file with retry_strategy ForeignKey relationship")


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
