import logging
import os
import sys
from collections.abc import Callable
from enum import Enum
from typing import Any, Generic, Optional, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

# Add the project root to sys.path to make the tests module importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the dummy models for demonstrating import functionality
from tests.dummy_models import ProviderCapabilities, RateLimitConfig  # noqa: E402

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("model_conversion")

# Configure more detailed logging for the import handler
import_handler_logger = logging.getLogger("pydantic2django.import_handler")
import_handler_logger.setLevel(logging.DEBUG)

# Type variable for model classes
T = TypeVar("T")
PromptType = TypeVar("PromptType", bound=Enum)

# Configure Django settings before importing any Django-related modules
import django  # noqa: E402
from django.conf import settings  # noqa: E402

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
from inspect import isclass  # noqa: E402

from pydantic2django import configure_type_handler_logging  # noqa: E402
from pydantic2django.relationships import RelationshipConversionAccessor  # noqa: E402
from pydantic2django.static_django_model_generator import StaticDjangoModelGenerator  # noqa: E402
from tests.mock_discovery import MockDiscovery, register_model  # noqa: E402


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
    input_transform: Optional[Callable[[ChainContext, Any], dict[str, Any]]] = None
    output_transform: Optional[Callable[[LLMResponse], T]] = None
    retry_strategy: RetryStrategy = Field(default_factory=RetryStrategy)

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)


class EnumValues(Enum):
    VALUE1 = "value1"
    VALUE2 = "value2"
    VALUE3 = "value3"


class EnumExample(BaseModel):
    enum_field: EnumValues


class Provider(BaseModel):
    """
    Example model using imported types to demonstrate import functionality.
    """

    name: str
    api_base: str
    capabilities: ProviderCapabilities = Field(default_factory=ProviderCapabilities)
    rate_limits: RateLimitConfig = Field(default_factory=RateLimitConfig)


class ComplexTypingExample(BaseModel, Generic[PromptType, T]):
    text: str
    prompt: type[PromptType]
    input_transform: Callable[[ChainContext, Any], dict[str, Any]]
    output_transform: Callable[[LLMResponse], T]
    retry_strategy: RetryStrategy


# +++ Add Generic Example Models +++
NodeT = TypeVar("NodeT")
EdgeT = TypeVar("EdgeT")


class SimpleGenericBase(BaseModel, Generic[NodeT, EdgeT]):
    """A simple generic base model for testing."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    nodes: list[NodeT]  # Use List from typing
    edges: list[EdgeT]  # Use List from typing
    # Add a context field to the base
    base_context_field: Callable[[str], int] = Field(exclude=True)
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ConcreteNode(BaseModel):
    """A concrete node type."""

    node_id: str
    data: str


class ConcreteEdge(BaseModel):
    """A concrete edge type."""

    edge_id: str
    from_node: str
    to_node: str


class SpecificGraph(SimpleGenericBase[ConcreteNode, ConcreteEdge]):
    """A specific graph implementation inheriting from the generic base."""

    graph_name: str
    description: Optional[str] = None
    # Add a specific context field
    specific_context_field: Callable[[int], str] = Field(exclude=True)  # Exclude from Django model
    model_config = ConfigDict(arbitrary_types_allowed=True)


class DoubleRelationModel(BaseModel):
    """A model with two fields pointing to the same related model."""

    related_1: BasePrompt
    related_2: BasePrompt
    description: str


class OptionalRelationExample(BaseModel):
    """Model with an optional relation field that requires context."""

    example_name: str
    # Replace optional_prompt with an optional context field
    optional_capability: Optional[ProviderCapabilities] = Field(default=None, exclude=True)
    # Add a context field to test context generation
    context_data: Callable[[int], int] = Field(exclude=True)
    model_config = ConfigDict(arbitrary_types_allowed=True)


# +++ End Generic Example Models +++


def is_persistent_model(obj: Any) -> bool:
    """
    Filter function that returns True if the object is a ChainStep, RetryStrategy,
    EnumExample, or Provider class.

    Args:
        obj: The object to check

    Returns:
        bool: True if obj is one of the specified model classes, False otherwise
    """
    # return isclass(obj) and (obj is ChainStep or obj is RetryStrategy or obj is EnumExample or obj is Provider)
    # Temporarily include all models for testing generics
    return isclass(obj) and issubclass(obj, BaseModel)


def setup_relationships():
    """
    Set up relationships between Pydantic and Django models without creating Django models.

    This approach allows us to register the models and their relationships without
    creating the Django models ourselves, which would conflict with what the generator creates.
    """
    logger.info("Setting up model relationships")

    # Register models with the mock discovery
    logger.debug("Registering models")
    register_model("ChainStep", ChainStep, has_context=True)
    register_model("RetryStrategy", RetryStrategy, has_context=False)
    register_model("EnumExample", EnumExample, has_context=False)
    register_model("Provider", Provider, has_context=True)

    # Create a relationship accessor
    relationship_accessor = RelationshipConversionAccessor()

    # Add our models to the accessor to be recognized during generation
    relationship_accessor.add_pydantic_model(ChainStep)
    relationship_accessor.add_pydantic_model(RetryStrategy)
    relationship_accessor.add_pydantic_model(EnumExample)
    relationship_accessor.add_pydantic_model(Provider)
    return relationship_accessor


def generate_models():
    """
    Generate Django models from Pydantic models using the mock discovery system.

    Note: This example only works with simple models (BasePrompt, EnumExample, RetryStrategy).
    More complex models like ChainStep and ComplexTypingExample cause errors due to the way
    they use field_type in the context fields. For those models, you'll need to use a different
    approach to handle their relationships.
    """
    logger.info("Starting model generation process")
    configure_type_handler_logging(level=logging.DEBUG)
    # Clear any previous registered models
    from tests.mock_discovery import clear

    clear()

    # Register models with the mock discovery
    logger.debug("Registering models")
    register_model("BasePrompt", BasePrompt, has_context=False)
    register_model("EnumExample", EnumExample, has_context=False)
    register_model("RetryStrategy", RetryStrategy, has_context=False)
    register_model("ComplexTypingExample", ComplexTypingExample, has_context=True)
    register_model("ChainStep", ChainStep, has_context=True)
    register_model("Provider", Provider, has_context=True)
    # Register generic models
    # register_model("SimpleGenericBase", SimpleGenericBase, has_context=True)
    # # Don't register the base generic directly? Let's see.
    register_model("ConcreteNode", ConcreteNode, has_context=False)
    register_model("ConcreteEdge", ConcreteEdge, has_context=False)
    register_model("SpecificGraph", SpecificGraph, has_context=True)  # Register the specific implementation
    register_model("DoubleRelationModel", DoubleRelationModel, has_context=False)  # Register the new model
    register_model("OptionalRelationExample", OptionalRelationExample, has_context=True)

    from tests.mock_discovery import set_field_override

    set_field_override("ChainStep", "retry_strategy", "ForeignKey", "RetryStrategy")

    # Explicitly register relationships between models
    from tests.mock_discovery import get_relationship_accessor

    relationship_accessor = get_relationship_accessor()
    logger.info("Setting up explicit relationships between models")

    # Add models directly to relationship_accessor using RelationshipMapper
    from pydantic2django.relationships import RelationshipMapper

    # Add our models
    relationship_accessor.available_relationships.append(RelationshipMapper(BasePrompt, None, None))
    relationship_accessor.available_relationships.append(RelationshipMapper(EnumExample, None, None))
    relationship_accessor.available_relationships.append(RelationshipMapper(RetryStrategy, None, None))
    relationship_accessor.available_relationships.append(RelationshipMapper(ChainStep, None, None))
    relationship_accessor.available_relationships.append(RelationshipMapper(Provider, None, None))
    relationship_accessor.available_relationships.append(
        RelationshipMapper(DoubleRelationModel, None, None)
    )  # Add new model here
    relationship_accessor.available_relationships.append(RelationshipMapper(OptionalRelationExample, None, None))

    # Set up Django models as well to establish the relationships

    # Set up base classes for our Django models

    # Define module mappings to fix imports
    module_mappings = {
        "__main__": "myapp.models",  # Replace __main__ imports with myapp.models
    }

    generator = StaticDjangoModelGenerator(
        output_path="generated_models.py",
        packages=["tests.django_llm"],  # This might need adjustment if models are defined here
        app_label="django_llm",
        # filter_function=is_persistent_model, # Use the modified filter temporarily
        filter_function=None,  # Remove filter to ensure all registered models are included
        discovery_module=MockDiscovery(),
        module_mappings=module_mappings,
        verbose=True,  # Add verbose logging
    )
    generator.generate()


if __name__ == "__main__":
    generate_models()
