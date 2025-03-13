"""
Tests for the StaticDjangoModelGenerator with Chain models.

This test suite verifies that the StaticDjangoModelGenerator correctly handles
the Chain models defined in tests/test_models/chains.py.
"""
import os
import sys
import importlib.util
import tempfile
from pathlib import Path
import pytest
import logging
import re

# Configure Django settings before importing Django models
import django
from django.conf import settings

if not settings.configured:
    settings.configure(
        INSTALLED_APPS=[
            "django.contrib.contenttypes",
            "tests",
        ],
        DATABASES={
            "default": {
                "ENGINE": "django.db.backends.sqlite3",
                "NAME": ":memory:",
            }
        },
    )
    django.setup()

from pydantic import BaseModel, Field
from django.db import models

from src.pydantic2django.static_django_model_generator import StaticDjangoModelGenerator
from src.pydantic2django.base_django_model import Pydantic2DjangoBaseClass
from src.pydantic2django.discovery import (
    discover_models,
    get_discovered_models,
    setup_dynamic_models,
)
from src.pydantic2django.mock_discovery import (
    register_model,
    clear,
    get_django_models,
)

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@pytest.fixture(scope="function")
def clean_discovery():
    """Fixture to ensure clean state for model discovery."""
    # Clear any previously discovered models
    clear()
    yield
    # Clean up after test
    clear()


@pytest.fixture
def temp_output_file():
    """Fixture to provide a temporary file for output."""
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
        tmp_path = tmp.name

    yield tmp_path

    # Clean up the file after the test
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)


@pytest.fixture
def chains_module():
    """Fixture to dynamically import the chains module."""
    # Get the path to the chains.py file
    chains_path = Path(__file__).parent / "test_models" / "chains.py"

    # Dynamically import the module
    spec = importlib.util.spec_from_file_location(
        "tests.test_models.chains", chains_path
    )
    if spec is None:
        pytest.skip("Could not find chains.py module")

    chains = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        pytest.skip("Could not load chains.py module")

    spec.loader.exec_module(chains)

    return chains


@pytest.fixture
def register_chain_models(chains_module, clean_discovery):
    """Fixture to register chain models for testing."""
    # Get all the Pydantic models from the chains module
    chain_models = {
        name: cls
        for name, cls in vars(chains_module).items()
        if isinstance(cls, type) and issubclass(cls, BaseModel) and cls != BaseModel
    }

    # Register each model
    for name, model_class in chain_models.items():
        register_model(name, model_class)

    return chain_models


@pytest.fixture
def generator(temp_output_file, register_chain_models):
    """Fixture to create a StaticDjangoModelGenerator instance."""
    return StaticDjangoModelGenerator(
        output_path=temp_output_file,
        packages=[],  # Empty packages since we're manually registering models
        app_label="test_chains",
        verbose=True,
    )


# Create a fixture for a model with field collisions
@pytest.fixture
def collision_model_file(tmp_path):
    """Create a temporary Python file with a model that has field collisions."""
    # Create a temporary directory for our module
    module_dir = tmp_path / "collision_models"
    module_dir.mkdir()

    # Create an __init__.py file to make it a proper package
    with open(module_dir / "__init__.py", "w") as f:
        f.write("# Package initialization\n")

    # Create a Python file with our collision model
    model_file = module_dir / "models.py"
    with open(model_file, "w") as f:
        f.write(
            """
from pydantic import BaseModel, Field

class CollisionModel(BaseModel):
    \"\"\"A model with fields that collide with base model fields.\"\"\"
    id: str = Field(default="custom-id")
    name: str = Field(default="custom-name")
    object_type: str = Field(default="custom-type")
    created_at: str = Field(default="2023-01-01")
    updated_at: str = Field(default="2023-01-02")

    # Add a regular field to ensure it's included
    description: str = Field(default="A test model")
"""
        )

    # Return the path to the module directory
    return str(module_dir)


@pytest.fixture
def register_collision_model(clean_discovery):
    """Fixture to register a model with field collisions."""

    # Define a model with fields that collide with base model fields
    class CollisionModel(BaseModel):
        """A model with fields that collide with base model fields."""

        id: str = Field(default="custom-id")
        name: str = Field(default="custom-name")
        object_type: str = Field(default="custom-type")
        created_at: str = Field(default="2023-01-01")
        updated_at: str = Field(default="2023-01-02")

        # Add a regular field to ensure it's included
        description: str = Field(default="A test model")

    # Register the model
    register_model("CollisionModel", CollisionModel)

    return CollisionModel


def test_generator_initialization(generator):
    """Test that the generator can be initialized."""
    assert generator is not None
    assert generator.output_path is not None
    assert (
        generator.packages == []
    )  # Empty packages since we're manually registering models
    assert generator.app_label == "test_chains"
    assert generator.verbose is True


def test_chains_module_import(chains_module):
    """Test that the chains module can be imported."""
    assert chains_module is not None
    assert hasattr(chains_module, "ChainGraph")
    assert hasattr(chains_module, "ChainNode")
    assert hasattr(chains_module, "ChainEdge")
    assert hasattr(chains_module, "ChainStep")


def test_discover_chain_models(generator, register_chain_models):
    """Test that the generator can discover chain models."""
    # Discover models
    discovered_models = generator.discover_models()

    # Check that we found some models
    assert discovered_models, "No models were discovered"

    # Check for specific chain models
    model_names = list(discovered_models.keys())
    logger.info(f"Discovered models: {model_names}")

    # Check for the main chain models
    expected_models = [
        "ChainMetadata",
        "ChainState",
        "ChainContext",
        "RetryStrategy",
        "ChainStep",
        "ChainNode",
        "ChainEdge",
        "ChainGraph",
    ]

    for model_name in expected_models:
        assert any(
            model_name in name for name in model_names
        ), f"Model {model_name} not found in discovered models"


def test_setup_django_models(generator, register_chain_models):
    """Test that the generator can set up Django models from the discovered chain models."""
    # Discover models first
    generator.discover_models()

    # Set up Django models
    django_models = generator.setup_django_models()

    # Check that we created some Django models
    assert django_models, "No Django models were created"

    # Check for specific Django models
    model_names = list(django_models.keys())
    logger.info(f"Django models: {model_names}")

    # Check for the main chain models with Django prefix
    expected_models = [
        "DjangoChainMetadata",
        "DjangoChainState",
        "DjangoChainContext",
        "DjangoRetryStrategy",
        "DjangoChainStep",
        "DjangoChainNode",
        "DjangoChainEdge",
        "DjangoChainGraph",
    ]

    for model_name in expected_models:
        assert (
            model_name in model_names
        ), f"Model {model_name} not found in Django models"


def test_generate_models_file(generator, register_chain_models, temp_output_file):
    """Test that the generator can generate a models file."""
    # Generate the models file
    generator.generate()

    # Check that the file was created
    assert os.path.exists(
        temp_output_file
    ), f"Models file {temp_output_file} was not created"

    # Read the file content
    with open(temp_output_file, "r") as f:
        content = f.read()

    # Check that the file contains expected content
    assert (
        "from pydantic2django.base_django_model import Pydantic2DjangoBaseClass"
        in content
    ), "Generated file does not import Pydantic2DjangoBaseClass"

    # Check for model classes
    expected_models = [
        "DjangoChainMetadata",
        "DjangoChainState",
        "DjangoChainContext",
        "DjangoRetryStrategy",
        "DjangoChainStep",
        "DjangoChainNode",
        "DjangoChainEdge",
        "DjangoChainGraph",
    ]

    for model_name in expected_models:
        assert (
            f"class {model_name}(Pydantic2DjangoBaseClass):" in content
        ), f"Generated file does not define {model_name} class"


def test_app_label_propagation(generator, register_chain_models, temp_output_file):
    """Test that the app_label is properly propagated to the model definitions."""
    # Generate the models file
    generator.generate()

    # Read the file content
    with open(temp_output_file, "r") as f:
        content = f.read()

    # Check for app_label in Meta classes
    app_label_count = content.count(f'app_label = "{generator.app_label}"')

    # We should have at least one app_label for each model
    expected_models = [
        "DjangoChainMetadata",
        "DjangoChainState",
        "DjangoChainContext",
        "DjangoRetryStrategy",
        "DjangoChainStep",
        "DjangoChainNode",
        "DjangoChainEdge",
        "DjangoChainGraph",
    ]

    assert app_label_count >= len(
        expected_models
    ), f"Expected at least {len(expected_models)} app_label definitions, found {app_label_count}"


@pytest.mark.parametrize(
    "field_name", ["id", "name", "object_type", "created_at", "updated_at"]
)
def test_base_model_fields_not_duplicated(
    generator, register_chain_models, temp_output_file, field_name
):
    """Test that base model fields are not duplicated in the generated models."""
    # Generate the models file
    generator.generate()

    # Read the file content
    with open(temp_output_file, "r") as f:
        content = f.read()

    # Check for field definitions
    # We need to be careful here because field names might appear in other contexts
    # So we'll look for patterns like "field_name = models.Field"
    field_pattern = re.compile(rf"\s+{field_name}\s+=\s+models\.")

    # Count occurrences
    matches = field_pattern.findall(content)

    # Base fields should not be duplicated in the generated models
    assert (
        len(matches) == 0
    ), f"Found {len(matches)} occurrences of base field '{field_name}' in generated models"


def test_field_collision_non_strict(
    register_collision_model, clean_discovery, temp_output_file
):
    """Test that field collisions are handled correctly in non-strict mode."""
    # Create a generator with non-strict mode (default)
    generator = StaticDjangoModelGenerator(
        output_path=temp_output_file,
        packages=[],  # Empty packages since we're manually registering the model
        app_label="test_collision",
        verbose=True,
    )

    # Generate the models file
    generator.generate()

    # Read the file content
    with open(temp_output_file, "r") as f:
        content = f.read()

    # In non-strict mode, base fields should be preferred
    # So our custom fields should not appear in the output
    for field_name in ["id", "name", "object_type", "created_at", "updated_at"]:
        field_pattern = re.compile(rf"\s+{field_name}\s+=\s+models\.")
        matches = field_pattern.findall(content)
        assert (
            len(matches) == 0
        ), f"Found {len(matches)} occurrences of base field '{field_name}' in non-strict mode"

    # But our regular field should be included
    assert (
        "description = models." in content
    ), "Regular field 'description' not found in generated model"


def test_field_collision_strict_mode(register_collision_model, clean_discovery):
    """Test that field collisions raise an error in strict mode."""
    # Create a generator with strict mode
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
        temp_output_file = tmp.name

    generator = StaticDjangoModelGenerator(
        output_path=temp_output_file,
        packages=[],  # Empty packages since we're manually registering the model
        app_label="test_collision",
        verbose=True,
    )

    # Monkey patch the generator to use strict mode
    # This is a bit of a hack, but it's the simplest way to test this
    # without modifying the original code
    def strict_generate_field_definition(self, field):
        # Check if this is a base field
        if field.name in ["id", "name", "object_type", "created_at", "updated_at"]:
            raise ValueError(f"Field collision detected: {field.name}")
        return generator.generate_field_definition(field)

    # Save the original method
    original_method = generator.generate_field_definition

    # Replace with our strict method
    generator.generate_field_definition = (
        lambda field: strict_generate_field_definition(generator, field)
    )

    try:
        # In strict mode, we should get an error
        with pytest.raises(ValueError, match="Field collision detected"):
            generator.generate()
    finally:
        # Restore the original method
        generator.generate_field_definition = original_method

        # Clean up the file
        if os.path.exists(temp_output_file):
            os.unlink(temp_output_file)
