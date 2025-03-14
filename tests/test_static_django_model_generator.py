"""
Tests for the StaticDjangoModelGenerator class.

These tests verify:
1. Basic initialization and configuration
2. Template availability
3. Model generation for different types of Pydantic models:
   - Simple models with basic field types
   - Models with relationships
   - Models requiring context for complex fields
"""
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime

import pytest
from pydantic2django.static_django_model_generator import StaticDjangoModelGenerator
from tests.mock_discovery import register_model, clear, get_model_has_context, get_django_models, MockDiscovery


# Configure logging for tests
logger = logging.getLogger(__name__)


@dataclass
class GeneratorParams:
    """Parameters for StaticDjangoModelGenerator initialization."""

    output_path: str
    packages: list[str]
    app_label: str
    filter_function: Optional[Any]
    verbose: bool


@dataclass
class ModelGenerationTestParams:
    """Parameters for model generation tests."""

    model_name: str
    expected_fields: list[str]
    expected_relationships: list[str] = field(default_factory=list)
    expected_context_fields: list[str] = field(default_factory=list)


@pytest.mark.parametrize(
    "params",
    [
        pytest.param(
            GeneratorParams(
                output_path="generated_models.py",
                packages=["pydantic_models"],
                app_label="django_app",
                filter_function=None,
                verbose=False,
            ),
            id="default_initialization",
        ),
        pytest.param(
            GeneratorParams(
                output_path="custom/path/models.py",
                packages=["custom_package"],
                app_label="custom_app",
                filter_function=None,
                verbose=True,
            ),
            id="custom_initialization",
        ),
    ],
)
def test_generator_params(tmp_path, params: GeneratorParams):
    """
    Test StaticDjangoModelGenerator initialization with different parameter sets.

    Args:
        tmp_path: Pytest fixture providing temporary directory
        params: Test parameters for generator initialization

    Verifies:
        - Generator is initialized with correct parameters
        - All attributes are set as expected
    """
    logger.info(f"Testing generator initialization with params: {params}")

    # Adjust output path for tmp_path when not using default
    if params.output_path != "generated_models.py":
        params.output_path = os.path.join(tmp_path, params.output_path)

    generator = StaticDjangoModelGenerator(
        output_path=params.output_path,
        packages=params.packages,
        app_label=params.app_label,
        filter_function=params.filter_function,
        verbose=params.verbose,
    )

    assert generator.output_path == params.output_path
    assert generator.packages == params.packages
    assert generator.app_label == params.app_label
    assert generator.filter_function == params.filter_function
    assert generator.verbose == params.verbose


@pytest.mark.parametrize(
    "template_name",
    [
        pytest.param("model_definition.py.j2", id="model_template"),
        pytest.param("context_class.py.j2", id="context_template"),
        pytest.param("models_file.py.j2", id="models_file_template"),
    ],
)
def test_jinja_templates(template_name: str):
    """
    Test that required Jinja2 templates are available.

    Args:
        template_name: Name of the template file to check

    Verifies:
        - Template file exists in generator's Jinja environment
    """
    logger.info(f"Checking for template: {template_name}")
    generator = StaticDjangoModelGenerator()
    assert template_name in generator.jinja_env.list_templates()


@pytest.mark.parametrize(
    "test_params",
    [
        pytest.param(
            ModelGenerationTestParams(
                model_name="BasicModel",
                expected_fields=[
                    "string_field = models.CharField",
                    "int_field = models.IntegerField",
                    "float_field = models.FloatField",
                    "bool_field = models.BooleanField",
                    "decimal_field = models.DecimalField",
                    "email_field = models.EmailField",
                ],
            ),
            id="basic_model",
        )
    ],
)
def test_simple_model_generation(tmp_path, basic_pydantic_model, test_params: ModelGenerationTestParams, caplog):
    """
    Test generation of a simple Django model from a basic Pydantic model.

    Args:
        tmp_path: Pytest fixture providing temporary directory
        basic_pydantic_model: Fixture providing a basic Pydantic model
        test_params: Test parameters defining expected model structure
        caplog: Pytest fixture for capturing log output

    Verifies:
        - Model file is generated
        - Model class is created with correct name
        - All expected fields are present with correct Django field types
    """
    caplog.set_level(logging.INFO)
    from tests.mock_discovery import register_model, clear, get_model_has_context, get_django_models
    from pydantic2django.context_storage import ContextRegistry
    from django.db import models

    logger.info(f"Testing simple model generation with params: {test_params}")

    # Setup
    clear()
    output_path = tmp_path / "generated_models.py"
    register_model(test_params.model_name, basic_pydantic_model)

    # Create and run generator
    generator = StaticDjangoModelGenerator(
        output_path=str(output_path),
        packages=["tests"],
        app_label="test_app",
        discovery_module=MockDiscovery(),
    )

    generator.generate()

    # Verify file was created
    assert output_path.exists()

    # Read and verify content
    content = output_path.read_text()
    logger.info(f"Generated model content length: {len(content)}")

    # Check for model definition and fields
    assert f"class {test_params.model_name}(models.Model):" in content
    for field in test_params.expected_fields:
        assert field in content, f"Expected field '{field}' not found"


@pytest.mark.parametrize(
    "test_params",
    [
        pytest.param(
            ModelGenerationTestParams(
                model_name="User",
                expected_fields=["name = models.CharField"],
                expected_relationships=[
                    "address = models.ForeignKey",
                    "profile = models.OneToOneField",
                    "tags = models.ManyToManyField",
                ],
            ),
            id="related_models",
        )
    ],
)
def test_relationship_model_generation(tmp_path, relationship_models, test_params: ModelGenerationTestParams, caplog):
    """
    Test generation of Django models with relationships from related Pydantic models.

    Args:
        tmp_path: Pytest fixture providing temporary directory
        relationship_models: Fixture providing related Pydantic models
        test_params: Test parameters defining expected model structure
        caplog: Pytest fixture for capturing log output

    Verifies:
        - Model file is generated
        - All related models are created
        - Relationships are properly defined with correct field types
    """
    caplog.set_level(logging.INFO)
    from tests.mock_discovery import register_model, clear, get_model_has_context, get_django_models
    from pydantic2django.context_storage import ContextRegistry
    from django.db import models

    logger.info(f"Testing relationship model generation with params: {test_params}")

    # Setup
    clear()
    output_path = tmp_path / "generated_models.py"

    # Register all models
    for name, model in relationship_models.items():
        register_model(name, model)
        logger.info(f"Registered model: {name}")

    # Create and run generator
    generator = StaticDjangoModelGenerator(
        output_path=str(output_path),
        packages=["tests"],
        app_label="test_app",
        discovery_module=MockDiscovery(),
    )

    generator.generate()

    # Verify file was created
    assert output_path.exists()

    # Read and verify content
    content = output_path.read_text()
    logger.info(f"Generated model content length: {len(content)}")

    # Check for model definition and fields
    assert f"class {test_params.model_name}(models.Model):" in content
    for field in test_params.expected_fields:
        assert field in content, f"Expected field '{field}' not found"

    # Check for relationships
    for relationship in test_params.expected_relationships:
        assert relationship in content, f"Expected relationship '{relationship}' not found"


@pytest.mark.parametrize(
    "test_params",
    [
        pytest.param(
            ModelGenerationTestParams(
                model_name="ContextTestModel",
                expected_fields=[
                    "name = models.CharField",
                    "value = models.IntegerField",
                ],
                expected_context_fields=[
                    "handler: ComplexHandler",
                    "processor: Callable",
                    "unserializable: UnserializableType",
                ],
            ),
            id="context_model",
        )
    ],
)
def test_context_model_generation(tmp_path, context_pydantic_model, test_params: ModelGenerationTestParams, caplog):
    """
    Test generation of Django models with context handling for complex fields.

    Args:
        tmp_path: Pytest fixture providing temporary directory
        context_pydantic_model: Fixture providing a Pydantic model with context fields
        test_params: Test parameters defining expected model structure
        caplog: Pytest fixture for capturing log output

    Verifies:
        - Model file is generated
        - Model class is created with correct fields
        - Context class is generated with all required fields
        - Complex field types are properly handled
    """
    caplog.set_level(logging.INFO)
    from tests.mock_discovery import register_model, clear, get_model_has_context, get_django_models
    from pydantic2django.context_storage import create_context_for_model, ContextRegistry
    from django.db import models

    logger.info(f"Testing context model generation with params: {test_params}")

    # Setup
    clear()
    ContextRegistry.clear()  # Clear any existing context
    output_path = tmp_path / "generated_models.py"
    register_model(test_params.model_name, context_pydantic_model, has_context=True)

    # Create a mock Django model for context creation
    class MockDjangoModel(models.Model):
        class Meta:
            app_label = "test_app"

    # Create context for the model
    context = create_context_for_model(MockDjangoModel, context_pydantic_model)
    ContextRegistry.register_context(test_params.model_name, context)

    # Create and run generator
    generator = StaticDjangoModelGenerator(
        output_path=str(output_path),
        packages=["tests"],
        app_label="test_app",
        verbose=True,
        discovery_module=MockDiscovery(),
    )

    generator.generate()

    # Verify file was created
    assert output_path.exists()

    # Read and verify content
    content = output_path.read_text()
    logger.info(f"Generated model content length: {len(content)}")

    # Check for model definition and fields
    assert f"class {test_params.model_name}(models.Model):" in content
    for field in test_params.expected_fields:
        assert field in content, f"Expected field '{field}' not found"

    # Check for context class and fields
    assert f"class {test_params.model_name}Context:" in content
    for context_field in test_params.expected_context_fields:
        assert context_field in content, f"Expected context field '{context_field}' not found"

    # Verify model is listed in __all__ with its context
    assert f'"{test_params.model_name}"' in content
    assert f'"{test_params.model_name}Context"' in content
