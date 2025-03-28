"""
Tests for the StaticDjangoModelGenerator class.

These tests verify:
1. Basic initialization and configuration
2. Template availability
3. Model generation for different types of Pydantic models:
   - Simple models with basic field types
   - Models with relationships
   - Models requiring context for complex fields
4. Proper handling of class object references and module paths
"""
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Callable
from datetime import datetime

import pytest
from pydantic import BaseModel
from pydantic2django.static_django_model_generator import StaticDjangoModelGenerator
from tests.mock_discovery import register_model, clear, get_model_has_context, get_django_models, MockDiscovery
from django.db import models


# Mock ContextRegistry for tests
class ContextRegistry:
    """Mock ContextRegistry class for testing."""

    _contexts = {}

    @classmethod
    def clear(cls):
        cls._contexts = {}

    @classmethod
    def register_context(cls, name, context):
        cls._contexts[name] = context

    @classmethod
    def get_context(cls, name):
        return cls._contexts.get(name)


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
                    "string_field = models.TextField",
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

    # Use the mock ContextRegistry defined in this file
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
    assert f"class Django{test_params.model_name}(Pydantic2DjangoBaseClass):" in content
    for field in test_params.expected_fields:
        assert field in content, f"Expected field '{field}' not found"


@pytest.mark.parametrize(
    "test_params",
    [
        pytest.param(
            ModelGenerationTestParams(
                model_name="User",
                expected_fields=[],
                expected_relationships=[
                    "address = models.ForeignKey",
                    "profile = models.ForeignKey",
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

    # Use the mock ContextRegistry defined in this file
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
    assert f"class Django{test_params.model_name}(Pydantic2DjangoBaseClass):" in content
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
                    "value = models.IntegerField",
                ],
                expected_context_fields=[
                    'field_name="handler"',
                    'field_name="processor"',
                    'field_name="unserializable"',
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
    from pydantic2django.context_storage import ModelContext

    # Import Django models
    from django.db import models

    # Since we don't have access to create_context_for_model and ContextRegistry,
    # we'll use our own minimal versions for testing
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
    assert f"class Django{test_params.model_name}(Pydantic2DjangoBaseClass):" in content
    for field in test_params.expected_fields:
        assert field in content, f"Expected field '{field}' not found"

    # Check for context class and fields
    assert f"class Django{test_params.model_name}Context(ModelContext):" in content
    for context_field in test_params.expected_context_fields:
        assert context_field in content, f"Expected context field '{context_field}' not found"

    # Verify model is listed in __all__ with its context
    assert f"'Django{test_params.model_name}'" in content
    assert f"'Django{test_params.model_name}Context'" in content


# Helper function for context model test
def create_context_for_model(django_model, pydantic_model):
    """Create a minimal context for testing purposes"""
    from pydantic2django.context_storage import ModelContext

    context = ModelContext(
        django_model=django_model,
        pydantic_class=pydantic_model,
    )

    # Add fields that need context
    for field_name, field_info in pydantic_model.model_fields.items():
        if field_name in ["handler", "processor", "unserializable"]:
            context.add_field(field_name=field_name, field_type=field_info.annotation, is_optional=False)

    return context


# Define Pydantic model classes with problematic field types for testing
class ChainNode:
    """Mock representation of a ChainNode class for testing."""

    pass


class ConversationNode:
    """Mock representation of a ConversationNode class for testing."""

    pass


class ComplexFieldPydanticModel(BaseModel):
    """Pydantic model with fields using complex class references."""

    name: str
    source: ChainNode
    target: ConversationNode

    model_config = {"arbitrary_types_allowed": True}


@pytest.fixture
def complex_field_pydantic_model():
    """Fixture to provide a Pydantic model with complex field types."""
    return ComplexFieldPydanticModel


class TestClassReferenceHandling:
    """Tests specifically for handling class reference issues in the generator."""

    def test_class_reference_field_types(self, tmp_path, complex_field_pydantic_model, caplog, monkeypatch):
        """
        Test that the generator properly handles class references in field types.

        This test verifies our fixes for the angle bracket issue where class object references
        would appear as <class 'module.path.ClassName'> in the output.
        """
        caplog.set_level(logging.INFO)
        from django.db import models
        from pydantic2django.context_storage import ModelContext

        # Setup
        clear()
        output_path = tmp_path / "class_ref_models.py"
        register_model("ComplexFieldModel", complex_field_pydantic_model, has_context=True)

        # Create a mock Django model
        class MockDjangoModel(models.Model):
            name = models.CharField(max_length=255)

            class Meta:
                app_label = "test_app"

        # Create context for the model
        context = ModelContext(
            django_model=MockDjangoModel,
            pydantic_class=complex_field_pydantic_model,
        )

        # Add the fields that would become context fields
        context.add_field(field_name="source", field_type=ChainNode, is_optional=False)
        context.add_field(field_name="target", field_type=ConversationNode, is_optional=False)

        # Create and run generator
        generator = StaticDjangoModelGenerator(
            output_path=str(output_path),
            packages=["tests"],
            app_label="test_app",
            verbose=True,
            discovery_module=MockDiscovery(),
        )

        # Mock the discovery contexts attribute
        monkeypatch.setattr(generator.discovery, "contexts", {"ComplexFieldModel": context})

        generator.generate()

        # Verify file was created
        assert output_path.exists()

        # Read and verify content
        content = output_path.read_text()

        # 1. Check that class references are properly quoted in context classes
        assert 'field_type="ChainNode"' in content, "Class reference not properly quoted in template"
        assert 'field_type="ConversationNode"' in content, "Class reference not properly quoted in template"

        # 2. Check that no angle bracket notation appears in the output
        assert "<class '" not in content, "Angle bracket class notation found in output"

        # 3. Check that typing is properly imported when needed
        if "typing.Dict" in content or "typing.List" in content:
            assert "import typing" in content, "Typing module not imported when needed"


class TestCategorizedImports:
    """Tests for proper import categorization."""

    def test_import_categorization(self, tmp_path, caplog, monkeypatch):
        """
        Test that imports are properly categorized in the generated model file.

        This test verifies our fixes for the issue where all imports were added to
        the typing section regardless of their source module.
        """
        caplog.set_level(logging.INFO)
        from django.db import models
        from pydantic2django.context_storage import ModelContext

        # Create mock class objects for import testing
        class CustomAgent:
            """Mock agent class."""

            pass

        class AgentPool:
            """Mock agent pool class."""

            pass

        # Create a Pydantic model with fields requiring module imports
        class ImportTestModel(BaseModel):
            """Model with fields that will generate module imports."""

            name: str
            agent: CustomAgent
            pool: Optional[AgentPool]
            handler: Callable[[Dict[str, Any]], List[str]]

            model_config = {"arbitrary_types_allowed": True}

        # Setup
        clear()
        output_path = tmp_path / "import_test_models.py"
        register_model("ImportTestModel", ImportTestModel, has_context=True)

        # Create a mock Django model
        class MockDjangoModel(models.Model):
            name = models.CharField(max_length=255)

            class Meta:
                app_label = "test_app"

        # Create context for the model
        context = ModelContext(
            django_model=MockDjangoModel,
            pydantic_class=ImportTestModel,
        )

        # Add fields with various import requirements
        context.add_field(field_name="agent", field_type=CustomAgent, is_optional=False)
        context.add_field(field_name="pool", field_type=Optional[AgentPool], is_optional=True)
        context.add_field(field_name="handler", field_type=Callable[[Dict[str, Any]], List[str]], is_optional=False)

        # Create and run generator
        generator = StaticDjangoModelGenerator(
            output_path=str(output_path),
            packages=["tests"],
            app_label="test_app",
            verbose=True,
            discovery_module=MockDiscovery(),
        )

        # Mock the discovery contexts attribute
        monkeypatch.setattr(generator.discovery, "contexts", {"ImportTestModel": context})

        generator.generate()

        # Verify file was created
        assert output_path.exists()

        # Read content
        content = output_path.read_text()

        # Parse and check the import sections
        content_lines = content.split("\n")

        # Find the import sections
        typing_import_section = []
        custom_import_section = []

        in_typing_section = False
        in_custom_section = False

        for line in content_lines:
            if "# Additional type imports" in line:
                in_typing_section = True
                in_custom_section = False
                continue
            elif "# Original Pydantic model imports" in line or "# Context class field type imports" in line:
                in_typing_section = False
                in_custom_section = True
                continue
            elif line.strip() == "" or line.startswith("#"):
                continue

            if in_typing_section:
                typing_import_section.append(line)
            elif in_custom_section:
                custom_import_section.append(line)

        # Analyze import sections
        typing_imports = " ".join(typing_import_section)
        custom_imports = " ".join(custom_import_section)

        # 1. Verify that typing constructs are imported from typing
        assert "Callable" in typing_imports, "Callable should be imported from typing"
        assert "Dict" in typing_imports, "Dict should be imported from typing"
        assert "List" in typing_imports, "List should be imported from typing"
        assert "Optional" in typing_imports, "Optional should be imported from typing"

        # 2. Verify that custom types are in the correct section and not in typing imports
        assert "CustomAgent" not in typing_imports, "CustomAgent should not be in typing imports"
        assert "AgentPool" not in typing_imports, "AgentPool should not be in typing imports"

        # 3. Verify typing module is imported when needed for dotted notation
        if "typing.Dict" in content or "typing.List" in content:
            assert "import typing" in content, "Typing module not imported when needed"
