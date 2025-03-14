"""
Tests for the StaticDjangoModelGenerator class.
"""
import os
from dataclasses import dataclass
from typing import Any, Optional

import pytest
from pydantic2django.static_django_model_generator import StaticDjangoModelGenerator


@dataclass
class GeneratorParams:
    """Parameters for StaticDjangoModelGenerator initialization."""

    output_path: str
    packages: list[str]
    app_label: str
    filter_function: Optional[Any]
    verbose: bool


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
            id="default_params",
        ),
        pytest.param(
            GeneratorParams(
                output_path="custom/path/models.py",
                packages=["custom_package"],
                app_label="custom_app",
                filter_function=None,
                verbose=True,
            ),
            id="custom_params",
        ),
    ],
)
def test_generator_params(tmp_path, params: GeneratorParams):
    """
    Test StaticDjangoModelGenerator initialization with different parameter sets.
    """
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
        "model_definition.py.j2",
        "context_class.py.j2",
        "models_file.py.j2",
    ],
)
def test_jinja_templates(template_name: str):
    """
    Test that required Jinja2 templates are available.
    """
    generator = StaticDjangoModelGenerator()
    assert template_name in generator.jinja_env.list_templates()


def test_simple_model_generation(tmp_path, basic_pydantic_model):
    """
    Test generation of a simple Django model from a basic Pydantic model.
    """
    from tests.mock_discovery import register_model, clear

    # Setup
    clear()
    output_path = tmp_path / "generated_models.py"
    register_model("BasicModel", basic_pydantic_model)

    # Create and run generator
    generator = StaticDjangoModelGenerator(
        output_path=str(output_path),
        packages=["tests"],
        app_label="test_app",
    )

    generator.generate_models()

    # Verify file was created
    assert output_path.exists()

    # Read and verify content
    content = output_path.read_text()
    assert "class BasicModel(models.Model):" in content
    assert "string_field = models.CharField" in content
    assert "int_field = models.IntegerField" in content
    assert "float_field = models.FloatField" in content
    assert "bool_field = models.BooleanField" in content
    assert "decimal_field = models.DecimalField" in content
    assert "email_field = models.EmailField" in content


def test_relationship_model_generation(tmp_path, relationship_models):
    """
    Test generation of Django models with relationships from related Pydantic models.
    """
    from tests.mock_discovery import register_model, clear

    # Setup
    clear()
    output_path = tmp_path / "generated_models.py"

    # Register all models
    for name, model in relationship_models.items():
        register_model(name, model)

    # Create and run generator
    generator = StaticDjangoModelGenerator(
        output_path=str(output_path),
        packages=["tests"],
        app_label="test_app",
    )

    generator.generate_models()

    # Verify file was created
    assert output_path.exists()

    # Read and verify content
    content = output_path.read_text()

    # Check for model definitions
    assert "class Address(models.Model):" in content
    assert "class Profile(models.Model):" in content
    assert "class Tag(models.Model):" in content
    assert "class User(models.Model):" in content

    # Check for relationships
    assert "address = models.ForeignKey" in content
    assert "profile = models.OneToOneField" in content
    assert "tags = models.ManyToManyField" in content


def test_context_model_generation(tmp_path, context_pydantic_model):
    """
    Test generation of Django models with context handling for complex fields.
    """
    from tests.mock_discovery import register_model, clear

    # Setup
    clear()
    output_path = tmp_path / "generated_models.py"
    register_model("ContextTestModel", context_pydantic_model)

    # Create and run generator
    generator = StaticDjangoModelGenerator(
        output_path=str(output_path),
        packages=["tests"],
        app_label="test_app",
        verbose=True,
    )

    generator.generate_models()

    # Verify file was created
    assert output_path.exists()

    # Read and verify content
    content = output_path.read_text()

    # Check for model definition
    assert "class ContextTestModel(models.Model):" in content

    # Check for regular fields
    assert "name = models.CharField" in content
    assert "value = models.IntegerField" in content

    # Check for context class generation
    assert "class ContextTestModelContext:" in content
    assert "handler: ComplexHandler" in content
    assert "processor: Callable" in content
    assert "unserializable: UnserializableType" in content
