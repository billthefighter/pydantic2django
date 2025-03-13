import os
import sys
import pytest
import tempfile
from typing import List, Optional
import src.pydantic2django.static_django_model_generator
import io
import logging
import importlib

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Configure Django settings before importing Django models
import django
from django.conf import settings

if not settings.configured:
    settings.configure(
        INSTALLED_APPS=[
            "django.contrib.contenttypes",
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
from src.pydantic2django.mock_discovery import (
    register_model,
    register_django_model,
    clear,
    get_django_models,
)


# Define a dummy Pydantic model for testing
class DummyPydanticModel(BaseModel):
    """A dummy Pydantic model for testing."""

    name: str
    description: Optional[str] = None
    count: int = 0
    tags: List[str] = Field(default_factory=list)
    is_active: bool = True


# Define a related model for testing many-to-many relationships
class RelatedModel(BaseModel):
    """A related model for testing many-to-many relationships."""

    name: str
    value: int = 0


# Define a model with a many-to-many relationship
class ModelWithM2M(BaseModel):
    """A model with a many-to-many relationship."""

    name: str
    related_items: List[RelatedModel] = Field(default_factory=list)


# Create a mock Django model that would be generated
class DjangoDummyPydanticModel(models.Model):
    """A mock Django model that would be generated."""

    id = models.UUIDField(primary_key=True)
    name = models.CharField(max_length=255)
    object_type = models.CharField(max_length=100)
    data = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    description = models.TextField(null=True, blank=True)
    count = models.IntegerField(default=0)
    is_active = models.BooleanField(default=True)

    class Meta:
        db_table = "dummy_pydantic_model"
        app_label = "test_app"
        verbose_name = "Dummy Pydantic Model"
        verbose_name_plural = "Dummy Pydantic Models"


# Create a mock Django model for the related model
class DjangoRelatedModel(models.Model):
    """A mock Django model for the related model."""

    id = models.UUIDField(primary_key=True)
    name = models.CharField(max_length=255)
    object_type = models.CharField(max_length=100)
    data = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    value = models.IntegerField(default=0)

    class Meta:
        db_table = "related_model"
        app_label = "test_app"
        verbose_name = "Related Model"
        verbose_name_plural = "Related Models"


# Create a mock Django model for the model with a many-to-many relationship
class DjangoModelWithM2M(models.Model):
    """A mock Django model for the model with a many-to-many relationship."""

    id = models.UUIDField(primary_key=True)
    name = models.CharField(max_length=255)
    object_type = models.CharField(max_length=100)
    data = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    related_items = models.ManyToManyField(DjangoRelatedModel, blank=True)

    class Meta:
        db_table = "model_with_m2m"
        app_label = "test_app"
        verbose_name = "Model With M2M"
        verbose_name_plural = "Models With M2M"


@pytest.fixture
def generator_setup():
    """Set up the test environment and return the generator and output path."""
    # Create a temporary directory for the output file
    temp_dir = tempfile.TemporaryDirectory()
    output_path = os.path.join(temp_dir.name, "generated_models.py")

    # Register our models with the mock discovery module
    clear()
    register_model("DummyPydanticModel", DummyPydanticModel)
    register_django_model("DjangoDummyPydanticModel", DjangoDummyPydanticModel)

    # Register the related models
    register_model("RelatedModel", RelatedModel)
    register_django_model("DjangoRelatedModel", DjangoRelatedModel)

    # Register the model with a many-to-many relationship
    register_model("ModelWithM2M", ModelWithM2M)
    register_django_model("DjangoModelWithM2M", DjangoModelWithM2M)

    # Monkey patch the setup_dynamic_models function to use our registered models
    def mock_setup_dynamic_models(app_label=None):
        return get_django_models()

    # Save the original function
    original_setup_dynamic_models = (
        src.pydantic2django.static_django_model_generator.setup_dynamic_models
    )

    # Replace with our mock function
    src.pydantic2django.static_django_model_generator.setup_dynamic_models = (
        mock_setup_dynamic_models
    )

    # Create the generator
    generator = StaticDjangoModelGenerator(
        output_path=output_path,
        packages=["tests"],
        app_label="test_app",
        verbose=True,
    )

    yield generator, output_path

    # Clean up after the test
    temp_dir.cleanup()
    clear()

    # Restore the original function
    src.pydantic2django.static_django_model_generator.setup_dynamic_models = (
        original_setup_dynamic_models
    )


@pytest.fixture
def generated_content(generator_setup):
    """Generate the models file and return its content."""
    generator, output_path = generator_setup

    # Generate the models file
    result = generator.generate()

    # Check that the file was created
    assert result == output_path
    assert os.path.exists(output_path)

    # Read the file contents
    with open(output_path, "r") as f:
        content = f.read()

    # Print the entire content for debugging
    print("\nGenerated file content:")
    print(content)

    return content


@pytest.mark.parametrize(
    "expected_content,description",
    [
        (
            "from pydantic2django.base_django_model import Pydantic2DjangoBaseClass",
            "imports Pydantic2DjangoBaseClass",
        ),
        (
            "class DjangoDummyPydanticModel(Pydantic2DjangoBaseClass):",
            "defines DjangoDummyPydanticModel class",
        ),
    ],
)
def test_generate_content(generated_content, expected_content, description):
    """Test that the generated file contains expected content."""
    assert (
        expected_content in generated_content
    ), f"Failed to find content that {description}"


def test_dummy_model_content(generator_setup):
    """Test that the DummyPydanticModel is properly generated with all expected fields and attributes."""
    generator, output_path = generator_setup

    # Generate the models file
    generator.generate()

    # Read the file contents
    with open(output_path, "r") as f:
        content = f.read()

    # Extract the DjangoDummyPydanticModel class content
    model_start = content.find("class DjangoDummyPydanticModel")
    if model_start == -1:
        pytest.fail("DjangoDummyPydanticModel class not found in generated content")

    model_end = content.find("class ", model_start + 1)
    if model_end == -1:
        model_end = content.find("__all__")

    model_content = content[model_start:model_end]

    # Check for the fields
    assert (
        "description = models.TextField(" in model_content
    ), "description field not found in DjangoDummyPydanticModel"
    assert "null=True" in model_content, "null parameter not found in description field"

    assert (
        "count = models.IntegerField(" in model_content
    ), "count field not found in DjangoDummyPydanticModel"
    assert (
        "default=0" in model_content
    ), "default parameter with value 0 not found in count field"

    assert (
        "is_active = models.BooleanField(" in model_content
    ), "is_active field not found in DjangoDummyPydanticModel"
    assert (
        "default=True" in model_content
    ), "default parameter with value True not found in is_active field"

    # Check for id field - this is now handled by the base class
    # assert (
    #     "id = models.BigAutoField(" in model_content
    # ), "id field not found in DjangoDummyPydanticModel"

    # Check for name field - this is now handled by the base class
    # assert (
    #     "name = models.CharField(" in model_content
    # ), "name field not found in DjangoDummyPydanticModel"
    # assert (
    #     "max_length=" in model_content
    # ), "max_length parameter not found in name field"

    # Check for tags field
    assert (
        "data = models.JSONField(" in model_content
    ), "data field not found in DjangoDummyPydanticModel"


@pytest.mark.parametrize(
    "expected_content,description",
    [
        (
            "class DjangoModelWithM2M(Pydantic2DjangoBaseClass):",
            "defines DjangoModelWithM2M class",
        ),
        (
            "class DjangoRelatedModel(Pydantic2DjangoBaseClass):",
            "defines DjangoRelatedModel class",
        ),
    ],
)
def test_many_to_many_relationship_content(
    generated_content, expected_content, description
):
    """Test that many-to-many relationships are properly generated."""
    assert (
        expected_content in generated_content
    ), f"Failed to find content that {description}"


def test_many_to_many_field_content(generator_setup):
    """Test that the many-to-many field is properly generated."""
    generator, output_path = generator_setup

    # Generate the models file
    generator.generate()

    # Read the file contents
    with open(output_path, "r") as f:
        content = f.read()

    # Extract the DjangoModelWithM2M class content
    m2m_model_start = content.find("class DjangoModelWithM2M")
    if m2m_model_start == -1:
        pytest.fail("DjangoModelWithM2M class not found in generated content")

    m2m_model_end = content.find("class ", m2m_model_start + 1)
    if m2m_model_end == -1:
        m2m_model_end = content.find("__all__")

    m2m_model_content = content[m2m_model_start:m2m_model_end]

    # Check for the many-to-many field
    assert (
        "related_items = models.ManyToManyField(" in m2m_model_content
    ), "ManyToManyField not found in DjangoModelWithM2M"

    # Print the actual content for debugging
    print("\nActual M2M field content:")
    print(m2m_model_content)

    # The field might reference DjangoRelatedModel in different ways
    assert any(
        rel_format in m2m_model_content
        for rel_format in [
            '"DjangoRelatedModel"',
            "'DjangoRelatedModel'",
            'to="DjangoRelatedModel"',
            "to='DjangoRelatedModel'",
            'to="test_app.DjangoRelatedModel"',
            "to='test_app.DjangoRelatedModel'",
        ]
    ), "DjangoRelatedModel not referenced in ManyToManyField"

    # Check for blank parameter (might be True or False)
    assert "blank=" in m2m_model_content, "blank parameter not found in ManyToManyField"

    # Extract the DjangoRelatedModel class content
    related_model_start = content.find("class DjangoRelatedModel")
    if related_model_start == -1:
        pytest.fail("DjangoRelatedModel class not found in generated content")

    related_model_end = content.find("class ", related_model_start + 1)
    if related_model_end == -1:
        related_model_end = content.find("__all__")

    related_model_content = content[related_model_start:related_model_end]

    # Check for the value field
    assert (
        "value = models.IntegerField(" in related_model_content
    ), "IntegerField not found in DjangoRelatedModel"
    assert (
        "default=" in related_model_content
    ), "default parameter not found in IntegerField"


def test_field_collision_handling():
    """Test that field collisions are properly handled during model generation.

    This test verifies that models with field collisions are still successfully generated
    by resolving the field collisions rather than skipping the models.
    """
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "models.py")

        # Create a test module with the chains module
        test_module = type(sys)("test_chains_module")

        # Import the chains module
        chains_module = importlib.import_module("tests.test_models.chains")

        # Copy all attributes from chains_module to test_module
        for name in dir(chains_module):
            if not name.startswith("__"):
                setattr(test_module, name, getattr(chains_module, name))

        # Add the module to sys.modules temporarily
        sys.modules["test_chains_module"] = test_module

        # Capture logs to analyze errors
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        logger = logging.getLogger("pydantic2django")
        logger.addHandler(handler)
        previous_level = logger.level
        logger.setLevel(logging.DEBUG)

        try:
            # Create a generator with verbose output
            generator = StaticDjangoModelGenerator(
                output_path=output_path,
                packages=["test_chains_module"],
                app_label="test_chains",
                verbose=True,
            )

            # Create a custom filter function to only include models with field collisions
            def filter_models_with_collisions(name, model):
                # Include models that have id, created_at, or name fields that might collide
                if not hasattr(model, "model_fields"):
                    return False
                field_names = model.model_fields.keys()
                collision_fields = {"id", "created_at", "updated_at", "name"}
                return bool(set(field_names) & collision_fields)

            # Set the filter function
            generator.filter_function = filter_models_with_collisions

            # Generate the models
            generator.generate()

            # Check that the output file was created
            assert os.path.exists(output_path), "Output file was not created"

            # Read the generated file to verify content
            with open(output_path, "r") as f:
                content = f.read()

            # Verify that models with field collisions were successfully generated
            # with the collisions properly resolved
            assert (
                "class DjangoBaseNode" in content
            ), "BaseNode model should be generated with field collisions resolved"
            assert (
                "class DjangoChainNode" in content
            ), "ChainNode model should be generated with field collisions resolved"
            assert (
                "class DjangoChainGraph" in content
            ), "ChainGraph model should be generated with field collisions resolved"

            # Verify that the field collisions were resolved by checking for renamed fields
            # or other resolution strategies
            assert (
                "id_field" in content
                or "custom_id" in content
                or "pydantic_id" in content
            ), "No renamed id field found"
            assert (
                "created_at_field" in content
                or "custom_created_at" in content
                or "pydantic_created_at" in content
            ), "No renamed created_at field found"

        finally:
            # Clean up logging
            logger.removeHandler(handler)
            logger.setLevel(previous_level)

            # Clean up the test module
            if "test_chains_module" in sys.modules:
                del sys.modules["test_chains_module"]


def test_app_label_configuration():
    """Test that app label configuration issues are properly handled during model generation.

    This test verifies that models from external packages are successfully generated
    by resolving app label configuration issues.
    """
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "models.py")

        # Create a test module with the chains module
        test_module = type(sys)("test_chains_module")

        # Import the chains module
        chains_module = importlib.import_module("tests.test_models.chains")

        # Copy all attributes from chains_module to test_module
        for name in dir(chains_module):
            if not name.startswith("__"):
                setattr(test_module, name, getattr(chains_module, name))

        # Add the module to sys.modules temporarily
        sys.modules["test_chains_module"] = test_module

        # Capture logs to analyze errors
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        logger = logging.getLogger("pydantic2django")
        logger.addHandler(handler)
        previous_level = logger.level
        logger.setLevel(logging.DEBUG)

        try:
            # Create a generator with verbose output
            generator = StaticDjangoModelGenerator(
                output_path=output_path,
                packages=["test_chains_module"],
                app_label="test_chains",
                verbose=True,
            )

            # Create a custom filter function to only include external models
            def filter_external_models(name, model):
                # Include models from external packages that might have app label issues
                return hasattr(model, "__module__") and "llmaestro" in model.__module__

            # Set the filter function
            generator.filter_function = filter_external_models

            # Generate the models
            generator.generate()

            # Check that the output file was created
            assert os.path.exists(output_path), "Output file was not created"

            # Read the generated file to verify content
            with open(output_path, "r") as f:
                content = f.read()

            # Verify that external models were successfully generated
            # with app label issues properly resolved
            assert (
                "class DjangoLLMResponse" in content
            ), "LLMResponse model should be generated with app label issues resolved"
            assert (
                "class DjangoBaseEdge" in content
            ), "BaseEdge model should be generated with app label issues resolved"

            # Verify that the app label was properly set in the Meta class
            assert (
                'app_label = "test_chains"' in content
            ), "app_label not properly set in Meta class"

        finally:
            # Clean up logging
            logger.removeHandler(handler)
            logger.setLevel(previous_level)

            # Clean up the test module
            if "test_chains_module" in sys.modules:
                del sys.modules["test_chains_module"]
