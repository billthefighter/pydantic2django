import os
import sys
import unittest
import tempfile
from typing import List, Optional

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
)


# Define a dummy Pydantic model for testing
class DummyPydanticModel(BaseModel):
    """A dummy Pydantic model for testing."""

    name: str
    description: Optional[str] = None
    count: int = 0
    tags: List[str] = Field(default_factory=list)
    is_active: bool = True


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


class TestStaticDjangoModelGenerator(unittest.TestCase):
    """Test the StaticDjangoModelGenerator class."""

    def setUp(self):
        """Set up the test."""
        # Create a temporary directory for the output file
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_path = os.path.join(self.temp_dir.name, "generated_models.py")

        # Register our models with the mock discovery module
        clear()
        register_model("DummyPydanticModel", DummyPydanticModel)
        register_django_model("DjangoDummyPydanticModel", DjangoDummyPydanticModel)

        # Create the generator
        self.generator = StaticDjangoModelGenerator(
            output_path=self.output_path,
            packages=["tests"],
            app_label="test_app",
            verbose=True,
        )

    def tearDown(self):
        """Clean up after the test."""
        self.temp_dir.cleanup()
        clear()

    def test_generate(self):
        """Test the generate method."""
        # Generate the models file
        result = self.generator.generate()

        # Check that the file was created
        self.assertEqual(result, self.output_path)
        self.assertTrue(os.path.exists(self.output_path))

        # Read the file contents
        with open(self.output_path, "r") as f:
            content = f.read()

        # Check that the file contains the expected content
        self.assertIn(
            "from pydantic2django.base_django_model import Pydantic2DjangoBaseClass",
            content,
        )
        self.assertIn(
            "class DjangoDummyPydanticModel(Pydantic2DjangoBaseClass):", content
        )
        self.assertIn('kwargs.setdefault("object_type", "DummyPydanticModel")', content)
        self.assertIn("def _get_module_path(self) -> str:", content)
        self.assertIn('return "tests.test_static_django_model_generator"', content)

        # Check that the model inherits from Pydantic2DjangoBaseClass
        self.assertIn("class Meta(Pydantic2DjangoBaseClass.Meta):", content)

        # Check that the model has the expected fields
        self.assertIn("description = models.TextField(null=True, blank=True)", content)
        self.assertIn("count = models.IntegerField(default=0)", content)
        self.assertIn("is_active = models.BooleanField(default=True)", content)

        # Check that the model has the expected meta attributes
        self.assertIn('db_table = "dummy_pydantic_model"', content)
        self.assertIn('app_label = "test_app"', content)
        self.assertIn('verbose_name = "Dummy Pydantic Model"', content)
        self.assertIn('verbose_name_plural = "Dummy Pydantic Models"', content)
        self.assertIn("abstract = False", content)


if __name__ == "__main__":
    unittest.main()
