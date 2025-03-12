import os
import sys
import unittest
import tempfile
from typing import List, Optional, Dict, Any, Type

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

# Import the base Django model class
from src.pydantic2django.base_django_model import Pydantic2DjangoBaseClass


# Define a dummy Pydantic model for testing
class DummyPydanticModel(BaseModel):
    """A dummy Pydantic model for testing."""

    name: str
    description: Optional[str] = None
    count: int = 0
    tags: List[str] = Field(default_factory=list)
    is_active: bool = True


# Create a Django model that inherits from Pydantic2DjangoBaseClass
class DjangoDummyPydanticModel(Pydantic2DjangoBaseClass):
    """A Django model that inherits from Pydantic2DjangoBaseClass."""

    description = models.TextField(null=True, blank=True)
    count = models.IntegerField(default=0)
    is_active = models.BooleanField(default=True)

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "dummy_pydantic_model"
        app_label = "test_app"
        verbose_name = "Dummy Pydantic Model"
        verbose_name_plural = "Dummy Pydantic Models"
        abstract = False

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("object_type", "DummyPydanticModel")
        super().__init__(*args, **kwargs)

    def _get_module_path(self) -> str:
        return "tests.simplified_test"


class TestPydantic2DjangoBaseClass(unittest.TestCase):
    """Test the Pydantic2DjangoBaseClass."""

    def test_from_pydantic(self):
        """Test the from_pydantic method."""
        # Create a Pydantic model
        pydantic_model = DummyPydanticModel(
            name="Test Model",
            description="A test model",
            count=42,
            tags=["test", "model"],
            is_active=True,
        )

        # Convert to Django model
        django_model = DjangoDummyPydanticModel.from_pydantic(pydantic_model)

        # Check that the conversion worked
        self.assertEqual(django_model.name, "Test Model")
        self.assertEqual(django_model.object_type, "DummyPydanticModel")
        self.assertEqual(django_model.data["description"], "A test model")
        self.assertEqual(django_model.data["count"], 42)
        self.assertEqual(django_model.data["tags"], ["test", "model"])
        self.assertEqual(django_model.data["is_active"], True)

    def test_to_pydantic(self):
        """Test the to_pydantic method."""
        # Create a Django model
        django_model = DjangoDummyPydanticModel(
            name="Test Model",
            object_type="DummyPydanticModel",
            data={
                "name": "Test Model",
                "description": "A test model",
                "count": 42,
                "tags": ["test", "model"],
                "is_active": True,
            },
        )

        # Convert to Pydantic model
        pydantic_model = django_model.to_pydantic()

        # Check that the conversion worked
        self.assertEqual(pydantic_model.name, "Test Model")
        self.assertEqual(pydantic_model.description, "A test model")
        self.assertEqual(pydantic_model.count, 42)
        self.assertEqual(pydantic_model.tags, ["test", "model"])
        self.assertEqual(pydantic_model.is_active, True)

    def test_update_from_pydantic(self):
        """Test the update_from_pydantic method."""
        # Create a Django model
        django_model = DjangoDummyPydanticModel(
            name="Test Model",
            object_type="DummyPydanticModel",
            data={
                "name": "Test Model",
                "description": "A test model",
                "count": 42,
                "tags": ["test", "model"],
                "is_active": True,
            },
        )

        # Create a new Pydantic model with updated data
        pydantic_model = DummyPydanticModel(
            name="Updated Model",
            description="An updated model",
            count=99,
            tags=["updated", "model"],
            is_active=False,
        )

        # Update the Django model
        django_model.update_from_pydantic(pydantic_model)

        # Check that the update worked
        self.assertEqual(django_model.data["name"], "Updated Model")
        self.assertEqual(django_model.data["description"], "An updated model")
        self.assertEqual(django_model.data["count"], 99)
        self.assertEqual(django_model.data["tags"], ["updated", "model"])
        self.assertEqual(django_model.data["is_active"], False)


if __name__ == "__main__":
    unittest.main()
