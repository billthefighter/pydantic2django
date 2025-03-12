import os
import sys
import unittest
import tempfile
import pytest
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
    data = models.JSONField(default=dict)  # Add data field for compatibility

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "dummy_pydantic_model"
        app_label = "test_app"
        verbose_name = "Dummy Pydantic Model"
        verbose_name_plural = "Dummy Pydantic Models"
        abstract = False

    def __init__(self, *args, **kwargs):
        # Set a default fully qualified object_type if not provided
        if "object_type" in kwargs and not "." in kwargs["object_type"]:
            # Convert simple class name to fully qualified name
            kwargs["object_type"] = f"tests.simplified_test.{kwargs['object_type']}"
        super().__init__(*args, **kwargs)

    def _get_data_for_pydantic(self) -> dict[str, Any]:
        """
        Get the data from Django fields for creating a Pydantic object.

        Returns:
            A dictionary of field values
        """
        # Start with an empty dictionary
        data = {
            "name": self.name,  # Always include the name field
        }

        # Get all fields from the Django model
        django_fields = {field.name: field for field in self._meta.fields}

        # Exclude these fields from consideration
        exclude_fields = {
            "id",
            "name",
            "object_type",
            "created_at",
            "updated_at",
            "data",
        }

        # Add each Django field value to the data dictionary
        for field_name, field in django_fields.items():
            if field_name in exclude_fields:
                continue

            # Get the value from the Django model
            value = getattr(self, field_name)

            # Add to data dictionary
            data[field_name] = value

        return data


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
        # Check that the object_type now contains the fully qualified name
        self.assertEqual(
            django_model.object_type, "tests.simplified_test.DummyPydanticModel"
        )
        self.assertEqual(django_model.description, "A test model")
        self.assertEqual(django_model.count, 42)
        self.assertEqual(django_model.is_active, True)

    def test_to_pydantic(self):
        """Test the to_pydantic method."""
        # Create a Django model
        django_model = DjangoDummyPydanticModel(
            name="Test Model",
            object_type="tests.simplified_test.DummyPydanticModel",
            description="A test model",
            count=42,
            is_active=True,
        )

        # Convert to Pydantic model
        pydantic_model = django_model.to_pydantic()

        # Check that the conversion worked
        self.assertEqual(pydantic_model.name, "Test Model")
        self.assertEqual(pydantic_model.description, "A test model")
        self.assertEqual(pydantic_model.count, 42)
        self.assertEqual(pydantic_model.tags, [])  # Default empty list
        self.assertEqual(pydantic_model.is_active, True)

    @pytest.mark.django_db
    def test_update_from_pydantic(self):
        """Test the update_from_pydantic method."""
        # Create a Django model
        django_model = DjangoDummyPydanticModel(
            name="Test Model",
            object_type="tests.simplified_test.DummyPydanticModel",
            description="A test model",
            count=42,
            is_active=True,
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
        self.assertEqual(django_model.name, "Updated Model")
        self.assertEqual(django_model.description, "An updated model")
        self.assertEqual(django_model.count, 99)
        self.assertEqual(django_model.is_active, False)


if __name__ == "__main__":
    unittest.main()
