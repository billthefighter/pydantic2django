import os
import sys
import unittest
import uuid
import importlib
from typing import List, Optional, Dict, Any, Type
from unittest.mock import patch

# Configure Django settings before importing Django models
import django
from django.conf import settings
from django.db import models

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


# Define a base Django model class similar to Pydantic2DjangoBaseClass
class BaseDjangoModel(models.Model):
    """
    Base class for storing Pydantic objects in the database.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    object_type = models.CharField(max_length=100)
    data = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True

    @classmethod
    def from_pydantic(
        cls, pydantic_obj: Any, name: Optional[str] = None
    ) -> "BaseDjangoModel":
        """
        Create a Django model instance from a Pydantic object.
        """
        object_type = pydantic_obj.__class__.__name__
        data = pydantic_obj.model_dump()

        # Use object_type as name if not provided and if object has a name attribute
        if name is None and hasattr(pydantic_obj, "name"):
            name = pydantic_obj.name
        elif name is None:
            name = object_type

        return cls(
            name=name,
            object_type=object_type,
            data=data,
        )

    def to_pydantic(self) -> Any:
        """
        Convert the stored data back to a Pydantic object.
        """
        # Import the appropriate class
        module_path = self._get_module_path()
        module = importlib.import_module(module_path)
        pydantic_class = getattr(module, self.object_type)

        # Get data with database field overrides
        data = self._get_data_with_db_overrides(pydantic_class)

        # Reconstruct the object
        return pydantic_class.model_validate(data)

    def _get_module_path(self) -> str:
        """
        Get the module path for the Pydantic class.
        """
        return "tests.standalone_test"

    def _get_data_with_db_overrides(self, pydantic_class: Any) -> dict[str, Any]:
        """
        Get the JSON data with database field overrides.
        """
        # Start with a copy of the stored JSON data
        data = self.data.copy()

        # Get all fields from the Pydantic model - we'll use the data keys as a fallback
        pydantic_field_names = set(data.keys())

        # Try to get field information from the Pydantic class
        try:
            # For Pydantic v2
            if hasattr(pydantic_class, "model_fields") and isinstance(
                pydantic_class.model_fields, dict
            ):
                pydantic_field_names.update(pydantic_class.model_fields.keys())
            # For Pydantic v1
            elif hasattr(pydantic_class, "__fields__") and isinstance(
                pydantic_class.__fields__, dict
            ):
                pydantic_field_names.update(pydantic_class.__fields__.keys())
        except Exception:
            # If we can't get the fields, just use what we have from the data
            pass

        # Get all fields from the Django model
        django_fields = {field.name: field for field in self._meta.fields}

        # Exclude these fields from consideration
        exclude_fields = {
            "id",
            "name",
            "object_type",
            "data",
            "created_at",
            "updated_at",
        }

        # Check each Django field to see if it matches a Pydantic field
        for field_name, field in django_fields.items():
            if field_name in exclude_fields:
                continue

            # Check if this field exists in the Pydantic model or data
            if field_name in pydantic_field_names:
                # Get the value from the Django model
                value = getattr(self, field_name)

                # Only override if the value is not None (unless the field in data is also None)
                if value is not None or (
                    field_name in data and data[field_name] is None
                ):
                    data[field_name] = value

        return data

    def update_from_pydantic(self, pydantic_obj: Any) -> None:
        """
        Update this object with new data from a Pydantic object.
        """
        self.data = pydantic_obj.model_dump()
        self.save()

    def save(self, *args, **kwargs):
        """
        Override save method for testing.
        """
        # In a real implementation, this would save to the database
        # For our test, we'll just do nothing
        pass


# Define a dummy Pydantic model for testing
class DummyPydanticModel(BaseModel):
    """A dummy Pydantic model for testing."""

    name: str
    description: Optional[str] = None
    count: int = 0
    tags: List[str] = Field(default_factory=list)
    is_active: bool = True


# Create a Django model that inherits from BaseDjangoModel
class DjangoDummyPydanticModel(BaseDjangoModel):
    """A Django model that inherits from BaseDjangoModel."""

    description = models.TextField(null=True, blank=True)
    count = models.IntegerField(default=0)
    is_active = models.BooleanField(default=True)

    class Meta(BaseDjangoModel.Meta):
        db_table = "dummy_pydantic_model"
        app_label = "test_app"
        verbose_name = "Dummy Pydantic Model"
        verbose_name_plural = "Dummy Pydantic Models"
        abstract = False

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("object_type", "DummyPydanticModel")
        super().__init__(*args, **kwargs)


class TestBaseDjangoModel(unittest.TestCase):
    """Test the BaseDjangoModel."""

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
