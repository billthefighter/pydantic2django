import os
import django
from django.conf import settings

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

import pytest
from django.db import models
from pydantic import BaseModel

from pydantic2django.field_utils import FieldSerializer
from pydantic2django.static_django_model_generator import StaticDjangoModelGenerator
from pydantic2django.factory import DjangoFieldFactory, DjangoModelFactory
from pydantic2django.relationships import RelationshipConversionAccessor


class TestModel(BaseModel):
    name: str


class DjangoTestModel(models.Model):
    name = models.CharField(max_length=255)

    class Meta:
        app_label = "test_app"


class RelatedModel(BaseModel):
    test_model: TestModel


class DjangoRelatedModel(models.Model):
    test_model = models.ForeignKey(DjangoTestModel, on_delete=models.CASCADE)

    class Meta:
        app_label = "test_app"


def test_field_serialization_no_duplicate_app_label():
    """Test that the field serializer doesn't produce fields with duplicate app labels."""
    # Create a ForeignKey field with an app_label already included
    field = models.ForeignKey(to="test_app.DjangoTestModel", on_delete=models.CASCADE)

    # Serialize the field
    field_def = FieldSerializer.serialize_field(field)

    # Check that there's no duplicate app label
    assert "to='test_app.test_app." not in field_def
    assert "to='test_app.DjangoTestModel'" in field_def


def test_static_generator_no_duplicate_app_label():
    """Test that the static generator doesn't produce fields with duplicate app labels."""
    # Create a ForeignKey field with an app_label already included
    field = models.ForeignKey(to="test_app.DjangoTestModel", on_delete=models.CASCADE)

    # Create a generator with app_label matching the field's app_label
    generator = StaticDjangoModelGenerator(app_label="test_app")

    # Generate a field definition
    field_def = generator.generate_field_definition(field)

    # Check that there's no duplicate app label
    assert "to='test_app.test_app." not in field_def
    assert "to='test_app.DjangoTestModel'" in field_def


def test_factory_relationship_field_no_duplicate_app_label():
    """Test that the field factory doesn't produce fields with duplicate app labels."""
    # Set up the accessor with test models
    accessor = RelationshipConversionAccessor()
    accessor.map_relationship(TestModel, DjangoTestModel)

    # Create the field factory with the accessor
    field_factory = DjangoFieldFactory(available_relationships=accessor)

    # Create a model that uses a related field that already has an app label
    field_info = RelatedModel.model_fields["test_model"]
    result = field_factory.convert_field(field_name="test_model", field_info=field_info, app_label="test_app")

    # Check the 'to' field value
    assert result.field_kwargs["to"] == "test_app.TestModel"

    # No duplicate app_label should be present
    assert "test_app.test_app" not in result.field_kwargs["to"]
