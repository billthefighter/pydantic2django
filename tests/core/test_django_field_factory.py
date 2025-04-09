import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Union
from uuid import UUID

import pytest
from django.db import models
from django.apps import apps
from pydantic import BaseModel, EmailStr, Field
from pydantic.fields import FieldInfo
from pydantic2django.core.factories import ConversionCarrier, FieldConversionResult, BaseFieldFactory
from pydantic2django.pydantic.factory import PydanticFieldFactory, PydanticModelFactory
from pydantic2django.django.mapping import TypeMapper
from pydantic2django.core.relationships import RelationshipConversionAccessor, RelationshipMapper
from pydantic2django.core.context import ModelContext


@dataclass
class DjangoFieldFactoryTestParams:
    """Test parameters for DjangoFieldFactory tests."""

    field_name: str
    field_info: FieldInfo
    app_label: str = "test_app"
    expected_django_field_type: Optional[type[models.Field]] = None
    expected_is_context_field: bool = False
    expected_error: bool = False


@pytest.fixture
def empty_relationship_accessor():
    """Fixture providing an empty RelationshipConversionAccessor."""
    return RelationshipConversionAccessor()


@pytest.fixture
def populated_relationship_accessor(relationship_models):
    """Fixture providing a RelationshipConversionAccessor with models from relationship_models."""
    accessor = RelationshipConversionAccessor()

    # Add all models to the relationship accessor
    for model_name, model_class in relationship_models.items():
        # Create fake Django models to pair with the Pydantic models
        django_model = type(
            f"Django{model_name}",
            (models.Model,),
            {"__module__": "tests.test_models", "Meta": type("Meta", (), {"app_label": "tests"})},
        )

        # Create a model context
        model_context = ModelContext(django_model=django_model, pydantic_class=model_class)

        # Add directly to the available_relationships list with proper context
        accessor.available_relationships.append(RelationshipMapper(model_class, django_model, model_context))

    return accessor


@pytest.fixture
def field_factory(populated_relationship_accessor):
    """Fixture providing a PydanticFieldFactory with populated relationships."""
    return PydanticFieldFactory(available_relationships=populated_relationship_accessor)


@pytest.fixture
def empty_field_factory(empty_relationship_accessor):
    """Fixture providing a PydanticFieldFactory with empty relationships."""
    return PydanticFieldFactory(available_relationships=empty_relationship_accessor)


@pytest.fixture(scope="function")
def dynamic_related_model():
    """Fixture to create and clean up a dynamic DjangoRelatedModel."""
    model_name = "DjangoRelatedModel"
    app_label = "test_app"
    try:
        # Ensure clean state before creating
        if hasattr(apps, "all_models") and app_label in apps.all_models:
            if model_name.lower() in apps.all_models[app_label]:
                del apps.all_models[app_label][model_name.lower()]
        apps.clear_cache()

        # Create the model
        django_related = type(
            model_name,
            (models.Model,),
            {"__module__": "tests.test_models", "Meta": type("Meta", (), {"app_label": app_label})},
        )
        yield django_related  # Provide the model to the test
    finally:
        # Cleanup: Remove model from registry after test
        if hasattr(apps, "all_models") and app_label in apps.all_models:
            if model_name.lower() in apps.all_models[app_label]:
                del apps.all_models[app_label][model_name.lower()]
        apps.clear_cache()


def test_convert_basic_fields(field_factory, basic_pydantic_model):
    """Test converting basic field types."""
    # Create a minimal carrier for this test, providing required args
    carrier = ConversionCarrier(
        source_model=basic_pydantic_model, meta_app_label="test_app", base_django_model=models.Model
    )
    for field_name, field_info in basic_pydantic_model.model_fields.items():
        # Call create_field instead of convert_field
        result = field_factory.create_field(
            field_info=field_info,
            model_name=basic_pydantic_model.__name__,  # Pass model_name, not field_name
            carrier=carrier,
        )

        # Basic assertions
        assert result is not None
        assert isinstance(result, FieldConversionResult)
        assert result.field_name == field_name
        assert result.field_info == field_info

        # Check that we got a valid type mapping and Django field
        assert result.django_field is not None
        assert result.context_field is None  # Should not be a context field

        # Verify field attributes
        if field_name == "string_field":
            assert isinstance(result.django_field, models.TextField)
        elif field_name == "int_field":
            assert isinstance(result.django_field, models.IntegerField)
        elif field_name == "float_field":
            assert isinstance(result.django_field, models.FloatField)
        elif field_name == "bool_field":
            assert isinstance(result.django_field, models.BooleanField)
        elif field_name == "decimal_field":
            assert isinstance(result.django_field, models.DecimalField)
        elif field_name == "email_field":
            assert isinstance(result.django_field, models.EmailField)


def test_convert_datetime_fields(field_factory, datetime_pydantic_model):
    """Test converting datetime-related field types."""
    # Create a minimal carrier for this test, providing required args
    carrier = ConversionCarrier(
        source_model=datetime_pydantic_model, meta_app_label="test_app", base_django_model=models.Model
    )
    for field_name, field_info in datetime_pydantic_model.model_fields.items():
        # Call create_field instead of convert_field
        result = field_factory.create_field(
            field_info=field_info,
            model_name=datetime_pydantic_model.__name__,  # Pass model_name
            carrier=carrier,
        )

        # Basic assertions
        assert result is not None
        assert isinstance(result, FieldConversionResult)

        # Check that we got a valid type mapping and Django field
        assert result.django_field is not None

        # Verify field types
        if field_name == "datetime_field":
            assert isinstance(result.django_field, models.DateTimeField)
        elif field_name == "date_field":
            assert isinstance(result.django_field, models.DateField)
        elif field_name == "time_field":
            assert isinstance(result.django_field, models.TimeField)
        elif field_name == "duration_field":
            assert isinstance(result.django_field, models.DurationField)


def test_convert_optional_fields(field_factory, optional_fields_model):
    """Test converting optional fields."""
    # Create a minimal carrier for this test, providing required args
    carrier = ConversionCarrier(
        source_model=optional_fields_model, meta_app_label="test_app", base_django_model=models.Model
    )
    for field_name, field_info in optional_fields_model.model_fields.items():
        # Call create_field instead of convert_field
        result = field_factory.create_field(
            field_info=field_info,
            model_name=optional_fields_model.__name__,  # Pass model_name
            carrier=carrier,
        )

        # Basic assertions
        assert result is not None
        assert isinstance(result, FieldConversionResult)

        # Check field_kwargs for null/blank values
        if field_name.startswith("optional_"):
            # Optional fields might be context fields if unmappable OR regular fields
            # If it's a regular field, null/blank should be set correctly by TypeMapper
            if result.django_field:
                assert result.django_field.null is True
                assert result.django_field.blank is True  # Assuming TypeMapper sets blank=True for optional
            elif result.context_field:
                # If it's a context field, this is also potentially valid
                pass
            else:
                # Should be either a django field or a context field
                pytest.fail(
                    f"Optional field '{field_name}' resulted in neither Django field nor context field: {result}"
                )
        else:
            # Required fields
            assert result.django_field is not None


def test_convert_constrained_fields(field_factory, constrained_fields_model):
    """Test converting fields with constraints."""
    # Create a minimal carrier for this test, providing required args
    carrier = ConversionCarrier(
        source_model=constrained_fields_model, meta_app_label="test_app", base_django_model=models.Model
    )
    for field_name, field_info in constrained_fields_model.model_fields.items():
        # Call create_field instead of convert_field
        result = field_factory.create_field(
            field_info=field_info,
            model_name=constrained_fields_model.__name__,  # Pass model_name
            carrier=carrier,
        )

        # Basic assertions
        assert result is not None
        assert isinstance(result, FieldConversionResult)

        # Check field attributes
        if field_name == "name":
            assert result.field_kwargs.get("help_text") == "Full name of the user"
            assert result.field_kwargs.get("verbose_name") == "Full Name"
        elif field_name == "age":
            assert result.field_kwargs.get("help_text") == "User's age in years"
            assert result.field_kwargs.get("verbose_name") == "Age"
        elif field_name == "balance":
            assert result.field_kwargs.get("help_text") == "Current account balance"
            assert result.field_kwargs.get("verbose_name") == "Account Balance"


def test_edge_cases(field_factory):
    """Test various edge cases and error handling."""

    # Case 1: Field with no annotation
    class NoAnnotationModel(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        field_without_annotation: Any

    field_info = NoAnnotationModel.model_fields["field_without_annotation"]
    # Create a carrier
    carrier_no_anno = ConversionCarrier(
        source_model=NoAnnotationModel, meta_app_label="test_app", base_django_model=models.Model
    )
    result = field_factory.create_field(
        field_info=field_info,
        model_name=NoAnnotationModel.__name__,
        carrier=carrier_no_anno,
    )

    assert result.django_field is None
    assert result.context_field is field_info  # Should be treated as context
    assert result.error_str is None

    # Case 2: Field with unmappable type should be treated as context field
    class UnmappableTypeModel(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        unmappable: object  # No direct mapping to Django field

    field_info = UnmappableTypeModel.model_fields["unmappable"]
    # Create a carrier
    carrier_unmap = ConversionCarrier(
        source_model=UnmappableTypeModel, meta_app_label="test_app", base_django_model=models.Model
    )
    result = field_factory.create_field(
        field_info=field_info,
        model_name=UnmappableTypeModel.__name__,
        carrier=carrier_unmap,
    )

    assert result.django_field is None
    assert result.context_field is field_info  # Should be treated as context
    assert result.error_str is None


def test_convert_simple_field(field_factory):
    """Test converting a simple field without relationships."""

    # Create a simple model
    class SimpleModel(BaseModel):
        name: str = Field(description="User's name", title="Full Name")

    # Use ConversionCarrier and add base_django_model
    carrier = ConversionCarrier(source_model=SimpleModel, meta_app_label="test_app", base_django_model=models.Model)
    field_info = SimpleModel.model_fields["name"]
    # Call create_field
    result = field_factory.create_field(
        field_info=field_info,
        model_name=SimpleModel.__name__,
        carrier=carrier,
    )

    assert result.django_field is not None
    assert isinstance(result.django_field, models.TextField)  # Default for str
    # Check attributes (verbose_name likely comes from title)
    assert result.django_field.verbose_name == "Full Name"
    # Description might be help_text
    assert result.django_field.help_text == "User's name"
    assert result.context_field is None
    assert result.error_str is None
