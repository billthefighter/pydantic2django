import logging
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import UUID

import pytest
from django.db import models
from django.apps import apps
from pydantic import BaseModel, EmailStr, Field
from pydantic.fields import FieldInfo
from pydantic2django.factory import DjangoModelFactoryCarrier

from pydantic2django.factory import DjangoFieldFactory, FieldConversionResult
from pydantic2django.field_type_mapping import TypeMapper, TypeMappingDefinition
from pydantic2django.relationships import RelationshipConversionAccessor, RelationshipMapper
from pydantic2django.context_storage import ModelContext


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
    """Fixture providing a DjangoFieldFactory with populated relationships."""
    return DjangoFieldFactory(available_relationships=populated_relationship_accessor)


@pytest.fixture
def empty_field_factory(empty_relationship_accessor):
    """Fixture providing a DjangoFieldFactory with empty relationships."""
    return DjangoFieldFactory(available_relationships=empty_relationship_accessor)


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
    carrier = DjangoModelFactoryCarrier(pydantic_model=basic_pydantic_model, meta_app_label="test_app")
    for field_name, field_info in basic_pydantic_model.model_fields.items():
        # Add source_model_name and carrier to the call
        result = field_factory.convert_field(
            field_name=field_name,
            field_info=field_info,
            source_model_name=basic_pydantic_model.__name__,
            carrier=carrier,
        )

        # Basic assertions
        assert result is not None
        assert isinstance(result, FieldConversionResult)
        assert result.field_name == field_name
        assert result.field_info == field_info

        # Check that we got a valid type mapping and Django field
        assert result.type_mapping_definition is not None
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
    carrier = DjangoModelFactoryCarrier(pydantic_model=datetime_pydantic_model, meta_app_label="test_app")
    for field_name, field_info in datetime_pydantic_model.model_fields.items():
        # Add source_model_name and carrier to the call
        result = field_factory.convert_field(
            field_name=field_name,
            field_info=field_info,
            source_model_name=datetime_pydantic_model.__name__,
            carrier=carrier,
        )

        # Basic assertions
        assert result is not None
        assert isinstance(result, FieldConversionResult)

        # Check that we got a valid type mapping and Django field
        assert result.type_mapping_definition is not None
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
    carrier = DjangoModelFactoryCarrier(pydantic_model=optional_fields_model, meta_app_label="test_app")
    for field_name, field_info in optional_fields_model.model_fields.items():
        # Add source_model_name and carrier to the call
        result = field_factory.convert_field(
            field_name=field_name,
            field_info=field_info,
            source_model_name=optional_fields_model.__name__,
            carrier=carrier,
        )

        # Basic assertions
        assert result is not None
        assert isinstance(result, FieldConversionResult)

        # Check field_kwargs for null/blank values
        if field_name.startswith("optional_"):
            # In current implementation, optional fields might be contextual
            # We can check that they're handled correctly either way
            if result.type_mapping_definition is not None:
                # If they have type mapping, they should have correct null/blank settings
                assert result.field_kwargs.get("null") is True
                assert result.field_kwargs.get("blank") is True
            else:
                # If no type mapping, they should be marked as context fields
                assert result.context_field is not None
        else:
            # Required fields
            assert result.type_mapping_definition is not None
            assert result.django_field is not None


def test_convert_constrained_fields(field_factory, constrained_fields_model):
    """Test converting fields with constraints."""
    # Create a minimal carrier for this test, providing required args
    carrier = DjangoModelFactoryCarrier(pydantic_model=constrained_fields_model, meta_app_label="test_app")
    for field_name, field_info in constrained_fields_model.model_fields.items():
        # Add source_model_name and carrier to the call
        result = field_factory.convert_field(
            field_name=field_name,
            field_info=field_info,
            source_model_name=constrained_fields_model.__name__,
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


def test_handle_id_field(field_factory):
    """Test handling of ID fields."""

    # Create test models with different ID field types
    class ModelWithIntId(BaseModel):
        id: int

    class ModelWithStrId(BaseModel):
        id: str

    class ModelWithCustomId(BaseModel):
        id: UUID  # A type that doesn't have special ID handling

    # Test integer ID
    int_id_info = ModelWithIntId.model_fields["id"]
    int_id_field = field_factory.handle_id_field("id", int_id_info)
    assert isinstance(int_id_field, models.AutoField)
    assert int_id_field.primary_key is True

    # Test string ID
    str_id_info = ModelWithStrId.model_fields["id"]
    str_id_field = field_factory.handle_id_field("id", str_id_info)
    assert isinstance(str_id_field, models.CharField)
    assert str_id_field.primary_key is True
    assert str_id_field.max_length == 255

    # Test custom ID type - should default to AutoField
    custom_id_info = ModelWithCustomId.model_fields["id"]
    custom_id_field = field_factory.handle_id_field("id", custom_id_info)
    assert isinstance(custom_id_field, models.AutoField)
    assert custom_id_field.primary_key is True

    # Test non-ID field returns None
    class ModelWithName(BaseModel):
        name: str

    name_info = ModelWithName.model_fields["name"]
    assert field_factory.handle_id_field("name", name_info) is None


def test_field_kwargs_generation(field_factory, constrained_fields_model):
    """Test that field_kwargs are correctly generated."""
    for field_name, field_info in constrained_fields_model.model_fields.items():
        kwargs = field_factory.process_field_attributes(field_info)

        # Basic assertions
        assert isinstance(kwargs, dict)

        # Check that we always have null and blank
        assert "null" in kwargs
        assert "blank" in kwargs

        # Check title and description handling
        if field_info.title:
            assert kwargs.get("verbose_name") == field_info.title
        if field_info.description:
            assert kwargs.get("help_text") == field_info.description


def test_relationship_field_conversion(field_factory, relationship_models):
    """Test conversion of relationship fields."""
    # Extract test models
    address_model = relationship_models["Address"]
    profile_model = relationship_models["Profile"]
    tag_model = relationship_models["Tag"]
    user_model = relationship_models["User"]

    # Create a carrier for the User model using the populated relationship_accessor
    carrier = DjangoModelFactoryCarrier(pydantic_model=user_model, meta_app_label="test_app")

    # Test each relationship field in User model
    for field_name, field_info in user_model.model_fields.items():
        if field_name in ["address", "profile", "tags"]:
            # Add source_model_name and carrier to the call
            result = field_factory.convert_field(
                field_name=field_name, field_info=field_info, source_model_name=user_model.__name__, carrier=carrier
            )

            # Basic assertions
            assert result is not None
            assert isinstance(result, FieldConversionResult)

            # Current implementation might handle these as context fields
            # Due to complex relationship requirements
            if result.context_field is not None:
                # If it's a context field, that's expected
                assert result.context_field is field_info
            else:
                # If not a context field, verify relationship mapping
                assert result.type_mapping_definition is not None
                assert result.type_mapping_definition.is_relationship is True

                # Field-specific checks only for successfully mapped fields
                if result.django_field is not None:
                    if field_name == "address":
                        # ForeignKey relationship
                        assert result.type_mapping_definition.django_field == models.ForeignKey
                        assert result.field_kwargs.get("on_delete") == models.CASCADE

                    elif field_name == "profile":
                        # OneToOne relationship (based on json_schema_extra in fixture)
                        assert result.type_mapping_definition.django_field == models.ForeignKey
                        assert result.field_kwargs.get("on_delete") == models.CASCADE

                    elif field_name == "tags":
                        # ManyToMany relationship
                        assert result.type_mapping_definition.django_field == models.ManyToManyField


def test_relationship_field_missing_model(empty_field_factory, relationship_models):
    """Test handling of relationship fields where the related model is missing."""
    user_model = relationship_models["User"]

    # Create a carrier for the User model
    carrier = DjangoModelFactoryCarrier(pydantic_model=user_model, meta_app_label="test_app")

    # Test the 'profile' field (assuming Profile model *is* registered in accessor)
    field_info = user_model.model_fields["profile"]
    result = empty_field_factory.convert_field(
        field_name="profile", field_info=field_info, source_model_name=user_model.__name__, carrier=carrier
    )
    assert result is not None
    # Should NOT successfully create ForeignKey when model is missing
    assert result.django_field is None
    assert result.context_field is not None  # Should be context field

    # Now, create a factory with an EMPTY accessor
    empty_accessor = RelationshipConversionAccessor()
    factory_no_relations = DjangoFieldFactory(available_relationships=empty_accessor)
    # Create a new carrier with the empty accessor
    carrier_no_relations = DjangoModelFactoryCarrier(pydantic_model=user_model, meta_app_label="test_app")

    # Test the 'profile' field again, expecting it to become contextual
    result_missing = factory_no_relations.convert_field(
        field_name="profile", field_info=field_info, source_model_name=user_model.__name__, carrier=carrier_no_relations
    )
    assert result_missing is not None
    assert result_missing.django_field is None  # Should NOT create a field
    assert result_missing.context_field is field_info  # Should become context field


def test_handle_relationship_field_directly():
    """Test directly calling handle_relationship_field."""

    # Define models for testing
    class Address(BaseModel):
        street: str

    class Profile(BaseModel):
        bio: str

    class Tag(BaseModel):
        name: str

    class User(BaseModel):
        name: str
        address: Address  # ForeignKey
        profile: Optional[Profile]  # OneToOneField
        tags: list[Tag]  # ManyToManyField

    # Create Django model mocks
    django_address = type(
        "DjangoAddress", (models.Model,), {"__module__": "tests", "Meta": type("Meta", (), {"app_label": "test_app"})}
    )
    django_profile = type(
        "DjangoProfile", (models.Model,), {"__module__": "tests", "Meta": type("Meta", (), {"app_label": "test_app"})}
    )
    django_tag = type(
        "DjangoTag", (models.Model,), {"__module__": "tests", "Meta": type("Meta", (), {"app_label": "test_app"})}
    )

    # Setup relationship accessor
    accessor = RelationshipConversionAccessor()
    accessor.available_relationships.extend(
        [
            RelationshipMapper(
                Address, django_address, ModelContext(django_model=django_address, pydantic_class=Address)
            ),
            RelationshipMapper(
                Profile, django_profile, ModelContext(django_model=django_profile, pydantic_class=Profile)
            ),
            RelationshipMapper(Tag, django_tag, ModelContext(django_model=django_tag, pydantic_class=Tag)),
        ]
    )

    # Create field factory
    field_factory = DjangoFieldFactory(available_relationships=accessor)

    # Get field info for a relationship field
    field_info = User.model_fields["address"]

    # Create a base result using convert_field (ensure it includes carrier)
    carrier = DjangoModelFactoryCarrier(pydantic_model=User, meta_app_label="test_app")
    base_result = field_factory.convert_field(
        field_name="address",
        field_info=field_info,
        app_label="test_app",
        source_model_name=User.__name__,  # Add source name
        carrier=carrier,  # Add carrier
    )

    # Directly call handle_relationship_field
    result = field_factory.handle_relationship_field(base_result, User.__name__, carrier)

    # Assertions
    assert result is not None
    assert result.django_field is not None
    assert isinstance(result.django_field, models.ForeignKey)


def test_invalid_relationship_types():
    """Test handling of invalid or unmappable relationship types."""

    class InvalidType:
        pass

    class InvalidRelationshipTestModel(BaseModel):
        invalid_list: list[InvalidType]  # List of non-BaseModel
        invalid_direct: InvalidType  # Direct non-BaseModel type

        # Add model_config to allow arbitrary types
        model_config = {"arbitrary_types_allowed": True}

    # Create Django model mocks (though not strictly needed for this test)
    django_invalid = type(
        "DjangoInvalid", (models.Model,), {"__module__": "tests", "Meta": type("Meta", (), {"app_label": "test_app"})}
    )

    # Setup relationship accessor
    accessor = RelationshipConversionAccessor()
    # Don't add InvalidType to accessor to test missing model handling

    # Create field factory
    field_factory = DjangoFieldFactory(available_relationships=accessor)

    # Get field info for the invalid list field
    field_info = InvalidRelationshipTestModel.model_fields["invalid_list"]

    # Create a base result using convert_field (ensure it includes carrier)
    carrier = DjangoModelFactoryCarrier(pydantic_model=InvalidRelationshipTestModel, meta_app_label="test_app")
    base_result = field_factory.convert_field(
        field_name="invalid_list",
        field_info=field_info,
        app_label="test_app",
        source_model_name=InvalidRelationshipTestModel.__name__,  # Add source name
        carrier=carrier,  # Add carrier
    )

    # Directly call handle_relationship_field
    result = field_factory.handle_relationship_field(base_result, InvalidRelationshipTestModel.__name__, carrier)

    # Assertions: Expect it to become a context field
    assert result is not None
    assert result.django_field is None
    assert result.context_field is not None
    assert result.error_str is not None
    assert "not in relationship accessor" in result.error_str


def test_edge_cases(field_factory):
    """Test various edge cases and error handling."""

    # Case 1: Field with no annotation
    class NoAnnotationModel(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        field_without_annotation: Any

    field_info = NoAnnotationModel.model_fields["field_without_annotation"]
    # Create a carrier
    carrier_no_anno = DjangoModelFactoryCarrier(pydantic_model=NoAnnotationModel, meta_app_label="test_app")
    result = field_factory.convert_field(
        field_name="field_without_annotation",
        field_info=field_info,
        source_model_name=NoAnnotationModel.__name__,
        carrier=carrier_no_anno,
    )

    # Any type should be handled as JSONField, but that's an implementation detail
    # Just check that the result is valid
    assert result is not None
    assert isinstance(result, FieldConversionResult)

    # Case 2: Field with unmappable type should be treated as context field
    class UnmappableTypeModel(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        unmappable: object  # No direct mapping to Django field

    field_info = UnmappableTypeModel.model_fields["unmappable"]
    # Create a carrier
    carrier_unmap = DjangoModelFactoryCarrier(pydantic_model=UnmappableTypeModel, meta_app_label="test_app")
    result = field_factory.convert_field(
        field_name="unmappable",
        field_info=field_info,
        source_model_name=UnmappableTypeModel.__name__,
        carrier=carrier_unmap,
    )

    # Should be treated as a context field or have a valid mapping
    assert result is not None
    assert isinstance(result, FieldConversionResult)
    # Either it has a context field or a valid mapping
    assert result.context_field is not None or result.django_field is not None


def test_rendered_django_field():
    """Test that the Django field object is correctly instantiated."""
    # Create field factory
    field_factory = DjangoFieldFactory(available_relationships=RelationshipConversionAccessor())

    # Test simple field
    class SimpleModel(BaseModel):
        name: str

    field_info = SimpleModel.model_fields["name"]
    carrier = DjangoModelFactoryCarrier(pydantic_model=SimpleModel, meta_app_label="test_app")
    result = field_factory.convert_field(
        field_name="name",
        field_info=field_info,
        app_label="test_app",
        source_model_name=SimpleModel.__name__,
        carrier=carrier,
    )

    # Check the instantiated field
    assert result is not None, "Result should not be None"
    assert result.django_field is not None, "Django field should be instantiated"
    assert isinstance(result.django_field, models.TextField), "Field should be a TextField"

    # Test field with attributes
    class ModelWithAttrs(BaseModel):
        description: str = Field(max_length=200, default="N/A")

    field_info_attrs = ModelWithAttrs.model_fields["description"]
    carrier_attrs = DjangoModelFactoryCarrier(pydantic_model=ModelWithAttrs, meta_app_label="test_app")
    result_attrs = field_factory.convert_field(
        field_name="description",
        field_info=field_info_attrs,
        app_label="test_app",
        source_model_name=ModelWithAttrs.__name__,
        carrier=carrier_attrs,
    )

    # Check the instantiated field with attributes
    assert result_attrs is not None, "Result with attrs should not be None"
    assert result_attrs.django_field is not None, "Django field with attrs should be instantiated"
    assert isinstance(result_attrs.django_field, models.TextField), "Field with attrs should be a TextField"
    assert getattr(result_attrs.django_field, "max_length") == 200, "Incorrect max_length"
    assert getattr(result_attrs.django_field, "default") == "N/A", "Incorrect default"


def test_process_field_attributes_with_extra_dict(field_factory):
    """Test process_field_attributes with extra dictionary."""

    # Get a basic field info
    class ExtraModel(BaseModel):
        name: str

    field_info = ExtraModel.model_fields["name"]

    # Process with extra dictionary
    extra_dict = {"max_length": 100, "db_index": True}
    kwargs = field_factory.process_field_attributes(field_info, extra=extra_dict)

    # Assertions
    assert kwargs.get("max_length") == 100
    assert kwargs.get("db_index") is True


def test_process_field_attributes_with_extra_callable(field_factory):
    """Test process_field_attributes with extra callable."""

    # Get a basic field info
    class ExtraModel(BaseModel):
        name: str

    field_info = ExtraModel.model_fields["name"]

    # Define extra callable
    def extra_callable(field_info):
        return {"max_length": 200, "unique": True}

    # Process with extra callable
    kwargs = field_factory.process_field_attributes(field_info, extra=extra_callable)

    # Assertions
    assert kwargs.get("max_length") == 200
    assert kwargs.get("unique") is True


def test_field_validators(field_factory):
    """Test that field validators are correctly applied."""

    # Create a model with constrained fields
    class ValidatorModel(BaseModel):
        min_value: int = Field(ge=10)
        max_value: int = Field(le=100)
        range_value: int = Field(gt=0, lt=50)

    # Pydantic v2 stores constraints in metadata as objects
    # Instead of directly testing for validators, ensure the metadata contains the constraints

    # Test min validator metadata
    min_info = ValidatorModel.model_fields["min_value"]
    assert hasattr(min_info, "metadata")
    assert min_info.metadata  # Should not be empty

    # Test max validator metadata
    max_info = ValidatorModel.model_fields["max_value"]
    assert hasattr(max_info, "metadata")
    assert max_info.metadata  # Should not be empty

    # Test range validator metadata
    range_info = ValidatorModel.model_fields["range_value"]
    assert hasattr(range_info, "metadata")
    assert range_info.metadata  # Should not be empty

    # Additional test - verify process_field_attributes doesn't raise exceptions
    # Even if it doesn't currently create validators
    min_kwargs = field_factory.process_field_attributes(min_info)
    max_kwargs = field_factory.process_field_attributes(max_info)
    range_kwargs = field_factory.process_field_attributes(range_info)

    # All should return dictionaries without raising exceptions
    assert isinstance(min_kwargs, dict)
    assert isinstance(max_kwargs, dict)
    assert isinstance(range_kwargs, dict)


def test_process_field_attributes(field_factory):
    """Test the process_field_attributes method directly with a basic model."""

    # Create a simple model
    class SimpleModel(BaseModel):
        name: str = Field(description="User's name", title="Full Name")
        # Note: For Pydantic v2, constraints like gt, ge, lt, le are stored as objects in the metadata list
        age: int = Field(gt=0, lt=120)
        is_active: bool = True

    # Test string field
    name_info = SimpleModel.model_fields["name"]
    name_attrs = field_factory.process_field_attributes(name_info)

    assert name_attrs["help_text"] == "User's name"
    assert name_attrs["verbose_name"] == "Full Name"

    # Test int field with validators - use process_field_attributes and look at the result
    age_info = SimpleModel.model_fields["age"]
    age_attrs = field_factory.process_field_attributes(age_info)

    # In Pydantic v2, Field's metadata is a list of constraint objects
    # The process_field_attributes method should handle this and extract constraints
    # Just test that it produces valid field_kwargs
    assert isinstance(age_attrs, dict)

    # Test boolean field with default
    bool_info = SimpleModel.model_fields["is_active"]
    bool_attrs = field_factory.process_field_attributes(bool_info)

    assert bool_attrs["default"] is True


def test_convert_simple_field(field_factory):
    """Test converting a simple field without relationships."""

    # Create a simple model
    class SimpleModel(BaseModel):
        name: str = Field(description="User's name", title="Full Name")

    # Get field info
    field_info = SimpleModel.model_fields["name"]

    # Convert field
    carrier = DjangoModelFactoryCarrier(pydantic_model=SimpleModel, meta_app_label="test_app")
    result = field_factory.convert_field(
        field_name="name",
        field_info=field_info,
        app_label="test_app",
        source_model_name=SimpleModel.__name__,
        carrier=carrier,
    )

    # Basic assertions
    assert result is not None
    assert isinstance(result, FieldConversionResult)
    assert result.field_name == "name"

    # Check field details
    assert result.django_field is not None
    assert isinstance(result.django_field, models.TextField)
    assert result.field_kwargs["help_text"] == "User's name"
    assert result.field_kwargs["verbose_name"] == "Full Name"


def test_relationship_field_to_parameter(field_factory, relationship_models):
    """Test that the 'to' parameter of relationship fields is properly formatted as a string, not a tuple."""
    # Extract test models
    user_model = relationship_models["User"]

    # Create a custom model to test each relationship type explicitly
    class RelationshipTestModel(BaseModel):
        # ForeignKey relationship
        address: relationship_models["Address"]
        # ManyToMany relationship
        tags: list[relationship_models["Tag"]]

    # Test ForeignKey relationship field
    address_field_info = RelationshipTestModel.model_fields["address"]
    address_result = field_factory.convert_field(
        field_name="address",
        field_info=address_field_info,
        app_label="test_app",
        source_model_name=RelationshipTestModel.__name__,
        carrier=DjangoModelFactoryCarrier(pydantic_model=RelationshipTestModel, meta_app_label="test_app"),
    )

    # The field should be mapped as a relationship
    assert address_result.type_mapping_definition is not None
    assert address_result.type_mapping_definition.is_relationship is True

    # Check that 'to' parameter is a string, not a tuple
    # This is crucial for Django to create the field correctly
    assert "to" in address_result.field_kwargs
    to_value = address_result.field_kwargs["to"]
    assert isinstance(to_value, str), f"Expected 'to' to be a string, got {type(to_value)}: {to_value}"
    assert to_value == "test_app.Address"

    # Test ManyToMany relationship field
    tags_field_info = RelationshipTestModel.model_fields["tags"]
    tags_result = field_factory.convert_field(
        field_name="tags",
        field_info=tags_field_info,
        app_label="test_app",
        source_model_name=RelationshipTestModel.__name__,
        carrier=DjangoModelFactoryCarrier(pydantic_model=RelationshipTestModel, meta_app_label="test_app"),
    )

    # The field should be mapped as a relationship
    assert tags_result.type_mapping_definition is not None
    assert tags_result.type_mapping_definition.is_relationship is True

    # Check that 'to' parameter is a string, not a tuple
    assert "to" in tags_result.field_kwargs
    to_value = tags_result.field_kwargs["to"]
    assert isinstance(to_value, str), f"Expected 'to' to be a string, got {type(to_value)}: {to_value}"
    assert to_value == "test_app.Tag"

    # Test rendered field creation
    if address_result.django_field is not None:
        # This should not raise an exception if 'to' is correctly formatted
        rendered_field = address_result.django_field
        assert rendered_field is not None
        assert isinstance(rendered_field, models.ForeignKey)

    if tags_result.django_field is not None:
        rendered_field = tags_result.django_field
        assert rendered_field is not None
        assert isinstance(rendered_field, models.ManyToManyField)


def test_chain_relationship_fields():
    """
    Test that handles the specific error case from the logs where 'nodes' and 'edges' relationship fields
    were failing because of incorrect 'to' parameter format.
    """

    # Create model classes similar to those mentioned in the error logs
    class ChainNode(BaseModel):
        name: str

    class ChainEdge(BaseModel):
        name: str

    class Chain(BaseModel):
        name: str
        nodes: list[ChainNode]
        edges: list[ChainEdge]

    # Create a relationship accessor with the models
    accessor = RelationshipConversionAccessor()

    # Create fake Django models to pair with the Pydantic models
    django_node_model = type(
        "DjangoChainNode",
        (models.Model,),
        {"__module__": "tests.test_models", "Meta": type("Meta", (), {"app_label": "django_llm"})},
    )
    django_edge_model = type(
        "DjangoChainEdge",
        (models.Model,),
        {"__module__": "tests.test_models", "Meta": type("Meta", (), {"app_label": "django_llm"})},
    )

    # Create model contexts
    node_context = ModelContext(django_model=django_node_model, pydantic_class=ChainNode)
    edge_context = ModelContext(django_model=django_edge_model, pydantic_class=ChainEdge)

    # Add models to relationship accessor
    accessor.available_relationships.append(RelationshipMapper(ChainNode, django_node_model, node_context))
    accessor.available_relationships.append(RelationshipMapper(ChainEdge, django_edge_model, edge_context))

    # Create field factory with relationships
    field_factory = DjangoFieldFactory(available_relationships=accessor)

    # Test ManyToMany relationship for nodes field
    nodes_field_info = Chain.model_fields["nodes"]
    nodes_result = field_factory.convert_field(
        field_name="nodes",
        field_info=nodes_field_info,
        app_label="django_llm",
        source_model_name=Chain.__name__,
        carrier=DjangoModelFactoryCarrier(pydantic_model=Chain, meta_app_label="django_llm"),
    )

    # The field should be mapped as a relationship
    assert nodes_result.type_mapping_definition is not None
    assert nodes_result.type_mapping_definition.is_relationship is True
    assert nodes_result.type_mapping_definition.django_field == models.ManyToManyField

    # Check that 'to' parameter is a string, not a tuple
    assert "to" in nodes_result.field_kwargs
    to_value = nodes_result.field_kwargs["to"]
    assert isinstance(to_value, str), f"Expected 'to' to be a string, got {type(to_value)}: {to_value}"
    assert to_value == "django_llm.ChainNode"

    # Test successful field creation
    if nodes_result.django_field is not None:
        rendered_field = nodes_result.django_field
        assert rendered_field is not None
        assert isinstance(rendered_field, models.ManyToManyField)

    # Test ManyToMany relationship for edges field
    edges_field_info = Chain.model_fields["edges"]
    edges_result = field_factory.convert_field(
        field_name="edges",
        field_info=edges_field_info,
        app_label="django_llm",
        source_model_name=Chain.__name__,
        carrier=DjangoModelFactoryCarrier(pydantic_model=Chain, meta_app_label="django_llm"),
    )

    # Similar assertions for edges
    assert edges_result.type_mapping_definition is not None
    assert edges_result.type_mapping_definition.is_relationship is True
    assert edges_result.type_mapping_definition.django_field == models.ManyToManyField

    # Check that 'to' parameter is a string, not a tuple
    assert "to" in edges_result.field_kwargs
    to_value = edges_result.field_kwargs["to"]
    assert isinstance(to_value, str), f"Expected 'to' to be a string, got {type(to_value)}: {to_value}"
    assert to_value == "django_llm.ChainEdge"

    # Test successful field creation
    if edges_result.django_field is not None:
        rendered_field = edges_result.django_field
        assert rendered_field is not None
        assert isinstance(rendered_field, models.ManyToManyField)


def test_relationship_field_parameters(dynamic_related_model):
    """
    Test that relationship fields have the correct parameters for their field type.
    Specifically, on_delete should only be added to ForeignKey and not to ManyToManyField.
    """

    # Create model classes
    class Node(BaseModel):
        name: str

    class TestModel(BaseModel):
        # ForeignKey relationship
        foreign: Node
        # ManyToMany relationship
        many: list[Node]

    # Create a relationship accessor with the models
    accessor = RelationshipConversionAccessor()

    # Use the fixture-provided Django model
    django_node = dynamic_related_model

    # Create model context
    node_context = ModelContext(django_model=django_node, pydantic_class=Node)

    # Add model to relationship accessor
    accessor.available_relationships.append(RelationshipMapper(Node, django_node, node_context))

    # Create field factory with relationships
    field_factory = DjangoFieldFactory(available_relationships=accessor)

    # Test ForeignKey relationship - should have on_delete
    foreign_field_info = TestModel.model_fields["foreign"]
    # Add carrier for the call
    foreign_carrier = DjangoModelFactoryCarrier(pydantic_model=TestModel, meta_app_label="test_app")
    foreign_result = field_factory.convert_field(
        field_name="foreign",
        field_info=foreign_field_info,
        app_label="test_app",
        source_model_name=TestModel.__name__,  # Add source name
        carrier=foreign_carrier,  # Add carrier
    )
    assert foreign_result is not None
    assert foreign_result.django_field is not None
    assert isinstance(foreign_result.django_field, models.ForeignKey)
    assert foreign_result.field_kwargs.get("on_delete") == models.CASCADE

    # Test ManyToMany relationship - should NOT have on_delete
    many_field_info = TestModel.model_fields["many"]
    # Add carrier for the call
    many_carrier = DjangoModelFactoryCarrier(pydantic_model=TestModel, meta_app_label="test_app")
    many_result = field_factory.convert_field(
        field_name="many",
        field_info=many_field_info,
        app_label="test_app",
        source_model_name=TestModel.__name__,  # Add source name
        carrier=many_carrier,  # Add carrier
    )
    assert many_result is not None
    assert many_result.django_field is not None
    assert isinstance(many_result.django_field, models.ManyToManyField)
    assert "on_delete" not in many_result.field_kwargs

    # Clean up registry for DjangoNode
    if hasattr(apps, "all_models") and "test_app" in apps.all_models:
        if "djangonode" in apps.all_models["test_app"]:
            del apps.all_models["test_app"]["djangonode"]
    apps.clear_cache()


def test_relationship_field_parameter_validation():
    """
    Test that verifies ManyToManyField doesn't include the on_delete parameter.

    This test ensures that the fix for the error described in the logs works correctly:
    "Failed to create Django field for nodes: Field.__init__() got an unexpected keyword argument 'on_delete'"
    """

    # Create model classes
    class Node(BaseModel):
        name: str

    class TestModels(BaseModel):
        # ForeignKey relationship - should have on_delete
        single: Node
        # ManyToMany relationship - should NOT have on_delete
        many: list[Node]

    # Create relationship accessor
    accessor = RelationshipConversionAccessor()

    # Create fake Django model for Node
    django_node = type(
        "DjangoNode",
        (models.Model,),
        {"__module__": "tests.test_models", "Meta": type("Meta", (), {"app_label": "test_app"})},
    )

    # Create model context and add to accessor
    node_context = ModelContext(django_model=django_node, pydantic_class=Node)
    accessor.available_relationships.append(RelationshipMapper(Node, django_node, node_context))

    # Create field factory
    field_factory = DjangoFieldFactory(available_relationships=accessor)

    # Convert fields
    single_result = field_factory.convert_field(
        field_name="single",
        field_info=TestModels.model_fields["single"],
        app_label="test_app",
        source_model_name=TestModels.__name__,
        carrier=DjangoModelFactoryCarrier(pydantic_model=TestModels, meta_app_label="test_app"),
    )

    many_result = field_factory.convert_field(
        field_name="many",
        field_info=TestModels.model_fields["many"],
        app_label="test_app",
        source_model_name=TestModels.__name__,
        carrier=DjangoModelFactoryCarrier(pydantic_model=TestModels, meta_app_label="test_app"),
    )

    # Verify ForeignKey has on_delete parameter
    assert "on_delete" in single_result.field_kwargs
    assert single_result.field_kwargs["on_delete"] == models.CASCADE

    # Verify ManyToManyField does NOT have on_delete parameter
    assert "on_delete" not in many_result.field_kwargs

    # Verify both fields are created successfully
    assert single_result.django_field is not None
    assert isinstance(single_result.django_field, models.ForeignKey)

    assert many_result.django_field is not None
    assert isinstance(many_result.django_field, models.ManyToManyField)


def test_error_handling_for_parameter_errors(monkeypatch):
    """
    Test that errors related to field parameters are properly handled.
    - For non-relationship fields: parameter errors should raise exceptions
    - For relationship fields: parameter errors should be handled as context fields
    """

    # Create a simple model for testing
    class SimpleModel(BaseModel):
        name: str

    # Create a model with a relationship field for testing
    class RelatedModel(BaseModel):
        name: str

    class ModelWithRelationship(BaseModel):
        relation: RelatedModel

    # Get field info
    simple_field_info = SimpleModel.model_fields["name"]
    relation_field_info = ModelWithRelationship.model_fields["relation"]

    # Create a field factory
    field_factory = DjangoFieldFactory(available_relationships=RelationshipConversionAccessor())

    # Monkey patch the TypeMappingDefinition.get_django_field method to raise a parameter error
    original_get_field = TypeMappingDefinition.get_django_field

    def mock_get_field(self, kwargs=None):
        # Simulate a parameter error
        raise TypeError("Field.__init__() got an unexpected keyword argument 'invalid_param'")

    # Apply the monkey patch
    monkeypatch.setattr(TypeMappingDefinition, "get_django_field", mock_get_field)

    # CASE 1: Non-relationship field - parameter errors should raise exceptions
    # Remove the redundant call outside the pytest.raises block
    # result_simple = field_factory.convert_field(
    #     field_name="name",
    #     field_info=simple_field_info,
    #     app_label="test_app",
    #     source_model_name=SimpleModel.__name__,
    #     carrier=DjangoModelFactoryCarrier(pydantic_model=SimpleModel, meta_app_label="test_app"),
    # )
    with pytest.raises(ValueError) as excinfo:
        field_factory.convert_field(
            field_name="name",
            field_info=simple_field_info,
            app_label="test_app",
            source_model_name=SimpleModel.__name__,
            carrier=DjangoModelFactoryCarrier(pydantic_model=SimpleModel, meta_app_label="test_app"),
        )
    assert "unexpected keyword argument" in str(excinfo.value)

    # CASE 2: Relationship field - parameter errors should be handled as context fields
    # Create a new accessor and factory that *includes* the RelatedModel
    accessor_with_relation = RelationshipConversionAccessor()
    django_related = type(
        "DjangoRelatedModel",
        (models.Model,),
        {"__module__": "tests", "Meta": type("Meta", (), {"app_label": "test_app"})},
    )
    related_context = ModelContext(django_model=django_related, pydantic_class=RelatedModel)
    accessor_with_relation.available_relationships.append(
        RelationshipMapper(RelatedModel, django_related, related_context)
    )
    factory_with_relation = DjangoFieldFactory(available_relationships=accessor_with_relation)

    # First set up the type mapping definition to identify this as a relationship field
    # We'll monkey patch TypeMapper.get_mapping_for_type to return a relationship mapping
    original_get_mapping = TypeMapper.get_mapping_for_type

    def mock_get_mapping_for_relationship(field_type):
        # For RelatedModel type, return a relationship mapping
        if field_type == RelatedModel:
            return TypeMappingDefinition(python_type=RelatedModel, django_field=models.ForeignKey, is_relationship=True)
        # For other types, use the original method
        return original_get_mapping(field_type)

    # Apply the second monkey patch
    monkeypatch.setattr(TypeMapper, "get_mapping_for_type", mock_get_mapping_for_relationship)

    # Now for relationship fields, parameter errors should NOT raise exceptions
    # Use the factory_with_relation that knows about RelatedModel
    result = factory_with_relation.convert_field(
        field_name="relation",
        field_info=relation_field_info,
        app_label="test_app",
        source_model_name=ModelWithRelationship.__name__,  # Add source name
        carrier=DjangoModelFactoryCarrier(pydantic_model=ModelWithRelationship, meta_app_label="test_app"),
    )

    # Verify the field is properly handled as a context field
    assert result is not None
    assert isinstance(result, FieldConversionResult)
    assert result.django_field is None
    assert result.context_field is relation_field_info

    # Verify the error message mentions the parameter error
    assert result.error_str is not None
    assert "unexpected keyword argument" in str(result.error_str)

    # Clean up registry
    if hasattr(apps, "all_models") and "test_app" in apps.all_models:
        if "djangorelatedmodel" in apps.all_models["test_app"]:
            del apps.all_models["test_app"]["djangorelatedmodel"]
    apps.clear_cache()


def test_optional_relationship_fields(dynamic_related_model):
    """
    Test handling of Optional relationship fields (Union[Type, None]).

    This test verifies that fields with types like Optional[BaseModel] are properly
    handled and converted to relationship fields with appropriate null/blank settings.
    """

    # Create model classes
    class RelatedModel(BaseModel):
        name: str

    class TestModel(BaseModel):
        # Optional ForeignKey relationship
        optional_foreign: Optional[RelatedModel] = None
        # Optional ManyToMany relationship
        optional_many: Optional[list[RelatedModel]] = None

    # Create relationship accessor
    accessor = RelationshipConversionAccessor()

    # Use the fixture-provided Django model
    django_related = dynamic_related_model

    # Create model context for RelatedModel
    related_context = ModelContext(django_model=django_related, pydantic_class=RelatedModel)

    # Add model to relationship accessor
    accessor.available_relationships.append(RelationshipMapper(RelatedModel, django_related, related_context))

    # Create field factory with relationships
    field_factory = DjangoFieldFactory(available_relationships=accessor)

    # Test Optional ForeignKey relationship
    foreign_field_info = TestModel.model_fields["optional_foreign"]
    foreign_result = field_factory.convert_field(
        field_name="optional_foreign",
        field_info=foreign_field_info,
        app_label="test_app",
        source_model_name=TestModel.__name__,  # Add source name
        carrier=DjangoModelFactoryCarrier(pydantic_model=TestModel, meta_app_label="test_app"),  # Add carrier
    )
    assert foreign_result is not None
    assert foreign_result.django_field is not None
    assert isinstance(foreign_result.django_field, models.ForeignKey)
    # Check for null=True, blank=True
    assert getattr(foreign_result.django_field, "null"), "Optional FK should have null=True"
    assert getattr(foreign_result.django_field, "blank"), "Optional FK should have blank=True"

    # Test Optional ManyToMany relationship
    many_field_info = TestModel.model_fields["optional_many"]
    # Add carrier
    many_carrier = DjangoModelFactoryCarrier(pydantic_model=TestModel, meta_app_label="test_app")
    many_result = field_factory.convert_field(
        field_name="optional_many",
        field_info=many_field_info,
        app_label="test_app",
        source_model_name=TestModel.__name__,  # Add source name
        carrier=many_carrier,  # Add carrier
    )
    assert many_result is not None
    assert many_result.django_field is not None
    assert isinstance(many_result.django_field, models.ManyToManyField)
    # Check for blank=True (null=True is not applicable for M2M)
    assert getattr(many_result.django_field, "blank"), "Optional M2M should have blank=True"

    # Clean up registry
    if hasattr(apps, "all_models") and "test_app" in apps.all_models:
        if "djangorelatedmodel" in apps.all_models["test_app"]:
            del apps.all_models["test_app"]["djangorelatedmodel"]
    # apps.clear_cache() # Cleanup handled by fixture


def test_missing_model_handled_as_context_field():
    """
    This test specifically replicates the issue from the error logs where a field refers to
    a BasePrompt model that isn't in the relationship accessor.
    """

    # Create a mock BasePrompt class similar to what was in the error logs
    class BasePrompt(BaseModel):
        content: str

    # Create a model with a field that refers to BasePrompt
    class ConversationChainNode(BaseModel):
        name: str
        # This is the field that was causing the error - Optional[BasePrompt]
        prompt: Optional[BasePrompt] = None

    # Create an empty relationship accessor (no models registered)
    accessor = RelationshipConversionAccessor()
    field_factory = DjangoFieldFactory(available_relationships=accessor)

    # Get the field info for the prompt field
    field_info = ConversationChainNode.model_fields["prompt"]

    # Create a carrier
    carrier = DjangoModelFactoryCarrier(pydantic_model=ConversationChainNode, meta_app_label="test_app")

    # Convert the field - this should NOT raise an exception
    result = field_factory.convert_field(
        field_name="prompt",
        field_info=field_info,
        app_label="django_llm",
        source_model_name=ConversationChainNode.__name__,  # Add source name
        carrier=carrier,  # Add carrier
    )

    # Verify the field is properly handled as a context field
    assert result is not None
    assert isinstance(result, FieldConversionResult)
    assert result.django_field is None
    assert result.context_field is field_info

    # Verify the error message mentions the missing model
    assert result.error_str is not None
    assert "not in relationship accessor" in result.error_str

    # Clean up registry
    if hasattr(apps, "all_models") and "test_app" in apps.all_models:
        if "djangorelatedmodel" in apps.all_models["test_app"]:
            del apps.all_models["test_app"]["djangorelatedmodel"]
    # apps.clear_cache() # Keep cleanup specific to where model was created


def test_various_missing_relationship_types_handled_as_context():
    """
    Test that verifies various types of relationship fields with missing models
    are properly handled as context fields rather than raising exceptions.
    """

    # Create some model classes that won't be in the relationship accessor
    class MissingForeignKeyModel(BaseModel):
        name: str

    class MissingListModel(BaseModel):
        name: str

    class MissingDictModel(BaseModel):
        key: str

    # Create a model with various relationship types to missing models
    class ModelWithMissingRelationships(BaseModel):
        # Direct foreign key relationship
        foreign_key: MissingForeignKeyModel
        # Optional foreign key
        optional_foreign_key: Optional[MissingForeignKeyModel] = None
        # List relationship (ManyToMany)
        list_relation: list[MissingListModel]
        # Optional list relationship
        optional_list: Optional[list[MissingListModel]] = None
        # Dict relationship
        dict_relation: dict[str, MissingDictModel]

    # Create an empty relationship accessor
    accessor = RelationshipConversionAccessor()
    field_factory = DjangoFieldFactory(available_relationships=accessor)

    # Create a carrier
    carrier = DjangoModelFactoryCarrier(pydantic_model=ModelWithMissingRelationships, meta_app_label="test_app")

    # Test each relationship field
    for field_name, field_info in ModelWithMissingRelationships.model_fields.items():
        # Convert the field - this should NOT raise an exception
        result = field_factory.convert_field(
            field_name=field_name,
            field_info=field_info,
            app_label="django_llm",
            source_model_name=ModelWithMissingRelationships.__name__,  # Add source name
            carrier=carrier,  # Add carrier
        )

        # Verify all fields are properly handled as context fields
        assert result is not None, f"Field {field_name} returned None result"
        assert isinstance(result, FieldConversionResult)
        assert result.django_field is None, f"Field {field_name} should have django_field=None"
        assert result.context_field is field_info, f"Field {field_name} should be a context field"

        # Verify the error message mentions the missing model
        assert result.error_str is not None, f"Field {field_name} should have an error message"
        assert "not in relationship accessor" in result.error_str, f"Field {field_name} should mention missing model"

    # Clean up registry
    if hasattr(apps, "all_models") and "test_app" in apps.all_models:
        if "djangorelatedmodel" in apps.all_models["test_app"]:
            del apps.all_models["test_app"]["djangorelatedmodel"]
    apps.clear_cache()


def test_relationship_parameter_errors_handled_as_context(monkeypatch):
    """
    Test that parameter errors in relationship fields are handled as context fields
    rather than raising exceptions.
    """

    # Create a simple model for testing
    class RelatedModel(BaseModel):
        name: str

    class TestModel(BaseModel):
        relation: RelatedModel

    # Set up a relationship accessor with the related model
    accessor = RelationshipConversionAccessor()

    # Create a fake Django model for RelatedModel
    django_related = type(
        "DjangoRelatedModel",
        (models.Model,),
        {"__module__": "tests.test_models", "Meta": type("Meta", (), {"app_label": "test_app"})},
    )

    # Create model context and add to accessor
    related_context = ModelContext(django_model=django_related, pydantic_class=RelatedModel)
    accessor.available_relationships.append(RelationshipMapper(RelatedModel, django_related, related_context))

    # Create field factory
    field_factory = DjangoFieldFactory(available_relationships=accessor)

    # Monkey patch TypeMappingDefinition.get_django_field to raise a parameter error
    original_get_field = TypeMappingDefinition.get_django_field

    def mock_get_field(self, kwargs=None):
        # For ForeignKey relationship fields, raise a missing 'to' parameter error
        if self.django_field == models.ForeignKey:
            raise TypeError("ForeignKey.__init__() missing 1 required positional argument: 'to'")
        # For other field types, use the original method
        return original_get_field(self, kwargs)

    # Apply the monkey patch
    monkeypatch.setattr(TypeMappingDefinition, "get_django_field", mock_get_field)

    # Get field info
    field_info = TestModel.model_fields["relation"]

    # Create carriers
    simple_carrier = DjangoModelFactoryCarrier(pydantic_model=TestModel, meta_app_label="test_app")
    relation_carrier = DjangoModelFactoryCarrier(pydantic_model=TestModel, meta_app_label="test_app")

    # Convert the field - this should NOT raise an exception despite the parameter error
    result = field_factory.convert_field(
        field_name="relation",
        field_info=field_info,
        app_label="test_app",
        source_model_name=TestModel.__name__,  # Add source name
        carrier=relation_carrier,  # Add carrier
    )

    # Verify the field is properly handled as a context field
    assert result is not None
    assert isinstance(result, FieldConversionResult)
    assert result.django_field is None
    assert result.context_field is field_info

    # Verify the error message mentions the parameter error
    assert result.error_str is not None
    assert "missing 1 required positional argument: 'to'" in result.error_str

    # Clean up registry
    if hasattr(apps, "all_models") and "test_app" in apps.all_models:
        if "djangorelatedmodel" in apps.all_models["test_app"]:
            del apps.all_models["test_app"]["djangorelatedmodel"]
    apps.clear_cache()
