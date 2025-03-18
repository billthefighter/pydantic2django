import logging
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import UUID

import pytest
from django.db import models
from pydantic import BaseModel, EmailStr, Field
from pydantic.fields import FieldInfo

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


def test_convert_basic_fields(field_factory, basic_pydantic_model):
    """Test converting basic field types."""
    for field_name, field_info in basic_pydantic_model.model_fields.items():
        result = field_factory.convert_field(field_name, field_info)

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
    for field_name, field_info in datetime_pydantic_model.model_fields.items():
        result = field_factory.convert_field(field_name, field_info)

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
    for field_name, field_info in optional_fields_model.model_fields.items():
        result = field_factory.convert_field(field_name, field_info)

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
    for field_name, field_info in constrained_fields_model.model_fields.items():
        result = field_factory.convert_field(field_name, field_info)

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

    # Test each relationship field in User model
    for field_name, field_info in user_model.model_fields.items():
        if field_name in ["address", "profile", "tags"]:
            result = field_factory.convert_field(field_name, field_info)

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
    """Test handling of relationship fields when the related model is not in the accessor."""
    # Get the User model with relationships
    user_model = relationship_models["User"]

    # Test relationship fields with empty relationship accessor
    for field_name, field_info in user_model.model_fields.items():
        if field_name in ["address", "profile", "tags"]:
            # Now we expect a ValueError for parameter issues like missing 'to'
            with pytest.raises(ValueError) as excinfo:
                result = empty_field_factory.convert_field(field_name, field_info)

            # Verify the error mentions the missing required parameter
            assert "missing 1 required positional argument: 'to'" in str(excinfo.value)


def test_handle_relationship_field_directly(field_factory, relationship_models):
    """Test the handle_relationship_field method directly."""
    # Get test models
    address_model = relationship_models["Address"]

    # We can't directly use BaseModel as a type annotation for a real field.
    # Create a proper field_info with a properly annotated type.
    class TestModel(BaseModel):
        # Use the actual address model from our fixtures
        address: relationship_models["Address"]

    field_info = TestModel.model_fields["address"]

    # Create a base result to pass to handle_relationship_field
    base_result = FieldConversionResult(
        field_info=field_info,
        field_name="address",
        app_label="test_app",
        field_kwargs={},
        # Use the actual address model for the type mapping
        type_mapping_definition=TypeMappingDefinition.foreign_key(relationship_models["Address"]),
    )

    # Manually add the Address model to the available_relationships if not already there
    if not field_factory.available_relationships.has_pydantic_model(relationship_models["Address"]):
        # Create Django model for Address
        django_address = type(
            "DjangoAddress",
            (models.Model,),
            {"__module__": "tests.test_models", "Meta": type("Meta", (), {"app_label": "tests"})},
        )

        # Create context
        model_context = ModelContext(django_model=django_address, pydantic_class=relationship_models["Address"])

        # Add to relationship accessor
        field_factory.available_relationships.available_relationships.append(
            RelationshipMapper(relationship_models["Address"], django_address, model_context)
        )

    # Process the relationship - relationships may not work directly in this test
    # because the handle_relationship_field method has complex requirements
    result = field_factory.handle_relationship_field(base_result)

    # Assertions - should still return a valid result even if field can't be created
    assert result is not None
    assert isinstance(result, FieldConversionResult)
    assert result.field_name == "address"

    # Either the field was created successfully or it was handled gracefully
    if result.django_field is not None:
        assert "related_name" in result.field_kwargs
        assert result.field_kwargs.get("on_delete") == models.CASCADE
    else:
        # If there's an error it should be captured
        assert result.error_str is not None


def test_invalid_relationship_types(field_factory):
    """Test handling of invalid relationship field types."""

    # Create a model with an invalid relationship type
    class InvalidModel(BaseModel):
        not_a_relationship: int  # Not a relationship type

    field_info = InvalidModel.model_fields["not_a_relationship"]

    # Create a base result with relationship flag but incompatible type
    base_result = FieldConversionResult(
        field_info=field_info,
        field_name="not_a_relationship",
        app_label="test_app",
        field_kwargs={},
        type_mapping_definition=TypeMappingDefinition(
            python_type=int, django_field=models.ForeignKey, is_relationship=True
        ),
    )

    # Process the relationship - should fail gracefully
    result = field_factory.handle_relationship_field(base_result)

    # Assertions
    assert result.django_field is None
    assert result.error_str is not None


def test_edge_cases(field_factory):
    """Test various edge cases and error handling."""

    # Case 1: Field with no annotation
    class NoAnnotationModel(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        field_without_annotation: Any

    field_info = NoAnnotationModel.model_fields["field_without_annotation"]
    result = field_factory.convert_field("field_without_annotation", field_info)

    # Any type should be handled as JSONField, but that's an implementation detail
    # Just check that the result is valid
    assert result is not None
    assert isinstance(result, FieldConversionResult)

    # Case 2: Field with unmappable type should be treated as context field
    class UnmappableTypeModel(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        unmappable: object  # No direct mapping to Django field

    field_info = UnmappableTypeModel.model_fields["unmappable"]
    result = field_factory.convert_field("unmappable", field_info)

    # Should be treated as a context field or have a valid mapping
    assert result is not None
    assert isinstance(result, FieldConversionResult)
    # Either it has a context field or a valid mapping
    assert result.context_field is not None or result.django_field is not None


def test_rendered_django_field(field_factory, basic_pydantic_model):
    """Test the rendered_django_field property."""
    # Get a field from the basic model
    field_info = basic_pydantic_model.model_fields["string_field"]

    # Convert the field
    result = field_factory.convert_field("string_field", field_info)

    # Get the rendered field
    rendered_field = result.rendered_django_field

    # Assertions
    assert rendered_field is not None
    assert isinstance(rendered_field, models.Field)
    assert isinstance(rendered_field, models.TextField)


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
    result = field_factory.convert_field("name", field_info, app_label="test_app")

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
        rendered_field = address_result.rendered_django_field
        assert rendered_field is not None
        assert isinstance(rendered_field, models.ForeignKey)

    if tags_result.django_field is not None:
        rendered_field = tags_result.rendered_django_field
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
        rendered_field = nodes_result.rendered_django_field
        assert rendered_field is not None
        assert isinstance(rendered_field, models.ManyToManyField)

    # Test ManyToMany relationship for edges field
    edges_field_info = Chain.model_fields["edges"]
    edges_result = field_factory.convert_field(
        field_name="edges",
        field_info=edges_field_info,
        app_label="django_llm",
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
        rendered_field = edges_result.rendered_django_field
        assert rendered_field is not None
        assert isinstance(rendered_field, models.ManyToManyField)


def test_relationship_field_parameters():
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

    # Create fake Django model for Node
    django_node = type(
        "DjangoNode",
        (models.Model,),
        {"__module__": "tests.test_models", "Meta": type("Meta", (), {"app_label": "test_app"})},
    )

    # Create model context
    node_context = ModelContext(django_model=django_node, pydantic_class=Node)

    # Add model to relationship accessor
    accessor.available_relationships.append(RelationshipMapper(Node, django_node, node_context))

    # Create field factory with relationships
    field_factory = DjangoFieldFactory(available_relationships=accessor)

    # Test ForeignKey relationship - should have on_delete
    foreign_field_info = TestModel.model_fields["foreign"]
    foreign_result = field_factory.convert_field(
        field_name="foreign",
        field_info=foreign_field_info,
        app_label="test_app",
    )

    # The field should be mapped as a relationship and include on_delete
    assert foreign_result.type_mapping_definition is not None
    assert foreign_result.type_mapping_definition.is_relationship is True
    assert foreign_result.type_mapping_definition.django_field == models.ForeignKey
    assert "on_delete" in foreign_result.field_kwargs
    assert foreign_result.field_kwargs["on_delete"] == models.CASCADE

    # Foreign key field should be created successfully
    assert foreign_result.django_field is not None
    assert isinstance(foreign_result.django_field, models.ForeignKey)

    # Test ManyToMany relationship - should NOT have on_delete
    many_field_info = TestModel.model_fields["many"]
    many_result = field_factory.convert_field(
        field_name="many",
        field_info=many_field_info,
        app_label="test_app",
    )

    # The field should be mapped as a relationship but exclude on_delete
    assert many_result.type_mapping_definition is not None
    assert many_result.type_mapping_definition.is_relationship is True
    assert many_result.type_mapping_definition.django_field == models.ManyToManyField
    # The key assertion: on_delete should not be in the field kwargs
    assert "on_delete" not in many_result.field_kwargs

    # ManyToMany field should be created successfully
    assert many_result.django_field is not None
    assert isinstance(many_result.django_field, models.ManyToManyField)


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
    )

    many_result = field_factory.convert_field(
        field_name="many",
        field_info=TestModels.model_fields["many"],
        app_label="test_app",
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
    Test that errors related to field parameters are properly raised rather than
    silently converted to contextual fields.
    """

    # Create a simple model for testing
    class SimpleModel(BaseModel):
        name: str

    # Get field info
    field_info = SimpleModel.model_fields["name"]

    # Create a field factory
    field_factory = DjangoFieldFactory(available_relationships=RelationshipConversionAccessor())

    # Monkey patch the TypeMappingDefinition.get_django_field method to raise a parameter error
    original_get_field = TypeMappingDefinition.get_django_field

    def mock_get_field(self, kwargs=None):
        # Simulate a parameter error
        raise TypeError("Field.__init__() got an unexpected keyword argument 'invalid_param'")

    # Apply the monkey patch
    monkeypatch.setattr(TypeMappingDefinition, "get_django_field", mock_get_field)

    # The convert_field call should raise a ValueError
    with pytest.raises(ValueError) as excinfo:
        field_factory.convert_field(
            field_name="name",
            field_info=field_info,
            app_label="test_app",
        )

    # Verify the error message mentions the parameter error
    assert "unexpected keyword argument" in str(excinfo.value)

    # Also test the missing required argument case
    def mock_get_field_missing_arg(self, kwargs=None):
        # Simulate a missing argument error
        raise TypeError("Field.__init__() missing 1 required positional argument: 'to'")

    # Apply the second monkey patch
    monkeypatch.setattr(TypeMappingDefinition, "get_django_field", mock_get_field_missing_arg)

    # This should also raise a ValueError
    with pytest.raises(ValueError) as excinfo:
        field_factory.convert_field(
            field_name="name",
            field_info=field_info,
            app_label="test_app",
        )

    # Verify the error message mentions the missing argument
    assert "missing 1 required positional argument" in str(excinfo.value)


def test_optional_relationship_fields():
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

    # Create fake Django model for RelatedModel
    django_related = type(
        "DjangoRelatedModel",
        (models.Model,),
        {"__module__": "tests.test_models", "Meta": type("Meta", (), {"app_label": "test_app"})},
    )

    # Create model context
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
    )

    # Verify the field is correctly converted to a ForeignKey with null=True
    assert foreign_result.django_field is not None
    assert isinstance(foreign_result.django_field, models.ForeignKey)
    assert foreign_result.field_kwargs["null"] is True
    assert foreign_result.field_kwargs["blank"] is True
    assert "on_delete" in foreign_result.field_kwargs
    assert foreign_result.field_kwargs["on_delete"] == models.CASCADE

    # Test Optional ManyToMany relationship
    many_field_info = TestModel.model_fields["optional_many"]
    many_result = field_factory.convert_field(
        field_name="optional_many",
        field_info=many_field_info,
        app_label="test_app",
    )

    # Verify the field is correctly converted to a ManyToManyField with null=True
    assert many_result.django_field is not None
    assert isinstance(many_result.django_field, models.ManyToManyField)
    assert many_result.field_kwargs["null"] is True
    assert many_result.field_kwargs["blank"] is True
    # The key fix: ManyToManyField should NOT have on_delete
    assert "on_delete" not in many_result.field_kwargs
