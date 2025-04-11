"""Tests for the core bidirectional mapper."""
import datetime
import logging
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Type, Union, Dict, Literal, Annotated
from uuid import UUID

import pytest
from django.db import models
from pydantic import BaseModel, EmailStr, Field, HttpUrl, IPvAnyAddress
from pydantic.fields import FieldInfo

# Classes to test
from pydantic2django.core.bidirectional_mapper import (
    BidirectionalTypeMapper,
    MappingError,
    TypeMappingUnit,  # Import base class for potential checks
    # Import specific units if needed for comparison, but mostly test via mapper
)
from pydantic2django.core.relationships import RelationshipConversionAccessor, RelationshipMapper


# --- Test Setup ---

logger = logging.getLogger(__name__)


# Define simple Django & Pydantic models for relationship testing
class RelatedDjangoModel(models.Model):
    name = models.CharField(max_length=50)

    class Meta:
        app_label = "test_app"


class TargetDjangoModel(models.Model):
    field1 = models.CharField(max_length=100)
    related_fk = models.ForeignKey(RelatedDjangoModel, on_delete=models.CASCADE, null=True, blank=True)
    related_o2o = models.OneToOneField(RelatedDjangoModel, on_delete=models.PROTECT, related_name="o2o_target")
    related_m2m = models.ManyToManyField(RelatedDjangoModel, related_name="m2m_targets")
    self_ref_fk = models.ForeignKey("self", on_delete=models.SET_NULL, null=True, blank=True)
    id_pk_int = models.AutoField(primary_key=True)
    uuid_pk = models.UUIDField(
        primary_key=True, default=UUID("a3a2a1a0-9b8c-7d6e-5f4a-3b2c1d0e9f8a")
    )  # Example non-int PK

    class Meta:
        app_label = "test_app"


class RelatedPydanticModel(BaseModel):
    id: int  # Assuming related model maps ID
    name: str


class TargetPydanticModel(BaseModel):
    field1: str
    related_fk: Optional[RelatedPydanticModel] = None
    related_o2o: RelatedPydanticModel
    related_m2m: List[RelatedPydanticModel] = Field(default_factory=list)
    self_ref_fk: Optional["TargetPydanticModel"] = None  # ForwardRef style
    id_pk_int: Optional[int] = Field(default=None, frozen=True)  # Auto PKs are optional and frozen
    uuid_pk: UUID


# Update ForwardRefs
TargetPydanticModel.model_rebuild()


# Add simple models for Union tests
class UnionModelA(BaseModel):
    name: str
    value_a: int


class UnionModelB(BaseModel):
    name: str
    value_b: bool


# Add dummy models for specific test cases
class TextPartDelta(BaseModel):
    text: str


class ToolCallPartDelta(BaseModel):
    tool_call_id: str
    function_name: str


class ToolCallPart(BaseModel):
    id: str
    type: Literal["function"] = "function"
    # function: FunctionCall # Assume FunctionCall is another model or complex type
    # For testing purposes, let's keep it simple
    function_name: str
    function_args: Dict[str, Any]


# --- Fixtures ---


@pytest.fixture(scope="module")
def relationship_accessor() -> RelationshipConversionAccessor:
    """Provides a RelationshipConversionAccessor with test models mapped."""
    accessor = RelationshipConversionAccessor()
    accessor.map_relationship(RelatedPydanticModel, RelatedDjangoModel)
    accessor.map_relationship(TargetPydanticModel, TargetDjangoModel)

    # Map the new union models (assuming dummy Django models exist or mapping doesn't strictly require them yet)
    # We might need dummy Django models if the accessor strictly validates
    class DummyDjangoUnionA(models.Model):
        class Meta:
            app_label = "test_app"  # type: ignore

    class DummyDjangoUnionB(models.Model):
        class Meta:
            app_label = "test_app"  # type: ignore

    # Define dummy Django models for the new test cases
    class DummyDjangoTextPartDelta(models.Model):
        class Meta:
            app_label = "test_app_delta"  # type: ignore

    class DummyDjangoToolCallPartDelta(models.Model):
        class Meta:
            app_label = "test_app_delta"  # type: ignore

    class DummyDjangoToolCallPart(models.Model):
        class Meta:
            app_label = "test_app_tool"  # type: ignore

    accessor.map_relationship(UnionModelA, DummyDjangoUnionA)
    accessor.map_relationship(UnionModelB, DummyDjangoUnionB)
    # Map the new dummy models
    accessor.map_relationship(TextPartDelta, DummyDjangoTextPartDelta)
    accessor.map_relationship(ToolCallPartDelta, DummyDjangoToolCallPartDelta)
    accessor.map_relationship(ToolCallPart, DummyDjangoToolCallPart)

    return accessor


@pytest.fixture(scope="module")
def mapper(relationship_accessor: RelationshipConversionAccessor) -> BidirectionalTypeMapper:
    """Provides an instance of the BidirectionalTypeMapper."""
    return BidirectionalTypeMapper(relationship_accessor=relationship_accessor)


# --- Test Parameter Dataclasses ---


@dataclass
class PydToDjParams:
    """Parameters for testing Pydantic type -> Django field mapping."""

    test_id: str
    python_type: Any
    # Non-default fields first
    expected_dj_type: Type[models.Field]
    expected_kwargs: Dict[str, Any] = field(default_factory=dict)  # Default factory
    # Optional/Defaulted fields last
    field_info: Optional[FieldInfo] = None
    raises_error: Optional[Type[Exception]] = None


@dataclass
class DjToPydParams:
    """Parameters for testing Django field -> Pydantic type mapping."""

    test_id: str
    django_field_instance: models.Field
    # Expected results
    expected_py_type: Any
    expected_field_info_kwargs: Dict[str, Any] = field(default_factory=dict)
    raises_error: Optional[Type[Exception]] = None


# --- Test Cases ---

# Pydantic -> Django Tests
# ------------------------

PYD_TO_DJ_SIMPLE_CASES = [
    # Basic Types
    PydToDjParams("int_to_int", int, models.IntegerField, {"null": False, "blank": False}),
    PydToDjParams("str_to_char", str, models.TextField, {"null": False, "blank": False}),
    PydToDjParams("bool_to_bool", bool, models.BooleanField, {"default": False, "null": False, "blank": False}),
    PydToDjParams("float_to_float", float, models.FloatField, {"null": False, "blank": False}),
    PydToDjParams("bytes_to_binary", bytes, models.BinaryField, {"null": False, "blank": False}),
    # Date/Time
    PydToDjParams("datetime_to_datetime", datetime.datetime, models.DateTimeField, {"null": False, "blank": False}),
    PydToDjParams("date_to_date", datetime.date, models.DateField, {"null": False, "blank": False}),
    PydToDjParams("time_to_time", datetime.time, models.TimeField, {"null": False, "blank": False}),
    PydToDjParams("timedelta_to_duration", datetime.timedelta, models.DurationField, {"null": False, "blank": False}),
    # Other Simple Types
    PydToDjParams("uuid_to_uuid", UUID, models.UUIDField, {"null": False, "blank": False}),
    PydToDjParams("path_to_filepath", Path, models.FilePathField, {"max_length": 100, "null": False, "blank": False}),
    # Pydantic Specific Types
    PydToDjParams("emailstr_to_email", EmailStr, models.EmailField, {"max_length": 254, "null": False, "blank": False}),
    PydToDjParams("httpurl_to_url", HttpUrl, models.URLField, {"max_length": 200, "null": False, "blank": False}),
    PydToDjParams("ipvany_to_generic_ip", IPvAnyAddress, models.GenericIPAddressField, {"null": False, "blank": False}),
    # Collections -> JSONField
    PydToDjParams("dict_to_json", dict, models.JSONField, {"null": False, "blank": False}),
    PydToDjParams("list_to_json", list, models.JSONField, {"null": False, "blank": False}),
    PydToDjParams("set_to_json", set, models.JSONField, {"null": False, "blank": False}),
    PydToDjParams("tuple_to_json", tuple, models.JSONField, {"null": False, "blank": False}),
    PydToDjParams("any_to_json", Any, models.JSONField, {"null": False, "blank": False}),
]

PYD_TO_DJ_OPTIONAL_CASES = [
    PydToDjParams("optional_int", Optional[int], models.IntegerField, {"null": True, "blank": True}),
    PydToDjParams("optional_str", Optional[str], models.TextField, {"null": True, "blank": True}),
    PydToDjParams("union_int_none", Union[int, None], models.IntegerField, {"null": True, "blank": True}),
]

PYD_TO_DJ_LITERAL_CASES = [
    # Literal string -> CharField with choices
    PydToDjParams(
        "literal_str_to_char_choices",
        Literal["alpha", "beta", "gamma"],
        models.CharField,
        {
            "max_length": 5,  # Longest option is "gamma"
            "choices": [("alpha", "alpha"), ("beta", "beta"), ("gamma", "gamma")],
            # Null/blank determined by Optional status, not part of Literal itself
            "null": False,
            "blank": False,
        },
    ),
    # Optional Literal string -> CharField with choices, null=True
    PydToDjParams(
        "optional_literal_str",
        Optional[Literal["on", "off"]],
        models.CharField,
        {
            "max_length": 3,  # Longest is "off"
            "choices": [("on", "on"), ("off", "off")],
            "null": True,
            "blank": True,
        },
    ),
    # NOTE: Literal with mixed types or non-string types might map differently (e.g., IntEnum vs Literal[1,2])
    # Add tests for those if needed.
]

PYD_TO_DJ_CONSTRAINT_CASES = [
    # Max Length - Corrected FieldInfo instantiation
    PydToDjParams(
        "str_with_max_length",
        str,
        models.CharField,
        {"max_length": 100, "null": False, "blank": False},
        field_info=FieldInfo(annotation=str, max_length=100),
    ),
    # This test name is now misleading, but keep ID for consistency.
    # Expect TextField based on current default preference for str.
    PydToDjParams(
        "str_to_textfield_no_max",
        str,
        models.TextField,  # Changed expected type
        {"null": False, "blank": False},  # Removed max_length expectation
        field_info=FieldInfo(annotation=str),
    ),
    # TODO: Need way to hint TextField? Or specific class? Assuming TextField needs separate handling/override.
    # Decimal - Corrected FieldInfo instantiation
    PydToDjParams(
        "decimal_default",
        Decimal,
        models.DecimalField,
        {"max_digits": 10, "decimal_places": 2, "null": False, "blank": False},
    ),
    PydToDjParams(
        "decimal_with_constraints",
        Decimal,
        models.DecimalField,
        {"max_digits": 10, "decimal_places": 4, "null": False, "blank": False},
        field_info=FieldInfo(annotation=Decimal, max_digits=10, decimal_places=4),
    ),
    # Defaults
    PydToDjParams(
        "int_with_default",
        int,
        models.IntegerField,
        {"default": 42, "null": False, "blank": False},
        field_info=FieldInfo(annotation=int, default=42),
    ),
    # str with default, no max_length -> TextField
    PydToDjParams(
        "str_with_default",
        str,
        models.TextField,  # Changed expected type
        {"default": "abc", "null": False, "blank": False},  # Removed max_length expectation
        field_info=FieldInfo(annotation=str, default="abc"),
    ),
    PydToDjParams(
        "optional_int_with_default",
        Optional[int],
        models.IntegerField,
        {"null": True, "blank": True, "default": 10},
        field_info=FieldInfo(annotation=Optional[int], default=10),
    ),
    PydToDjParams(
        "optional_int_with_none_default",
        Optional[int],
        models.IntegerField,
        {"null": True, "blank": True, "default": None},
        field_info=FieldInfo(annotation=Optional[int], default=None),
    ),
    PydToDjParams(
        "int_with_factory",
        int,
        models.IntegerField,
        {"null": False, "blank": False},
        field_info=FieldInfo(annotation=int, default_factory=lambda: 1),
    ),
    # Title/Description - str with no max_length -> TextField
    PydToDjParams(
        "str_with_title_desc",
        str,
        models.TextField,  # Changed expected type
        {
            "verbose_name": "Field Title",
            "help_text": "Helpful text",
            "null": False,
            "blank": False,
        },  # Removed max_length expectation
        field_info=FieldInfo(annotation=str, title="Field Title", description="Helpful text"),
    ),
    # Str -> TextField (no max_length) - This test now aligns with default behavior
    PydToDjParams(
        "str_to_textfield_implicit",
        str,
        models.TextField,  # Expect TextField (consistent with default)
        {"null": False, "blank": False},
        field_info=FieldInfo(annotation=str),  # No max_length specified
    ),
    # URL with max_length
    PydToDjParams(
        "url_with_max_length",
        HttpUrl,
        models.URLField,
        {"max_length": 500, "null": False, "blank": False},
        field_info=FieldInfo(annotation=HttpUrl, max_length=500),
    ),
    # FilePath with max_length
    PydToDjParams(
        "filepath_with_max_length",
        Path,
        models.FilePathField,
        {"max_length": 150, "null": False, "blank": False},
        field_info=FieldInfo(annotation=Path, max_length=150),
    ),
    # Bool with default=True
    PydToDjParams(
        "bool_with_true_default",
        bool,
        models.BooleanField,
        {"default": True, "null": False, "blank": False},
        field_info=FieldInfo(annotation=bool, default=True),
    ),
    # Optional Bool with default=None (should behave same as just Optional[bool])
    PydToDjParams(
        "optional_bool_with_none_default",
        Optional[bool],
        models.BooleanField,
        {"default": None, "null": True, "blank": True},  # Explicit default=None
        field_info=FieldInfo(annotation=Optional[bool], default=None),
    ),
    # Positive Int constraint (ge=0)
    PydToDjParams(
        "int_with_ge0_constraint",
        int,
        models.PositiveIntegerField,  # Expect PositiveIntegerField
        {"null": False, "blank": False},
        field_info=FieldInfo(annotation=int, ge=0),  # Note: ge=0 is a validator, not direct field kwarg
    ),
]


PYD_TO_DJ_DEFAULT_FACTORY_CASES = [
    PydToDjParams(
        "uuid_default_factory",
        UUID,
        models.UUIDField,
        {"null": False, "blank": False},  # No 'default' expected
        field_info=FieldInfo(annotation=UUID, default_factory=UUID),
    ),
    PydToDjParams(
        "datetime_default_factory",
        datetime.datetime,
        models.DateTimeField,
        {"null": False, "blank": False},  # No 'default' expected
        field_info=FieldInfo(annotation=datetime.datetime, default_factory=datetime.datetime.now),
    ),
    PydToDjParams(
        "list_default_factory",
        list[int],
        models.JSONField,
        {"null": False, "blank": False},  # No 'default' expected
        field_info=FieldInfo(annotation=list[int], default_factory=list),
    ),
]


class ColorEnum(Enum):
    RED = "r"
    GREEN = "g"
    BLUE = "b"


class IntEnum(Enum):
    ONE = 1
    TWO = 2


PYD_TO_DJ_ENUM_CASES = [
    PydToDjParams(
        "str_enum",
        ColorEnum,
        models.CharField,
        {"max_length": 1, "choices": [("r", "RED"), ("g", "GREEN"), ("b", "BLUE")], "null": False, "blank": False},
    ),
    PydToDjParams(
        "int_enum", IntEnum, models.IntegerField, {"choices": [(1, "ONE"), (2, "TWO")], "null": False, "blank": False}
    ),
]

PYD_TO_DJ_RELATIONSHIP_CASES = [
    # Simple Pydantic Model -> Django ForeignKey
    PydToDjParams(
        "fk_simple",
        RelatedPydanticModel,
        models.ForeignKey,
        {"to": "test_app.relateddjangomodel", "on_delete": models.CASCADE, "null": False, "blank": False},
        field_info=FieldInfo(annotation=RelatedPydanticModel),  # Add FieldInfo
    ),
    # Optional Pydantic Model -> Django ForeignKey (nullable)
    PydToDjParams(
        "fk_optional",
        Optional[RelatedPydanticModel],
        models.ForeignKey,
        {"to": "test_app.relateddjangomodel", "on_delete": models.SET_NULL, "null": True, "blank": True},
        field_info=FieldInfo(annotation=Optional[RelatedPydanticModel]),  # Add FieldInfo
    ),
    # List of Pydantic Models -> Django ManyToManyField
    PydToDjParams(
        "m2m_list",
        list[RelatedPydanticModel],
        models.ManyToManyField,
        {"to": "test_app.relateddjangomodel", "blank": True},  # blank=True often default for M2M
        field_info=FieldInfo(annotation=list[RelatedPydanticModel]),  # Add FieldInfo
    ),
    # Optional List of Pydantic Models -> Django ManyToManyField (blank=True)
    PydToDjParams(
        "m2m_optional_list",
        Optional[list[RelatedPydanticModel]],
        models.ManyToManyField,
        {"to": "test_app.relateddjangomodel", "blank": True},
        field_info=FieldInfo(annotation=Optional[list[RelatedPydanticModel]]),  # Add FieldInfo
    ),
    # Simple Pydantic Model (intended as O2O) -> Currently maps to ForeignKey
    # TODO: Update this test when O2O detection is implemented
    PydToDjParams(
        "o2o_simple_as_fk",
        RelatedPydanticModel,
        models.ForeignKey,  # Currently maps to FK, update if O2O logic added
        {"to": "test_app.relateddjangomodel", "on_delete": models.CASCADE, "null": False, "blank": False},
        # Update expected type if needed
        # field_info=FieldInfo(annotation=RelatedPydanticModel, json_schema_extra={"relation_type": "one_to_one"})
    ),
    # Optional Pydantic Model (intended as O2O) -> Currently maps to ForeignKey (nullable)
    PydToDjParams(
        "o2o_optional_as_fk",
        Optional[RelatedPydanticModel],
        models.ForeignKey,  # Currently maps to FK, update if O2O logic added
        {"to": "test_app.relateddjangomodel", "on_delete": models.SET_NULL, "null": True, "blank": True},
        # Update expected type if needed
        # field_info=FieldInfo(annotation=Optional[RelatedPydanticModel], json_schema_extra={"relation_type": "one_to_one"})
    ),
    # Self-referencing ForeignKey (Optional)
    PydToDjParams(
        "self_ref_fk",
        Optional[TargetPydanticModel],
        models.ForeignKey,
        {"to": "self", "on_delete": models.SET_NULL, "null": True, "blank": True},
        field_info=FieldInfo(annotation=Optional[TargetPydanticModel]),  # Add FieldInfo
    ),
]

# Test cases for Complex Union Types
PYD_TO_DJ_UNION_CASES = [
    # Union of BaseModels -> Expect placeholder + signal for multi-FK generation
    PydToDjParams(
        "union_model_a_b",
        Union[UnionModelA, UnionModelB],
        models.JSONField,  # Placeholder type
        {
            "_union_details": {
                "type": "multi_fk",
                "models": [UnionModelA, UnionModelB],
                "is_optional": False,
            },
            "null": True,  # Base nullability (optional applied later)
            "blank": True,
        },
        field_info=FieldInfo(annotation=Union[UnionModelA, UnionModelB]),  # Add FieldInfo
    ),
    PydToDjParams(
        "optional_union_model_a_b",
        Optional[Union[UnionModelA, UnionModelB]],
        models.JSONField,  # Placeholder type
        {
            "_union_details": {
                "type": "multi_fk",
                "models": [UnionModelA, UnionModelB],
                "is_optional": True,
            },
            "null": True,  # Optionality reflected
            "blank": True,
        },
        field_info=FieldInfo(annotation=Optional[Union[UnionModelA, UnionModelB]]),  # Add FieldInfo
    ),
    # Union involving Literal and str -> Expect TextField (str usually dominates)
    PydToDjParams(
        "union_literal_str",
        Union[Literal["A", "B"], str],
        models.TextField,  # Expect TextField as str is present
        {"null": False, "blank": False},
        field_info=FieldInfo(annotation=str),  # Use str as annotation, Union passed to mapper
    ),
    # Union involving List and str -> Expect JSONField (common fallback)
    PydToDjParams(
        "union_list_str",
        Union[List[int], str],
        models.JSONField,  # Expect JSONField as it handles lists and mixed types
        {"null": False, "blank": False},
        field_info=FieldInfo(annotation=Any),  # Use Any for list/str union annotation
    ),
    # Union involving Any and str -> Expect TextField? Or JSON? Let's expect JSON for now.
    PydToDjParams(
        "union_any_str",
        Union[Any, str],
        models.JSONField,  # JSONField is safer for Any
        {"null": False, "blank": False},
        field_info=FieldInfo(annotation=Any),  # Use Any for Any/str union annotation
    ),
    # -- Specific cases from user issue --
    # Union[TextPartDelta, ToolCallPartDelta] -> Multi-FK
    PydToDjParams(
        "union_text_tool_deltas",
        Union[TextPartDelta, ToolCallPartDelta],
        models.JSONField,  # Placeholder
        {
            "_union_details": {
                "type": "multi_fk",
                "models": [TextPartDelta, ToolCallPartDelta],
                "is_optional": False,
            },
            "null": True,  # Always nullable for multi-fk
            "blank": True,
        },
        field_info=FieldInfo(annotation=Union[TextPartDelta, ToolCallPartDelta]),
    ),
    # Optional Union[TextPartDelta, ToolCallPartDelta] -> Multi-FK (Optional)
    PydToDjParams(
        "optional_union_text_tool_deltas",
        Optional[Union[TextPartDelta, ToolCallPartDelta]],
        models.JSONField,  # Placeholder
        {
            "_union_details": {
                "type": "multi_fk",
                "models": [TextPartDelta, ToolCallPartDelta],
                "is_optional": True,
            },
            "null": True,  # Always nullable for multi-fk
            "blank": True,
        },
        field_info=FieldInfo(annotation=Optional[Union[TextPartDelta, ToolCallPartDelta]]),
    ),
    # str | dict[str, Any] -> JSONField (using UnionType syntax)
    PydToDjParams(
        "union_str_dict_uniontype",
        str | dict[str, Any],  # Use | syntax
        models.JSONField,
        {"null": False, "blank": False},
        field_info=FieldInfo(annotation=Any),  # Annotate as Any
    ),
    # str | list[str] -> JSONField (using UnionType syntax)
    PydToDjParams(
        "union_str_list_uniontype",
        str | list[str],  # Use | syntax
        models.JSONField,
        {"null": False, "blank": False},
        field_info=FieldInfo(annotation=Any),  # Annotate as Any
    ),
    # str | Literal[...] -> TextField (using UnionType syntax)
    PydToDjParams(
        "union_str_literal_uniontype",
        str | Literal["Opt1", "Opt2"],  # Use | syntax
        models.TextField,
        {"null": False, "blank": False},
        field_info=FieldInfo(annotation=str),  # Annotate as str
    ),
]

PYD_TO_DJ_ANNOTATED_CASES = [
    # Annotated Union of BaseModels -> Expect placeholder + signal for multi-FK generation
    PydToDjParams(
        "annotated_union_model_a_b",
        Annotated[Union[UnionModelA, UnionModelB], "SomeMetadata"],  # Using simple metadata
        models.JSONField,  # Placeholder type
        {
            "_union_details": {
                "type": "multi_fk",
                "models": [UnionModelA, UnionModelB],
                "is_optional": False,
            },
            "null": True,  # Base nullability for multi-FK
            "blank": True,
        },
        # FieldInfo not strictly needed here as type itself is complex
        # field_info=FieldInfo(annotation=Annotated[Union[UnionModelA, UnionModelB], "SomeMetadata"])
    ),
    # Optional Annotated Union
    PydToDjParams(
        "optional_annotated_union_model_a_b",
        Optional[Annotated[Union[UnionModelA, UnionModelB], "SomeMetadata"]],
        models.JSONField,  # Placeholder type
        {
            "_union_details": {
                "type": "multi_fk",
                "models": [UnionModelA, UnionModelB],
                "is_optional": True,  # Reflects the Optional wrapper
            },
            "null": True,  # Always nullable for multi-FK
            "blank": True,
        },
    ),
]

# --- Test Functions ---


@pytest.mark.parametrize(
    "params", PYD_TO_DJ_SIMPLE_CASES + PYD_TO_DJ_OPTIONAL_CASES + PYD_TO_DJ_LITERAL_CASES, ids=lambda p: p.test_id
)
def test_get_django_mapping_simple_optional_literal(mapper: BidirectionalTypeMapper, params: PydToDjParams):
    """Tests mapping simple, Optional, and Literal Python/Pydantic types to Django fields."""
    logger.debug(f"Testing: {params.test_id}")
    if params.raises_error:
        with pytest.raises(params.raises_error):
            mapper.get_django_mapping(params.python_type, params.field_info)
    else:
        dj_type, dj_kwargs = mapper.get_django_mapping(params.python_type, params.field_info)
        assert dj_type is params.expected_dj_type, f"Expected {params.expected_dj_type}, got {dj_type}"
        # Check kwargs equality - ignore related_name for simplicity here
        dj_kwargs.pop("related_name", None)
        params.expected_kwargs.pop("related_name", None)
        assert dj_kwargs == params.expected_kwargs, f"Expected {params.expected_kwargs}, got {dj_kwargs}"


@pytest.mark.parametrize("params", PYD_TO_DJ_CONSTRAINT_CASES, ids=lambda p: p.test_id)
def test_get_django_mapping_constraints(mapper: BidirectionalTypeMapper, params: PydToDjParams):
    """Tests mapping Pydantic field constraints (max_length, decimal places, default) to Django kwargs."""
    logger.debug(f"Testing: {params.test_id}")
    dj_type, dj_kwargs = mapper.get_django_mapping(params.python_type, params.field_info)
    assert dj_type is params.expected_dj_type
    assert dj_kwargs == params.expected_kwargs


@pytest.mark.parametrize("params", PYD_TO_DJ_ENUM_CASES, ids=lambda p: p.test_id)
def test_get_django_mapping_enums(mapper: BidirectionalTypeMapper, params: PydToDjParams):
    """Tests mapping Pydantic Enum types to Django fields with choices."""
    logger.debug(f"Testing: {params.test_id}")
    dj_type, dj_kwargs = mapper.get_django_mapping(params.python_type, params.field_info)
    assert dj_type is params.expected_dj_type
    assert dj_kwargs == params.expected_kwargs


@pytest.mark.parametrize("params", PYD_TO_DJ_UNION_CASES, ids=lambda p: p.test_id)
def test_get_django_mapping_unions(mapper: BidirectionalTypeMapper, params: PydToDjParams):
    """Tests mapping complex Pydantic Union types to Django fields."""
    logger.debug(f"Testing: {params.test_id}")
    # Determine parent model (only needed for self-ref checks, unlikely in basic union tests)
    parent_model = None

    # Pass parent_model to the mapper method
    dj_type, dj_kwargs = mapper.get_django_mapping(
        params.python_type, params.field_info, parent_pydantic_model=parent_model
    )

    # Special check for multi-FK union signals
    if "_union_details" in params.expected_kwargs:
        assert "_union_details" in dj_kwargs, "Missing '_union_details' signal in kwargs"
        # Sort model lists before comparison for stability
        expected_models = sorted(params.expected_kwargs["_union_details"]["models"], key=lambda m: m.__name__)
        actual_models = sorted(dj_kwargs["_union_details"]["models"], key=lambda m: m.__name__)
        assert actual_models == expected_models, "Model list in '_union_details' does not match"
        # Compare the rest of the _union_details dict
        assert dj_kwargs["_union_details"]["type"] == params.expected_kwargs["_union_details"]["type"]
        assert dj_kwargs["_union_details"]["is_optional"] == params.expected_kwargs["_union_details"]["is_optional"]
        # Remove _union_details before comparing the rest of the kwargs
        del dj_kwargs["_union_details"]
        del params.expected_kwargs["_union_details"]
    # For non-multi-FK unions, check type directly
    else:
        assert dj_type is params.expected_dj_type

    # Compare remaining kwargs
    dj_kwargs.pop("related_name", None)  # Ignore related_name
    params.expected_kwargs.pop("related_name", None)
    assert (
        dj_kwargs == params.expected_kwargs
    ), f"Remaining kwargs mismatch. Expected {params.expected_kwargs}, got {dj_kwargs}"


@pytest.mark.parametrize("params", PYD_TO_DJ_RELATIONSHIP_CASES, ids=lambda p: p.test_id)
def test_get_django_mapping_relationships(mapper: BidirectionalTypeMapper, params: PydToDjParams):
    """Tests mapping Pydantic relationship types (BaseModel, List[BaseModel]) to Django fields."""
    logger.debug(f"Testing: {params.test_id}")

    # Determine parent model for self-reference check
    parent_model = None
    if params.test_id == "self_ref_fk":
        parent_model = TargetPydanticModel

    # Pass parent_model to the mapper method
    dj_type, dj_kwargs = mapper.get_django_mapping(
        params.python_type, params.field_info, parent_pydantic_model=parent_model
    )
    assert dj_type is params.expected_dj_type
    # Don't check related_name here, it's dynamically generated/checked in factory
    dj_kwargs.pop("related_name", None)
    params.expected_kwargs.pop("related_name", None)
    assert dj_kwargs == params.expected_kwargs


@pytest.mark.parametrize("params", PYD_TO_DJ_DEFAULT_FACTORY_CASES, ids=lambda p: p.test_id)
def test_get_django_mapping_default_factories(mapper: BidirectionalTypeMapper, params: PydToDjParams):
    """Tests that Pydantic default_factory does not result in a Django default kwarg."""
    logger.debug(f"Testing: {params.test_id}")
    dj_type, dj_kwargs = mapper.get_django_mapping(params.python_type, params.field_info)
    assert dj_type is params.expected_dj_type
    assert (
        "default" not in dj_kwargs
    ), f"Django default kwarg should not be present when Pydantic uses default_factory (got: {dj_kwargs.get('default')})"
    # Check other expected kwargs are still present
    expected_kwargs_no_default = params.expected_kwargs.copy()
    expected_kwargs_no_default.pop("default", None)
    kwargs_no_default = dj_kwargs.copy()
    kwargs_no_default.pop("default", None)
    assert (
        kwargs_no_default == expected_kwargs_no_default
    ), f"Other kwargs mismatch. Expected {expected_kwargs_no_default}, got {kwargs_no_default}"


@pytest.mark.parametrize("params", PYD_TO_DJ_ANNOTATED_CASES, ids=lambda p: p.test_id)
def test_get_django_mapping_annotated_unions(mapper: BidirectionalTypeMapper, params: PydToDjParams):
    """Tests mapping complex Pydantic Annotated[Union[...]] types."""
    logger.debug(f"Testing Annotated Union: {params.test_id}")
    parent_model = None  # Not testing self-ref here

    # Pass parent_model to the mapper method
    dj_type, dj_kwargs = mapper.get_django_mapping(
        params.python_type, params.field_info, parent_pydantic_model=parent_model
    )

    # Expect JSONField placeholder and _union_details signal
    assert dj_type is params.expected_dj_type, f"Expected placeholder type {params.expected_dj_type}, got {dj_type}"
    assert "_union_details" in dj_kwargs, "Missing '_union_details' signal in kwargs for Annotated[Union[...]]"

    # Sort model lists before comparison for stability
    expected_models = sorted(params.expected_kwargs["_union_details"]["models"], key=lambda m: m.__name__)
    actual_models = sorted(dj_kwargs["_union_details"]["models"], key=lambda m: m.__name__)
    assert actual_models == expected_models, "Model list in '_union_details' does not match"

    # Compare the rest of the _union_details dict
    assert dj_kwargs["_union_details"]["type"] == params.expected_kwargs["_union_details"]["type"]
    assert dj_kwargs["_union_details"]["is_optional"] == params.expected_kwargs["_union_details"]["is_optional"]

    # Remove _union_details before comparing the rest of the kwargs
    del dj_kwargs["_union_details"]
    expected_kwargs_no_details = params.expected_kwargs.copy()
    del expected_kwargs_no_details["_union_details"]

    # Compare remaining kwargs (like null/blank)
    dj_kwargs.pop("related_name", None)  # Ignore related_name
    expected_kwargs_no_details.pop("related_name", None)
    assert (
        dj_kwargs == expected_kwargs_no_details
    ), f"Remaining kwargs mismatch. Expected {expected_kwargs_no_details}, got {dj_kwargs}"


# Django -> Pydantic Tests
# ------------------------

# --- Test Data Setup ---

# Basic Field Instances
DJ_CHARFIELD = models.CharField(max_length=100, help_text="Test Help")
DJ_CHARFIELD_NULL = models.CharField(max_length=50, null=True, blank=True)
DJ_TEXTFIELD = models.TextField(verbose_name="Notes Field")
DJ_INTFIELD = models.IntegerField(default=0)
DJ_FLOATFIELD = models.FloatField()
DJ_BOOLFIELD = models.BooleanField(default=True)
DJ_DECIMALFIELD = models.DecimalField(max_digits=10, decimal_places=2, null=True)
DJ_DATEFIELD = models.DateField(auto_now_add=True)  # auto_now_add implies not editable
DJ_DATETIMEFIELD = models.DateTimeField(null=True)
DJ_UUIDFIELD = models.UUIDField(default=UUID("12345678-1234-5678-1234-567812345678"))
DJ_EMAILFIELD = models.EmailField(unique=True)
DJ_URLFIELD = models.URLField(max_length=300)
DJ_IPFIELD = models.GenericIPAddressField(protocol="ipv4")
DJ_FILEFIELD = models.FileField(upload_to="files/", null=True)
DJ_IMAGEFIELD = models.ImageField(upload_to="images/")
DJ_JSONFIELD = models.JSONField(default=dict)
DJ_BINARYFIELD = models.BinaryField()
# Positive Fields
DJ_POS_INTFIELD = models.PositiveIntegerField()
DJ_POS_SMALLINTFIELD = models.PositiveSmallIntegerField()
# PK Fields
DJ_AUTO_PK = TargetDjangoModel._meta.get_field("id_pk_int")  # Get actual AutoField
DJ_UUID_PK = TargetDjangoModel._meta.get_field("uuid_pk")
# Choice Fields
DJ_CHOICE_CHAR = models.CharField(max_length=1, choices=[("A", "Alpha"), ("B", "Beta")])
DJ_CHOICE_INT = models.IntegerField(choices=[(1, "One"), (2, "Two")], null=True)
# Choice field specifically for Literal mapping test
DJ_CHOICE_CHAR_FOR_LITERAL = models.CharField(max_length=5, choices=[("r", "Red"), ("g", "Green"), ("b", "Blue")])
# Relationship Fields (get from TargetDjangoModel)
DJ_FK = TargetDjangoModel._meta.get_field("related_fk")
DJ_O2O = TargetDjangoModel._meta.get_field("related_o2o")
DJ_M2M = TargetDjangoModel._meta.get_field("related_m2m")
DJ_SELF_FK = TargetDjangoModel._meta.get_field("self_ref_fk")


DJ_TO_PYD_SIMPLE_CASES = [
    DjToPydParams("char_to_str", DJ_CHARFIELD, str, {"max_length": 100, "description": "Test Help"}),
    DjToPydParams("char_null_to_optional_str", DJ_CHARFIELD_NULL, Optional[str], {"max_length": 50}),
    DjToPydParams("text_to_str", DJ_TEXTFIELD, str, {"title": "Notes field"}),
    DjToPydParams("int_to_int", DJ_INTFIELD, int, {"default": 0}),
    DjToPydParams("float_to_float", DJ_FLOATFIELD, float, {}),
    DjToPydParams("bool_to_bool", DJ_BOOLFIELD, bool, {"default": True}),
    DjToPydParams(
        "decimal_null_to_optional_decimal",
        DJ_DECIMALFIELD,
        Optional[Decimal],
        {"max_digits": 10, "decimal_places": 2},
    ),
    DjToPydParams("date_to_date", DJ_DATEFIELD, datetime.date, {}),
    DjToPydParams("datetime_null_to_optional_datetime", DJ_DATETIMEFIELD, Optional[datetime.datetime], {}),
    DjToPydParams(
        "uuid_to_uuid",
        DJ_UUIDFIELD,
        UUID,
        {},
    ),
    DjToPydParams("email_to_emailstr", DJ_EMAILFIELD, EmailStr, {"max_length": 254}),
    DjToPydParams("url_to_httpurl", DJ_URLFIELD, HttpUrl, {"max_length": 300}),
    DjToPydParams("ip_to_ipvany", DJ_IPFIELD, IPvAnyAddress, {}),
    DjToPydParams(
        "file_null_to_optional_str", DJ_FILEFIELD, Optional[str], {"json_schema_extra": {"format": "binary"}}
    ),
    DjToPydParams("image_to_str", DJ_IMAGEFIELD, str, {"json_schema_extra": {"format": "image"}}),
    DjToPydParams("json_to_any", DJ_JSONFIELD, Any, {"default_factory": dict}),
    DjToPydParams("binary_to_bytes", DJ_BINARYFIELD, bytes, {}),
    # Positive fields
    DjToPydParams("posint_to_int_ge0", DJ_POS_INTFIELD, int, {"ge": 0}),
    DjToPydParams("possmallint_to_int_ge0", DJ_POS_SMALLINTFIELD, int, {"ge": 0}),
    # PK fields - these have titles via model definition
    DjToPydParams(
        "auto_pk_to_optional_int_frozen",
        DJ_AUTO_PK,
        Optional[int],
        {"default": None, "frozen": True, "title": "Id pk int"},
    ),
    DjToPydParams(
        "uuid_pk_to_uuid",
        DJ_UUID_PK,
        UUID,
        {"title": "Uuid pk"},
    ),
    # Choices - Assume verbose_name might be missing, remove expected title
    DjToPydParams(
        "choice_char_to_literal",
        DJ_CHOICE_CHAR_FOR_LITERAL,
        str,
        {"json_schema_extra": {"choices": DJ_CHOICE_CHAR_FOR_LITERAL.choices}},
    ),
    DjToPydParams(
        "choice_int_null_to_optional_int",
        DJ_CHOICE_INT,
        Optional[int],
        {"json_schema_extra": {"choices": DJ_CHOICE_INT.choices}},
    ),
]

DJ_TO_PYD_RELATIONSHIP_CASES = [
    DjToPydParams("fk_null_to_optional_model", DJ_FK, Optional[RelatedPydanticModel], {"title": "Related fk"}),
    DjToPydParams("o2o_to_model", DJ_O2O, RelatedPydanticModel, {"title": "Related o2o"}),
    DjToPydParams(
        "m2m_to_list_model", DJ_M2M, list[RelatedPydanticModel], {"title": "Related m2m", "default_factory": list}
    ),
    DjToPydParams(
        "self_fk_null_to_optional_model", DJ_SELF_FK, Optional[TargetPydanticModel], {"title": "Self ref fk"}
    ),
]


@pytest.mark.parametrize("params", DJ_TO_PYD_SIMPLE_CASES, ids=lambda p: p.test_id)
def test_get_pydantic_mapping_simple_constraints(mapper: BidirectionalTypeMapper, params: DjToPydParams):
    """Tests mapping simple Django fields (and constraints) to Pydantic types and FieldInfo kwargs."""
    logger.debug(f"Testing: {params.test_id}")
    if params.raises_error:
        with pytest.raises(params.raises_error):
            mapper.get_pydantic_mapping(params.django_field_instance)
    else:
        py_type, field_info_kwargs = mapper.get_pydantic_mapping(params.django_field_instance)
        assert py_type == params.expected_py_type, f"Expected type {params.expected_py_type}, got {py_type}"
        assert (
            field_info_kwargs == params.expected_field_info_kwargs
        ), f"Expected kwargs {params.expected_field_info_kwargs}, got {field_info_kwargs}"


@pytest.mark.parametrize("params", DJ_TO_PYD_RELATIONSHIP_CASES, ids=lambda p: p.test_id)
def test_get_pydantic_mapping_relationships(mapper: BidirectionalTypeMapper, params: DjToPydParams):
    """Tests mapping Django relationship fields to Pydantic types."""
    logger.debug(f"Testing: {params.test_id}")
    py_type, field_info_kwargs = mapper.get_pydantic_mapping(params.django_field_instance)
    assert py_type == params.expected_py_type, f"Expected type {params.expected_py_type}, got {py_type}"
    assert (
        field_info_kwargs == params.expected_field_info_kwargs
    ), f"Expected kwargs {params.expected_field_info_kwargs}, got {field_info_kwargs}"


# TODO: Add tests for unmapped types / error conditions / edge cases
# e.g., Unmapped Pydantic type -> Fallback to JSONField
# e.g., Unmapped Django type -> Fallback to Any
# e.g., Relationship mapping missing from accessor
