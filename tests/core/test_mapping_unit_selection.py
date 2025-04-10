import pytest
from typing import Any, Optional, Type, Literal, List, Dict, Set, Tuple, get_origin, get_args
from pydantic import Field, EmailStr, HttpUrl, IPvAnyAddress
from pydantic.fields import FieldInfo
import logging
from enum import Enum
from pathlib import Path
from uuid import UUID
import datetime
from decimal import Decimal

# Import the necessary classes from your project
from pydantic2django.core.bidirectional_mapper import BidirectionalTypeMapper, MappingError
from pydantic2django.core.mapping_units import (
    TypeMappingUnit,
    StrFieldMapping,
    TextFieldMapping,
    EmailFieldMapping,
    SlugFieldMapping,
    IntFieldMapping,
    FloatFieldMapping,
    BoolFieldMapping,
    JsonFieldMapping,
    EnumFieldMapping,
    URLFieldMapping,
    IPAddressFieldMapping,
    FilePathFieldMapping,
    UUIDFieldMapping,
    DateTimeFieldMapping,
    DateFieldMapping,
    TimeFieldMapping,
    DurationFieldMapping,
    DecimalFieldMapping,
    BinaryFieldMapping,
    BigIntFieldMapping,
    SmallIntFieldMapping,
    PositiveIntFieldMapping,  # Add others as needed
    PositiveSmallIntFieldMapping,
    PositiveBigIntFieldMapping,
    AutoFieldMapping,
    BigAutoFieldMapping,
    SmallAutoFieldMapping,
    FileFieldMapping,
    ImageFieldMapping,
    # Relationships handled separately by mapper logic usually
    # ForeignKeyMapping, OneToOneFieldMapping, ManyToManyFieldMapping
)

# Assuming a basic RelationshipConversionAccessor for standalone tests
from pydantic2django.core.relationships import RelationshipConversionAccessor

logger = logging.getLogger(__name__)


# Fixture for the mapper instance
@pytest.fixture
def mapper() -> BidirectionalTypeMapper:
    # You might need a more sophisticated setup depending on relationship needs
    return BidirectionalTypeMapper(RelationshipConversionAccessor())


# --- Helper Enums/Types for Testing ---
class ColorEnum(Enum):
    RED = "R"
    GREEN = "G"
    BLUE = "B"


class IntEnum(Enum):
    ONE = 1
    TWO = 2


# Test case structure
class SelectionTestCase:
    def __init__(
        self,
        test_id: str,
        py_type: Any,
        field_info: Optional[FieldInfo],
        expected_unit: Optional[Type[TypeMappingUnit]],
        raises_error: Optional[Type[Exception]] = None,
    ):
        self.test_id = test_id
        self.py_type = py_type
        self.field_info = field_info
        self.expected_unit = expected_unit
        self.raises_error = raises_error


# --- Test Cases ---
SELECTION_TEST_CASES = [
    # --- Simple Types ---
    SelectionTestCase("simple_int", int, None, IntFieldMapping),
    SelectionTestCase("simple_str_no_info", str, None, TextFieldMapping),
    SelectionTestCase("simple_str_with_max_length", str, Field(max_length=100), StrFieldMapping),
    SelectionTestCase("simple_bool", bool, None, BoolFieldMapping),
    SelectionTestCase("simple_float", float, None, FloatFieldMapping),
    SelectionTestCase("simple_bytes", bytes, None, BinaryFieldMapping),
    SelectionTestCase("simple_uuid", UUID, None, UUIDFieldMapping),
    SelectionTestCase("simple_datetime", datetime.datetime, None, DateTimeFieldMapping),
    SelectionTestCase("simple_date", datetime.date, None, DateFieldMapping),
    SelectionTestCase("simple_time", datetime.time, None, TimeFieldMapping),
    SelectionTestCase("simple_timedelta", datetime.timedelta, None, DurationFieldMapping),
    SelectionTestCase("simple_decimal", Decimal, None, DecimalFieldMapping),
    SelectionTestCase("simple_path", Path, None, FilePathFieldMapping),
    SelectionTestCase("any_type", Any, None, JsonFieldMapping),  # Any maps to JSON
    # --- Pydantic Specific Types ---
    SelectionTestCase("pydantic_email", EmailStr, None, EmailFieldMapping),
    SelectionTestCase("pydantic_url", HttpUrl, None, URLFieldMapping),
    SelectionTestCase("pydantic_ipvany", IPvAnyAddress, None, IPAddressFieldMapping),
    # --- String Subtype Discrimination ---
    SelectionTestCase("str_slug_hint_title", str, Field(title="Item Slug"), SlugFieldMapping),
    SelectionTestCase("str_slug_hint_desc", str, Field(description="A unique slug"), SlugFieldMapping),
    SelectionTestCase("str_slug_hint_pattern", str, Field(pattern=r"^[-\\w]+$"), SlugFieldMapping),
    SelectionTestCase(
        "str_email_hint_no_match", str, Field(title="User email address"), TextFieldMapping
    ),  # Don't match str based on hint
    SelectionTestCase(
        "str_url_hint_no_match", str, Field(description="Website URL"), TextFieldMapping
    ),  # Don't match str based on hint
    # --- Numeric Type Discrimination (Ensure correct Int mapping is chosen) ---
    # Base `int` without specific hints should map to IntFieldMapping by default
    SelectionTestCase("int_no_hints", int, None, IntFieldMapping),
    # TODO: Add tests if PositiveIntField etc. add `matches` overrides based on ge/gt hints
    # --- Enum / Literal ---
    SelectionTestCase("enum_str", ColorEnum, None, EnumFieldMapping),
    SelectionTestCase("enum_int", IntEnum, None, EnumFieldMapping),
    SelectionTestCase("literal_str", Literal["A", "B", "C"], None, EnumFieldMapping),
    SelectionTestCase("literal_int", Literal[1, 2, 3], None, EnumFieldMapping),
    SelectionTestCase("literal_mixed", Literal["A", 1, True], None, EnumFieldMapping),
    # --- Collections (should map to JSON unless it's List[KnownModel]) ---
    SelectionTestCase("collection_list_int", List[int], None, JsonFieldMapping),
    SelectionTestCase("collection_dict_str_int", Dict[str, int], None, JsonFieldMapping),
    SelectionTestCase("collection_set_float", Set[float], None, JsonFieldMapping),
    SelectionTestCase("collection_tuple_bool", Tuple[bool, ...], None, JsonFieldMapping),
    SelectionTestCase("collection_bare_list", list, None, JsonFieldMapping),
    SelectionTestCase("collection_bare_dict", dict, None, JsonFieldMapping),
    # --- Optional Types (Selection should ignore Optional wrapper) ---
    SelectionTestCase("optional_int", Optional[int], None, IntFieldMapping),
    SelectionTestCase("optional_str_no_info", Optional[str], None, TextFieldMapping),
    SelectionTestCase("optional_str_with_max_length", Optional[str], Field(max_length=50), StrFieldMapping),
    SelectionTestCase("optional_email", Optional[EmailStr], None, EmailFieldMapping),
    SelectionTestCase("optional_enum", Optional[ColorEnum], None, EnumFieldMapping),
    SelectionTestCase("optional_literal", Optional[Literal["X", "Y"]], None, EnumFieldMapping),
    SelectionTestCase("optional_list", Optional[List[int]], None, JsonFieldMapping),  # Optional list of simple types
    # --- File/Image Fields (Map to StrFieldMapping or FileFieldMapping depending on overrides) ---
    # Default FileFieldMapping maps python_type = str
    # SelectionTestCase("file_path_type", ???, Field(format="file-path"), FileFieldMapping), # Need way to signal file/image
    # Current FileFieldMapping maps `str`, so need overrides or specific type
    SelectionTestCase("file_field_str", str, Field(json_schema_extra={"format": "binary"}), FileFieldMapping),
    SelectionTestCase("image_field_str", str, Field(json_schema_extra={"format": "image"}), ImageFieldMapping),
]


@pytest.mark.parametrize("params", SELECTION_TEST_CASES, ids=[c.test_id for c in SELECTION_TEST_CASES])
def test_find_unit_for_pydantic_type_selection(mapper: BidirectionalTypeMapper, params: SelectionTestCase):
    """Tests the selection logic of _find_unit_for_pydantic_type based on type and FieldInfo."""
    logger.debug(f"Testing selection: {params.test_id} for type {params.py_type}")

    # NOTE: Removed incorrect unwrapping logic. The tested function
    # _find_unit_for_pydantic_type is expected (in normal flow)
    # to receive the *unwrapped* type from get_django_mapping.
    # However, for robustness, _find_unit_for_pydantic_type *itself*
    # should handle being passed an Optional type directly.
    py_type_to_find = params.py_type  # Pass the potentially Optional type directly for this test
    # origin = get_origin(py_type_to_find)
    # if origin is Optional:
    #     args = get_args(py_type_to_find)
    #     py_type_to_find = next((arg for arg in args if arg is not type(None)), Any)

    if params.raises_error:
        with pytest.raises(params.raises_error):
            # Access the protected method for testing purposes
            mapper._find_unit_for_pydantic_type(py_type_to_find, params.field_info)
    else:
        # Access the protected method for testing purposes
        selected_unit = mapper._find_unit_for_pydantic_type(py_type_to_find, params.field_info)

        # Add logging for easier debugging on failure
        if selected_unit is not params.expected_unit:
            logger.error(f"Test Failure: {params.test_id}")
            logger.error(f"  Input Type: {params.py_type}")
            logger.error(f"  Type Passed to Find: {py_type_to_find}")
            logger.error(f"  Field Info: {params.field_info}")
            logger.error(f"  Expected Unit: {params.expected_unit}")
            logger.error(f"  Selected Unit: {selected_unit}")

            # Log scores for debugging the selection
            scores = {}
            for unit_cls in mapper._registry:
                # Need to handle potential errors in matches method during logging
                try:
                    # Unwrap Optional *inside* the score logging for accurate debug info
                    current_py_type = py_type_to_find
                    origin = get_origin(current_py_type)
                    if origin is Optional:
                        args = get_args(current_py_type)
                        current_py_type = next((arg for arg in args if arg is not type(None)), Any)

                    score = unit_cls.matches(current_py_type, params.field_info)
                    if score > 0:
                        scores[unit_cls.__name__] = score
                except Exception as e:
                    scores[unit_cls.__name__] = f"ERROR: {e}"
            logger.error(f"  Scores: {scores}")

        assert (
            selected_unit is params.expected_unit
        ), f"Test ID '{params.test_id}': Expected {params.expected_unit}, got {selected_unit}"


# TODO: Add tests specifically for interaction between different overrides (e.g., slug hint + max_length)
# TODO: Add tests for caching behavior (though harder to test reliably)
# TODO: Add tests for relationship types if needed (though handled earlier in get_django_mapping)
