import pytest
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union
from pydantic import BaseModel

from pydantic2django.field_type_resolver import is_serializable_type
from .fixtures import SerializableType, UnserializableType


class SimpleEnum(Enum):
    A = "a"
    B = "b"


class SimplePydanticModel(BaseModel):
    value: str


@pytest.mark.parametrize(
    "field_type,expected",
    [
        # Basic Python types
        (str, True),
        (int, True),
        (float, True),
        (bool, True),
        (dict, True),
        (list, True),
        (set, True),
        # Optional types
        (Optional[str], True),
        (Optional[int], True),
        (Union[str, None], True),
        # Collection types
        (List[str], True),
        (List[int], True),
        (Dict[str, int], True),
        (Set[str], True),
        (List[SimplePydanticModel], True),  # List of Pydantic models
        # Pydantic models
        (SimplePydanticModel, True),
        (SerializableType, True),
        # Enums
        (SimpleEnum, True),
        # Non-serializable types
        (Callable, False),
        (UnserializableType, False),
        (Any, False),
        # Complex nested types
        (Dict[str, List[int]], True),
        (List[Optional[str]], True),
        (Dict[str, SimplePydanticModel], True),
        # Edge cases
        (type(None), True),  # NoneType is serializable
        (type, False),  # type objects are not serializable
    ],
)
def test_is_serializable_type(field_type: Any, expected: bool):
    """
    Test is_serializable_type with various field types.

    Args:
        field_type: The type to test
        expected: Whether the type should be considered serializable
    """
    assert is_serializable_type(field_type) == expected
