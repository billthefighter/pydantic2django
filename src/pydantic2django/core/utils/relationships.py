from typing import Any, Optional, TypeAlias, Union, get_args, get_origin

from django.db import models
from pydantic import BaseModel

# Type aliases (can potentially be moved to core.defs)
RelationshipFieldType: TypeAlias = Optional[
    type[models.ForeignKey] | type[models.ManyToManyField] | type[models.OneToOneField]
]
RelationshipFieldKwargs: TypeAlias = dict[str, Any]
RelationshipFieldDetectionResult: TypeAlias = tuple[RelationshipFieldType, RelationshipFieldKwargs]


def get_relationship_metadata(field_type: Any) -> dict[str, Any]:
    """
    Analyze a type hint to extract relationship metadata (if any).

    Determines if the type represents a relationship (e.g., Optional[OtherModel], List[OtherModel])
    and extracts the target model type.

    Args:
        field_type: The type annotation to analyze.

    Returns:
        A dictionary containing metadata: { 'is_relationship': bool, 'target_model': type | None, 'is_many': bool }
    """
    metadata = {"is_relationship": False, "target_model": None, "is_many": False}
    origin = get_origin(field_type)
    args = get_args(field_type)

    target_type = None

    # Handle Optional[T]
    if origin is Union and type(None) in args and len(args) == 2:
        target_type = next(arg for arg in args if arg is not type(None))
    # Handle List[T] or list[T]
    elif origin is list and args:
        target_type = args[0]
        metadata["is_many"] = True
    # Handle direct type T
    elif origin is None and isinstance(field_type, type):
        target_type = field_type

    # Check if the target type is a Pydantic model
    if target_type and isinstance(target_type, type):
        try:
            if issubclass(target_type, BaseModel):
                metadata["is_relationship"] = True
                metadata["target_model"] = target_type
        except TypeError:
            pass  # issubclass fails if target_type is not a class

    return metadata


# Need to import issubclass helper or define it here
# Minimal check, should ideally use the one from core.defs
def _is_pydantic_model(obj: Any) -> bool:
    try:
        return issubclass(obj, BaseModel)
    except TypeError:
        return False
