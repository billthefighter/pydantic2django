"""
Context storage system for handling non-serializable fields in Pydantic2Django.

This module provides the core functionality for managing context fields and their
mapping back to Pydantic objects. It handles the storage and retrieval of context
information needed for field reconstruction.
"""
from dataclasses import dataclass, field
from typing import Any, Optional, TypeVar

from django.db import models
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from .field_type_mapping import TypeMapper

T = TypeVar("T", bound=BaseModel)


@dataclass
class FieldContext:
    """
    Represents context information for a single field.
    """

    field_name: str
    field_type: type[Any]
    is_optional: bool = False
    is_list: bool = False
    additional_metadata: dict[str, Any] = field(default_factory=dict)
    value: Optional[Any] = None


@dataclass
class ModelContext:
    """
    Base class for model context classes.
    Stores context information for a Django model's fields that require special handling
    during conversion back to Pydantic objects.
    """

    model_name: str = ""
    pydantic_class: type[BaseModel] = BaseModel
    context_fields: dict[str, FieldContext] = field(default_factory=dict)
    required_context_keys: set[str] = field(default_factory=set)

    def add_field(self, field_name: str, field_type: type[Any], **kwargs) -> None:
        """
        Add a field to the context storage.

        Args:
            field_name: Name of the field
            field_type: Type of the field
            **kwargs: Additional metadata for the field
        """
        self.context_fields[field_name] = FieldContext(field_name=field_name, field_type=field_type, **kwargs)
        self.required_context_keys.add(field_name)

    def validate_context(self, context: dict[str, Any]) -> None:
        """
        Validate that all required context fields are present.

        Args:
            context: The context dictionary to validate

        Raises:
            ValueError: If required context fields are missing
        """
        missing_fields = self.required_context_keys - set(context.keys())
        if missing_fields:
            raise ValueError(f"Missing required context fields: {', '.join(missing_fields)}")

    def get_field_type(self, field_name: str) -> Optional[type[Any]]:
        """
        Get the type of a context field.

        Args:
            field_name: Name of the field

        Returns:
            The field type if it exists in the context, None otherwise
        """
        if field_name in self.context_fields:
            return self.context_fields[field_name].field_type
        return None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert context to a dictionary format suitable for to_pydantic().

        Returns:
            Dictionary containing all context values
        """
        return {field_name: field.value for field_name, field in self.context_fields.items() if field.value is not None}

    def set_value(self, field_name: str, value: Any) -> None:
        """
        Set the value for a context field.

        Args:
            field_name: Name of the field
            value: Value to set

        Raises:
            ValueError: If the field doesn't exist in the context
        """
        if field_name not in self.context_fields:
            raise ValueError(f"Field {field_name} not found in context")
        self.context_fields[field_name].value = value

    def get_value(self, field_name: str) -> Optional[Any]:
        """
        Get the value of a context field.

        Args:
            field_name: Name of the field

        Returns:
            The field value if it exists and has been set, None otherwise
        """
        if field_name in self.context_fields:
            return self.context_fields[field_name].value
        return None


def create_context_for_model(django_model: type[models.Model], pydantic_model: type[BaseModel]) -> ModelContext:
    """
    Create a context object for a Django model.

    Args:
        django_model: The Django model class
        pydantic_model: The corresponding Pydantic model class

    Returns:
        A ModelContext object containing context information for the model
    """
    context = ModelContext(model_name=django_model.__name__, pydantic_class=pydantic_model)

    # Analyze fields and add context information
    for field_name, pydantic_field in pydantic_model.model_fields.items():
        if not isinstance(pydantic_field, FieldInfo):
            continue

        field_type = pydantic_field.annotation
        if field_type is None:
            continue

        # Skip if the type is supported by TypeMapper (can be stored in the database)
        if TypeMapper.is_type_supported(field_type):
            continue

        # If we get here, the type needs context
        context.add_field(
            field_name=field_name,
            field_type=field_type,
            is_optional=not pydantic_field.is_required(),
            is_list=getattr(pydantic_field, "is_list", False),
            additional_metadata={"field_info": pydantic_field.json_schema_extra or {}},
        )

    return context
