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

    django_model: type[models.Model]
    pydantic_class: type[BaseModel]
    context_fields: list[FieldContext] = field(default_factory=list)

    @property
    def required_context_keys(self) -> set[str]:
        required_fields = {x.field_name for x in self.context_fields if not x.is_optional}
        return required_fields

    def add_field(self, field_name: str, field_type: type[Any], **kwargs) -> None:
        """
        Add a field to the context storage.

        Args:
            field_name: Name of the field
            field_type: Type of the field
            **kwargs: Additional metadata for the field
        """
        self.context_fields.append(FieldContext(field_name=field_name, field_type=field_type, **kwargs))

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
