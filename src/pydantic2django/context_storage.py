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


class ContextRegistry:
    """
    Global registry for model contexts.
    """

    _contexts: dict[str, ModelContext] = {}

    @classmethod
    def register_context(cls, model_name: str, context: ModelContext) -> None:
        """
        Register a model context.

        Args:
            model_name: Name of the model
            context: Context object for the model
        """
        cls._contexts[model_name] = context

    @classmethod
    def get_context(cls, model_name: str) -> Optional[ModelContext]:
        """
        Get a model's context.

        Args:
            model_name: Name of the model

        Returns:
            The model context if it exists, None otherwise
        """
        return cls._contexts.get(model_name)

    @classmethod
    def clear(cls) -> None:
        """Clear all registered contexts."""
        cls._contexts.clear()


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
    for context_field in django_model._meta.get_fields():
        if getattr(context_field, "is_relationship", False):
            # Get the original Pydantic field type
            pydantic_field = pydantic_model.model_fields.get(context_field.name)
            if pydantic_field and isinstance(pydantic_field, FieldInfo):
                field_type = pydantic_field.annotation
                if field_type is not None:
                    context.add_field(
                        field_name=context_field.name,
                        field_type=field_type,
                        is_optional=not pydantic_field.is_required(),
                        is_list=getattr(context_field, "is_list", False),
                        additional_metadata={"field_info": pydantic_field.json_schema_extra or {}},
                    )

    # Register the context
    ContextRegistry.register_context(django_model.__name__, context)
    return context
