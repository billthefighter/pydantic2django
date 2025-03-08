from typing import Any, Generic, TypeVar, cast

from django.db import models
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class DjangoBaseModel(models.Model, Generic[T]):
    """
    Base class for Django models generated from Pydantic models.
    Provides type-safe conversion methods and proper IDE completion.
    """

    class Meta:
        abstract = True
        managed = False  # Don't create tables for the base class
        app_label = "pydantic2django"  # Explicitly set the app label

    _pydantic_model: type[T]  # Class variable to store reference to Pydantic model

    def __new__(cls, *args, **kwargs):
        """
        Override __new__ to ensure subclasses are not abstract.
        """
        if cls is DjangoBaseModel:
            raise TypeError("DjangoBaseModel cannot be instantiated directly")

        # Ensure subclasses are not abstract
        if hasattr(cls, "Meta"):
            cls.Meta.abstract = False
            cls.Meta.managed = True

        return super().__new__(cls)

    def __getattr__(self, name: str) -> Any:
        """
        Forward method calls to the Pydantic model implementation.
        This enables proper type checking for methods defined in the Pydantic model.
        """
        # Get the Pydantic model class
        pydantic_cls = self._pydantic_model

        # Check if the attribute exists in the Pydantic model
        if hasattr(pydantic_cls, name):
            # Get the attribute from the Pydantic model
            attr = getattr(pydantic_cls, name)

            # If it's a property, we need to create an instance to access it
            if isinstance(attr, property):
                # Convert to Pydantic instance to access property
                pydantic_instance = self.to_pydantic()
                return getattr(pydantic_instance, name)

            # If it's a method, bind it to a Pydantic instance
            elif callable(attr):

                def wrapped_method(*args, **kwargs):
                    pydantic_instance = self.to_pydantic()
                    return getattr(pydantic_instance, name)(*args, **kwargs)

                return wrapped_method

            return attr

        # If attribute not found, raise AttributeError
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    @classmethod
    def from_pydantic(cls, pydantic_instance: T) -> "DjangoBaseModel[T]":
        """Convert a Pydantic instance to Django model instance."""
        from .methods import convert_pydantic_to_django

        # Get the app_label, defaulting to 'tests' for concrete models
        app_label = "tests" if cls is not DjangoBaseModel else cls._meta.app_label

        return cast(
            DjangoBaseModel[T],
            convert_pydantic_to_django(pydantic_instance, app_label=app_label),
        )

    def to_pydantic(self) -> T:
        """Convert Django model instance to Pydantic model instance."""
        data = {
            field.name: getattr(self, field.name)
            for field in self._meta.fields
            if not field.primary_key  # Exclude primary key by default
        }
        return self._pydantic_model(**data)

    class Config:
        """Configuration for the base model."""

        arbitrary_types_allowed = True
