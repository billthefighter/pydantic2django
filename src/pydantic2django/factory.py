from collections.abc import Callable
from typing import Any, Generic, Optional, cast

from django.db import models

# Remove direct import from core to avoid circular import
# from .core import make_django_model
from .types import DjangoBaseModel, T

# Forward reference to make_django_model to be imported at runtime
make_django_model: Optional[Callable] = None


def _init_imports():
    """Initialize imports that might cause circular imports."""
    global make_django_model
    # Import at runtime to avoid circular imports
    from pydantic2django import make_django_model as _make_django_model

    make_django_model = _make_django_model


# Initialize imports at module level
_init_imports()


class DjangoModelFactory(Generic[T]):
    """
    Factory for creating Django models with proper type hints and IDE support.
    Builds on top of make_django_model while providing better IDE integration.
    """

    @classmethod
    def create_model(
        cls,
        pydantic_model: type[T],
        *,
        app_label: str,
        base_django_model: Optional[type[models.Model]] = None,
        check_migrations: bool = True,
        skip_relationships: bool = False,
        **options: Any,
    ) -> tuple[type[DjangoBaseModel[T]], Optional[dict[str, models.Field]]]:
        """
        Create a Django model class with proper type hints.

        Args:
            pydantic_model: The Pydantic model to convert
            app_label: The Django app label for the model
            base_django_model: Optional base Django model to inherit from
            check_migrations: Whether to check for needed migrations
            skip_relationships: Whether to skip relationship fields
            **options: Additional options for model creation

        Returns:
            A tuple of (django_model, field_updates) where django_model has proper type hints
        """
        # Ensure imports are initialized
        if make_django_model is None:
            _init_imports()
            if make_django_model is None:
                raise ImportError("Failed to import make_django_model")

        # Ensure DjangoBaseModel is in the inheritance chain
        if base_django_model:
            if not issubclass(base_django_model, DjangoBaseModel):
                # Create a new base class that includes DjangoBaseModel
                base_name = f"Base{base_django_model.__name__}"
                # Create Meta class
                meta_attrs = {
                    "abstract": False,
                    "managed": True,
                }
                meta = type("Meta", (), meta_attrs)
                base_django_model = type(
                    base_name,
                    (DjangoBaseModel, base_django_model),
                    {
                        "__module__": base_django_model.__module__,
                        "Meta": meta,
                    },
                )

        # Call the original make_django_model
        django_model, field_updates = make_django_model(
            pydantic_model=pydantic_model,
            base_django_model=base_django_model or DjangoBaseModel,
            check_migrations=check_migrations,
            skip_relationships=skip_relationships,
            app_label=app_label,
            **options,
        )

        # Store reference to the Pydantic model in the object_type field
        # Use fully qualified module path
        module_name = pydantic_model.__module__
        class_name = pydantic_model.__name__
        fully_qualified_name = f"{module_name}.{class_name}"

        # Set the object_type field
        django_model.object_type = fully_qualified_name

        # Ensure the model is not abstract by setting Meta attributes
        meta_attrs = {
            "abstract": False,
            "managed": True,
            "app_label": app_label,
        }

        # Create a new Meta class with our attributes
        meta = type("Meta", (), meta_attrs)

        # Use setattr to avoid linter errors
        django_model.Meta = meta

        # Cast to proper type for IDE support
        return cast(type[DjangoBaseModel[T]], django_model), field_updates

    @classmethod
    def create_abstract_model(
        cls,
        pydantic_model: type[T],
        **kwargs: Any,
    ) -> type[DjangoBaseModel[T]]:
        """
        Create an abstract Django model from a Pydantic model.

        Args:
            pydantic_model: The Pydantic model to convert
            **kwargs: Additional options for model creation

        Returns:
            An abstract Django model class with proper type hints
        """
        # Ensure imports are initialized
        if make_django_model is None:
            _init_imports()
            if make_django_model is None:
                raise ImportError("Failed to import make_django_model")

        # Set abstract=True in Meta options
        kwargs["abstract"] = True

        # Create the model
        django_model, _ = make_django_model(
            pydantic_model=pydantic_model,
            base_django_model=DjangoBaseModel,
            check_migrations=False,
            **kwargs,
        )

        # Store reference to the Pydantic model in the object_type field
        # Use fully qualified module path
        module_name = pydantic_model.__module__
        class_name = pydantic_model.__name__
        fully_qualified_name = f"{module_name}.{class_name}"

        # Set the object_type field
        django_model.object_type = fully_qualified_name

        return django_model
