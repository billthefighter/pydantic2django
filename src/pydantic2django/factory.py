import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, Optional, Union, cast, get_args, get_origin

from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from pydantic.errors import PydanticUndefined
from pydantic.fields import FieldInfo

from pydantic2django.field_type_mapping import TypeMapper
from pydantic2django.field_utils import RelationshipFieldHandler

from .base_django_model import Pydantic2DjangoBaseClass
from .context_storage import ModelContext, create_context_for_model
from .types import T

# Cache for converted models to prevent duplicate conversions
_converted_models: dict[str, type[models.Model]] = {}

logger = logging.getLogger(__name__)


@dataclass
class DjangoModelFactory(Generic[T]):
    """
    Factory for creating Django models with proper type hints and IDE support.
    Builds on top of make_django_model while providing better IDE integration.
    """

    relationship_field_handler: RelationshipFieldHandler = field(default_factory=RelationshipFieldHandler)

    def create_model(
        self,
        pydantic_model: type[T],
        *,
        app_label: str,
        base_django_model: Optional[type[models.Model]] = None,
        check_migrations: bool = True,
        skip_relationships: bool = False,
        **options: Any,
    ) -> tuple[type[Pydantic2DjangoBaseClass[T]], Optional[dict[str, models.Field]]]:
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
        # Ensure Pydantic2DjangoBaseClass is in the inheritance chain
        if base_django_model:
            if not issubclass(base_django_model, Pydantic2DjangoBaseClass):
                # Create a new base class that includes Pydantic2DjangoBaseClass
                base_name = f"Base{base_django_model.__name__}"
                # Create Meta class
                meta_attrs = {
                    "abstract": False,
                    "managed": True,
                }
                meta = type("Meta", (), meta_attrs)
                base_django_model = type(
                    base_name,
                    (Pydantic2DjangoBaseClass, base_django_model),
                    {
                        "__module__": base_django_model.__module__,
                        "Meta": meta,
                    },
                )

        # Call the original make_django_model
        django_model, field_updates, context = self.make_django_model(
            pydantic_model=pydantic_model,
            base_django_model=base_django_model or Pydantic2DjangoBaseClass,
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
        if hasattr(django_model, "object_type"):
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
        return cast(type[Pydantic2DjangoBaseClass[T]], django_model), field_updates

    def make_django_model(
        self,
        pydantic_model: type[T],
        base_django_model: Optional[type[models.Model]] = None,
        existing_model: Optional[type[models.Model]] = None,
        class_name_prefix: str = "Django",
        strict: bool = False,
        **options: Any,
    ) -> tuple[type[models.Model], Optional[dict[str, models.Field]], Optional[ModelContext]]:
        """
        Convert a Pydantic model to a Django model, with optional base Django model inheritance.

        Args:
            pydantic_model: The Pydantic model class to convert
            base_django_model: Optional base Django model to inherit from
            existing_model: Optional existing model to update with new fields
            class_name_prefix: Prefix to use for the generated Django model class name
            strict: If True, raise an error on field collisions; if False, keep base model fields
            **options: Additional options for customizing the conversion

        Returns:
            A tuple of (django_model, field_updates, model_context) where:
            - django_model is the Django model class that corresponds to the Pydantic model
            - field_updates is a dict of fields that need to be added to an existing model, or None
            - model_context is the ModelContext object containing context information, or None if not needed

        Raises:
            ValueError: If app_label is not provided in options or if field type cannot be mapped
        """
        # Create context for the model to hold all non-serializable fields

        if "app_label" not in options:
            raise ValueError("app_label must be provided in options")

        logger.debug(f"Converting Pydantic model {pydantic_model.__name__}")
        if base_django_model:
            logger.debug(f"Using base Django model {base_django_model.__name__}")

        # Check if model was already converted and we're not updating an existing model
        model_key = f"{pydantic_model.__module__}.{pydantic_model.__name__}"
        if model_key in _converted_models and not existing_model:
            logger.debug(f"Returning cached model for {model_key}")
            return _converted_models[model_key], None, None

        # Get all fields from the Pydantic model
        pydantic_fields = pydantic_model.model_fields
        logger.debug(f"Processing {len(pydantic_fields)} fields from Pydantic model")

        # Create Django model fields
        django_fields = {}
        relationship_fields = {}
        context_fields = {}
        invalid_fields = []

        for field_name, field_info in pydantic_fields.items():
            try:
                # Skip id field if we're updating an existing model
                if field_name == "id" and existing_model:
                    continue

                # Create the Django field
                django_field = self.convert_field(
                    field_name,
                    field_info,
                    app_label=options["app_label"],
                )

                if not django_field:
                    context_fields[field_name] = field_info
                    continue

                # Handle relationship fields differently based on skip_relationships
                if isinstance(
                    django_field,
                    (models.ForeignKey, models.ManyToManyField, models.OneToOneField),
                ):
                    relationship_fields[field_name] = django_field
                    continue

                django_fields[field_name] = django_field

            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping field {field_name}: {str(e)}")
                invalid_fields.append((field_name, str(e)))
                continue

        # We're currently using None for error handling, but this code exists for reference
        ## If we have invalid fields, raise a ValueError
        # if invalid_fields and not skip_relationships and not options.get("ignore_errors", False):
        #    error_msg = "Failed to convert the following fields:\n"
        #    for field_name, error in invalid_fields:
        #        error_msg += f"  - {field_name}: {error}\n"
        #    raise ValueError(error_msg)

        # If we're updating an existing model, return only the relationship fields
        if existing_model:
            logger.debug(f"Returning relationship fields for existing model {existing_model.__name__}")
            return existing_model, relationship_fields, None

        # Check for field collisions if a base Django model is provided
        if base_django_model:
            # Use hasattr to safely check for _meta
            if hasattr(base_django_model, "_meta"):
                base_fields = base_django_model._meta.get_fields()
                base_field_names = {field.name for field in base_fields}
                logger.debug(f"Checking field collisions with base model {base_django_model.__name__}")

                # Check for collisions
                collision_fields = set(django_fields.keys()) & base_field_names
                if collision_fields:
                    if strict:
                        # In strict mode, raise an error with helpful message
                        error_msg = (
                            f"Field collision detected with base model: {collision_fields}. "
                            f"Options: 1) Change the base model fields, 2) Rename the Pydantic fields, "
                            f"or 3) Set strict=False to keep base model fields and discard Pydantic fields."
                        )
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                    else:
                        # In non-strict mode, keep base model fields and discard Pydantic fields
                        logger.warning(
                            f"Field collision detected with base model: {collision_fields}. "
                            f"Keeping base model fields and discarding Pydantic fields."
                        )
                        for field_name in collision_fields:
                            django_fields.pop(field_name, None)

        # TODO: Got this far on Saturday

        # Determine base classes
        base_classes = [base_django_model] if base_django_model else [models.Model]
        logger.debug(f"Using base classes: {[cls.__name__ for cls in base_classes]}")

        # Set up Meta options
        meta_app_label = options["app_label"]
        meta_db_table = options.get("db_table", f"{meta_app_label}_{pydantic_model.__name__.lower()}")

        # Create Meta class
        meta_attrs = {
            "app_label": meta_app_label,
            "db_table": meta_db_table,
            "abstract": False,  # Ensure model is not abstract
            "managed": True,  # Ensure model is managed by Django
        }

        # Add verbose names if available
        doc = (getattr(pydantic_model, "__doc__", "") or "").strip()
        meta_attrs["verbose_name"] = doc or pydantic_model.__name__
        meta_attrs["verbose_name_plural"] = f"{meta_attrs['verbose_name']}s"

        # If inheriting from an abstract model, we still need to set app_label
        # to avoid Django's error about missing app_label
        if (
            base_django_model
            and hasattr(base_django_model, "_meta")
            and getattr(base_django_model._meta, "abstract", False)
        ):
            # Keep app_label even for abstract base models
            logger.debug("Keeping app_label for model with abstract base")

        # Create Meta class
        if base_django_model and hasattr(base_django_model, "_meta"):
            # Inherit from base model's Meta class if it exists
            base_meta = getattr(base_django_model._meta, "original_attrs", {})
            meta_attrs.update(base_meta)
            # Ensure model is not abstract even if base model is
            meta_attrs["abstract"] = False
            meta_attrs["managed"] = True
            # Always ensure app_label is set
            meta_attrs["app_label"] = meta_app_label
            Meta = type("Meta", (object,), meta_attrs)
            logger.debug(f"Created Meta class inheriting from {base_django_model.__name__}")
        else:
            Meta = type("Meta", (), meta_attrs)
            logger.debug("Created new Meta class")

        # Create the model attributes
        attrs = {
            "__module__": pydantic_model.__module__,  # Use the Pydantic model's module
            "Meta": Meta,
            **django_fields,
        }

        # Create the Django model
        model_name = f"{class_name_prefix}{pydantic_model.__name__}"

        # Use the correct base class
        bases = tuple(base_classes)

        # Create the model class
        model = type(model_name, bases, attrs)
        django_model = cast(type[models.Model], model)

        # Create context object if needed
        model_context = None
        if any(getattr(field, "is_relationship", False) for field in django_fields.values()):
            model_context = create_context_for_model(django_model, pydantic_model)

        logger.debug(f"Created Django model {model_name}")

        # Register the model with Django if it has a Meta class with app_label
        if hasattr(django_model, "_meta") and hasattr(django_model._meta, "app_label"):
            from django.apps import apps

            app_label = django_model._meta.app_label

            try:
                apps.get_registered_model(app_label, model_name)
            except LookupError:
                apps.register_model(app_label, django_model)
                logger.debug(f"Registered model {model_name} with app {app_label}")

        # Cache the model if not updating an existing one
        if not existing_model:
            _converted_models[model_key] = django_model
            logger.debug(f"Cached model {model_key}")

        return django_model, relationship_fields, model_context

    def handle_id_field(field_name: str, field_info: FieldInfo) -> Optional[models.Field]:
        """
        Handle potential ID field naming conflicts with Django's automatic primary key.

        Args:
            field_name: The original field name
            field_info: The Pydantic field info

        Returns:
            A Django field instance configured as a primary key
        """
        # Check if this is an ID field (case insensitive)
        if field_name.lower() == "id":
            field_type = field_info.annotation

            # Determine the field type based on the annotation
            if field_type is int:
                field_class = models.AutoField
            elif field_type is str:
                field_class = models.CharField
            else:
                # Default to AutoField for other types
                field_class = models.AutoField

            # Create field kwargs
            field_kwargs = {
                "primary_key": True,
                "verbose_name": f"Custom {field_name} (used as primary key)",
            }

            # Add max_length for CharField
            if field_class is models.CharField:
                field_kwargs["max_length"] = 255

            # Create and return the field
            return field_class(**field_kwargs)

        return None

    def handle_enum_field(self, field_type: type[Enum], kwargs: dict[str, Any]) -> models.Field:
        """
        Create a Django field for an Enum type.

        Args:
            field_type: The Enum type
            kwargs: Additional field attributes

        Returns:
            A Django field for the Enum
        """
        # Get all enum values
        enum_values = [item.value for item in field_type]

        # Determine the type of the enum values
        if all(isinstance(val, int) for val in enum_values):
            # Integer enum
            return models.IntegerField(
                choices=[(item.value, item.name) for item in field_type],
                **kwargs,
            )
        elif all(isinstance(val, (str, int)) for val in enum_values):
            # String enum
            max_length = max(len(val) for val in enum_values)
            return models.CharField(
                max_length=max_length,
                choices=[(item.value, item.name) for item in field_type],
                **kwargs,
            )
        else:
            # Mixed type enum - use TextField with choices
            return models.TextField(
                choices=[(str(item.value), item.name) for item in field_type],
                **kwargs,
            )

    def convert_field(
        self,
        field_name: str,
        field_info: FieldInfo,
        app_label: str = "django_llm",
        model_name: Optional[str] = None,
    ) -> Optional[models.Field]:
        """
        Convert a Pydantic field to a Django field.
        This is the main entry point for field conversion.

        Args:
            field_name: The name of the field
            field_info: The Pydantic field info
            app_label: The app label to use for model registration
            model_name: The name of the model to reference (for relationships)

        Returns:
            A Django field instance or None if the field should be skipped

        Raises:
            ValueError: If the field type cannot be mapped to a Django field
        """
        # Handle potential ID field naming conflicts
        id_field = self.handle_id_field(field_name, field_info)
        if id_field:
            return id_field

        # Get field type from annotation
        field_type = field_info.annotation
        is_optional = False

        # Handle Optional types
        origin = get_origin(field_type)
        if origin is Union:
            args = get_args(field_type)
            if len(args) == 2 and type(None) in args:
                # This is an Optional type
                field_type = next(arg for arg in args if arg is not type(None))
                is_optional = True

        # Handle Enum types before falling back to TypeMapper
        if isinstance(field_type, type) and issubclass(field_type, Enum):
            # Get field attributes from FieldAttributeHandler
            kwargs = self.handle_field_attributes(field_info)
            if is_optional:
                kwargs["null"] = True
                kwargs["blank"] = True
            return self.handle_enum_field(field_type, kwargs)

        # Get the field mapping from TypeMapper
        mapping = TypeMapper.get_mapping_for_type(field_type)
        if not mapping:
            logger.warning(f"Could not map field type {field_type} to a Django field, must be contextual")
            return None

        # Get field attributes from both TypeMapper and FieldAttributeHandler
        kwargs = TypeMapper.get_field_attributes(field_type)

        # Add field attributes from field_info
        field_attrs = self.handle_field_attributes(field_info)
        kwargs.update(field_attrs)

        # For Optional types, set null and blank to True
        if is_optional:
            kwargs["null"] = True
            kwargs["blank"] = True

        # For relationship fields, use RelationshipFieldHandler
        if mapping.is_relationship:
            return RelationshipFieldHandler.create_field(
                field_name=field_name,
                field_info=field_info,
                field_type=field_type,
                app_label=app_label,
                model_name=model_name,
            )

        # Create and return the field
        return mapping.django_field(**kwargs)

    @staticmethod
    def handle_field_attributes(
        field_info: FieldInfo,
        extra: Optional[Union[dict[str, Any], Callable[[FieldInfo], dict[str, Any]]]] = None,
    ) -> dict[str, Any]:
        """
        Extract and process field attributes from Pydantic field info.

        Args:
            field_info: The Pydantic field info
            extra: Optional extra attributes or callable to get extra attributes

        Returns:
            A dictionary of field attributes
        """
        kwargs = {}

        # Handle null/blank based on whether the field is optional
        is_optional = not field_info.is_required
        kwargs["null"] = is_optional
        kwargs["blank"] = is_optional

        if (
            field_info.default is not None
            and field_info.default != Ellipsis
            and field_info.default != PydanticUndefined
        ):
            kwargs["default"] = field_info.default
        else:
            kwargs["default"] = None

        # Handle description as help_text
        if field_info.description:
            kwargs["help_text"] = field_info.description

        # Handle title as verbose_name
        if field_info.title:
            kwargs["verbose_name"] = field_info.title

        # Handle validators from field constraints
        # In Pydantic v2, constraints are stored in the field's metadata
        metadata = field_info.metadata
        if isinstance(metadata, dict):
            gt = metadata.get("gt")
            if gt is not None:
                kwargs.setdefault("validators", []).append(MinValueValidator(limit_value=gt))

            ge = metadata.get("ge")
            if ge is not None:
                kwargs.setdefault("validators", []).append(MinValueValidator(limit_value=ge))

            lt = metadata.get("lt")
            if lt is not None:
                kwargs.setdefault("validators", []).append(MaxValueValidator(limit_value=lt))

            le = metadata.get("le")
            if le is not None:
                kwargs.setdefault("validators", []).append(MaxValueValidator(limit_value=le))

        # Process extra attributes
        if extra:
            if callable(extra):
                extra_kwargs = extra(field_info)
                kwargs.update(extra_kwargs)
            else:
                kwargs.update(extra)

        return kwargs
