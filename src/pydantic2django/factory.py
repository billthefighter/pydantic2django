import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, Optional, Union, cast, get_args, get_origin

from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from pydantic import BaseModel
from pydantic.errors import PydanticUndefined
from pydantic.fields import FieldInfo

from pydantic2django.field_type_mapping import TypeMapper
from pydantic2django.field_utils import RelationshipFieldHandler

from .base_django_model import Pydantic2DjangoBaseClass
from .context_storage import ModelContext
from .types import T

logger = logging.getLogger(__name__)


@dataclass
class DjangoModelFactoryCarrier:
    """
    Carrier class for creating a Django model from a Pydantic model.

    Also carries the result of the conversion.

    This dataclass encapsulates all the necessary parameters for converting a Pydantic model
    into a Django model, including configuration for model naming, inheritance, and conflict resolution.

    Attributes:
        pydantic_model: The Pydantic model class to convert to a Django model.
        meta_app_label: The Django app label to use for the model's Meta class.
        base_django_model: Base Django model class to inherit from. Defaults to Pydantic2DjangoBaseClass.
        existing_model: Optional existing Django model to update with new fields.
        class_name_prefix: Prefix to use for the generated Django model class name. Defaults to "Django".
        strict: If True, raise errors on field collisions; if False, keep base model fields. Defaults to False.
    """

    pydantic_model: type[BaseModel]
    meta_app_label: str
    base_django_model: type[models.Model] = Pydantic2DjangoBaseClass
    existing_model: Optional[type[models.Model]] = None
    class_name_prefix: str = "Django"
    strict: bool = False

    def __post_init__(self):
        self.django_fields: dict[str, models.Field] = {}
        self.relationship_fields: dict[str, models.Field] = {}
        self.context_fields: dict[str, FieldInfo] = {}
        self.invalid_fields: list[tuple[str, str]] = []
        self.django_meta_class: Optional[type] = None
        self.django_model: Optional[type[models.Model]] = None
        self.model_context: Optional[ModelContext] = None

    @property
    def model_key(self):
        return f"{self.pydantic_model.__module__}.{self.pydantic_model.__name__}"


# Cache for converted models to prevent duplicate conversions
_converted_models: dict[str, DjangoModelFactoryCarrier] = {}


@dataclass
class DjangoModelFactory(Generic[T]):
    """
    Factory for creating Django models with proper type hints and IDE support.
    Builds on top of make_django_model while providing better IDE integration.
    """

    relationship_field_handler: RelationshipFieldHandler = field(default_factory=RelationshipFieldHandler)

    @classmethod
    def make_django_model(
        cls,
        carrier: DjangoModelFactoryCarrier,
    ) -> DjangoModelFactoryCarrier:
        """
        Convert a Pydantic model to a Django model, with optional base Django model inheritance.

        Returns:
            A tuple of (django_model, field_updates, model_context) where:
            - django_model is the Django model class that corresponds to the Pydantic model
            - field_updates is a dict of fields that need to be added to an existing model, or None
            - model_context is the ModelContext object containing context information, or None if not needed

        Raises:
            ValueError: If app_label is not provided in options or if field type cannot be mapped
        """

        logger.debug(f"Converting Pydantic model {carrier.pydantic_model.__name__}")
        if carrier.base_django_model:
            logger.debug(f"Using base Django model {carrier.base_django_model.__name__}")

        # Check if model was already converted and we're not updating an existing model

        if carrier.model_key in _converted_models and not carrier.existing_model:
            logger.debug(f"Returning cached model for {carrier.model_key}")
            return _converted_models[carrier.model_key]

        carrier = cls.handle_fields(carrier)
        carrier = cls.handle_field_collisions(carrier)
        carrier = cls.handle_django_meta(carrier)
        # TODO: Figure out what to do with existing_model
        # if existing_model:
        #    logger.debug(f"Returning relationship fields for existing model {existing_model.__name__}")
        #     return existing_model, relationship_fields, None

        # Check for field collisions if a base Django model is provided
        # Cache the model if not updating an existing one
        if not carrier.existing_model:
            _converted_models[carrier.model_key] = carrier
            logger.debug(f"Cached model {carrier.model_key}")

        return carrier

    @classmethod
    def handle_field_collisions(cls, carrier: DjangoModelFactoryCarrier):
        if carrier.base_django_model:
            # Use hasattr to safely check for _meta
            if hasattr(carrier.base_django_model, "_meta"):
                base_fields = carrier.base_django_model._meta.get_fields()
                base_field_names = {field.name for field in base_fields}
                logger.debug(f"Checking field collisions with base model {carrier.base_django_model.__name__}")

                # Check for collisions
                collision_fields = set(carrier.django_fields.keys()) & base_field_names
                if collision_fields:
                    if carrier.strict:
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
                            carrier.django_fields.pop(field_name, None)
        return carrier

    @classmethod
    def handle_django_meta(cls, carrier: DjangoModelFactoryCarrier):
        # Set up Meta options
        meta_db_table = f"{carrier.meta_app_label}_{carrier.pydantic_model.__name__.lower()}"

        # Create Meta class
        meta_attrs = {
            "app_label": carrier.meta_app_label,
            "db_table": meta_db_table,
            "abstract": False,  # Ensure model is not abstract
            "managed": True,  # Ensure model is managed by Django
        }

        # Add verbose names if available
        # Doc is too long, usually - we'll use the model name as verbose name
        # doc = (getattr(pydantic_model, "__doc__", "") or "").strip()
        meta_attrs["verbose_name"] = carrier.pydantic_model.__name__

        # Create Meta class
        if carrier.base_django_model and hasattr(carrier.base_django_model, "_meta"):
            # Inherit from base model's Meta class if it exists
            base_meta = getattr(carrier.base_django_model._meta, "original_attrs", {})
            meta_attrs.update(base_meta)
            # Ensure model is not abstract even if base model is
            meta_attrs["abstract"] = False
            meta_attrs["managed"] = True
            # Always ensure app_label is set
            meta_attrs["app_label"] = carrier.meta_app_label
            carrier.django_meta_class = type("Meta", (object,), meta_attrs)
            logger.debug(f"Created Meta class inheriting from {carrier.base_django_model.__name__}")
        else:
            carrier.django_meta_class = type("Meta", (), meta_attrs)
            logger.debug("Created new Meta class")
        return carrier

    @classmethod
    def assemble_django_model(cls, carrier: DjangoModelFactoryCarrier):
        # Create the model attributes
        attrs = {
            "__module__": carrier.pydantic_model.__module__,  # Use the Pydantic model's module
            "Meta": carrier.django_meta_class,
            **carrier.django_fields,
        }

        # Determine base classes
        base_classes = [carrier.base_django_model] if carrier.base_django_model else [models.Model]
        logger.debug(f"Using base classes: {[cls.__name__ for cls in base_classes]}")

        # Create the Django model
        model_name = f"{carrier.class_name_prefix}{carrier.pydantic_model.__name__}"

        # Use the correct base class
        bases = tuple(base_classes)

        # Create the model class
        carrier.django_model = cast(type[models.Model], type(model_name, bases, attrs))

        return carrier

    @classmethod
    def handle_fields(cls, carrier: DjangoModelFactoryCarrier) -> DjangoModelFactoryCarrier:
        for field_name, field_info in carrier.pydantic_model.model_fields.items():
            try:
                # Skip id field if we're updating an existing model
                if field_name == "id" and carrier.existing_model:
                    continue

                # Create the Django field
                django_field = DjangoFieldFactory.convert_field(
                    field_name=field_name,
                    field_info=field_info,
                    app_label=carrier.meta_app_label,
                )
                # Convert field returns none if it can't find a mapping
                if not django_field:
                    carrier.context_fields[field_name] = field_info
                    continue

                # Handle relationship fields differently based on skip_relationships
                if isinstance(
                    django_field,
                    (models.ForeignKey, models.ManyToManyField, models.OneToOneField),
                ):
                    carrier.relationship_fields[field_name] = django_field
                    continue

                carrier.django_fields[field_name] = django_field

            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping field {field_name}: {str(e)}")
                carrier.invalid_fields.append((field_name, str(e)))
                continue
        return carrier


class DjangoFieldFactory:
    @classmethod
    def convert_field(
        cls,
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
        if id_field := cls.handle_id_field(field_name, field_info):
            return id_field

        # Get field type from annotation
        field_type = field_info.annotation

        # Get unified field attributes
        kwargs = cls.process_field_attributes(field_info)

        # Handle Enum types before falling back to TypeMapper
        if isinstance(field_type, type) and issubclass(field_type, Enum):
            return cls.handle_enum_field(field_type, kwargs)
        # TODO: Got here saturday night. Need to consider the logic and flow.
        # Enum handling and other weird handling is tricky. Maybe move all to
        # TypeMapper.get_field_attributes?

        # Get the field mapping from TypeMapper
        mapping = TypeMapper.get_mapping_for_type(field_type)
        if not mapping:
            logger.warning(f"Could not map field type {field_type} to a Django field, must be contextual")
            return None

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

    @classmethod
    def process_field_attributes(
        cls,
        field_info: FieldInfo,
        extra: Optional[Union[dict[str, Any], Callable[[FieldInfo], dict[str, Any]]]] = None,
    ) -> dict[str, Any]:
        """
        Unified method to process all field attributes from both Pydantic field info and type mapping.

        Args:
            field_info: The Pydantic field info
            field_type: The Python type of the field
            extra: Optional extra attributes or callable to get extra attributes

        Returns:
            A dictionary of field attributes
        """
        field_type = field_info.annotation
        kwargs = {}

        # 1. Get type-specific attributes from TypeMapper
        type_kwargs = TypeMapper.get_field_attributes(field_type)
        kwargs.update(type_kwargs)

        # 2. Handle null/blank based on whether the field is optional
        # method 1: Handle Optional types
        origin = get_origin(field_type)
        is_optional = False
        if origin is Union:
            args = get_args(field_type)
            if len(args) == 2 and type(None) in args:
                # This is an Optional type
                field_type = next(arg for arg in args if arg is not type(None))
                is_optional = True
        # Method 2: Check if field is required in Pydantic
        is_optional = is_optional or not field_info.is_required
        kwargs["null"] = is_optional
        kwargs["blank"] = is_optional

        # 3. Handle default values
        if (
            field_info.default is not None
            and field_info.default != Ellipsis
            and field_info.default != PydanticUndefined
        ):
            kwargs["default"] = field_info.default
        else:
            kwargs["default"] = None

        # 4. Handle description as help_text
        if field_info.description:
            kwargs["help_text"] = field_info.description

        # 5. Handle title as verbose_name
        if field_info.title:
            kwargs["verbose_name"] = field_info.title

        # 6. Handle validators from field constraints
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

        # 7. Process extra attributes
        if extra:
            if callable(extra):
                extra_kwargs = extra(field_info)
                kwargs.update(extra_kwargs)
            else:
                kwargs.update(extra)

        return kwargs

    @classmethod
    def handle_id_field(cls, field_name: str, field_info: FieldInfo) -> Optional[models.Field]:
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

    @classmethod
    def handle_enum_field(cls, enum_instance: type[Enum], kwargs: dict[str, Any]) -> models.Field:
        # TODO: MOving this to TypeMappingDefinition.enum_field?
        """
        Create a Django field for an enum.

        Args:
            enum_instance: The Enum instance
            kwargs: Additional field attributes

        Returns:
            A Django field for the Enum
        """
        # Get all enum values
        enum_values = [item.value for item in enum_instance]

        # Determine the type of the enum values
        if all(isinstance(val, int) for val in enum_values):
            # Integer enum
            return models.IntegerField(
                choices=[(item.value, item.name) for item in enum_instance],
                **kwargs,
            )
        elif all(isinstance(val, (str, int)) for val in enum_values):
            # String enum
            max_length = max(len(str(val)) for val in enum_values if isinstance(val, str))
            return models.CharField(
                max_length=max_length,
                choices=[(item.value, item.name) for item in enum_instance],
                **kwargs,
            )
        else:
            # Mixed type enum - use TextField with choices
            return models.TextField(
                choices=[(str(item.value), item.name) for item in enum_instance],
                **kwargs,
            )
