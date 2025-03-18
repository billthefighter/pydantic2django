import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional, Union, cast, get_args, get_origin

from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

# Use absolute imports for these dependencies
try:
    # Try relative imports first (normal case)
    from .base_django_model import Pydantic2DjangoBaseClass
    from .context_storage import ModelContext
    from .field_type_mapping import TypeMapper, TypeMappingDefinition
    from .field_utils import is_pydantic_model_field_optional, sanitize_related_name
    from .relationships import RelationshipConversionAccessor
except ImportError:
    # Fall back to absolute imports (for direct module execution)
    from pydantic2django.base_django_model import Pydantic2DjangoBaseClass
    from pydantic2django.context_storage import ModelContext
    from pydantic2django.field_type_mapping import TypeMapper, TypeMappingDefinition
    from pydantic2django.field_utils import is_pydantic_model_field_optional, sanitize_related_name
    from pydantic2django.relationships import RelationshipConversionAccessor

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

    def build_model_context(self):
        if not self.pydantic_model or not self.django_model:
            logger.exception(
                f"Model context cannot be built for {self.model_key} - missing pydantic_model or django_model"
            )
        else:
            model_context = ModelContext(
                django_model=self.django_model,
                pydantic_class=self.pydantic_model,
            )
            for field_name, field_info in self.context_fields.items():
                if field_info.annotation is None:
                    logger.warning(f"Field {field_name} has no annotation, skipping")
                    continue
                else:
                    optional = is_pydantic_model_field_optional(field_info.annotation)
                    model_context.add_field(
                        field_name=field_name, field_type=field_info.annotation, is_optional=optional
                    )
            self.model_context = model_context

    def __str__(self):
        if self.django_model:
            return f"{self.pydantic_model.__name__} -> {self.django_model.__name__}"
        else:
            return f"{self.pydantic_model.__name__} -> None"


# Cache for converted models to prevent duplicate conversions
_converted_models: dict[str, DjangoModelFactoryCarrier] = {}


@dataclass
class FieldConversionResult:
    field_info: FieldInfo
    field_name: str
    app_label: str
    type_mapping_definition: Optional[TypeMappingDefinition] = None
    field_kwargs: dict[str, Any] = field(default_factory=dict)
    django_field: Optional[models.Field] = None
    context_field: Optional[FieldInfo] = None
    error_str: Optional[str] = None

    @property
    def rendered_django_field(self) -> models.Field:
        if self.django_field and self.type_mapping_definition:
            return self.type_mapping_definition.get_django_field(self.field_kwargs)
        else:
            errorstr = "Django field or type mapping definition not found: "
            if not self.django_field:
                errorstr += "django_field is None"
            if not self.type_mapping_definition:
                errorstr += "type_mapping_definition is None"
            raise ValueError(errorstr)

    def __str__(self):
        return (
            f"FieldConversionResult(field_name={self.field_name}, "
            f"django_field={self.django_field}, "
            f"context_field={self.context_field}, "
            f"error_str={self.error_str})"
        )


@dataclass
class DjangoFieldFactory:
    """
    Factory for creating Django fields from Pydantic fields.

    Requires a available_relationships dict to be passed in - this is expected to be
    pre-populated with a dict of available relationships.
    """

    available_relationships: RelationshipConversionAccessor

    def convert_field(
        self,
        field_name: str,
        field_info: FieldInfo,
        app_label: str = "django_llm",
    ) -> FieldConversionResult:
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
        # Create an empty result object to return
        result = FieldConversionResult(field_info=field_info, app_label=app_label, field_name=field_name)
        # Handle potential ID field naming conflicts
        try:
            if id_field := self.handle_id_field(field_name, field_info):
                result.django_field = id_field

            # Get field type from annotation
            field_type = field_info.annotation

            # Get unified field attributes
            result.field_kwargs = self.process_field_attributes(field_info)

            # Get the field mapping from TypeMapper
            result.type_mapping_definition = TypeMapper.get_mapping_for_type(field_type)
            if not result.type_mapping_definition:
                logger.warning(f"Could not map {field_name} of type {field_type} to a Django field, must be contextual")
                # Mark this as a context field since we can't map it
                result.context_field = field_info
                return result

            # For relationship fields, use RelationshipFieldHandler
            if result.type_mapping_definition and result.type_mapping_definition.is_relationship:
                result = self.handle_relationship_field(result)
                if not result.django_field:
                    logger.warning(f"Could not create relationship field for {field_name}, must be contextual")
                    # Mark unmappable relationship fields as context fields
                    result.context_field = field_info

            # Try to create a Django field from the mapping
            if result.type_mapping_definition and not result.django_field:
                try:
                    # Instantiate the field class instead of just assigning the class itself
                    result.django_field = result.type_mapping_definition.get_django_field(result.field_kwargs)
                except Exception as e:
                    # Don't silently fall back to contextual fields for parameter errors
                    # These indicate a bug that should be fixed
                    error_msg = f"Failed to convert Django field for {field_name}: {e}"
                    logger.warning(f"{error_msg} - saving this result to context.")

                    # For parameter errors, raise an exception
                    if "got an unexpected keyword argument" in str(
                        e
                    ) or "missing 1 required positional argument" in str(e):
                        logger.error(f"Parameter error detected: {e}")
                        raise ValueError(error_msg) from e

                    # For other errors, still mark as contextual field
                    result.context_field = field_info

            return result

        except Exception as e:
            # Create a more detailed error message
            detailed_msg = f"Error converting field '{field_name}' of type '{field_info.annotation}': {e}"

            # Add context information
            if result.type_mapping_definition:
                detailed_msg += f"\nMapping found: {result.type_mapping_definition.django_field.__name__}"
            else:
                detailed_msg += "\nNo type mapping found in TypeMapper"

            # Add field details for debugging
            detailed_msg += f"\nField info: {field_info}"
            detailed_msg += f"\nField default: {field_info.default}"
            detailed_msg += f"\nField metadata: {field_info.metadata}"

            # Log the detailed message
            logger.error(detailed_msg)
            logger.error("Result dump:")
            # Convert dataclass to dict for safe logging
            from dataclasses import asdict

            try:
                # Try using asdict, which handles nested dataclasses
                result_dict = asdict(result)
                result_dict["error_str"] = str(e)
                logger.error(f"Result: {result_dict}")
            except Exception as dict_err:
                # Fallback to manually logging attributes
                logger.error(f"Could not convert result to dict: {dict_err}")
                for attr_name, attr_value in vars(result).items():
                    try:
                        logger.error(f"  {attr_name}: {attr_value}")
                    except Exception:
                        logger.error(f"  {attr_name}: <unprintable value>")

            # Include detailed error message with the exception
            result.error_str = detailed_msg

            # Don't fall back to contextual fields for parameter errors
            if "got an unexpected keyword argument" in str(e) or "missing 1 required positional argument" in str(e):
                logger.error(f"Parameter error detected: {e}")
                raise ValueError(detailed_msg) from e

            # For other errors, still mark as contextual field and return
            result.context_field = field_info
            return result

    def handle_relationship_field(self, result: FieldConversionResult) -> FieldConversionResult:
        field_info = result.field_info
        field_kwargs = result.field_kwargs
        field_type = field_info.annotation
        field_name = result.field_name

        # Get the django field class
        if not result.type_mapping_definition:
            es1 = "Relationship field should not be called without a type "
            es2 = "mapping definition - something must have gone wrong."
            raise ValueError(f"{es1} {es2}")

        # Get the model class based on the field type
        origin = get_origin(field_type)
        args = get_args(field_type)

        # Handle different relationship types based on origin
        if origin is list and args:
            # For list[Model] - many-to-many relationship
            model_class = args[0]
        elif origin is dict and len(args) == 2:
            # For dict[str, Model] - many-to-many with key
            model_class = args[1]
        elif origin is None and inspect.isclass(field_type) and issubclass(field_type, BaseModel):
            # Direct model reference - foreign key
            model_class = field_type
        else:
            logger.warning(f"Invalid model class type for field {field_name}: {origin} {args}")
            result.error_str = f"Invalid model class type for field {field_name}: {origin} {args}"
            result.django_field = None
            return result

        # Handle case where model is not in relationship accessor
        if model_class not in self.available_relationships.available_pydantic_models:
            logger.warning(f"Model {model_class} not in relationship accessor")
            result.django_field = None
            result.error_str = f"Model {model_class} not in relationship accessor"
            return result

        # Get the model name, handling both string and class references
        if isinstance(model_class, str):
            target_model_name = model_class
        elif inspect.isclass(model_class):
            target_model_name = model_class.__name__
        else:
            logger.warning(f"Invalid model class type for field {field_name}: {type(model_class)}")
            result.django_field = None
            return result

        # Get the related name
        related_name = sanitize_related_name(
            getattr(field_info, "related_name", ""),
            target_model_name or "",
            field_name,
        )

        field_kwargs["related_name"] = related_name

        # Handle to_field behavior - Fix: Use a string instead of a tuple
        field_kwargs["to"] = f"{result.app_label}.{target_model_name}"

        # Add on_delete only for ForeignKey and OneToOneField, not for ManyToManyField
        django_field = result.type_mapping_definition.django_field
        if django_field == models.ForeignKey or django_field == models.OneToOneField:
            field_kwargs["on_delete"] = models.CASCADE

        # Set the django_field directly from the type mapping definition's django_field class
        if result.type_mapping_definition and result.type_mapping_definition.django_field:
            try:
                # First populate the field, then create it
                result.django_field = result.type_mapping_definition.get_django_field(field_kwargs)
            except Exception as e:
                # Log the error
                error_msg = f"Failed to create Django field for {field_name}: {e}"
                logger.warning(error_msg)
                result.error_str = error_msg

                # Don't fall back to contextual fields for parameter errors - these indicate a bug
                # that should be fixed rather than silently treating as contextual
                if "got an unexpected keyword argument" in str(e) or "missing 1 required positional argument" in str(e):
                    raise ValueError(error_msg) from e

                # For other errors, still set as None to allow graceful fallback
                result.django_field = None

        return result

    def process_field_attributes(
        self,
        field_info: FieldInfo,
        extra: Optional[Union[dict[str, Any], Callable[[FieldInfo], dict[str, Any]]]] = None,
    ) -> dict[str, Any]:
        """
        Unified method to process all field attributes from both Pydantic field info and type mapping.

        Note: relationship fields are handled by the RelationshipFieldHandler, which is passed a list of
        relationship fields and their kwargs.

        Args:
            field_info: The Pydantic field info
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
            and field_info.default is not PydanticUndefined
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

    def handle_id_field(self, field_name: str, field_info: FieldInfo) -> Optional[models.Field]:
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


@dataclass
class DjangoModelFactory:
    """
    Factory for creating Django models with proper type hints and IDE support.
    """

    field_factory: DjangoFieldFactory

    def make_django_model(
        self,
        carrier: DjangoModelFactoryCarrier,
    ) -> DjangoModelFactoryCarrier:
        """
        Convert a Pydantic model to a Django model, with optional base Django model inheritance.

        Important note: Relationship fields must be handled in order to be mapped correctly.
        The discovery module has a registration order that can be used to ensure that relationship fields
        are processed in the correct order.

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

        carrier = self.handle_fields(carrier)
        carrier = self.handle_field_collisions(carrier)
        carrier = self.handle_django_meta(carrier)
        carrier = self.assemble_django_model(carrier)
        carrier.build_model_context()
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

    def handle_field_collisions(self, carrier: DjangoModelFactoryCarrier):
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

    def handle_django_meta(self, carrier: DjangoModelFactoryCarrier):
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

    def assemble_django_model(self, carrier: DjangoModelFactoryCarrier):
        # Create the model attributes
        model_attrs = {**carrier.django_fields}
        if carrier.django_meta_class:
            model_attrs["Meta"] = carrier.django_meta_class
        if carrier.pydantic_model.__module__:
            model_attrs["__module__"] = carrier.pydantic_model.__module__
        # Add  attribute that refers to the FQN
        model_attrs["object_type"] = f"{carrier.pydantic_model.__module__}.{carrier.pydantic_model.__name__}"

        # Determine base classes
        base_classes = [carrier.base_django_model] if carrier.base_django_model else [models.Model]
        logger.debug(f"Using base classes: {[self.__name__ for self in base_classes]}")

        # Create the Django model
        model_name = f"{carrier.class_name_prefix}{carrier.pydantic_model.__name__}"

        # Use the correct base class
        bases = tuple(base_classes)

        # Create the model class
        carrier.django_model = cast(type[models.Model], type(model_name, bases, model_attrs))

        return carrier

    def handle_fields(self, carrier: DjangoModelFactoryCarrier) -> DjangoModelFactoryCarrier:
        for field_name, field_info in carrier.pydantic_model.model_fields.items():
            try:
                # Skip id field if we're updating an existing model
                if field_name == "id" and carrier.existing_model:
                    continue

                # Create the Django field
                conversion_result = self.field_factory.convert_field(
                    field_name=field_name,
                    field_info=field_info,
                    app_label=carrier.meta_app_label,
                )

                # If the field is marked as a context field, add it to context_fields and continue
                if conversion_result.context_field:
                    carrier.context_fields[field_name] = conversion_result.context_field
                    continue

                # If we don't have a type mapping definition, mark this as a contextual field
                if not conversion_result.type_mapping_definition:
                    carrier.context_fields[field_name] = field_info
                    logger.info(f"Field '{field_name}' of type '{field_info.annotation}' treated as contextual field")
                    continue

                # Only try to render the Django field if we have a type mapping definition
                try:
                    django_field = conversion_result.rendered_django_field
                    carrier.django_fields[field_name] = django_field
                except ValueError as e:
                    # If we can't render the field, treat it as contextual
                    logger.warning(f"Could not render Django field for '{field_name}': {e}")
                    carrier.context_fields[field_name] = field_info

            except (ValueError, TypeError) as e:
                # Create a more detailed error message
                detailed_msg = f"Error converting field '{field_name}' in model '{carrier.pydantic_model.__name__}':"
                detailed_msg += f"\n  - Error type: {type(e).__name__}"
                detailed_msg += f"\n  - Error message: {str(e)}"
                detailed_msg += f"\n  - Field type: {field_info.annotation}"

                # Add information about the field
                detailed_msg += f"\n  - Field is_required: {field_info.is_required}"
                detailed_msg += f"\n  - Field has default: {'Yes' if field_info.default is not None else 'No'}"

                # Add information about available mappings
                from .field_type_mapping import TypeMapper

                mapping = TypeMapper.get_mapping_for_type(field_info.annotation)
                if mapping:
                    detailed_msg += f"\n  - Found mapping to: {mapping.django_field.__name__}"
                else:
                    detailed_msg += "\n  - No TypeMapper mapping available for this type"

                # Log the error
                logger.error(detailed_msg)

                # Add to invalid fields
                carrier.invalid_fields.append((field_name, detailed_msg))
                continue
        return carrier
