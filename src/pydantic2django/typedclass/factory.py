import datetime
import inspect
import logging
from dataclasses import dataclass
from typing import Any, Optional, Union, get_args, get_origin

from django.db import models

from pydantic2django.core.factories import (
    IMPORT_MAPPING,
    BaseFieldFactory,
    BaseModelFactory,
    ConversionCarrier,
)
from pydantic2django.core.relationships import RelationshipConversionAccessor
from pydantic2django.core.type_translators import BaseTypeTranslator
from pydantic2django.core.utils import (
    generate_meta_class_string,
)

from .discovery import TypedClassType  # Assuming TypedClassType is defined in discovery

logger = logging.getLogger(__name__)


# Placeholder for a more sophisticated FieldInfo if needed for generic classes
@dataclass
class TypedClassFieldInfo:
    """Holds information about a discovered attribute in a generic class."""

    name: str
    type_hint: Any  # The type hint for the attribute
    default_value: Any = inspect.Parameter.empty  # Default value from __init__ or class
    is_from_init: bool = True  # True if from __init__, False if class var not in __init__
    # Potentially add other metadata, like if it's a class variable, etc.


class TypedClassTypeTranslator(BaseTypeTranslator):
    """Translates Python types from generic classes to Django field parameters."""

    def __init__(self, relationship_accessor: RelationshipConversionAccessor):
        super().__init__(relationship_accessor)
        # Add any TypedClass specific mappings or overrides
        self.type_mapping.update(
            {
                # Example: maybe some specific handling for common non-serializable types
                # in reckless mode later.
            }
        )

    def translate_type(self, field_type: Any, field_name: str, carrier: ConversionCarrier) -> dict[str, Any]:
        """
        Translates a Python type to Django field parameters.
        Focuses on __init__ args and class vars.
        """
        # For TypedClass, we primarily rely on direct type hints.
        # This will be expanded for "reckless" mode.
        origin = get_origin(field_type)
        args = get_args(field_type)

        if origin is Union and type(None) in args:  # Optional field
            # Get the actual type from Union[T, None]
            actual_type = next(arg for arg in args if arg is not type(None))
            params = self._get_django_field_params(actual_type, field_name, carrier)
            params["null"] = True
            params["blank"] = True  # Often good practice for optional fields
            return params

        # Handle basic types
        params = self._get_django_field_params(field_type, field_name, carrier)

        # If no direct mapping, and we're not in "reckless mode" (to be added)
        # we might return a default or raise an error/warning.
        # For now, if _get_django_field_params returns a TextField, that's the fallback.
        if not params.get("field_class"):
            logger.warning(
                f"Could not map type {field_type} for field {field_name} in {carrier.source_model_name}. Defaulting to TextField."
            )
            params["field_class"] = models.TextField
            params["default"] = "None"  # Or some indication it's a placeholder
            carrier.add_django_field_import(models.TextField)

        # Placeholder for relationship handling (ForeignKey, OneToOne, ManyToMany)
        # This will require checking if 'field_type' or 'actual_type' is a discovered TypedClassType
        # and using the relationship_accessor.
        # For now, assume simple fields or fallback to TextField/JSONField.

        return params


class TypedClassFieldFactory(BaseFieldFactory[TypedClassFieldInfo, TypedClassType]):
    """Generates Django model field definitions from TypedClassFieldInfo."""

    def __init__(self, relationship_accessor: RelationshipConversionAccessor):
        super().__init__(
            type_translator=TypedClassTypeTranslator(relationship_accessor),
            relationship_accessor=relationship_accessor,
        )

    def create_field_definition(
        self, field_info: TypedClassFieldInfo, carrier: ConversionCarrier[TypedClassType]
    ) -> str:
        """
        Creates a Django field definition string.
        Example: "my_field = models.CharField(max_length=100, null=True)"
        """
        params = self.type_translator.translate_type(field_info.type_hint, field_info.name, carrier)

        field_class = params.pop("field_class", models.TextField)  # Default fallback
        # Ensure the import for the field_class is added
        carrier.add_django_field_import(field_class)

        # Handle default values - this needs careful conversion from Python defaults to Django defaults
        # For now, a simple string representation or skip if complex.
        default_value_str = ""
        if field_info.default_value is not inspect.Parameter.empty:
            if isinstance(field_info.default_value, (str, int, float, bool, datetime.date, datetime.datetime)):
                default_value_str = f", default={repr(field_info.default_value)}"
            elif field_info.default_value is None:
                default_value_str = ", default=None"
            # else: default for complex types is tricky, might need callable or be handled by null=True

        # Constructing the field string:
        # e.g. my_field = models.CharField(max_length=255, null=True, blank=True, default='foo')
        param_str = []
        for k, v in params.items():
            if k in ["null", "blank"] and v:  # Add null=True, blank=True
                param_str.append(f"{k}=True")
            elif k == "to" and isinstance(v, str):  # For relationships, 'to' model name
                param_str.append(f"to='{v}'")
            elif isinstance(v, str) and k not in ["related_name"]:  # other string params
                param_str.append(f"{k}='{v}'")
            elif isinstance(v, type) and issubclass(v, models.Model):  # 'to' can be a model class
                param_str.append(f"to={v.__name__}")  # This assumes 'to' model is in same app or imported
            elif v is not None:  # For other types like max_length
                param_str.append(f"{k}={v}")

        # Add related_name for relationships if present
        if "related_name" in params and params["related_name"]:
            param_str.append(f"related_name='{params['related_name']}'")

        field_str = f"{field_info.name} = {IMPORT_MAPPING[field_class.__module__ + '.' + field_class.__name__]}({', '.join(param_str)}{default_value_str})"

        # Placeholder for "reckless mode" JSONField and serialization methods
        # if 'is_reckless' and not directly_mappable:
        #    field_str = f"{field_info.name} = models.JSONField(null=True, blank=True)"
        #    carrier.add_django_field_import(models.JSONField)
        #    # Add placeholder for serialization/deserialization methods to carrier
        #    carrier.add_custom_method(f"def serialize_{field_info.name}(self): ...")
        #    carrier.add_custom_method(f"@property def {field_info.name}_deserialized(self): ...")

        return field_str


class TypedClassModelFactory(BaseModelFactory[TypedClassType, TypedClassFieldInfo]):
    """Creates Django model definitions from generic Python classes."""

    def __init__(
        self,
        field_factory: TypedClassFieldFactory,
        relationship_accessor: RelationshipConversionAccessor,
        # Add 'reckless_mode: bool = False' if implementing that flag
    ):
        super().__init__(field_factory, relationship_accessor)
        # self.reckless_mode = reckless_mode

    def _get_model_fields_info(
        self, model_class: TypedClassType, carrier: ConversionCarrier
    ) -> list[TypedClassFieldInfo]:
        """
        Extracts attribute information from a generic class.
        Prioritizes __init__ signature, then class-level annotations.
        """
        field_infos = []
        processed_params = set()

        # 1. Inspect __init__ method
        try:
            init_signature = inspect.signature(model_class.__init__)
            for name, param in init_signature.parameters.items():
                if name == "self":
                    continue

                type_hint = param.annotation if param.annotation is not inspect.Parameter.empty else Any
                default_val = param.default if param.default is not inspect.Parameter.empty else inspect.Parameter.empty

                field_infos.append(
                    TypedClassFieldInfo(name=name, type_hint=type_hint, default_value=default_val, is_from_init=True)
                )
                processed_params.add(name)
        except (ValueError, TypeError) as e:
            logger.debug(
                f"Could not inspect __init__ for {model_class.__name__}: {e}. Proceeding with class annotations."
            )

        # 2. Inspect class-level annotations (for attributes not in __init__)
        try:
            annotations = inspect.get_annotations(model_class, eval_str=True)
            for name, type_hint in annotations.items():
                if name not in processed_params and not name.startswith("_"):  # Avoid private/protected by convention
                    default_val = getattr(model_class, name, inspect.Parameter.empty)
                    field_infos.append(
                        TypedClassFieldInfo(
                            name=name, type_hint=type_hint, default_value=default_val, is_from_init=False
                        )
                    )
        except Exception as e:  # Broad exception as get_annotations can fail in various ways
            logger.debug(f"Could not get class annotations for {model_class.__name__}: {e}")

        logger.debug(f"Discovered field infos for {model_class.__name__}: {field_infos}")
        return field_infos

    def create_model_definition(
        self,
        model_class: TypedClassType,
        app_label: str,
        base_model_class: type[models.Model],
        module_mappings: Optional[dict[str, str]] = None,
    ) -> ConversionCarrier[TypedClassType]:
        """
        Generates a ConversionCarrier containing the Django model string and related info.
        """
        model_name = model_class.__name__
        django_model_name = f"{model_name}DjangoModel"  # Or some other naming convention

        carrier = ConversionCarrier(
            source_model=model_class,
            source_model_name=model_name,
            django_model_name=django_model_name,
            app_label=app_label,
            module_mappings=module_mappings or {},
            relationship_accessor=self.relationship_accessor,
        )

        # Add import for the base model class
        carrier.add_django_model_import(base_model_class)

        field_definitions = []
        model_fields_info = self._get_model_fields_info(model_class, carrier)

        if not model_fields_info:
            logger.warning(f"No fields discovered for class {model_name}. Generating an empty Django model.")
            # Optionally, add a default placeholder field if empty models are problematic
            # field_definitions.append("    # No convertible fields found")

        for field_info in model_fields_info:
            try:
                field_def_str = self.field_factory.create_field_definition(field_info, carrier)
                field_definitions.append(f"    {field_def_str}")
            except Exception as e:
                logger.error(
                    f"Error creating field definition for {field_info.name} in {model_name}: {e}", exc_info=True
                )
                # Optionally, add a placeholder or skip this field
                field_definitions.append(f"    # Error processing field: {field_info.name} - {e}")

        carrier.django_field_definitions = field_definitions

        # Meta class
        carrier.meta_class_string = generate_meta_class_string(
            app_label=app_label,
            django_model_name=django_model_name,  # Use the generated Django model name
            verbose_name=model_name,
        )

        # __str__ method
        # Heuristic: use 'name' or 'id' attribute if present in field_infos, else default
        str_field = "id"  # Django models get 'id' by default from models.Model
        for finfo in model_fields_info:
            if finfo.name in ["name", "title", "identifier"]:  # common __str__ candidates
                str_field = finfo.name
                break

        carrier.str_method_string = f"    def __str__(self):\n        return str(self.{str_field})"

        logger.info(f"Prepared ConversionCarrier for {model_name} -> {django_model_name}")
        return carrier
