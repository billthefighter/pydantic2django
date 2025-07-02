"""
XML Schema factory module.
Creates Django fields and models from XML Schema definitions.
"""
import logging
from dataclasses import dataclass

from django.db import models

from ..core.context import ModelContext
from ..core.factories import (
    BaseFieldFactory,
    BaseModelFactory,
    ConversionCarrier,
    FieldConversionResult,
)
from ..core.relationships import RelationshipConversionAccessor
from .models import XmlSchemaAttribute, XmlSchemaComplexType, XmlSchemaElement, XmlSchemaType

logger = logging.getLogger(__name__)


@dataclass
class XmlSchemaFieldInfo:
    """Field info wrapper for XML Schema elements and attributes"""

    element: XmlSchemaElement | None = None
    attribute: XmlSchemaAttribute | None = None

    @property
    def name(self) -> str:
        return (self.element.name if self.element else self.attribute.name) if (self.element or self.attribute) else ""

    @property
    def is_required(self) -> bool:
        if self.element:
            return self.element.is_required
        elif self.attribute:
            return self.attribute.is_required
        return False


class XmlSchemaFieldFactory(BaseFieldFactory[XmlSchemaFieldInfo]):
    """Creates Django fields from XML Schema elements and attributes."""

    def __init__(self, relationship_accessor: RelationshipConversionAccessor):
        self.relationship_accessor = relationship_accessor

        # XML Schema type to Django field mapping
        self.type_mapping = {
            XmlSchemaType.STRING: models.CharField,
            XmlSchemaType.NORMALIZEDSTRING: models.CharField,
            XmlSchemaType.TOKEN: models.CharField,
            XmlSchemaType.INTEGER: models.IntegerField,
            XmlSchemaType.LONG: models.BigIntegerField,
            XmlSchemaType.SHORT: models.SmallIntegerField,
            XmlSchemaType.BYTE: models.SmallIntegerField,
            XmlSchemaType.UNSIGNEDINT: models.PositiveIntegerField,
            XmlSchemaType.UNSIGNEDLONG: models.PositiveBigIntegerField,
            XmlSchemaType.POSITIVEINTEGER: models.PositiveIntegerField,
            XmlSchemaType.NONNEGATIVEINTEGER: models.PositiveIntegerField,
            XmlSchemaType.DECIMAL: models.DecimalField,
            XmlSchemaType.FLOAT: models.FloatField,
            XmlSchemaType.DOUBLE: models.FloatField,
            XmlSchemaType.BOOLEAN: models.BooleanField,
            XmlSchemaType.DATE: models.DateField,
            XmlSchemaType.DATETIME: models.DateTimeField,
            XmlSchemaType.TIME: models.TimeField,
            XmlSchemaType.DURATION: models.DurationField,
            XmlSchemaType.ANYURI: models.URLField,
            XmlSchemaType.BASE64BINARY: models.BinaryField,
            XmlSchemaType.HEXBINARY: models.BinaryField,
        }

    def create_field(
        self, field_info: XmlSchemaFieldInfo, model_name: str, carrier: ConversionCarrier[XmlSchemaComplexType]
    ) -> FieldConversionResult:
        """Convert XML Schema element/attribute to Django field."""

        result = FieldConversionResult(field_info=field_info, field_name=field_info.name)

        if field_info.element:
            return self._create_element_field(field_info.element, model_name, carrier, result)
        elif field_info.attribute:
            return self._create_attribute_field(field_info.attribute, model_name, carrier, result)
        else:
            result.error_str = "Invalid field info: no element or attribute"
            return result

    def _create_element_field(
        self,
        element: XmlSchemaElement,
        model_name: str,
        carrier: ConversionCarrier[XmlSchemaComplexType],
        result: FieldConversionResult,
    ) -> FieldConversionResult:
        """Create Django field from XML Schema element."""

        # Handle basic types
        if element.base_type:
            field_class = self.type_mapping.get(element.base_type, models.CharField)
            kwargs = {}

            # Apply common field properties
            if not element.is_required:
                kwargs["null"] = True
                kwargs["blank"] = True

            if element.default_value:
                kwargs["default"] = element.default_value

            # Apply restrictions
            if element.restrictions:
                self._apply_restrictions(element.restrictions, field_class, kwargs)

            # Handle list types (multiple occurrences)
            if element.is_list:
                # For lists, we'll use a TextField with JSON serialization
                # or create a separate related model (implementation choice)
                field_class = models.TextField
                kwargs["help_text"] = f"List field (maxOccurs={element.max_occurs})"

            try:
                result.django_field = field_class(**kwargs)
                result.field_kwargs = kwargs
                result.field_definition_str = self._generate_field_definition(field_class, kwargs)
            except Exception as e:
                result.error_str = f"Failed to create field: {e}"
                logger.error(f"Error creating field for {model_name}.{element.name}: {e}")

        # Handle relationships (references to other complex types)
        elif element.type_name:
            return self._create_relationship_field(element, model_name, carrier, result)

        else:
            # Unknown or complex inline type
            result.error_str = f"Unsupported element type: {element.type_name}"
            logger.warning(f"Unsupported element type in {model_name}.{element.name}: {element.type_name}")

        return result

    def _create_attribute_field(
        self,
        attribute: XmlSchemaAttribute,
        model_name: str,
        carrier: ConversionCarrier[XmlSchemaComplexType],
        result: FieldConversionResult,
    ) -> FieldConversionResult:
        """Create Django field from XML Schema attribute."""

        if attribute.base_type:
            field_class = self.type_mapping.get(attribute.base_type, models.CharField)
            kwargs = {}

            # Attributes are typically shorter, use CharField with reasonable max_length
            if field_class == models.CharField and "max_length" not in kwargs:
                kwargs["max_length"] = 255

            if not attribute.is_required:
                kwargs["null"] = True
                kwargs["blank"] = True

            if attribute.default_value:
                kwargs["default"] = attribute.default_value

            try:
                result.django_field = field_class(**kwargs)
                result.field_kwargs = kwargs
                result.field_definition_str = self._generate_field_definition(field_class, kwargs)
            except Exception as e:
                result.error_str = f"Failed to create attribute field: {e}"
        else:
            result.error_str = f"Unsupported attribute type: {attribute.type_name}"

        return result

    def _create_relationship_field(
        self,
        element: XmlSchemaElement,
        model_name: str,
        carrier: ConversionCarrier[XmlSchemaComplexType],
        result: FieldConversionResult,
    ) -> FieldConversionResult:
        """Create relationship field for references to other complex types."""

        target_model_name = element.type_name.split(":")[-1]  # Remove namespace

        # Determine relationship type
        if element.is_list:
            field_class = models.ManyToManyField
            kwargs = {"to": target_model_name, "blank": True}
        else:
            field_class = models.ForeignKey
            kwargs = {
                "to": target_model_name,
                "on_delete": models.SET_NULL if not element.is_required else models.CASCADE,
            }
            if not element.is_required:
                kwargs["null"] = True
                kwargs["blank"] = True

        try:
            result.django_field = field_class(**kwargs)
            result.field_kwargs = kwargs
            result.field_definition_str = self._generate_field_definition(field_class, kwargs)
        except Exception as e:
            result.error_str = f"Failed to create relationship field: {e}"

        return result

    def _apply_restrictions(self, restrictions, field_class, kwargs):
        """Apply XML Schema restrictions to Django field kwargs."""
        if restrictions.max_length and field_class in (models.CharField, models.TextField):
            kwargs["max_length"] = restrictions.max_length

        if restrictions.min_length and field_class in (models.CharField, models.TextField):
            # Django doesn't have min_length on fields, could add validator
            pass

        if restrictions.max_inclusive and field_class in (models.IntegerField, models.FloatField, models.DecimalField):
            # Could add validators for numeric constraints
            pass

    def _generate_field_definition(self, field_class, kwargs) -> str:
        """Generate string representation of field definition."""
        class_name = field_class.__name__
        kwargs_str = ", ".join(f"{k}={repr(v)}" for k, v in kwargs.items())
        return f"models.{class_name}({kwargs_str})"


class XmlSchemaModelFactory(BaseModelFactory[XmlSchemaComplexType, XmlSchemaFieldInfo]):
    """Creates Django models from XML Schema complex types."""

    def __init__(self, field_factory: XmlSchemaFieldFactory, relationship_accessor: RelationshipConversionAccessor):
        self.field_factory = field_factory
        self.relationship_accessor = relationship_accessor

    def create_model_definition(
        self, source_model: XmlSchemaComplexType, app_label: str, base_model_class: type[models.Model]
    ) -> ConversionCarrier[XmlSchemaComplexType]:
        """
        Create a Django model definition carrier from an XML Schema complex type.
        """
        carrier = ConversionCarrier[XmlSchemaComplexType](
            source_model=source_model,
            meta_app_label=app_label,
            base_django_model=base_model_class,
            class_name_prefix="",  # Use XML type names directly
        )

        # Process fields, create meta, and assemble the model
        self._process_source_fields(carrier)
        self._create_django_meta(carrier)
        self._assemble_django_model_class(carrier)
        self._build_model_context(carrier)

        return carrier

    def _process_source_fields(self, carrier: ConversionCarrier[XmlSchemaComplexType]):
        """Process elements and attributes of the complex type to create Django fields."""
        source_model = carrier.source_model
        model_name = source_model.name

        # Process attributes
        for attribute in source_model.attributes.values():
            field_info = XmlSchemaFieldInfo(attribute=attribute)
            result = self.field_factory.create_field(field_info, model_name, carrier)
            if result.django_field:
                carrier.django_fields[result.field_name] = result.django_field
                if result.field_definition_str:
                    carrier.django_field_definitions[result.field_name] = result.field_definition_str
            elif result.error_str:
                carrier.invalid_fields.append((result.field_name, result.error_str))

        # Process elements
        for element in source_model.elements:
            field_info = XmlSchemaFieldInfo(element=element)
            result = self.field_factory.create_field(field_info, model_name, carrier)
            is_relationship = isinstance(
                result.django_field, (models.ForeignKey, models.ManyToManyField, models.OneToOneField)
            )

            if result.django_field:
                if is_relationship:
                    carrier.relationship_fields[result.field_name] = result.django_field
                else:
                    carrier.django_fields[result.field_name] = result.django_field

                if result.field_definition_str:
                    carrier.django_field_definitions[result.field_name] = result.field_definition_str
            elif result.error_str:
                carrier.invalid_fields.append((result.field_name, result.error_str))

    def _build_model_context(self, carrier: ConversionCarrier[XmlSchemaComplexType]):
        """Builds the ModelContext for an XML Schema-based model."""
        if not carrier.django_model:
            return

        source_model = carrier.source_model
        context = ModelContext(
            django_model=carrier.django_model,
            source_class=source_model,
        )
        carrier.model_context = context
