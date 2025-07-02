"""
XML Schema factory module.
Creates Django fields and models from XML Schema definitions.
"""
import logging
from dataclasses import dataclass

from django.db import models

from ..core.factories import (
    BaseFieldFactory,
    BaseModelFactory,
    ConversionCarrier,
    FieldConversionResult,
)
from .models import (
    XmlSchemaAttribute,
    XmlSchemaComplexType,
    XmlSchemaElement,
    XmlSchemaSimpleType,
    XmlSchemaType,
)

logger = logging.getLogger(__name__)


@dataclass
class XmlSchemaFieldInfo:
    """Holds information about an XML Schema field (element or attribute)."""

    name: str
    element: XmlSchemaElement | None = None
    attribute: XmlSchemaAttribute | None = None


class XmlSchemaFieldFactory(BaseFieldFactory[XmlSchemaFieldInfo]):
    """Creates Django fields from XML Schema elements and attributes."""

    FIELD_TYPE_MAP = {
        XmlSchemaType.STRING: models.CharField,
        XmlSchemaType.INTEGER: models.IntegerField,
        XmlSchemaType.POSITIVEINTEGER: models.PositiveIntegerField,
        XmlSchemaType.DECIMAL: models.DecimalField,
        XmlSchemaType.BOOLEAN: models.BooleanField,
        XmlSchemaType.DATE: models.DateField,
        XmlSchemaType.DATETIME: models.DateTimeField,
        XmlSchemaType.TIME: models.TimeField,
        XmlSchemaType.GYEAR: models.IntegerField,
        XmlSchemaType.ID: models.CharField,  # Often used as PK
        XmlSchemaType.IDREF: models.CharField,  # Foreign key reference
        XmlSchemaType.HEXBINARY: models.BinaryField,
    }

    def create_field(
        self, field_info: XmlSchemaFieldInfo, model_name: str, carrier: ConversionCarrier[XmlSchemaComplexType]
    ) -> FieldConversionResult:
        """Convert XML Schema element/attribute to Django field."""

        result = FieldConversionResult(field_info=field_info, field_name=field_info.name)
        schema_def = carrier.source_model.schema_def

        # Determine the source element or attribute and its type name
        source_field = field_info.element if field_info.element else field_info.attribute
        if not source_field:
            return result  # Should not happen

        field_type_name = source_field.type_name

        # Check if the field's type is a defined simpleType (e.g., for enums)
        if field_type_name and schema_def.simple_types.get(field_type_name):
            simple_type = schema_def.simple_types[field_type_name]
            if simple_type.enumeration:
                # This is an enum, create a CharField with choices
                self._create_enum_field(simple_type, field_info, model_name, result)
            elif simple_type.restrictions:
                # Handle other restrictions like pattern for validators
                self._apply_simple_type_restrictions(simple_type, field_info, result)
            return result

        # Fallback to creating fields from elements or attributes directly
        if field_info.element:
            self._create_element_field(field_info.element, model_name, carrier, result)
        elif field_info.attribute:
            self._create_attribute_field(field_info.attribute, model_name, result)

        return result

    def _create_enum_field(
        self,
        simple_type: XmlSchemaSimpleType,
        field_info: XmlSchemaFieldInfo,
        model_name: str,
        result: FieldConversionResult,
    ):
        """Create a CharField with choices for an enumeration."""
        max_length = max(len(val) for val, _ in simple_type.enumeration) if simple_type.enumeration else 255

        # Add a Choices class to the carrier to be generated later
        enum_class_name = f"{model_name}{field_info.name.capitalize()}Choices"
        result.context["choices_class"] = {
            "name": enum_class_name,
            "choices": simple_type.enumeration,
        }

        result.field_type = models.CharField
        result.field_kwargs["max_length"] = max_length
        result.field_kwargs["choices"] = f"{enum_class_name}.choices"

        if field_info.element and field_info.element.default:
            result.field_kwargs["default"] = f"{enum_class_name}.{field_info.element.default.upper()}"
        elif field_info.attribute and field_info.attribute.default_value:
            result.field_kwargs["default"] = f"{enum_class_name}.{field_info.attribute.default_value.upper()}"

    def _apply_simple_type_restrictions(
        self,
        simple_type: XmlSchemaSimpleType,
        field_info: XmlSchemaFieldInfo,
        result: FieldConversionResult,
    ):
        """Apply validators and other constraints from simpleType restrictions."""
        if simple_type.restrictions and simple_type.restrictions.pattern:
            # This logic needs to be improved to add a RegexValidator instance
            # For now, we are just noting it.
            result.field_type = models.CharField
            result.field_kwargs.setdefault("max_length", 255)
            result.context["validator_pattern"] = simple_type.restrictions.pattern
            # This part will need to be connected to the generator to add the validator
            # to the field definition. For now, it's a placeholder.
            pass

    def _create_element_field(
        self,
        element: XmlSchemaElement,
        model_name: str,
        carrier: ConversionCarrier[XmlSchemaComplexType],
        result: FieldConversionResult,
    ):
        """Creates a Django field from an XmlSchemaElement."""
        field_type = self.FIELD_TYPE_MAP.get(element.base_type, models.CharField)
        result.field_type = field_type

        # Handle common properties
        if element.nillable or element.min_occurs == 0:
            result.field_kwargs["null"] = True
            result.field_kwargs["blank"] = True

        if field_type == models.CharField:
            result.field_kwargs.setdefault("max_length", 255)

        if element.default:
            result.field_kwargs["default"] = element.default

        # Basic relationship handling placeholder
        if element.base_type is None and element.type_name:
            # Could be a foreign key if type_name matches another complexType
            pass

    def _create_attribute_field(self, attribute: XmlSchemaAttribute, model_name: str, result: FieldConversionResult):
        """Creates a Django field from an XmlSchemaAttribute."""
        field_type = self.FIELD_TYPE_MAP.get(attribute.base_type, models.CharField)
        result.field_type = field_type

        # Handle common properties
        if attribute.use == "optional":
            result.field_kwargs["null"] = True
            result.field_kwargs["blank"] = True

        if field_type == models.CharField:
            result.field_kwargs.setdefault("max_length", 255)

        if attribute.default_value:
            result.field_kwargs["default"] = attribute.default_value

        # Placeholder for ID/PK handling
        if attribute.base_type == XmlSchemaType.ID:
            result.field_kwargs["primary_key"] = True


class XmlSchemaModelFactory(BaseModelFactory[XmlSchemaComplexType, XmlSchemaFieldInfo]):
    """Creates a Django model definition from an XML Schema complex type."""

    def __init__(self, app_label: str):
        self.field_factory = XmlSchemaFieldFactory()
        super().__init__(app_label=app_label, field_factory=self.field_factory)

    def _process_source_fields(self, carrier: ConversionCarrier[XmlSchemaComplexType]):
        """
        Iterate over the elements and attributes of the complex type and create
        fields.
        """
        source_model = carrier.source_model

        # Process elements
        for element in source_model.elements:
            field_info = XmlSchemaFieldInfo(name=element.name, element=element)
            result = self.field_factory.create_field(field_info, source_model.name, carrier)
            self.add_field_to_carrier(result, carrier)

        # Process attributes
        for attribute in source_model.attributes.values():
            field_info = XmlSchemaFieldInfo(name=attribute.name, attribute=attribute)
            result = self.field_factory.create_field(field_info, source_model.name, carrier)
            self.add_field_to_carrier(result, carrier)

    def _build_model_context(self, carrier: ConversionCarrier[XmlSchemaComplexType]):
        """Build the template context for the XML Schema model."""
        super()._build_model_context(carrier)

        # Add choices classes and their imports
        if "choices_classes" in carrier.model_context:
            for choices_info in carrier.model_context["choices_classes"]:
                carrier.model_context.add_text_choices_class(choices_info["name"], choices_info["choices"])
                carrier.model_context.add_import("django.db", "models")

    def create_model(
        self, source_model: XmlSchemaComplexType, base_classes: list[type] | None = None
    ) -> ConversionCarrier[XmlSchemaComplexType]:
        carrier = super().create_model(source_model, base_classes)

        # Add any generated choices classes to the model's context
        if not hasattr(carrier, "model_context") or carrier.model_context is None:
            carrier.model_context = {}

        for field_result in carrier.field_results:
            if "choices_class" in field_result.context:
                choices_info = field_result.context["choices_class"]
                if "choices_classes" not in carrier.model_context:
                    carrier.model_context["choices_classes"] = []
                carrier.model_context["choices_classes"].append(choices_info)

        return carrier
