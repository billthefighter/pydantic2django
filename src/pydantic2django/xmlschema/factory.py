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
        XmlSchemaType.IDREF: models.ForeignKey,
        XmlSchemaType.HEXBINARY: models.BinaryField,
    }

    def create_field(
        self, field_info: XmlSchemaFieldInfo, model_name: str, carrier: ConversionCarrier[XmlSchemaComplexType]
    ) -> FieldConversionResult:
        """Convert XML Schema element/attribute to Django field."""

        result = FieldConversionResult(field_info=field_info, field_name=field_info.name)
        schema_def = carrier.source_model.schema_def

        source_field = field_info.element if field_info.element else field_info.attribute
        if not source_field:
            return result

        field_type_name = source_field.type_name
        if field_type_name and ":" in field_type_name:
            field_type_name = field_type_name.split(":", 1)[1]

        field_class, kwargs = None, {}

        # Check for keyref first to determine if this is a ForeignKey
        keyref = next(
            (kr for kr in schema_def.keyrefs if field_info.name in kr.fields),
            None,
        )

        if keyref:
            field_class, kwargs = self._create_foreign_key_field(field_info, model_name, carrier)
        elif source_field and source_field.base_type == XmlSchemaType.IDREF:
            # Fallback for IDREFs not part of an explicit keyref
            field_class, kwargs = self._create_foreign_key_field(field_info, model_name, carrier)
        else:
            # Resolve simple types
            simple_type = self._resolve_simple_type(source_field.type_name, schema_def)
            if simple_type:
                if simple_type.restriction and simple_type.restriction.enumeration:
                    field_class, kwargs = self._create_enum_field(simple_type, field_info, model_name, carrier)
                elif simple_type.restriction:
                    field_class, kwargs = self._apply_simple_type_restrictions(simple_type, field_info)
            elif field_info.element:
                field_class, kwargs = self._create_element_field(field_info.element, model_name, carrier)
            elif field_info.attribute:
                field_class, kwargs = self._create_attribute_field(field_info.attribute, model_name)

        if field_class:
            try:
                result.django_field = field_class(**kwargs)
                result.field_kwargs = kwargs
                result.field_definition_str = self._generate_field_def_string(result, carrier.meta_app_label)
            except Exception as e:
                result.error_str = f"Failed to instantiate {field_class.__name__}: {e}"
        else:
            result.context_field = field_info

        return result

    def _create_enum_field(
        self,
        simple_type: XmlSchemaSimpleType,
        field_info: XmlSchemaFieldInfo,
        model_name: str,
        carrier: ConversionCarrier,
    ):
        """Create a CharField with choices for an enumeration."""
        max_length = (
            max(len(val) for val, _ in simple_type.restriction.enumeration)
            if simple_type.restriction and simple_type.restriction.enumeration
            else 255
        )

        # Get or create a shared enum class for this simpleType
        enum_class_name, is_new = self._get_or_create_enum_class(simple_type, field_info, carrier)

        kwargs = {
            "max_length": max_length,
            "choices": f"{enum_class_name}.choices",
        }

        default_val = None
        if field_info.element:
            default_val = field_info.element.default_value
        elif field_info.attribute:
            default_val = field_info.attribute.default_value

        if default_val:
            kwargs["default"] = f"{enum_class_name}.{default_val.upper()}"

        return models.CharField, kwargs

    def _apply_simple_type_restrictions(
        self,
        simple_type: XmlSchemaSimpleType,
        field_info: XmlSchemaFieldInfo,
    ):
        """Apply validators and other constraints from simpleType restrictions."""
        kwargs = {"max_length": 255}  # Default, can be overridden
        if simple_type.restriction:
            if simple_type.restriction.pattern:
                # Import RawCode for proper validator serialization
                from ..django.utils.serialization import RawCode

                # Use RawCode to ensure validator is not quoted as string
                kwargs["validators"] = [RawCode(f"RegexValidator(r'{simple_type.restriction.pattern}')")]
            if simple_type.restriction.max_length:
                kwargs["max_length"] = int(simple_type.restriction.max_length)
            # Add other restrictions as needed (e.g., min_length)

        # Determine the base Django field type
        base_field_class = self.FIELD_TYPE_MAP.get(simple_type.base_type, models.CharField)

        if simple_type.base_type == XmlSchemaType.STRING and (
            not simple_type.restriction or not simple_type.restriction.max_length
        ):
            # If a pattern is present, it's likely a constrained string that should be a CharField
            if not (simple_type.restriction and simple_type.restriction.pattern):
                base_field_class = models.TextField
                # TextField doesn't accept max_length, so remove it if it was defaulted
                kwargs.pop("max_length", None)

        if field_info.element and (field_info.element.nillable or field_info.element.min_occurs == 0):
            kwargs["null"] = True
            kwargs["blank"] = True
        elif field_info.attribute and field_info.attribute.use == "optional":
            kwargs["null"] = True
            kwargs["blank"] = True

        return base_field_class, kwargs

    def _create_element_field(
        self,
        element: XmlSchemaElement,
        model_name: str,
        carrier: ConversionCarrier[XmlSchemaComplexType],
    ):
        """Creates a Django field from an XmlSchemaElement."""
        field_class = self.FIELD_TYPE_MAP.get(element.base_type, models.CharField)
        kwargs = {}

        if element.nillable or element.min_occurs == 0:
            kwargs["null"] = True
            kwargs["blank"] = True

        if field_class == models.CharField:
            if element.nillable:
                field_class = models.TextField
                kwargs.pop("max_length", None)
            else:
                kwargs.setdefault("max_length", 255)

        if element.default_value:
            kwargs["default"] = element.default_value

        if element.base_type is None and element.type_name:
            pass

        return field_class, kwargs

    def _create_attribute_field(self, attribute: XmlSchemaAttribute, model_name: str):
        """Creates a Django field from an XmlSchemaAttribute."""
        field_class = self.FIELD_TYPE_MAP.get(attribute.base_type, models.CharField)
        kwargs = {}

        if attribute.use == "optional":
            kwargs["null"] = True
            kwargs["blank"] = True

        if field_class == models.CharField:
            kwargs.setdefault("max_length", 255)

        if attribute.default_value:
            kwargs["default"] = attribute.default_value

        if attribute.base_type == XmlSchemaType.ID:
            kwargs["primary_key"] = True

        return field_class, kwargs

    def _create_foreign_key_field(
        self,
        field_info: XmlSchemaFieldInfo,
        model_name: str,
        carrier: ConversionCarrier[XmlSchemaComplexType],
    ) -> tuple[type[models.ForeignKey], dict]:
        """Creates a ForeignKey field from an IDREF attribute or element."""
        schema_def = carrier.source_model.schema_def
        kwargs = {"on_delete": models.CASCADE}

        # Find the keyref that applies to this field
        # Handle namespace prefixes in keyref fields (e.g., 'tns:author_ref' matches 'author_ref')
        keyref = next(
            (
                kr
                for kr in schema_def.keyrefs
                if any(field_info.name == field_path.split(":")[-1] for field_path in kr.fields)
            ),
            None,
        )

        if keyref:
            # Find the corresponding key
            refer_name = keyref.refer.split(":")[-1]
            key = next((k for k in schema_def.keys if k.name == refer_name), None)
            if key:
                # Determine the target model by resolving the selector xpath
                # key.selector is like ".//tns:Author", extract "Author"
                selector_target = key.selector.split(":")[-1]

                # The selector typically refers to elements of a specific type
                # Try multiple resolution strategies:

                # Strategy 1: Direct complex type match (Author -> AuthorType)
                if f"{selector_target}Type" in schema_def.complex_types:
                    target_model_name = f"{selector_target}Type"
                # Strategy 2: Look for global element with that name
                elif selector_target in schema_def.elements:
                    target_element = schema_def.elements[selector_target]
                    if target_element.type_name:
                        target_model_name = target_element.type_name.split(":")[-1]
                    else:
                        target_model_name = f"{selector_target}Type"
                # Strategy 3: Check if selector_target itself is a complex type
                elif selector_target in schema_def.complex_types:
                    target_model_name = selector_target
                else:
                    # Final fallback - use the selector target name + "Type"
                    target_model_name = f"{selector_target}Type"

                kwargs["to"] = f"{carrier.meta_app_label}.{target_model_name}"

                # Use the keyref selector to generate a better related_name
                if keyref.selector:
                    related_name_base = keyref.selector.split(":")[-1].replace(".//", "")
                    related_name = f"{related_name_base.lower()}s"
                    kwargs["related_name"] = related_name
                else:
                    kwargs["related_name"] = f"{model_name.lower()}s"

        # Fallback if keyref resolution fails
        if "to" not in kwargs:
            kwargs["to"] = f"'{carrier.meta_app_label}.OtherModel'"
            logger.warning(
                "Could not fully resolve keyref for field '%s' in model '%s'. Using placeholder.",
                field_info.name,
                model_name,
            )

        return models.ForeignKey, kwargs

    def _resolve_simple_type(
        self, type_name: str | None, schema_def: "XmlSchemaDefinition"
    ) -> XmlSchemaSimpleType | None:
        """Looks up a simple type by its name in the schema definition."""
        if not type_name:
            return None
        local_name = type_name.split(":")[-1]
        return schema_def.simple_types.get(local_name)

    def _get_or_create_enum_class(
        self, simple_type: XmlSchemaSimpleType, field_info: XmlSchemaFieldInfo, carrier: ConversionCarrier
    ) -> tuple[str, bool]:
        """
        Get or create a TextChoices enum class for a simpleType with enumeration.
        Returns the class name and a boolean indicating if it was newly created.
        """
        # Generate a more readable name, e.g., 'BookGenre' from 'BookType' and 'genre'
        enum_name_base = field_info.name.replace("_", " ").title().replace(" ", "")

        # Add a suffix to avoid clashes with model names
        enum_class_name = f"{enum_name_base}"

        # Store enums in the context_data of the carrier to share them across the generation process
        if "enums" not in carrier.context_data:
            carrier.context_data["enums"] = {}

        if enum_class_name in carrier.context_data["enums"]:
            return enum_class_name, False

        choices = []
        if simple_type.restriction and simple_type.restriction.enumeration:
            for value, label in simple_type.restriction.enumeration:
                # ('fiction', 'Fiction') -> FICTION = 'fiction', 'Fiction'
                enum_member_name = label.replace("-", " ").upper().replace(" ", "_")
                choices.append({"name": enum_member_name, "value": value, "label": label})

        carrier.context_data["enums"][enum_class_name] = {"name": enum_class_name, "choices": choices}

        return enum_class_name, True

    def _generate_field_def_string(self, result: FieldConversionResult, app_label: str) -> str:
        # Avoid circular import
        from ..django.utils.serialization import generate_field_definition_string

        return generate_field_definition_string(
            field_class=result.django_field.__class__,
            field_kwargs=result.field_kwargs,
            app_label=app_label,
        )


class XmlSchemaModelFactory(BaseModelFactory[XmlSchemaComplexType, XmlSchemaFieldInfo]):
    """Creates Django `Model` instances from `XmlSchemaComplexType` definitions."""

    def __init__(self, app_label: str):
        self.app_label = app_label
        self.field_factory = XmlSchemaFieldFactory()

    def _handle_field_result(self, result: FieldConversionResult, carrier: ConversionCarrier[XmlSchemaComplexType]):
        """Handle the result of field conversion and add to appropriate carrier containers."""
        if result.django_field:
            carrier.django_fields[result.field_name] = result.django_field
            if result.field_definition_str:
                carrier.django_field_definitions[result.field_name] = result.field_definition_str
        elif result.context_field:
            carrier.context_fields[result.field_name] = result.context_field
        elif result.error_str:
            carrier.invalid_fields.append((result.field_name, result.error_str))

    def _process_source_fields(self, carrier: ConversionCarrier[XmlSchemaComplexType]):
        """Processes elements and attributes to create Django fields."""
        complex_type = carrier.source_model

        # Get the model name from the source model
        model_name = getattr(carrier.source_model, "__name__", "UnknownModel")

        # Process attributes
        for attr_name, attribute in complex_type.attributes.items():
            field_info = XmlSchemaFieldInfo(name=attr_name, attribute=attribute)
            result = self.field_factory.create_field(field_info, model_name, carrier)
            self._handle_field_result(result, carrier)

        # Process elements
        for element in complex_type.elements:
            field_info = XmlSchemaFieldInfo(name=element.name, element=element)
            result = self.field_factory.create_field(field_info, model_name, carrier)
            self._handle_field_result(result, carrier)

    def _build_model_context(self, carrier: ConversionCarrier[XmlSchemaComplexType]):
        """Builds the final ModelContext for the Django model."""
        if not carrier.django_model:
            logger.debug("Skipping context build: missing django model.")
            return

        # Create ModelContext with correct parameters
        carrier.model_context = ModelContext(
            django_model=carrier.django_model,
            source_class=carrier.source_model,
            context_fields={},  # Will be populated if needed
            context_data=carrier.context_data,  # Pass through any context data like enums
        )
