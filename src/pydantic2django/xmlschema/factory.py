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
from ..django.timescale.heuristics import should_invert_fk, should_soft_reference
from .models import (
    XmlSchemaAttribute,
    XmlSchemaComplexType,
    XmlSchemaDefinition,
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
        XmlSchemaType.LONG: models.BigIntegerField,
        XmlSchemaType.SHORT: models.SmallIntegerField,
        XmlSchemaType.BYTE: models.SmallIntegerField,
        XmlSchemaType.UNSIGNEDINT: models.PositiveIntegerField,
        XmlSchemaType.UNSIGNEDLONG: models.PositiveBigIntegerField,
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

    def __init__(
        self,
        *,
        nested_relationship_strategy: str = "auto",
        list_relationship_style: str = "child_fk",
        nesting_depth_threshold: int = 1,
        included_model_names: set[str] | None = None,
        invert_fk_to_timescale: bool = True,
    ):
        super().__init__()
        self.nested_relationship_strategy = nested_relationship_strategy
        self.list_relationship_style = list_relationship_style
        self.nesting_depth_threshold = max(0, int(nesting_depth_threshold))
        # Track which Django validators are needed based on restrictions encountered
        self.used_validators: set[str] = set()
        # Limit FK generation only to models that will actually be generated
        self.included_model_names: set[str] | None = included_model_names
        # Control whether to invert FKs targeting hypertables (hypertable -> dimension)
        self.invert_fk_to_timescale = invert_fk_to_timescale

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
                restr = getattr(simple_type, "restrictions", None)
                if restr and getattr(restr, "enumeration", None):
                    field_class, kwargs = self._create_enum_field(simple_type, field_info, model_name, carrier)
                else:
                    # Always apply base-type mapping and any available restrictions
                    field_class, kwargs = self._apply_simple_type_restrictions(simple_type, field_info)
            # Inline restrictions on element/attribute (no named simpleType)
            elif getattr(source_field, "restrictions", None):
                # Build a temporary simple type holder to reuse restriction logic
                tmp_simple = XmlSchemaSimpleType(
                    name=field_info.name,
                    base_type=getattr(source_field, "base_type", None),
                    restrictions=getattr(source_field, "restrictions", None),
                )
                restr = tmp_simple.restrictions
                if restr and getattr(restr, "enumeration", None):
                    field_class, kwargs = self._create_enum_field(tmp_simple, field_info, model_name, carrier)
                else:
                    field_class, kwargs = self._apply_simple_type_restrictions(tmp_simple, field_info)
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
        restr = getattr(simple_type, "restrictions", None)
        max_length = 255
        if restr and getattr(restr, "enumeration", None):
            try:
                max_length = max(len(val) for val, _ in restr.enumeration)
            except Exception:
                max_length = 255

        # Get or create a shared enum class for this simpleType
        enum_class_name, is_new = self._get_or_create_enum_class(simple_type, field_info, carrier)

        # Use RawCode so choices/default emit without quotes
        from ..django.utils.serialization import RawCode

        kwargs = {
            "max_length": max_length,
            "choices": RawCode(f"{enum_class_name}.choices"),
        }

        default_val = None
        if field_info.element:
            default_val = field_info.element.default_value
        elif field_info.attribute:
            default_val = field_info.attribute.default_value

        if default_val:
            kwargs["default"] = RawCode(f"{enum_class_name}.{default_val.upper()}")

        return models.CharField, kwargs

    def _apply_simple_type_restrictions(
        self,
        simple_type: XmlSchemaSimpleType,
        field_info: XmlSchemaFieldInfo,
    ):
        """Apply validators and other constraints from simpleType restrictions."""
        # Import RawCode for proper validator serialization
        from ..django.utils.serialization import RawCode

        kwargs = {"max_length": 255}  # Default for string-like; will be dropped for numeric fields
        restr = getattr(simple_type, "restrictions", None)
        if restr:
            if getattr(restr, "pattern", None):
                # Use RawCode to ensure validator is not quoted as string
                kwargs["validators"] = [RawCode(f"RegexValidator(r'{restr.pattern}')")]
                # Note usage for later import emission
                try:
                    self.used_validators.add("RegexValidator")
                except Exception:
                    pass
            if getattr(restr, "max_length", None):
                kwargs["max_length"] = int(restr.max_length)
            # Numeric bounds -> validators
            min_val = getattr(restr, "min_inclusive", None)
            max_val = getattr(restr, "max_inclusive", None)
            min_ex = getattr(restr, "min_exclusive", None)
            max_ex = getattr(restr, "max_exclusive", None)
            validators_list = kwargs.setdefault("validators", [])
            if min_val is not None:
                validators_list.append(
                    RawCode(f"MinValueValidator({int(min_val) if isinstance(min_val, int) else min_val})")
                )
                try:
                    self.used_validators.add("MinValueValidator")
                except Exception:
                    pass
            if max_val is not None:
                validators_list.append(
                    RawCode(f"MaxValueValidator({int(max_val) if isinstance(max_val, int) else max_val})")
                )
                try:
                    self.used_validators.add("MaxValueValidator")
                except Exception:
                    pass
            if min_ex is not None:
                adj = int(min_ex) + 1 if isinstance(min_ex, int) else min_ex
                validators_list.append(RawCode(f"MinValueValidator({adj})"))
                try:
                    self.used_validators.add("MinValueValidator")
                except Exception:
                    pass
            if max_ex is not None:
                adj = int(max_ex) - 1 if isinstance(max_ex, int) else max_ex
                validators_list.append(RawCode(f"MaxValueValidator({adj})"))
                try:
                    self.used_validators.add("MaxValueValidator")
                except Exception:
                    pass

        # Determine the base Django field type
        base_field_class = self.FIELD_TYPE_MAP.get(simple_type.base_type, models.CharField)

        if simple_type.base_type == XmlSchemaType.STRING and (not restr or not getattr(restr, "max_length", None)):
            # If a pattern is present, it's likely a constrained string that should be a CharField
            if not (restr and getattr(restr, "pattern", None)):
                base_field_class = models.TextField
                # TextField doesn't accept max_length, so remove it if it was defaulted
                kwargs.pop("max_length", None)

        # Remove max_length for numeric fields and optionally map precision
        if issubclass(base_field_class, (models.IntegerField, models.BigIntegerField, models.SmallIntegerField)):
            kwargs.pop("max_length", None)
        if issubclass(base_field_class, models.DecimalField):
            kwargs.pop("max_length", None)
            # Map precision if available
            total = getattr(restr, "total_digits", None)
            frac = getattr(restr, "fraction_digits", None)
            if total is not None:
                kwargs["max_digits"] = int(total)
            if frac is not None:
                kwargs["decimal_places"] = int(frac)

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
        schema_def = carrier.source_model.schema_def

        # Handle references to complex types (nested objects)
        if element.base_type is None and element.type_name and schema_def:
            target_type_name = element.type_name.split(":")[-1]
            # Only create relations to complex types that are slated for generation
            allowed = True
            if self.included_model_names is not None and target_type_name not in self.included_model_names:
                allowed = False

            if allowed and target_type_name in schema_def.complex_types:
                strategy = self._decide_nested_strategy(current_depth=1)
                # Repeating complex elements
                if element.is_list:
                    if strategy == "json" or self.list_relationship_style == "json" or not allowed:
                        return self._make_json_field_kwargs(element)
                    if self.list_relationship_style == "m2m":
                        kwargs = {"to": f"{carrier.meta_app_label}.{target_type_name}", "blank": True}
                        return models.ManyToManyField, kwargs
                    # Default: child_fk -> defer actual FK creation to child model via finalize step
                    # Store pending relation on the model factory via carrier.context_data
                    pending = carrier.context_data.setdefault("_pending_child_fk", [])
                    pending.append(
                        {
                            "child": target_type_name,
                            "parent": getattr(carrier.source_model, "__name__", model_name),
                            "element_name": element.name,
                        }
                    )
                    # Represent on parent as a reverse accessor only; no concrete field needed
                    # Use a JSONField as placeholder if desired by strategy
                    return self._make_json_field_kwargs(element) if strategy == "json" else (None, {})

                # Single nested complex element
                if strategy == "json" or not allowed:
                    return self._make_json_field_kwargs(element)
                # FK on parent to child, unless hypertable -> hypertable
                roles = carrier.context_data.get("_timescale_roles", {})
                src_name = str(getattr(carrier.source_model, "__name__", model_name))
                # Hypertable->Hypertable => soft reference
                if should_soft_reference(src_name, str(target_type_name), roles):
                    # Soft reference field with index; default to UUID representation
                    soft_kwargs = {"null": True, "blank": True}
                    try:
                        soft_kwargs["db_index"] = True
                    except Exception:
                        pass
                    return models.UUIDField, soft_kwargs
                # Dimension->Hypertable => optionally invert FK to hypertable
                if self.invert_fk_to_timescale and should_invert_fk(src_name, str(target_type_name), roles):
                    # Record pending inverted FK to inject later on the hypertable model
                    pending_inv = carrier.context_data.setdefault("_pending_inverted_fk", [])
                    pending_inv.append(
                        {
                            "hypertable": str(target_type_name),
                            "dimension": src_name,
                            "field_name": src_name.lower(),
                        }
                    )
                    # Do not create a FK on the dimension side
                    return None, {}
                kwargs = {"to": f"{carrier.meta_app_label}.{target_type_name}", "on_delete": models.SET_NULL}
                kwargs["null"] = True
                kwargs["blank"] = True
                return models.ForeignKey, kwargs

        # Default simple-type mapping
        field_class = self.FIELD_TYPE_MAP.get(element.base_type, models.CharField)
        kwargs = {}

        # Treat element-based xs:ID as a primary key using CharField
        if element.base_type == XmlSchemaType.ID:
            field_class = models.CharField
            kwargs["max_length"] = 255
            kwargs["primary_key"] = True
            # ID fields should not be nullable
            kwargs.pop("null", None)
            kwargs.pop("blank", None)
            return field_class, kwargs

        if element.nillable or element.min_occurs == 0:
            kwargs["null"] = True
            kwargs["blank"] = True

        # Apply inline restrictions if present (including enums, patterns, bounds)
        if element.restrictions:
            tmp_simple = XmlSchemaSimpleType(
                name=element.name,
                base_type=element.base_type,
                restrictions=element.restrictions,
            )
            restr = element.restrictions
            if getattr(restr, "enumeration", None):
                return self._create_enum_field(
                    tmp_simple, XmlSchemaFieldInfo(name=element.name, element=element), model_name, carrier
                )
            # Otherwise apply restriction mapping (validators, lengths, numeric bounds)
            return self._apply_simple_type_restrictions(
                tmp_simple, XmlSchemaFieldInfo(name=element.name, element=element)
            )

        if field_class == models.CharField:
            if element.nillable:
                field_class = models.TextField
                kwargs.pop("max_length", None)
            else:
                kwargs.setdefault("max_length", 255)

        if element.default_value:
            kwargs["default"] = element.default_value

        if element.base_type is None and element.type_name:
            # If we reach here, the referenced type couldn't be resolved to a complex/simple mapping
            # Fallback to JSON for safety
            json_field, json_kwargs = self._make_json_field_kwargs(element)
            return json_field, json_kwargs

        return field_class, kwargs

    def _decide_nested_strategy(self, current_depth: int) -> str:
        """Decide how to represent nested complex types based on configuration."""
        if self.nested_relationship_strategy in {"fk", "json"}:
            return self.nested_relationship_strategy
        # auto: depth-based
        return "fk" if current_depth <= self.nesting_depth_threshold else "json"

    def _make_json_field_kwargs(self, element: XmlSchemaElement):
        kwargs: dict = {}
        if element.nillable or element.min_occurs == 0:
            kwargs["null"] = True
            kwargs["blank"] = True
        return models.JSONField, kwargs

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
    ) -> tuple[type[models.Field], dict]:
        """Creates a ForeignKey field from an IDREF attribute or element."""
        schema_def = carrier.source_model.schema_def
        # Default to CASCADE; may be overridden to SET_NULL if optional
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

                # Decide on FK vs soft reference using Timescale classification
                roles = carrier.context_data.get("_timescale_roles", {})
                src_name = str(getattr(carrier.source_model, "__name__", model_name))
                if should_soft_reference(src_name, str(target_model_name), roles):
                    # Emit a soft reference field (UUID) instead of FK
                    soft_kwargs = {"null": True, "blank": True, "db_index": True}
                    # Return UUIDField tuple indicator by piggy-backing the return contract
                    return models.UUIDField, soft_kwargs
                kwargs["to"] = f"{carrier.meta_app_label}.{target_model_name}"

                # Use the keyref selector to generate a better related_name
                if keyref.selector:
                    related_name_base = keyref.selector.split(":")[-1].replace(".//", "")
                    related_name = f"{related_name_base.lower()}s"
                    kwargs["related_name"] = related_name
                else:
                    kwargs["related_name"] = f"{model_name.lower()}s"

        # Determine optionality to adjust on_delete/null/blank
        is_optional = False
        if field_info.attribute is not None:
            is_optional = field_info.attribute.use == "optional"
        elif field_info.element is not None:
            is_optional = field_info.element.nillable or field_info.element.min_occurs == 0

        if is_optional:
            kwargs["on_delete"] = models.SET_NULL
            kwargs["null"] = True
            kwargs["blank"] = True

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
        self, type_name: str | None, schema_def: XmlSchemaDefinition
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
        # Derive enum class name from the field name via core naming utilities
        try:
            from ..core.utils.naming import enum_class_name_from_field

            enum_name_base = enum_class_name_from_field(field_info.name)
        except Exception:
            enum_name_base = str(field_info.name).replace("_", " ").title().replace(" ", "")

        # Add a suffix to avoid clashes with model names
        enum_class_name = f"{enum_name_base}"

        # Store enums in the context_data of the carrier to share them across the generation process
        if "enums" not in carrier.context_data:
            carrier.context_data["enums"] = {}

        if enum_class_name in carrier.context_data["enums"]:
            return enum_class_name, False

        choices = []
        restr = getattr(simple_type, "restrictions", None)
        if restr and getattr(restr, "enumeration", None):
            for value, label in restr.enumeration:
                # Use the value to form the enum member name, label for human-readable
                enum_member_name = value.replace("-", " ").upper().replace(" ", "_")
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

    def __init__(
        self,
        app_label: str,
        *,
        nested_relationship_strategy: str = "auto",
        list_relationship_style: str = "child_fk",
        nesting_depth_threshold: int = 1,
    ):
        self.app_label = app_label
        self.field_factory = XmlSchemaFieldFactory(
            nested_relationship_strategy=nested_relationship_strategy,
            list_relationship_style=list_relationship_style,
            nesting_depth_threshold=nesting_depth_threshold,
        )
        # Track pending child FK relations to inject after all models exist
        self._pending_child_fk: list[dict] = []

    def _handle_field_result(self, result: FieldConversionResult, carrier: ConversionCarrier[XmlSchemaComplexType]):
        """Handle the result of field conversion and add to appropriate carrier containers."""
        if result.django_field:
            # Normalize XML names to valid Django-style field names
            try:
                from ..core.utils.naming import sanitize_field_identifier

                out_field_name = sanitize_field_identifier(result.field_name)
            except Exception:
                # Fallback: basic lowering
                out_field_name = str(result.field_name).lower()
            # Avoid generating a plain 'id' field unless it's a primary key; rename to prevent Django E004
            try:
                if out_field_name.lower() == "id" and not getattr(result.django_field, "primary_key", False):
                    out_field_name = "xml_id"
            except Exception:
                pass

            carrier.django_fields[out_field_name] = result.django_field
            if result.field_definition_str:
                carrier.django_field_definitions[out_field_name] = result.field_definition_str
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
            # Collect pending child FK relations from this carrier's context
            pending = carrier.context_data.pop("_pending_child_fk", [])
            if pending:
                self._pending_child_fk.extend(pending)

        # Ensure a concrete model exists even if no fields were produced, to satisfy relations.
        if not carrier.django_fields and not carrier.relationship_fields:
            try:
                # Add a primary key to register a concrete model
                carrier.django_fields["id"] = models.AutoField(primary_key=True)
                # Also register a definition string for static emission
                from ..django.utils.serialization import generate_field_definition_string

                pk_def = generate_field_definition_string(models.AutoField, {"primary_key": True}, self.app_label)
                carrier.django_field_definitions["id"] = pk_def
            except Exception:
                # As a fallback, keep it context-only
                pass

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

    # --- Post-processing for cross-model relationships ---
    def finalize_relationships(
        self, carriers_by_name: dict[str, ConversionCarrier[XmlSchemaComplexType]], app_label: str
    ):
        """
        After all models are built, inject ForeignKey fields into child models
        for repeating nested complex elements (one-to-many parent->child).
        """
        # Process pending child FKs first (if any)
        for rel in self._pending_child_fk:
            child_name = rel.get("child")
            parent_name = rel.get("parent")
            element_name = rel.get("element_name", "items")
            child_carrier = carriers_by_name.get(child_name)
            if not child_carrier or not child_carrier.django_model:
                logger.warning(f"Could not inject child FK: missing carrier for {child_name}")
                continue
            # Build kwargs for FK on child -> parent
            rn_base = element_name.lower()
            if not rn_base.endswith("s"):
                rn_base = rn_base + "s"
            related_name = rn_base
            # Use Timescale roles to decide FK vs soft reference
            roles = child_carrier.context_data.get("_timescale_roles", {})
            child_name_str = str(child_name) if child_name else ""
            parent_name_str = str(parent_name) if parent_name else ""
            if should_soft_reference(child_name_str, parent_name_str, roles):
                # Soft reference: UUID field with index
                soft_kwargs = {"db_index": True}
                # Soft refs on child->parent are often required; keep nullable off by default? Use safe default True.
                soft_kwargs["null"] = True
                soft_kwargs["blank"] = True
                from ..django.utils.serialization import generate_field_definition_string

                soft_def = generate_field_definition_string(models.UUIDField, soft_kwargs, app_label)
                field_name = f"{parent_name.lower()}"
                child_carrier.django_field_definitions[field_name] = soft_def
            else:
                fk_kwargs = {
                    "to": f"{app_label}.{parent_name}",
                    "on_delete": "models.CASCADE",
                    "related_name": related_name,
                }
                from ..django.utils.serialization import generate_field_definition_string

                fk_def = generate_field_definition_string(models.ForeignKey, fk_kwargs, app_label)
                field_name = f"{parent_name.lower()}"
                child_carrier.django_field_definitions[field_name] = fk_def

        # Handle inverted Timescale relationships: add FK on hypertable -> dimension
        # and emit recommended indexes
        for carrier in carriers_by_name.values():
            pending_inverted = carrier.context_data.get("_pending_inverted_fk", [])
            for inv in pending_inverted:
                hyper = inv.get("hypertable")
                dim = inv.get("dimension")
                field_name = inv.get("field_name", (dim or "").lower())
                hyper_carrier = carriers_by_name.get(str(hyper))
                if not hyper_carrier or not hyper_carrier.django_model:
                    logger.warning(f"Could not inject inverted FK: missing hypertable carrier for {hyper}")
                    continue
                # Build FK kwargs: SET_NULL safe policy
                from ..django.utils.serialization import RawCode

                fk_kwargs = {
                    "to": f"{app_label}.{dim}",
                    "on_delete": RawCode("models.SET_NULL"),
                    "null": True,
                    "blank": True,
                }
                from ..django.utils.serialization import generate_field_definition_string

                fk_def = generate_field_definition_string(models.ForeignKey, fk_kwargs, app_label)
                hyper_carrier.django_field_definitions[field_name] = fk_def

                # Auto-indexes: (dimension_id) and (dimension_id, -time) if time field exists
                idx_list: list[str] = hyper_carrier.context_data.setdefault("meta_indexes", [])
                # Single-column index on FK
                idx_list.append(f"models.Index(fields=['{field_name}'])")
                # Composite with time if defined on the hypertable
                has_time = "time" in hyper_carrier.django_field_definitions or "time" in hyper_carrier.django_fields
                if has_time:
                    idx_list.append(f"models.Index(fields=['{field_name}', '-time'])")

    # --- Override meta class creation to keep dynamic models abstract (avoid registry conflicts) ---
    def _create_django_meta(self, carrier: ConversionCarrier[XmlSchemaComplexType]):
        """Create the Meta class for the generated Django model (abstract for dynamic xmlschema models)."""
        source_name = getattr(carrier.source_model, "__name__", "UnknownSourceModel")
        source_model_name_cleaned = source_name.replace("_", " ")
        meta_attrs = {
            "app_label": self.app_label,
            "db_table": f"{self.app_label}_{source_name.lower()}",
            # Keep abstract to avoid polluting Django's app registry during generation
            "abstract": True,
            "managed": True,
            "verbose_name": source_model_name_cleaned,
            "verbose_name_plural": source_model_name_cleaned + "s",
            "ordering": ["pk"],
        }

        logger.debug("Creating Meta class for xmlschema model (abstract=True)")
        carrier.django_meta_class = type("Meta", (), meta_attrs)
