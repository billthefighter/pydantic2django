"""
XML Schema data models for representing parsed XSD structures.
These models represent the parsed XML Schema definitions before Django model conversion.
"""
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class XmlSchemaType(Enum):
    """XML Schema built-in data types."""

    # String types
    STRING = "xs:string"
    NORMALIZEDSTRING = "xs:normalizedString"
    TOKEN = "xs:token"

    # Numeric types
    INTEGER = "xs:integer"
    LONG = "xs:long"
    INT = "xs:int"
    SHORT = "xs:short"
    BYTE = "xs:byte"
    DECIMAL = "xs:decimal"
    FLOAT = "xs:float"
    DOUBLE = "xs:double"

    # Unsigned numeric types
    UNSIGNEDLONG = "xs:unsignedLong"
    UNSIGNEDINT = "xs:unsignedInt"
    UNSIGNEDSHORT = "xs:unsignedShort"
    UNSIGNEDBYTE = "xs:unsignedByte"

    # Specialized numeric types
    POSITIVEINTEGER = "xs:positiveInteger"
    NONNEGATIVEINTEGER = "xs:nonNegativeInteger"
    NEGATIVEINTEGER = "xs:negativeInteger"
    NONPOSITIVEINTEGER = "xs:nonPositiveInteger"

    # Boolean
    BOOLEAN = "xs:boolean"

    # Date and time types
    DATE = "xs:date"
    DATETIME = "xs:dateTime"
    TIME = "xs:time"
    DURATION = "xs:duration"
    GYEAR = "xs:gYear"
    GMONTH = "xs:gMonth"
    GDAY = "xs:gDay"
    GYEARMONTH = "xs:gYearMonth"
    GMONTHDAY = "xs:gMonthDay"

    # Binary types
    BASE64BINARY = "xs:base64Binary"
    HEXBINARY = "xs:hexBinary"

    # URI types
    ANYURI = "xs:anyURI"

    # Other types
    QNAME = "xs:QName"
    NOTATION = "xs:NOTATION"
    ID = "xs:ID"
    IDREF = "xs:IDREF"


@dataclass
class XmlSchemaImport:
    """Represents an XSD import statement."""

    namespace: str | None
    schema_location: str | None


@dataclass
class XmlSchemaRestriction:
    """Represents XML Schema restrictions/facets."""

    base: str | None = None
    min_length: int | None = None
    max_length: int | None = None
    length: int | None = None
    pattern: str | None = None
    enumeration: list[tuple[str, str]] = field(default_factory=list)
    min_inclusive: int | float | None = None
    max_inclusive: int | float | None = None
    min_exclusive: int | float | None = None
    max_exclusive: int | float | None = None
    total_digits: int | None = None
    fraction_digits: int | None = None
    white_space: str | None = None  # preserve, replace, collapse


@dataclass
class XmlSchemaKey:
    """Represents an <xs:key> definition."""

    name: str
    selector: str
    fields: list[str] = field(default_factory=list)


@dataclass
class XmlSchemaKeyRef:
    """Represents an <xs:keyref> definition."""

    name: str
    refer: str
    selector: str
    fields: list[str] = field(default_factory=list)


@dataclass
class XmlSchemaElement:
    """Represents an XML Schema element."""

    name: str
    base_type: XmlSchemaType | None = None
    type_name: str | None = None  # For references to complex types
    min_occurs: int = 1
    max_occurs: int | str = 1  # Can be "unbounded"
    is_list: bool = False
    default_value: Any | None = None
    fixed_value: Any | None = None
    nillable: bool = False
    abstract: bool = False
    substitution_group: str | None = None
    restrictions: XmlSchemaRestriction | None = None
    complex_type: "XmlSchemaComplexType | None" = None
    documentation: str | None = None
    namespace: str | None = None
    schema_location: str | None = None
    keys: list[XmlSchemaKey] = field(default_factory=list)
    keyrefs: list[XmlSchemaKeyRef] = field(default_factory=list)

    @property
    def is_optional(self) -> bool:
        """Check if element is optional (min_occurs = 0)."""
        return self.min_occurs == 0

    @property
    def is_required(self) -> bool:
        """Check if element is required (min_occurs > 0)."""
        return self.min_occurs > 0

    # @property
    # def is_list(self) -> bool:
    #     """Check if element can have multiple occurrences."""
    #     return (isinstance(self.max_occurs, int) and self.max_occurs > 1) or self.max_occurs == "unbounded"


@dataclass
class XmlSchemaAttribute:
    """Represents an XML Schema attribute."""

    name: str
    base_type: XmlSchemaType | None = None
    type_name: str | None = None  # For references to simple types
    use: str = "optional"  # required, optional, prohibited
    default_value: Any | None = None
    fixed_value: Any | None = None
    restrictions: XmlSchemaRestriction | None = None
    documentation: str | None = None
    namespace: str | None = None

    @property
    def is_required(self) -> bool:
        """Check if attribute is required."""
        return self.use == "required"

    @property
    def is_optional(self) -> bool:
        """Check if attribute is optional."""
        return self.use == "optional"


@dataclass
class XmlSchemaComplexType:
    """Represents an XML Schema complex type."""

    name: str
    namespace: str | None = None
    elements: list[XmlSchemaElement] = field(default_factory=list)
    attributes: dict[str, XmlSchemaAttribute] = field(default_factory=dict)
    base_type: str | None = None  # For inheritance
    mixed: bool = False  # Mixed content model
    abstract: bool = False
    sequence: bool = False
    choice: bool = False
    all_elements: bool = False
    documentation: str | None = None
    schema_location: str | None = None
    schema_def: "XmlSchemaDefinition | None" = field(default=None, repr=False)

    @property
    def __name__(self) -> str:
        return self.name

    # Schema linking fields
    conversion_session_id: str | None = field(default_factory=lambda: str(uuid.uuid4()))
    schema_source_file: str | None = None

    def add_element(self, element: XmlSchemaElement) -> None:
        """Add an element to this complex type."""
        self.elements.append(element)

    def add_attribute(self, attribute: XmlSchemaAttribute) -> None:
        """Add an attribute to this complex type."""
        self.attributes[attribute.name] = attribute

    def get_required_elements(self) -> list[XmlSchemaElement]:
        """Get all required elements (min_occurs > 0)."""
        return [elem for elem in self.elements if elem.is_required]

    def get_optional_elements(self) -> list[XmlSchemaElement]:
        """Get all optional elements (min_occurs = 0)."""
        return [elem for elem in self.elements if elem.is_optional]

    def get_required_attributes(self) -> list[XmlSchemaAttribute]:
        """Get all required attributes."""
        return [attr for attr in self.attributes.values() if attr.is_required]

    def has_list_elements(self) -> bool:
        """Check if any elements can have multiple occurrences."""
        return any(elem.is_list for elem in self.elements)

    def __hash__(self):
        return hash((self.name, self.namespace, self.schema_location))

    def __eq__(self, other):
        if not isinstance(other, XmlSchemaComplexType):
            return NotImplemented
        return (
            self.name == other.name
            and self.namespace == other.namespace
            and self.schema_location == other.schema_location
        )


@dataclass
class XmlSchemaSimpleType:
    """Represents an XML Schema simple type."""

    name: str
    namespace: str | None = None
    base_type: XmlSchemaType | None = None
    restrictions: XmlSchemaRestriction | None = None
    schema_location: str | None = None

    @property
    def __name__(self) -> str:
        return self.name

    # --- Backward-compatibility aliases/proxies ---
    @property
    def restriction(self) -> XmlSchemaRestriction | None:  # pragma: no cover - compatibility shim
        return self.restrictions

    @restriction.setter
    def restriction(self, value: XmlSchemaRestriction | None) -> None:  # pragma: no cover - compatibility shim
        self.restrictions = value

    @property
    def enumeration(self) -> list[tuple[str, str]]:  # pragma: no cover - compatibility shim
        return self.restrictions.enumeration if self.restrictions and self.restrictions.enumeration else []

    @enumeration.setter
    def enumeration(self, values: list[tuple[str, str]]) -> None:  # pragma: no cover - compatibility shim
        if self.restrictions is None:
            self.restrictions = XmlSchemaRestriction()
        self.restrictions.enumeration = list(values)


@dataclass
class XmlSchemaDefinition:
    """Represents a complete XML Schema definition."""

    schema_location: str
    target_namespace: str | None = None
    element_form_default: str = "unqualified"
    attribute_form_default: str = "unqualified"
    complex_types: dict[str, XmlSchemaComplexType] = field(default_factory=dict)
    simple_types: dict[str, XmlSchemaSimpleType] = field(default_factory=dict)
    elements: dict[str, XmlSchemaElement] = field(default_factory=dict)
    attributes: dict[str, XmlSchemaAttribute] = field(default_factory=dict)
    imports: list[XmlSchemaImport] = field(default_factory=list)
    includes: list[str] = field(default_factory=list)
    keys: list[XmlSchemaKey] = field(default_factory=list)
    keyrefs: list[XmlSchemaKeyRef] = field(default_factory=list)

    # Schema linking fields
    conversion_session_id: str | None = field(default_factory=lambda: str(uuid.uuid4()))

    def add_complex_type(self, complex_type: XmlSchemaComplexType) -> None:
        """Add a complex type to this schema."""
        complex_type.schema_location = self.schema_location
        complex_type.namespace = complex_type.namespace or self.target_namespace
        complex_type.conversion_session_id = self.conversion_session_id
        complex_type.schema_source_file = self.schema_location
        complex_type.schema_def = self
        self.complex_types[complex_type.name] = complex_type

    def add_simple_type(self, simple_type: XmlSchemaSimpleType) -> None:
        """Add a simple type to this schema."""
        simple_type.schema_location = self.schema_location
        simple_type.namespace = simple_type.namespace or self.target_namespace
        self.simple_types[simple_type.name] = simple_type

    def add_element(self, element: XmlSchemaElement) -> None:
        """Add a global element to this schema."""
        element.namespace = element.namespace or self.target_namespace
        self.elements[element.name] = element

    def get_all_complex_types(self) -> list[XmlSchemaComplexType]:
        """Get all complex types in this schema."""
        return list(self.complex_types.values())

    def get_all_simple_types(self) -> list[XmlSchemaSimpleType]:
        """Get all simple types in this schema."""
        return list(self.simple_types.values())

    def find_complex_type(self, name: str, namespace: str | None = None) -> XmlSchemaComplexType | None:
        """Find a complex type by name and optional namespace."""
        # Try exact match first
        if name in self.complex_types:
            complex_type = self.complex_types[name]
            if namespace is None or complex_type.namespace == namespace:
                return complex_type

        # Try namespace-qualified lookup
        for complex_type in self.complex_types.values():
            if complex_type.name == name and (namespace is None or complex_type.namespace == namespace):
                return complex_type

        return None


@dataclass
class XmlSchemaConversionSession:
    """
    Represents a complete conversion session linking all schema elements together.
    This provides the mechanism to reference all schema elements for a particular conversion.
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    schema_files: list[str] = field(default_factory=list)
    schemas: list[XmlSchemaDefinition] = field(default_factory=list)
    target_namespace: str | None = None
    app_label: str = "xmlschema_app"
    created_at: str | None = None  # ISO timestamp
    conversion_metadata: dict[str, Any] = field(default_factory=dict)

    def add_schema(self, schema: XmlSchemaDefinition) -> None:
        """Add a schema to this conversion session."""
        schema.conversion_session_id = self.session_id
        self.schemas.append(schema)
        if schema.schema_location not in self.schema_files:
            self.schema_files.append(schema.schema_location)

    def get_all_complex_types(self) -> list[XmlSchemaComplexType]:
        """Get all complex types from all schemas in this session."""
        all_types = []
        for schema in self.schemas:
            all_types.extend(schema.get_all_complex_types())
        return all_types

    def find_complex_type_across_schemas(self, name: str, namespace: str | None = None) -> XmlSchemaComplexType | None:
        """Find a complex type across all schemas in this session."""
        for schema in self.schemas:
            complex_type = schema.find_complex_type(name, namespace)
            if complex_type:
                return complex_type
        return None

    def get_conversion_statistics(self) -> dict[str, Any]:
        """Get statistics about this conversion session."""
        total_complex_types = sum(len(schema.complex_types) for schema in self.schemas)
        total_simple_types = sum(len(schema.simple_types) for schema in self.schemas)
        total_elements = sum(len(schema.elements) for schema in self.schemas)

        return {
            "session_id": self.session_id,
            "total_schemas": len(self.schemas),
            "total_complex_types": total_complex_types,
            "total_simple_types": total_simple_types,
            "total_elements": total_elements,
            "schema_files": self.schema_files,
            "target_namespace": self.target_namespace,
            "app_label": self.app_label,
        }
