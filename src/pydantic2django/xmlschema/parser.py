"""
XML Schema parser that converts XSD files into internal representation models.
Uses lxml for XML parsing and provides the foundation for the discovery pipeline.
"""
import logging
from pathlib import Path

try:
    from lxml import etree
except ImportError:
    etree = None

from .models import (
    XmlSchemaAttribute,
    XmlSchemaComplexType,
    XmlSchemaDefinition,
    XmlSchemaElement,
    XmlSchemaImport,
    XmlSchemaRestriction,
    XmlSchemaSimpleType,
    XmlSchemaType,
)

logger = logging.getLogger(__name__)


class XmlSchemaParseError(Exception):
    """Exception raised when XML Schema parsing fails"""

    pass


class XmlSchemaParser:
    """
    Parses XML Schema (XSD) files into internal representation models.
    Supports local files and basic schema imports/includes.
    """

    # XML Schema namespace
    XS_NAMESPACE = "http://www.w3.org/2001/XMLSchema"

    def __init__(self):
        if etree is None:
            raise ImportError("lxml is required for XML Schema parsing. Install with: pip install lxml")
        self.parsed_schemas: dict[str, XmlSchemaDefinition] = {}
        self.type_mappings = self._build_type_mappings()

    def _build_type_mappings(self) -> dict[str, XmlSchemaType]:
        """Build mapping from XSD type names to our enum values"""
        return {
            "string": XmlSchemaType.STRING,
            "normalizedString": XmlSchemaType.NORMALIZEDSTRING,
            "token": XmlSchemaType.TOKEN,
            "int": XmlSchemaType.INTEGER,
            "integer": XmlSchemaType.INTEGER,
            "long": XmlSchemaType.LONG,
            "short": XmlSchemaType.SHORT,
            "byte": XmlSchemaType.BYTE,
            "unsignedInt": XmlSchemaType.UNSIGNEDINT,
            "unsignedLong": XmlSchemaType.UNSIGNEDLONG,
            "positiveInteger": XmlSchemaType.POSITIVEINTEGER,
            "nonNegativeInteger": XmlSchemaType.NONNEGATIVEINTEGER,
            "decimal": XmlSchemaType.DECIMAL,
            "float": XmlSchemaType.FLOAT,
            "double": XmlSchemaType.DOUBLE,
            "boolean": XmlSchemaType.BOOLEAN,
            "date": XmlSchemaType.DATE,
            "dateTime": XmlSchemaType.DATETIME,
            "time": XmlSchemaType.TIME,
            "duration": XmlSchemaType.DURATION,
            "anyURI": XmlSchemaType.ANYURI,
            "base64Binary": XmlSchemaType.BASE64BINARY,
            "hexBinary": XmlSchemaType.HEXBINARY,
            "QName": XmlSchemaType.QNAME,
            "ID": XmlSchemaType.STRING,
        }

    def parse_schema_file(self, schema_path: str | Path) -> XmlSchemaDefinition:
        """
        Parse a single XSD file into an XmlSchemaDefinition.

        Args:
            schema_path: Path to the XSD file

        Returns:
            Parsed schema definition

        Raises:
            XmlSchemaParseError: If parsing fails
        """
        schema_path = Path(schema_path)

        if not schema_path.exists():
            raise XmlSchemaParseError(f"Schema file not found: {schema_path}")

        try:
            logger.info(f"Parsing XML Schema: {schema_path}")

            # Parse the XML document
            with open(schema_path, "rb") as f:
                tree = etree.parse(f)
            root = tree.getroot()

            # Verify it's a schema document
            if root.tag != f"{{{self.XS_NAMESPACE}}}schema":
                raise XmlSchemaParseError(f"Not a valid XML Schema document: {schema_path}")

            # Create schema definition
            schema_def = XmlSchemaDefinition(
                schema_location=str(schema_path),
                target_namespace=root.get("targetNamespace"),
                element_form_default=root.get("elementFormDefault", "unqualified"),
                attribute_form_default=root.get("attributeFormDefault", "unqualified"),
            )

            # Parse schema contents
            self._parse_schema_contents(root, schema_def)

            # Cache the parsed schema
            self.parsed_schemas[str(schema_path)] = schema_def

            logger.info(f"Successfully parsed schema with {len(schema_def.complex_types)} complex types")
            return schema_def

        except etree.XMLSyntaxError as e:
            raise XmlSchemaParseError(f"XML syntax error in {schema_path}: {e}") from e
        except Exception as e:
            raise XmlSchemaParseError(f"Failed to parse schema {schema_path}: {e}") from e

    def _parse_schema_contents(self, schema_root: "etree.Element", schema_def: XmlSchemaDefinition):
        """Parse the contents of a schema element"""
        logger.info(f"Parsing contents of schema: {schema_def.schema_location}")
        for child in schema_root:
            # Skip comments and other non-element nodes
            if not isinstance(child.tag, str):
                continue

            tag_name = self._get_local_name(child.tag)

            if tag_name == "complexType":
                complex_type = self._parse_complex_type(child, schema_def)
                if complex_type:
                    logger.info(f"Parsed complexType: {complex_type.name}")
                    schema_def.complex_types[complex_type.name] = complex_type

            elif tag_name == "simpleType":
                simple_type = self._parse_simple_type(child, schema_def)
                if simple_type:
                    logger.info(f"Parsed simpleType: {simple_type.name}")
                    schema_def.simple_types[simple_type.name] = simple_type

            elif tag_name == "element":
                element = self._parse_element(child, schema_def)
                if element:
                    logger.info(f"Parsed element: {element.name}")
                    schema_def.elements[element.name] = element

            elif tag_name == "attribute":
                attribute = self._parse_attribute(child, schema_def)
                if attribute:
                    schema_def.attributes[attribute.name] = attribute

            elif tag_name == "import":
                import_info = XmlSchemaImport(
                    namespace=child.get("namespace"), schema_location=child.get("schemaLocation")
                )
                schema_def.imports.append(import_info)

            elif tag_name == "include":
                schema_location = child.get("schemaLocation")
                if schema_location:
                    schema_def.includes.append(schema_location)

    def _parse_complex_type(
        self, element: "etree.Element", schema_def: XmlSchemaDefinition, name_override: str | None = None
    ) -> XmlSchemaComplexType | None:
        """Parse a complexType element"""
        name = name_override or element.get("name")
        if not name:
            logger.warning("Skipping anonymous complex type")
            return None

        complex_type = XmlSchemaComplexType(
            name=name,
            abstract=element.get("abstract", "false").lower() == "true",
            mixed=element.get("mixed", "false").lower() == "true",
            namespace=schema_def.target_namespace,
            schema_location=schema_def.schema_location,
        )

        # Parse documentation
        doc_elem = element.find(f"{{{self.XS_NAMESPACE}}}annotation/{{{self.XS_NAMESPACE}}}documentation")
        if doc_elem is not None and doc_elem.text:
            complex_type.documentation = doc_elem.text.strip()

        logger.debug(f"Parsing content for complexType: {name}")
        # Parse content model (sequence, choice, all)
        for child in element:
            tag_name = self._get_local_name(child.tag)
            logger.debug(f"  - Found tag in {name}: {tag_name}")

            if tag_name == "sequence":
                complex_type.sequence = True
                complex_type.choice = False
                self._parse_particle_content(child, complex_type, schema_def)

            elif tag_name == "choice":
                complex_type.choice = True
                complex_type.sequence = False
                self._parse_particle_content(child, complex_type, schema_def)

            elif tag_name == "all":
                complex_type.all_elements = True
                complex_type.sequence = False
                self._parse_particle_content(child, complex_type, schema_def)

            elif tag_name == "attribute":
                attribute = self._parse_attribute(child, schema_def)
                if attribute:
                    complex_type.attributes[attribute.name] = attribute

            elif tag_name == "complexContent":
                # Handle inheritance/extension
                self._parse_complex_content(child, complex_type, schema_def)

            elif tag_name == "simpleContent":
                # Handle simple content with attributes
                self._parse_simple_content(child, complex_type, schema_def)

        return complex_type

    def _parse_particle_content(
        self, particle: "etree.Element", complex_type: XmlSchemaComplexType, schema_def: XmlSchemaDefinition
    ):
        """Parse sequence, choice, or all content"""
        logger.debug(f"Parsing particle content for {complex_type.name}")
        for child in particle:
            tag_name = self._get_local_name(child.tag)
            logger.debug(f"  - Found particle child in {complex_type.name}: {tag_name}")

            if tag_name == "element":
                element = self._parse_element(child, schema_def)
                if element:
                    complex_type.elements.append(element)

            elif tag_name in ("sequence", "choice", "all"):
                # Nested groups
                self._parse_particle_content(child, complex_type, schema_def)

        logger.debug(f"Finished particle content for {complex_type.name}. Total elements: {len(complex_type.elements)}")

    def _parse_element(self, element: "etree.Element", schema_def: XmlSchemaDefinition) -> XmlSchemaElement | None:
        """Parse an element definition"""
        name = element.get("name")
        ref = element.get("ref")

        if not name and not ref:
            logger.warning("Element without name or ref, skipping")
            return None

        xml_element = XmlSchemaElement(
            name=name or ref,
            type_name=element.get("type"),
            min_occurs=int(element.get("minOccurs", "1")),
            max_occurs=element.get("maxOccurs", "1"),
            nillable=element.get("nillable", "false").lower() == "true",
            default_value=element.get("default"),
            fixed_value=element.get("fixed"),
            abstract=element.get("abstract", "false").lower() == "true",
            namespace=schema_def.target_namespace,
            schema_location=schema_def.schema_location,
        )

        # Handle maxOccurs="unbounded"
        if xml_element.max_occurs == "unbounded":
            pass  # Keep as string
        else:
            try:
                xml_element.max_occurs = int(xml_element.max_occurs)
            except (ValueError, TypeError):
                xml_element.max_occurs = 1

        # Parse documentation
        doc_elem = element.find(f"{{{self.XS_NAMESPACE}}}annotation/{{{self.XS_NAMESPACE}}}documentation")
        if doc_elem is not None and doc_elem.text:
            xml_element.documentation = doc_elem.text.strip()

        # Map built-in types
        if xml_element.type_name:
            type_local_name = self._get_local_name(xml_element.type_name)
            if type_local_name in self.type_mappings:
                xml_element.base_type = self.type_mappings[type_local_name]

        # Parse inline complex type
        inline_complex = element.find(f"{{{self.XS_NAMESPACE}}}complexType")
        if inline_complex is not None:
            # Create a synthetic name for the inline type
            synthetic_name = f"{name}_Type" if name else "InlineType"
            inline_type = self._parse_complex_type(inline_complex, schema_def, name_override=synthetic_name)
            if inline_type:
                schema_def.complex_types[inline_type.name] = inline_type
                xml_element.complex_type = inline_type

        # Parse restrictions (simpleType with restrictions)
        simple_type_elem = element.find(f"{{{self.XS_NAMESPACE}}}simpleType")
        if simple_type_elem is not None:
            restriction_elem = simple_type_elem.find(f"{{{self.XS_NAMESPACE}}}restriction")
            if restriction_elem is not None:
                xml_element.restrictions = self._parse_restriction(restriction_elem)
                base_type = restriction_elem.get("base")
                if base_type:
                    type_local_name = self._get_local_name(base_type)
                    if type_local_name in self.type_mappings:
                        xml_element.base_type = self.type_mappings[type_local_name]

        return xml_element

    def _parse_attribute(self, element: "etree.Element", schema_def: XmlSchemaDefinition) -> XmlSchemaAttribute | None:
        """Parse an attribute definition"""
        name = element.get("name")
        ref = element.get("ref")

        if not name and not ref:
            logger.warning("Attribute without name or ref, skipping")
            return None

        attribute = XmlSchemaAttribute(
            name=name or ref,
            type_name=element.get("type"),
            use=element.get("use", "optional"),
            default_value=element.get("default"),
            fixed_value=element.get("fixed"),
            namespace=schema_def.target_namespace,
        )

        # Map built-in types
        if attribute.type_name:
            type_local_name = self._get_local_name(attribute.type_name)
            if type_local_name in self.type_mappings:
                attribute.base_type = self.type_mappings[type_local_name]

        return attribute

    def _parse_simple_type(
        self, element: "etree.Element", schema_def: XmlSchemaDefinition
    ) -> XmlSchemaSimpleType | None:
        """Parse a simpleType definition"""
        name = element.get("name")
        if not name:
            logger.warning("Skipping anonymous simple type")
            return None

        simple_type = XmlSchemaSimpleType(
            name=name,
            namespace=schema_def.target_namespace,
        )

        # Parse documentation
        doc_elem = element.find(f"{{{self.XS_NAMESPACE}}}annotation/{{{self.XS_NAMESPACE}}}documentation")
        if doc_elem is not None and doc_elem.text:
            simple_type.documentation = doc_elem.text.strip()

        # Parse restriction
        restriction_elem = element.find(f"{{{self.XS_NAMESPACE}}}restriction")
        if restriction_elem is not None:
            base_type = restriction_elem.get("base")
            if base_type:
                type_local_name = self._get_local_name(base_type)
                if type_local_name in self.type_mappings:
                    simple_type.base_type = self.type_mappings[type_local_name]

            simple_type.restrictions = self._parse_restriction(restriction_elem)

            # Parse enumeration values separately
            enum_values = []
            for facet in restriction_elem.findall(f"{{{self.XS_NAMESPACE}}}enumeration"):
                value = facet.get("value")
                if value:
                    doc_elem = facet.find(f"{{{self.XS_NAMESPACE}}}annotation/{{{self.XS_NAMESPACE}}}documentation")
                    label = doc_elem.text.strip() if doc_elem is not None and doc_elem.text else value
                    enum_values.append((value, label))
            if enum_values:
                simple_type.enumeration = enum_values

        return simple_type

    def _parse_restriction(self, restriction_elem: "etree.Element") -> XmlSchemaRestriction:
        """Parse a restriction element"""
        restriction = XmlSchemaRestriction(base=restriction_elem.get("base"))

        # Map facet elements to restriction attributes
        facet_mapping = {
            "minLength": ("min_length", int),
            "maxLength": ("max_length", int),
            "length": ("length", int),
            "minInclusive": ("min_inclusive", float),
            "maxInclusive": ("max_inclusive", float),
            "minExclusive": ("min_exclusive", float),
            "maxExclusive": ("max_exclusive", float),
            "pattern": ("pattern", str),
            "fractionDigits": ("fraction_digits", int),
            "totalDigits": ("total_digits", int),
            "whiteSpace": ("white_space", str),
        }

        for facet_elem in restriction_elem:
            facet_name = self._get_local_name(facet_elem.tag)
            if facet_name in facet_mapping:
                attr_name, converter = facet_mapping[facet_name]
                value = facet_elem.get("value")
                if value:
                    try:
                        setattr(restriction, attr_name, converter(value))
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid {facet_name} value '{value}': {e}")
            elif facet_name == "enumeration":
                # Handle enumeration separately as it can have multiple values
                if restriction.enumeration is None:
                    restriction.enumeration = []
                value = facet_elem.get("value")
                if value:
                    doc_elem = facet_elem.find(
                        f"{{{self.XS_NAMESPACE}}}annotation/{{{self.XS_NAMESPACE}}}documentation"
                    )
                    label = doc_elem.text.strip() if doc_elem is not None and doc_elem.text else value
                    restriction.enumeration.append((value, label))

        return restriction

    def _parse_complex_content(
        self, complex_content: "etree.Element", complex_type: XmlSchemaComplexType, schema_def: XmlSchemaDefinition
    ):
        """Parse complexContent (inheritance/extension)"""
        extension = complex_content.find(f"{{{self.XS_NAMESPACE}}}extension")
        if extension is not None:
            base_type = extension.get("base")
            if base_type:
                complex_type.base_type = base_type

            # Parse attributes in the extension
            for child in extension:
                if self._get_local_name(child.tag) == "attribute":
                    attribute = self._parse_attribute(child, schema_def)
                    if attribute:
                        complex_type.attributes[attribute.name] = attribute

            # Parse additional content in the extension
            self._parse_particle_content(extension, complex_type, schema_def)

    def _parse_simple_content(
        self, simple_content: "etree.Element", complex_type: XmlSchemaComplexType, schema_def: XmlSchemaDefinition
    ):
        """Parse simpleContent (simple content with attributes)"""
        extension = simple_content.find(f"{{{self.XS_NAMESPACE}}}extension")
        if extension is not None:
            base_type = extension.get("base")
            if base_type:
                complex_type.base_type = base_type

            # Parse attributes in the extension
            for child in extension:
                if self._get_local_name(child.tag) == "attribute":
                    attribute = self._parse_attribute(child, schema_def)
                    if attribute:
                        complex_type.attributes[attribute.name] = attribute

    def _get_local_name(self, qname: str) -> str:
        """Extract local name from a qualified name"""
        if "}" in qname:
            return qname.split("}")[-1]
        return qname.split(":")[-1]

    def parse_multiple_schemas(self, schema_paths: list[str | Path]) -> list[XmlSchemaDefinition]:
        """
        Parse multiple XSD files and return their definitions.

        Args:
            schema_paths: List of paths to XSD files

        Returns:
            List of parsed schema definitions
        """
        schemas = []
        for path in schema_paths:
            try:
                schema = self.parse_schema_file(path)
                schemas.append(schema)
            except XmlSchemaParseError as e:
                logger.error(f"Failed to parse schema {path}: {e}")
                # Continue with other schemas
                continue

        return schemas
