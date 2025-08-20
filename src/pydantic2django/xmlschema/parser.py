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
    XmlSchemaKey,
    XmlSchemaKeyRef,
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
            "gYear": XmlSchemaType.GYEAR,
            "anyURI": XmlSchemaType.ANYURI,
            "base64Binary": XmlSchemaType.BASE64BINARY,
            "hexBinary": XmlSchemaType.HEXBINARY,
            "QName": XmlSchemaType.QNAME,
            "ID": XmlSchemaType.ID,
            "IDREF": XmlSchemaType.IDREF,
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
        logger.debug(f"Root element: {schema_root.tag}")
        for child in schema_root:
            if not isinstance(child.tag, str):
                logger.debug(f"Skipping non-string tag: {child.tag}")
                continue
            tag_name = self._get_local_name(child.tag)
            logger.debug(f"Found child tag: {tag_name}")

            if tag_name == "complexType":
                complex_type = self._parse_complex_type(child, schema_def)
                if complex_type:
                    schema_def.add_complex_type(complex_type)
            elif tag_name == "simpleType":
                simple_type = self._parse_simple_type(child, schema_def)
                if simple_type:
                    schema_def.add_simple_type(simple_type)
            elif tag_name == "element":
                element = self._parse_element(child, schema_def)
                if element:
                    schema_def.add_element(element)
            elif tag_name == "key":
                key = self._parse_key(child)
                if key:
                    schema_def.keys.append(key)
            elif tag_name == "keyref":
                keyref = self._parse_keyref(child)
                if keyref:
                    schema_def.keyrefs.append(keyref)

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

        # Keys and keyrefs can be defined within an element
        for key_elem in element.findall(f"{{{self.XS_NAMESPACE}}}key"):
            key = self._parse_key(key_elem)
            if key:
                schema_def.keys.append(key)

        for keyref_elem in element.findall(f"{{{self.XS_NAMESPACE}}}keyref"):
            keyref = self._parse_keyref(keyref_elem)
            if keyref:
                schema_def.keyrefs.append(keyref)

        # If it's a reference, we might need to look it up later
        if ref:
            # For now, just store the ref. Resolution can happen in a later pass.
            pass

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
            xml_element.is_list = True
            xml_element.max_occurs = -1  # Or some indicator of unbounded
        else:
            xml_element.max_occurs = int(xml_element.max_occurs)

        # Map built-in types
        if xml_element.type_name:
            type_local_name = self._get_local_name(xml_element.type_name)
            if type_local_name in self.type_mappings:
                xml_element.base_type = self.type_mappings[type_local_name]

        # Look for an inline complexType and parse it
        inline_complex_type_elem = element.find(f"{{{self.XS_NAMESPACE}}}complexType")
        if inline_complex_type_elem is not None:
            # Synthesize a name for the inline complex type
            type_name = f"{xml_element.name}_Type"
            logger.debug(f"Found inline complexType for element '{xml_element.name}'. Creating type '{type_name}'.")

            # Parse the complex type
            complex_type = self._parse_complex_type(inline_complex_type_elem, schema_def, name_override=type_name)

            if complex_type:
                schema_def.add_complex_type(complex_type)
                xml_element.type_name = type_name  # Associate element with new type

        # Handle inline simpleType as well
        inline_simple_type_elem = element.find(f"{{{self.XS_NAMESPACE}}}simpleType")
        if inline_simple_type_elem is not None:
            # Synthesize a name for the inline simple type
            type_name = f"{xml_element.name}_SimpleType"
            simple_type = self._parse_simple_type(inline_simple_type_elem, schema_def, name_override=type_name)
            if simple_type:
                schema_def.add_simple_type(simple_type)
                xml_element.type_name = type_name

        # Parse documentation
        doc_elem = element.find(f"{{{self.XS_NAMESPACE}}}annotation/{{{self.XS_NAMESPACE}}}documentation")
        if doc_elem is not None and doc_elem.text:
            xml_element.documentation = doc_elem.text.strip()

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
        self, element: "etree.Element", schema_def: XmlSchemaDefinition, name_override: str | None = None
    ) -> XmlSchemaSimpleType | None:
        """Parse a simpleType element."""
        name = name_override or element.get("name")
        if not name:
            logger.warning("Skipping anonymous simple type.")
            return None

        simple_type = XmlSchemaSimpleType(name=name)

        # Parse documentation
        doc_elem = element.find(f"{{{self.XS_NAMESPACE}}}annotation/{{{self.XS_NAMESPACE}}}documentation")
        if doc_elem is not None and doc_elem.text:
            simple_type.documentation = doc_elem.text.strip()

        # Parse content (restriction, list, union)
        for child in element:
            tag_name = self._get_local_name(child.tag)
            if tag_name == "restriction":
                simple_type.restrictions = self._parse_restriction(child)
                if simple_type.restrictions and simple_type.restrictions.base:
                    type_local_name = self._get_local_name(simple_type.restrictions.base)
                    if type_local_name in self.type_mappings:
                        simple_type.base_type = self.type_mappings[type_local_name]
            # TODO: Add support for list and union simpleTypes

        return simple_type

    def _parse_restriction(self, restriction_elem: "etree.Element") -> XmlSchemaRestriction:
        """Parse a restriction element"""
        base_type_qname = restriction_elem.get("base")
        base_type_name = self._get_local_name(base_type_qname) if base_type_qname else None
        base_type = self._get_base_type(base_type_name)

        restriction = XmlSchemaRestriction(base=base_type.value if base_type else None)

        for child in restriction_elem:
            tag_name = self._get_local_name(child.tag)
            value = child.get("value")

            if tag_name == "enumeration":
                if value:
                    # Try to read per-enumeration documentation label
                    doc_elem = child.find(f"{{{self.XS_NAMESPACE}}}annotation/{{{self.XS_NAMESPACE}}}documentation")
                    if doc_elem is not None and doc_elem.text:
                        label = doc_elem.text.strip()
                    else:
                        label = value.replace("_", " ").title()
                    restriction.enumeration.append((value, label))
            elif tag_name == "pattern":
                if value:
                    restriction.pattern = value
            elif tag_name == "minLength":
                restriction.min_length = int(value)
            elif tag_name == "maxLength":
                restriction.max_length = int(value)
            elif tag_name == "length":
                restriction.length = int(value)
            elif tag_name == "minInclusive":
                restriction.min_inclusive = float(value)
            elif tag_name == "maxInclusive":
                restriction.max_inclusive = float(value)
            elif tag_name == "minExclusive":
                restriction.min_exclusive = float(value)
            elif tag_name == "maxExclusive":
                restriction.max_exclusive = float(value)
            elif tag_name == "fractionDigits":
                restriction.fraction_digits = int(value)
            elif tag_name == "totalDigits":
                restriction.total_digits = int(value)
            elif tag_name == "whiteSpace":
                restriction.white_space = value

        return restriction

    def _parse_complex_content(
        self, complex_content: "etree.Element", complex_type: XmlSchemaComplexType, schema_def: XmlSchemaDefinition
    ):
        """Parse complexContent, which typically implies extension or restriction."""
        # TODO: Implement complexContent parsing if needed
        logger.warning(f"complexContent in {complex_type.name} is not fully supported yet.")

    def _parse_simple_content(
        self, simple_content: "etree.Element", complex_type: XmlSchemaComplexType, schema_def: XmlSchemaDefinition
    ):
        """Parse simpleContent, which adds attributes to a simple type."""
        # Minimal support: capture attributes declared on the extension node
        # so that types like HeaderType (attributes-only) are not filtered out.
        try:
            ext = simple_content.find(f"{{{self.XS_NAMESPACE}}}extension")
            if ext is None:
                logger.warning(f"simpleContent in {complex_type.name} has no extension; limited support")
                return
            for attr_elem in ext.findall(f"{{{self.XS_NAMESPACE}}}attribute"):
                attribute = self._parse_attribute(attr_elem, schema_def)
                if attribute:
                    complex_type.attributes[attribute.name] = attribute
        except Exception:
            logger.warning(f"simpleContent in {complex_type.name} is not fully supported yet.")

    def _get_local_name(self, qname: str) -> str:
        """Extract the local name from a qualified name (e.g., {http://...}string -> string)."""
        if "}" in qname:
            return qname.split("}", 1)[-1]
        if ":" in qname:
            return qname.split(":", 1)[-1]
        return qname

    def _parse_key(self, element: "etree.Element") -> XmlSchemaKey | None:
        """Parse a key element."""
        name = element.get("name")
        if not name:
            logger.warning("Key without name, skipping.")
            return None

        selector_elem = element.find(f"{{{self.XS_NAMESPACE}}}selector")
        selector = selector_elem.get("xpath") if selector_elem is not None else None

        fields = [
            field.get("xpath") for field in element.findall(f"{{{self.XS_NAMESPACE}}}field") if field.get("xpath")
        ]

        if not selector or not fields:
            logger.warning(f"Key '{name}' is missing selector or fields, skipping.")
            return None

        return XmlSchemaKey(name=name, selector=selector, fields=fields)

    def _parse_keyref(self, element: "etree.Element") -> XmlSchemaKeyRef | None:
        """Parse a keyref element."""
        name = element.get("name")
        refer = element.get("refer")

        if not name or not refer:
            logger.warning("KeyRef without name or refer, skipping.")
            return None

        selector_elem = element.find(f"{{{self.XS_NAMESPACE}}}selector")
        selector = selector_elem.get("xpath") if selector_elem is not None else None

        fields = [
            field.get("xpath") for field in element.findall(f"{{{self.XS_NAMESPACE}}}field") if field.get("xpath")
        ]

        if not selector or not fields:
            logger.warning(f"KeyRef '{name}' is missing selector or fields, skipping.")
            return None

        return XmlSchemaKeyRef(name=name, refer=refer, selector=selector, fields=fields)

    def parse_multiple_schemas(self, schema_paths: list[str | Path]) -> list[XmlSchemaDefinition]:
        """
        Parse multiple XSD files and return a list of schema definitions.

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

    def _get_base_type(self, type_name: str | None) -> XmlSchemaType:
        """Resolve XML Schema built-in type to an XmlSchemaType enum."""
        if not type_name:
            return XmlSchemaType.STRING  # Default

        logger.debug(f"Attempting to resolve base type for: {type_name}")
        # Remove namespace prefix (e.g., 'xs:')
        local_name = type_name.split(":")[-1].upper()

        try:
            resolved_type = XmlSchemaType[local_name]
            logger.debug(f"Resolved '{type_name}' to '{resolved_type}'")
            return resolved_type
        except KeyError:
            # It's not a built-in, might be a custom simpleType or complexType
            # The factory will need to resolve this later.
            # For now, we can return a default or a special "unresolved" type
            logger.debug(f"Could not resolve '{type_name}' to a built-in type. Defaulting to STRING.")
            return XmlSchemaType.STRING  # Fallback
