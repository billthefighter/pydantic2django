"""
Unit tests for the XmlSchemaParser, ensuring it correctly parses
complex and simple types from XSD files.
"""

import pytest
from pathlib import Path
import logging

from pydantic2django.xmlschema.parser import XmlSchemaParser
from pydantic2django.xmlschema.models import XmlSchemaType

# Configure logging to see parser output during tests
parser_logger = logging.getLogger("pydantic2django.xmlschema.parser")
parser_logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
parser_logger.addHandler(handler)
parser_logger.propagate = False


@pytest.fixture(scope="module")
def comprehensive_xsd_path() -> Path:
    """Provides the path to the comprehensive test XSD file."""
    return Path(__file__).parent / "fixtures" / "comprehensive_schema.xsd"


@pytest.fixture(scope="module")
def parsed_schema(comprehensive_xsd_path: Path):
    """Parses the comprehensive schema once for all tests in this module."""
    parser = XmlSchemaParser()
    return parser.parse_schema_file(comprehensive_xsd_path)


def test_parser_finds_complex_types(parsed_schema):
    """Tests that the parser discovers the top-level complex types."""
    assert "AuthorType" in parsed_schema.complex_types
    assert "BookType" in parsed_schema.complex_types
    assert len(parsed_schema.complex_types) == 3  # Includes Library's inline type


def test_parser_finds_simple_types(parsed_schema):
    """Tests that the parser discovers the top-level simple types."""
    assert "AuthorStatus" in parsed_schema.simple_types
    assert "BookGenre" in parsed_schema.simple_types
    assert "emailType" in parsed_schema.simple_types
    assert len(parsed_schema.simple_types) == 3


def test_author_type_has_correct_elements(parsed_schema):
    """Tests that the AuthorType complex type has the correct elements."""
    author_type = parsed_schema.complex_types["AuthorType"]
    element_names = {element.name for element in author_type.elements}
    assert element_names == {"name", "email", "bio", "status"}
    assert len(author_type.elements) == 4


def test_author_type_has_correct_attributes(parsed_schema):
    """Tests that the AuthorType complex type has the correct attributes."""
    author_type = parsed_schema.complex_types["AuthorType"]
    attribute_names = set(author_type.attributes.keys())
    assert attribute_names == {"author_id"}
    assert len(author_type.attributes) == 1


def test_book_type_has_correct_attributes(parsed_schema):
    """
    Tests that the BookType complex type has the correct attributes.
    """
    book_type = parsed_schema.complex_types["BookType"]
    attribute_names = set(book_type.attributes.keys())
    assert attribute_names == {"isbn", "publication_date", "genre"}
    assert len(book_type.attributes) == 3


def test_author_status_enum_is_parsed_correctly(parsed_schema):
    """
    Tests that enum labels are parsed from documentation.
    """
    author_status = parsed_schema.simple_types["AuthorStatus"]
    assert author_status.base_type == XmlSchemaType.STRING
    expected_enums = {
        ("active", "Active and contributing"),
        ("inactive", "No new works"),
        ("deceased", "Deceased"),
    }
    assert set(author_status.enumeration) == expected_enums
