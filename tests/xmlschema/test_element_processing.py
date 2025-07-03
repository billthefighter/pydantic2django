"""
Test file to debug specific element processing issues in XML schema factory.
"""

import pytest
from pathlib import Path
import logging

from pydantic2django.xmlschema.parser import XmlSchemaParser
from pydantic2django.xmlschema.factory import XmlSchemaModelFactory
from pydantic2django.xmlschema.models import XmlSchemaComplexType
from pydantic2django.core.context import ModelContext

# Configure logging
logger = logging.getLogger("pydantic2django.xmlschema")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False


@pytest.fixture(scope="module")
def comprehensive_xsd_path() -> Path:
    """Provides the path to the comprehensive test XSD file."""
    return Path(__file__).parent / "fixtures" / "comprehensive_schema.xsd"


@pytest.fixture(scope="module")
def parsed_schema(comprehensive_xsd_path: Path):
    """Parses the comprehensive schema once for all tests in this module."""
    parser = XmlSchemaParser()
    return parser.parse_schema_file(comprehensive_xsd_path)


def test_element_objects_are_not_strings(parsed_schema):
    """Test that elements in complex types are proper objects, not strings."""
    author_type = parsed_schema.complex_types["AuthorType"]

    print(f"AuthorType elements count: {len(author_type.elements)}")
    print(f"AuthorType elements: {author_type.elements}")

    for i, element in enumerate(author_type.elements):
        print(f"Element {i}: type={type(element)}, value={element}")
        if hasattr(element, 'name'):
            print(f"  - name: {element.name}")
        else:
            print(f"  - ERROR: element is {type(element)} with no 'name' attribute")

    # Verify that elements are not strings
    for element in author_type.elements:
        assert not isinstance(element, str), f"Element {element} should not be a string"
        assert hasattr(element, 'name'), f"Element {element} should have a 'name' attribute"


def test_complex_type_structure(parsed_schema):
    """Test the structure of a complex type to understand element storage."""
    author_type = parsed_schema.complex_types["AuthorType"]

    print(f"AuthorType type: {type(author_type)}")
    print(f"AuthorType attributes: {dir(author_type)}")
    print(f"AuthorType.elements type: {type(author_type.elements)}")

    if hasattr(author_type, '__dict__'):
        print(f"AuthorType.__dict__: {author_type.__dict__}")


def test_factory_element_processing(parsed_schema):
    """Test how the factory processes elements from a complex type."""
    author_type = parsed_schema.complex_types["AuthorType"]

    # Create a factory instance
    factory = XmlSchemaModelFactory(
        app_label="test_app"
    )

    # Create a mock model context
    context = ModelContext(
        name="AuthorType",
        original_class=author_type,
        app_label="test_app"
    )

    # Try to debug what happens in the factory
    print(f"About to process elements for AuthorType")
    print(f"AuthorType.elements: {author_type.elements}")
    print(f"Type of elements: {[type(e) for e in author_type.elements]}")

    # This should reveal where the string conversion happens
    try:
        # We can't call _process_source_fields directly as it expects a carrier,
        # but we can check the elements directly
        for element in author_type.elements:
            print(f"Processing element: {element} (type: {type(element)})")
            if hasattr(element, 'name'):
                print(f"  Element name: {element.name}")
            else:
                print(f"  ERROR: Element {element} has no name attribute")
    except Exception as e:
        print(f"Error processing elements: {e}")
        import traceback
        traceback.print_exc()
