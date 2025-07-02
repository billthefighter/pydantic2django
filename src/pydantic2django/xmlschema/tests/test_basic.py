"""
Basic tests for XML Schema functionality.
"""
import unittest
from unittest.mock import Mock, patch

from ..models import XmlSchemaComplexType, XmlSchemaElement, XmlSchemaType
from ..discovery import XmlSchemaDiscovery


class TestXmlSchemaModels(unittest.TestCase):
    """Test XML Schema model classes."""

    def test_complex_type_creation(self):
        """Test creating an XML Schema complex type."""
        complex_type = XmlSchemaComplexType(
            name="TestType",
            namespace="http://example.com/test"
        )

        self.assertEqual(complex_type.name, "TestType")
        self.assertEqual(complex_type.namespace, "http://example.com/test")
        self.assertFalse(complex_type.abstract)
        self.assertEqual(len(complex_type.elements), 0)

    def test_element_properties(self):
        """Test XML Schema element properties."""
        element = XmlSchemaElement(
            name="testElement",
            base_type=XmlSchemaType.STRING,
            min_occurs=0,
            max_occurs="unbounded"
        )

        self.assertEqual(element.name, "testElement")
        self.assertTrue(element.is_optional)
        self.assertTrue(element.is_list)
        self.assertFalse(element.is_required)


class TestXmlSchemaDiscovery(unittest.TestCase):
    """Test XML Schema discovery functionality."""

    def test_discovery_initialization(self):
        """Test discovery can be initialized."""
        discovery = XmlSchemaDiscovery()
        self.assertIsNotNone(discovery)
        self.assertEqual(len(discovery.schema_files), 0)

    def test_is_target_model(self):
        """Test _is_target_model method."""
        discovery = XmlSchemaDiscovery()

        # Create a valid complex type
        valid_type = XmlSchemaComplexType(
            name="ValidType",
            elements=[XmlSchemaElement(name="field1", base_type=XmlSchemaType.STRING)]
        )
        self.assertTrue(discovery._is_target_model(valid_type))

        # Create an abstract type (should be rejected)
        abstract_type = XmlSchemaComplexType(
            name="AbstractType",
            abstract=True,
            elements=[XmlSchemaElement(name="field1", base_type=XmlSchemaType.STRING)]
        )
        self.assertFalse(discovery._is_target_model(abstract_type))

        # Create an empty type (should be rejected)
        empty_type = XmlSchemaComplexType(name="EmptyType")
        self.assertFalse(discovery._is_target_model(empty_type))

    @patch('pydantic2django.xmlschema.discovery.XmlSchemaParser')
    def test_discover_models_basic(self, mock_parser_class):
        """Test basic model discovery."""
        # Mock parser
        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser

        # Mock schema definition
        mock_schema = Mock()
        mock_schema.get_all_complex_types.return_value = [
            XmlSchemaComplexType(
                name="TestType",
                namespace="http://example.com",
                elements=[XmlSchemaElement(name="field1", base_type=XmlSchemaType.STRING)]
            )
        ]
        mock_parser.parse_schema_file.return_value = mock_schema

        # Test discovery
        discovery = XmlSchemaDiscovery()
        discovery.discover_models(
            packages=["test.xsd"],
            app_label="test_app"
        )

        # Verify parser was called
        mock_parser.parse_schema_file.assert_called_once_with("test.xsd")

        # Verify models were discovered
        self.assertEqual(len(discovery.all_models), 1)
        self.assertIn("http://example.com.TestType", discovery.all_models)


if __name__ == '__main__':
    unittest.main()
