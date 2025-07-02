# XML Schema Usage Guide

## Installation

```bash
# Install with XML Schema support
pip install pydantic2django[xmlschema]

# Or install manually
pip install lxml
```

## Quick Example

```python
from pydantic2django.xmlschema import XmlSchemaDjangoModelGenerator

# Generate Django models from XML Schema
generator = XmlSchemaDjangoModelGenerator(
    schema_files=["path/to/schema.xsd"],
    output_path="myapp/models.py",
    app_label="myapp",
    verbose=True
)

# Generate the models
generator.generate()
```

## Features Implemented

✅ **Core Architecture**
- `XmlSchemaDiscovery` - Discovers complex types from XSD files
- `XmlSchemaFieldFactory` - Converts XML elements to Django fields
- `XmlSchemaModelFactory` - Creates Django model definitions
- `XmlSchemaDjangoModelGenerator` - Main generator class

✅ **XML Schema Support**
- Parse XSD files using lxml
- Extract complex type definitions
- Map XML Schema types to Django field types
- Handle relationships between types
- Support for constraints (minOccurs, maxOccurs, etc.)

✅ **Generated Models**
- Inherit from `Xml2DjangoBaseClass`
- Include XML conversion methods (`from_xml_dict`, `to_xml_dict`)
- Proper field types and constraints
- Relationship handling (ForeignKey, ManyToManyField)

## Type Mapping

| XML Schema Type | Django Field |
|-----------------|--------------|
| xs:string | CharField |
| xs:int, xs:integer | IntegerField |
| xs:decimal | DecimalField |
| xs:boolean | BooleanField |
| xs:date | DateField |
| xs:dateTime | DateTimeField |
| xs:time | TimeField |
| Type references | ForeignKey/ManyToManyField |

## Architecture

The XML Schema module follows the same proven three-stage pipeline as the existing Pydantic and dataclass modules:

1. **Discovery** - Parse XSD files and identify target complex types
2. **Factory** - Convert XML Schema definitions to Django model components
3. **Generator** - Generate final Python code using Jinja2 templates

This ensures consistency with the rest of the pydantic2django library and leverages all existing core functionality.
