# XML Schema to Django Models

This module provides automatic Django model generation from XML Schema (XSD) files. It follows the same architectural patterns as the Pydantic and dataclass modules, providing a seamless way to generate Django models from XML Schema definitions.

## Overview

The XML Schema module parses XSD files and generates Django models that represent the complex types defined in the schema. It handles:

- **Complex Types**: Converted to Django models
- **Elements**: Converted to Django fields with appropriate types
- **Attributes**: Converted to Django fields
- **Type References**: Converted to Django relationships (ForeignKey/ManyToManyField)
- **Inheritance**: Basic support for XML Schema type inheritance
- **Constraints**: XML Schema restrictions mapped to Django field constraints

## Quick Start

### Basic Usage

```python
from pydantic2django.xmlschema import XmlSchemaDjangoModelGenerator

# Generate Django models from XML Schema files
generator = XmlSchemaDjangoModelGenerator(
    schema_files=["path/to/schema1.xsd", "path/to/schema2.xsd"],
    output_path="myapp/models.py",
    app_label="myapp",
    verbose=True
)

# Generate the models file
generator.generate()
```

### Example Schema

Given this XML Schema (`person.xsd`):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
           targetNamespace="http://example.com/person"
           elementFormDefault="qualified">

  <xs:complexType name="PersonType">
    <xs:sequence>
      <xs:element name="firstName" type="xs:string"/>
      <xs:element name="lastName" type="xs:string"/>
      <xs:element name="age" type="xs:int" minOccurs="0"/>
      <xs:element name="emails" type="xs:string" maxOccurs="unbounded" minOccurs="0"/>
      <xs:element name="address" type="AddressType" minOccurs="0"/>
    </xs:sequence>
    <xs:attribute name="id" type="xs:string" use="required"/>
  </xs:complexType>

  <xs:complexType name="AddressType">
    <xs:sequence>
      <xs:element name="street" type="xs:string"/>
      <xs:element name="city" type="xs:string"/>
      <xs:element name="zipCode" type="xs:string"/>
    </xs:sequence>
  </xs:complexType>
</xs:schema>
```

The generator would produce Django models like:

```python
class AddressType(Xml2DjangoBaseClass):
    street = models.CharField(max_length=255)
    city = models.CharField(max_length=255)
    zip_code = models.CharField(max_length=255)

    class Meta:
        app_label = "myapp"

class PersonType(Xml2DjangoBaseClass):
    first_name = models.CharField(max_length=255)
    last_name = models.CharField(max_length=255)
    age = models.IntegerField(null=True, blank=True)
    emails = models.TextField(help_text="List field (maxOccurs=unbounded)")
    address = models.ForeignKey(AddressType, on_delete=models.SET_NULL, null=True, blank=True)
    id = models.CharField(max_length=255)  # From attribute

    class Meta:
        app_label = "myapp"
```

## Features

### Supported XML Schema Constructs

#### Data Types
- **String types**: `string`, `normalizedString`, `token` → `CharField`
- **Numeric types**: `int`, `long`, `decimal`, `float`, `double` → `IntegerField`, `DecimalField`, etc.
- **Date/Time types**: `date`, `dateTime`, `time` → `DateField`, `DateTimeField`, `TimeField`
- **Boolean**: `boolean` → `BooleanField`
- **Binary**: `base64Binary`, `hexBinary` → `BinaryField`
- **URI**: `anyURI` → `URLField`

#### Structure
- **Complex Types**: Converted to Django models
- **Elements**: Converted to model fields
- **Attributes**: Converted to model fields
- **Sequences**: Field order preserved
- **Choices**: Supported (with limitations)

#### Constraints
- **minOccurs/maxOccurs**: Mapped to `null`, `blank`, and list handling
- **Length restrictions**: Mapped to `max_length` on `CharField`
- **Numeric ranges**: Can be extended with custom validators
- **Patterns**: Can be extended with custom validators
- **Enumerations**: Can be extended with `choices` parameter

#### Relationships
- **Type References**: Become `ForeignKey` or `ManyToManyField`
- **Cardinality**: `maxOccurs="unbounded"` becomes `ManyToManyField`
- **Optional References**: `minOccurs="0"` adds `null=True, blank=True`

### Advanced Usage

#### Filtering Complex Types

```python
def only_large_types(complex_type):
    """Only include types with more than 3 elements"""
    return len(complex_type.elements) > 3

generator = XmlSchemaDjangoModelGenerator(
    schema_files=["schema.xsd"],
    filter_function=only_large_types,
    # ... other args
)
```

#### Custom Base Model

```python
from django.db import models

class MyXmlBaseModel(models.Model):
    xml_source = models.CharField(max_length=255, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        abstract = True

generator = XmlSchemaDjangoModelGenerator(
    schema_files=["schema.xsd"],
    base_model_class=MyXmlBaseModel,
    # ... other args
)
```

#### Schema Validation and Statistics

```python
generator = XmlSchemaDjangoModelGenerator(schema_files=["schema.xsd"])

# Validate schemas
validation_messages = generator.validate_schemas()
for message in validation_messages:
    print(f"Warning: {message}")

# Get statistics
stats = generator.get_schema_statistics()
print(f"Generated {stats['generated_models']} models from {stats['total_schemas']} schemas")
```

## Architecture

The XML Schema module follows the same three-stage pipeline as other modules:

### 1. Discovery (`XmlSchemaDiscovery`)
- Parses XSD files using `lxml`
- Identifies complex types suitable for model generation
- Builds dependency graph between types
- Applies filtering logic

### 2. Factory (`XmlSchemaFieldFactory` & `XmlSchemaModelFactory`)
- Converts XML Schema elements to Django fields
- Handles type mapping and constraints
- Creates relationship fields for type references
- Builds model definitions

### 3. Generator (`XmlSchemaDjangoModelGenerator`)
- Orchestrates the entire process
- Generates final Python code using Jinja2 templates
- Handles imports and model ordering
- Writes output to file

## Dependencies

The XML Schema module requires additional dependencies:

```bash
pip install lxml  # For XML parsing
```

Or install with XML support:

```bash
pip install pydantic2django[xmlschema]
```

## Limitations

### Current Limitations

1. **Namespaces**: Basic namespace support; complex namespace scenarios may need manual adjustment
2. **Schema Imports**: Limited support for cross-schema references
3. **Choice Groups**: Mapped to individual optional fields (may not preserve exact semantics)
4. **Mixed Content**: Limited support for elements with mixed text/element content
5. **Substitution Groups**: Not currently supported
6. **Complex Constraints**: Advanced restrictions may need custom Django validators

### Future Enhancements

- Better namespace handling
- Schema import resolution
- Custom constraint mapping
- XML serialization/deserialization methods
- Integration with Django REST framework serializers

## Examples

### E-commerce Schema

```python
# Generate models from e-commerce schemas
generator = XmlSchemaDjangoModelGenerator(
    schema_files=[
        "schemas/product.xsd",
        "schemas/order.xsd",
        "schemas/customer.xsd"
    ],
    output_path="ecommerce/models.py",
    app_label="ecommerce",
    verbose=True
)

models_file = generator.generate()
print(f"Generated models: {models_file}")

# Check what was generated
stats = generator.get_schema_statistics()
print(f"Created {stats['generated_models']} Django models")
```

### Web Service Schema

```python
# Generate from WSDL/XSD definitions
generator = XmlSchemaDjangoModelGenerator(
    schema_files=["webservice.xsd"],
    filter_function=lambda ct: not ct.name.endswith("Request"),  # Skip request types
    output_path="api/models.py",
    app_label="webapi"
)

generator.generate()
```

## Integration with Other Modules

The XML Schema module works alongside other Pydantic2Django modules:

```python
# Generate from multiple sources
from pydantic2django.pydantic import StaticPydanticModelGenerator
from pydantic2django.xmlschema import XmlSchemaDjangoModelGenerator

# Generate from Pydantic models
pydantic_gen = StaticPydanticModelGenerator(
    packages=["myapp.pydantic_models"],
    output_path="myapp/pydantic_models.py",
    app_label="myapp"
)

# Generate from XML Schema
xml_gen = XmlSchemaDjangoModelGenerator(
    schema_files=["schemas/external_api.xsd"],
    output_path="myapp/xml_models.py",
    app_label="myapp"
)

# Generate both
pydantic_gen.generate()
xml_gen.generate()
```

## Contributing

When extending the XML Schema module:

1. Follow the existing patterns in `pydantic/` and `dataclass/` modules
2. Add comprehensive tests for new XML Schema constructs
3. Update this README with new features
4. Consider backward compatibility

See the main project README for general contribution guidelines.
