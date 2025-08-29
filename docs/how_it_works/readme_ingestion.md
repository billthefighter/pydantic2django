# Ingestion Pattern Across Modules

This document describes the ingestion mechanism used to create Django model instances from external representations (XML instances, Pydantic models, and Python dataclasses).

## Goals

- Provide a clear, consistent adapter pattern for "ingest" (external → Django) and "emit" (Django → external).
- Keep module-specific logic encapsulated while preserving a common mental model across modules.

## Common Pattern

- Base classes expose high-level helpers:
  - Pydantic: `from_pydantic(...)`, `to_pydantic(...)`
  - Dataclasses: `from_dataclass(...)`, `to_dataclass(...)`
  - XML: `from_xml_dict(...)`, `from_xml_string(...)`, `to_xml_dict(...)`, `to_xml_string(...)`

- For complex sources that need schema-aware walking (XML), a dedicated ingestor class is used:
  - XML: `XmlInstanceIngestor` parses an XML document with `lxml`, consults the parsed `XmlSchemaDefinition`/`XmlSchemaComplexType` graph, and materializes Django instances following the same relationship strategy used during model generation.

## XML Ingestion

`XmlInstanceIngestor` lives in `pydantic2django.xmlschema.ingestor` and:

- Accepts:
  - `schema_files`: list of XSD files used during generation
  - `app_label`: Django app label where generated models live
- Resolves the root complex type via global `xs:element` or matching complex type name
- Creates the root Django instance and recursively processes nested elements:
  - Simple elements/attributes → mapped to Django fields (camelCase → snake_case)
  - Single nested complex types → stored as FK on parent
  - Repeated nested complex types (`maxOccurs="unbounded"`) → child instances created with FK to parent (`child_fk` strategy)

Future extensions:
- Namespace-scoped matching beyond simple local-name stripping
- Post-pass to resolve `xs:key` / `xs:keyref` (e.g., ID/IDREF) lookups
- Configurable relationship strategies at ingest time (e.g., JSON/M2M)

### Note on Timescale soft references

When generation replaces illegal hypertable→hypertable FKs with soft references (e.g., `UUIDField(db_index=True)`), the ingestor will persist the identifier value. If you need strong integrity across hypertables, add an application-level validator or a periodic job that checks the referenced IDs exist (or maintain a regular “latest snapshot” table which hypertables can FK to).

## Pydantic and Dataclasses

- Ingestion is simpler:
  - Pydantic: `from_pydantic(model)` maps fields and stores types using `serialize_value`, `to_pydantic()` reconstructs via `model_validate`
  - Dataclasses: `from_dataclass(instance)` uses `dataclasses.asdict()`, `to_dataclass()` reconstructs using a direct constructor

Both benefit from `ModelContext` when non-serializable values need to be provided round-trip.

## Consistency Checklist

- Each module offers `from_*` and `to_*` helpers on its base class.
- Complex adapters (XML) provide an external ingestor class with a narrow API.
- Relationship handling during ingestion mirrors the generation factory’s strategy.
- Name conversion rules are mirrored in both directions (e.g., XML name ↔ Django field name).
