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

### Singleton-style reuse and warmup

Long-lived processes can avoid repeated schema discovery/model registration using process-wide helpers in `pydantic2django.xmlschema.ingestor`:

- `warmup_xmlschema_models(schema_files, app_label=...)`
  - Pre-generates and registers XML-derived Django model classes in an in-memory registry
  - No-op on subsequent calls for the same `(app_label, schema file set + mtimes)`

- `get_shared_ingestor(schema_files=..., app_label=...)`
  - Returns a shared `XmlInstanceIngestor` keyed by `(app_label, normalized schema paths, schema mtimes)`
  - Reuses the same instance across calls; entries are cached with an LRU + TTL policy
  - Default cache policy: LRU maxsize=4, TTL=600s; configurable via `set_ingestor_cache()`
  - Public controls:
    - `set_ingestor_cache(maxsize: int | None = None, ttl_seconds: float | None = None)`
    - `clear_ingestor_cache()` to clear (useful in tests)
    - `ingestor_cache_stats()` for diagnostics

Example (task runner):

```python
from pydantic2django.xmlschema.ingestor import warmup_xmlschema_models, get_shared_ingestor

SCHEMAS = [
    "/abs/path/MTConnectStreams_1.7.xsd",
    "/abs/path/MTConnectDevices_1.7.xsd",
    "/abs/path/MTConnectAssets_1.7.xsd",
    "/abs/path/MTConnectError_1.7.xsd",
]
APP_LABEL = "tests"

# At process start
warmup_xmlschema_models(SCHEMAS, app_label=APP_LABEL)

# In each task
ingestor = get_shared_ingestor(schema_files=SCHEMAS, app_label=APP_LABEL)
root = ingestor.ingest_from_file("/abs/path/example.xml", save=False)
```

Notes:
- `save=True` requires concrete, migrated models in an installed app
- Warmup/registry enables object instantiation without DB tables for dry-runs and analysis paths
- You can tune cache behavior globally at process start:

```python
from pydantic2django.xmlschema.ingestor import set_ingestor_cache

# Keep up to 8 different ingestors for 10 minutes each
set_ingestor_cache(maxsize=8, ttl_seconds=600)
```

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

### Identifier normalization and enum choices

When generating Django models from XML Schema, identifiers are normalized to valid Python/Django field names, and enumerations are emitted as `TextChoices`.

- **Field name normalization**:
  - **Namespace separators and punctuation** (`:`, `.`, `-`, spaces) become `_`.
  - **CamelCase → snake_case** (e.g., `camelCase` → `camel_case`).
  - **Invalid characters** are replaced with `_`; multiple `_` are collapsed.
  - **Leading character** is ensured to be a letter or `_` (prefix `_` if needed).
  - **Lowercased** output (e.g., `xlink:type` → `xlink_type`).

- **Special cases**:
  - Element named `id` that is not an XML `xs:ID` is renamed to `xml_id` to avoid Django primary key conflicts.
  - Names like `type` remain `type` (no suffixing), but their related enums use a proper class symbol (see below).

- **Enum emission**:
  - For `xs:simpleType` with `xs:restriction`/`xs:enumeration`, a `models.TextChoices` class is generated.
  - The enum class name is derived from the (cleaned) field name in PascalCase (e.g., `type` → `Type`).
  - Fields reference choices via the enum symbol: `choices=Type.choices` and default via `Type.MEMBER`.

Example:

```python
class CoordinateSystemType(Xml2DjangoBaseClass):
    # Generated alongside the model
    class Type(models.TextChoices):
        CARTESIAN = "cartesian", "Cartesian"
        POLAR = "polar", "Polar"

    # Field named 'type' correctly references the enum class
    type = models.CharField(choices=Type.choices, max_length=9)

class FileLocationType(Xml2DjangoBaseClass):
    # Namespaced attribute xlink:type → xlink_type
    xlink_type = models.CharField(max_length=255, null=True, blank=True)
```

Notes:
- The normalization is applied uniformly to elements and attributes before code generation.
- If you encounter a source name that still produces an invalid identifier, please report it with the original XML/XSD name so we can extend the rules.
