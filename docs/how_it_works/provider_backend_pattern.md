## Provider/backends pattern

This document explains the common pattern shared by the provider backends in `src/pydantic2django/`:

- `dataclass/`
- `pydantic/`
- `typedclass/`
- `xmlschema/`

These backends implement the same high-level pipeline and reuse the core abstractions in `src/pydantic2django/core/`.

### The shared pipeline
1) **Discovery** (find source models)
- Each backend provides a `Discovery` that subclasses `core.discovery.BaseDiscovery[TModel]`.
- It scans the provided packages or inputs, applies default and user filters, and builds a dependency graph.

2) **Factory** (turn fields into Django `Field`s, then a model)
- Each backend provides a `FieldFactory` and a `ModelFactory` that subclass `core.factories.BaseFieldFactory` and `BaseModelFactory`.
- Field factories translate a single source field to a Django field and kwargs.
- Model factories iterate fields, handle collisions, build a `Meta` class, assemble the Django `Model` type, and construct optional context.

3) **Generator** (orchestrate and render)
- Each backend provides a `...ModelGenerator` that subclasses `core.base_generator.BaseStaticGenerator`.
- It calls discovery, invokes the model factory for each source model to produce carriers, renders Jinja templates (`django/templates/*.j2`), and writes the output `models.py`.

### Core building blocks reused by all backends
- **`BaseDiscovery`**: common scanning/filtering hooks and dependency analysis entrypoints.
- **`BaseModelFactory` / `BaseFieldFactory`**: single-field conversion and whole-model assembly with a `ConversionCarrier` for state.
- **`BidirectionalTypeMapper` + `mapping_units`**: type mapping for Pydantic/Dataclass fields to Django fields, including enums, collections, and relationships.
- **`RelationshipConversionAccessor`**: resolves and tracks mappings between source models and generated Django models, used by mappers/factories.
- **`ImportHandler`**: collects and de-duplicates imports across definitions and context classes.
- **`ModelContext` / `ContextClassGenerator`**: holds non-serializable context and renders companion context classes (primarily for Pydantic).
- **`TypeHandler`**: normalizes types (Optional/List/Union/Annotated), produces readable type strings, and calculates required imports.

### Backend specifics
#### Pydantic (`pydantic/`)
- Discovery: `PydanticDiscovery` finds `BaseModel` classes and builds a dependency graph from field annotations.
- Factories: `PydanticFieldFactory` + `PydanticModelFactory` use `BidirectionalTypeMapper` to convert `FieldInfo` to Django fields (includes union/multi-FK handling, relationships, enums/choices, defaults).
- Generator: `StaticPydanticModelGenerator` orchestrates generation and also renders per-model context classes where needed.

#### Dataclass (`dataclass/`)
- Discovery: `DataclassDiscovery` identifies Python dataclasses and derives dependencies from dataclass field annotations.
- Factories: `DataclassFieldFactory` + `DataclassModelFactory` use the same bidirectional mapper; default/metadata resolution is tailored to dataclasses.
- Generator: `DataclassDjangoModelGenerator` follows the shared flow; context classes are typically not emitted.

#### TypedClass (`typedclass/`)
- Discovery: `TypedClassDiscovery` targets plain Python classes (not Pydantic/dataclasses), leveraging `__init__` parameters and class annotations for fields/deps.
- Factories: `TypedClassFieldFactory` + `TypedClassModelFactory` translate type hints using a type translator suited to generic classes.
- Generator: `TypedClassDjangoModelGenerator` reuses the shared pipeline; focuses on simple type-hint based field generation.

#### XML Schema (`xmlschema/`)
- Discovery: `XmlSchemaDiscovery` parses XSDs via `XmlSchemaParser`, gathers complex types, applies filters, and computes inter-type dependencies.
- Factories: `XmlSchemaFieldFactory` + `XmlSchemaModelFactory` map XSD simple/complex types to Django fields. Handles enumerations (TextChoices), nillability/occurrence constraints, and post-pass injection of relationships (e.g., child FKs) via `finalize_relationships`.
- Generator: `XmlSchemaDjangoModelGenerator` extends the shared generator logic, calling `finalize_relationships` before rendering and passing enum/context info to templates.

### Relationship handling
- Pydantic/Dataclass: `BidirectionalTypeMapper` inspects type hints and uses `RelationshipConversionAccessor` to resolve related model references (FK, O2O, M2M). It detects lists of known models for M2M and unions of models for multi-FK patterns.
- XML Schema: relationships are derived from `key`/`keyref`, element nesting, and configuration (fk/json/m2m). Some relations are injected after all models exist.

### Imports and templates
- `ImportHandler` accumulates typing/pydantic/context and model imports during field conversion and context rendering, then de-duplicates for the templates.
- All generators render with the shared Jinja templates in `src/pydantic2django/django/templates/` (`model_definition.py.j2`, `models_file.py.j2`, `context_class.py.j2`, `imports_block.py.j2`).

### Adding a new backend
- Implement a `Discovery` subclass to find your source models and compute dependencies.
- Implement a `FieldFactory` + `ModelFactory` pairing to convert fields and assemble models.
- Implement a `...ModelGenerator` subclass of `BaseStaticGenerator` and wire your discovery/factory and any special context/relationship steps.
- Reuse `BidirectionalTypeMapper` where possible; add new `TypeMappingUnit`s if your source introduces new type shapes.
- Leverage `ImportHandler`, `RelationshipConversionAccessor`, and `TypeHandler` to keep logic consistent and DRY.

### API reference
For full, up-to-date API details of these modules and classes, see the generated reference: [API Reference](/reference/).
