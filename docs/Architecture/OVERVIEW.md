## Architecture overview

This document explains how `pydantic2django` is organized, what the shared core is responsible for, and what each implementation (Pydantic, Dataclass, TypedClass) has in common.

For inlined API links below, we use mkdocstrings’ Python handler syntax; see “mkdocstrings usage” for details. Reference: [mkdocstrings usage](https://mkdocstrings.github.io/python/usage/).

### Big picture

- **Goal**: Convert typed Python models into Django models and make them interoperable in both directions.
- **Layers**:
  - **Core**: Shared building blocks (discovery, factories, bidirectional type mapping, typing utils, import aggregation, context handling, code generation base).
  - **Implementations**: Source-specific adapters that plug into the core (Pydantic, Dataclass, TypedClass).
  - **Django base models**: Ready-to-extend base classes that store or map typed objects inside Django models.

High-level flow (static code generation path):

1. Discover source models in Python packages.
2. For each model, convert its fields to Django fields via the bidirectional type mapper.
3. Create an in-memory Django model class and collect rendered definitions using Jinja templates.
4. Write a `models.py` file including imports, model classes, and optional context classes.

Templates live in `src/pydantic2django/django/templates` and are used by the generator base.

### Core responsibilities

- **Static generation orchestration**
  - ::: pydantic2django.core.base_generator.BaseStaticGenerator

- **Model discovery (abstract + per-source dependency graph)**
  - ::: pydantic2django.core.discovery.BaseDiscovery

- **Model/field factories and the conversion carrier**
  - ::: pydantic2django.core.factories.BaseModelFactory
  - ::: pydantic2django.core.factories.BaseFieldFactory
  - ::: pydantic2django.core.factories.ConversionCarrier
  - ::: pydantic2django.core.factories.FieldConversionResult

- **Bidirectional type mapping (Python/Pydantic ↔ Django.Field)**
  - ::: pydantic2django.core.bidirectional_mapper.BidirectionalTypeMapper
  - ::: pydantic2django.core.mapping_units.TypeMappingUnit

- **Relationships and cross-model resolution**
  - ::: pydantic2django.core.relationships.RelationshipConversionAccessor
  - ::: pydantic2django.core.relationships.RelationshipMapper

- **Typing utilities**
  - ::: pydantic2django.core.typing.TypeHandler

- **Import aggregation for generated code**
  - ::: pydantic2django.core.imports.ImportHandler

- **Context handling for non-serializable fields**
  - ::: pydantic2django.core.context.ModelContext
  - ::: pydantic2django.core.context.ContextClassGenerator

- **Serialization helpers**
  - ::: pydantic2django.core.serialization.serialize_value
  - ::: pydantic2django.core.serialization.get_serialization_method

### What all implementations have in common

Each implementation integrates the same core concepts:

- A `Discovery` that lists eligible source models and computes a safe registration order.
- A `ModelFactory` and `FieldFactory` pair that build Django fields using the shared `BidirectionalTypeMapper`.
- A `Generator` that subclasses `BaseStaticGenerator`, wires up discovery/factories, and prepares template context.
- Use of `RelationshipConversionAccessor` so relationship fields (FK/O2O/M2M) can be resolved across generated models.
- Shared `ImportHandler`, `TypeHandler`, and `ModelContext` mechanisms.

### Implementations

#### Pydantic

- Discovery, factory, and generator:
  - ::: pydantic2django.pydantic.discovery.PydanticDiscovery
  - ::: pydantic2django.pydantic.factory.PydanticModelFactory
  - ::: pydantic2django.pydantic.factory.PydanticFieldFactory
  - ::: pydantic2django.pydantic.generator.StaticPydanticModelGenerator

- Notes:
  - Relies on Pydantic `FieldInfo` for field metadata and constraints.
  - Generates optional per-model context classes when non-serializable fields are detected.

#### Dataclass

- Discovery, factory, and generator:
  - ::: pydantic2django.dataclass.discovery.DataclassDiscovery
  - ::: pydantic2django.dataclass.factory.DataclassModelFactory
  - ::: pydantic2django.dataclass.factory.DataclassFieldFactory
  - ::: pydantic2django.dataclass.generator.DataclassDjangoModelGenerator

- Notes:
  - Reads field metadata and optional per-field `django` overrides from `dataclasses.Field.metadata`.
  - Uses `typing.get_type_hints` to resolve forward references.

#### TypedClass (experimental)

- Discovery and factory (subject to change as APIs stabilize):
  - ::: pydantic2django.typedclass.discovery.TypedClassDiscovery
  - ::: pydantic2django.typedclass.factory.TypedClassModelFactory

- Notes:
  - Targets arbitrary typed Python classes (non-Pydantic, non-dataclass). APIs are evolving.

### Django base models (runtime mapping and storage)

These abstract models provide runtime persistence patterns for typed objects.

- Store entire objects as JSON or map their fields to columns. See:
  - ::: pydantic2django.django.models.Pydantic2DjangoBaseClass
  - ::: pydantic2django.django.models.Dataclass2DjangoBaseClass
  - ::: pydantic2django.django.models.TypedClass2DjangoBaseClass

Additional docs:

- Context handling: `docs/Architecture/CONTEXT_HANDLING.md`
- Pydantic ↔ Django methods: `docs/Architecture/README_PYDANTIC_DJANGO_METHODS.md`
- Django storage options: `docs/Architecture/README_DJANGO_STORAGE_OPTIONS.md`

### Extending the system

- To support a new source type, implement:
  - A `Discovery` subclass of `BaseDiscovery`.
  - A `FieldFactory` and `ModelFactory` pair using `BidirectionalTypeMapper`.
  - A `Generator` subclass of `BaseStaticGenerator` that wires everything together and prepares template context.
- To add or refine type mappings, provide new `TypeMappingUnit` subclasses and ensure `BidirectionalTypeMapper` registry order or match scores select them appropriately.

### mkdocstrings usage

Each `:::` block above renders live API documentation using mkdocstrings with the Python handler, configured in `mkdocs.yml` with `handlers.python.paths: [src]`. See the official guide: [mkdocstrings usage](https://mkdocstrings.github.io/python/usage/).
