# Pydantic2Django Relationships Accessor Details

This document explains how relationships between source models (Pydantic/Dataclass) and Django models are tracked and resolved via `RelationshipMapper` and `RelationshipConversionAccessor`.

## Core Data Structures

```15:31:src/pydantic2django/core/relationships.py
@dataclass
class RelationshipMapper:
    """
    Bidirectional mapper between source models (Pydantic/Dataclass) and Django models.
    """

    # Allow storing either source type
    pydantic_model: Optional[type[BaseModel]] = None
    dataclass_model: Optional[type] = None
    django_model: Optional[type[models.Model]] = None
    context: Optional[ModelContext] = None  # Keep context if needed later

    @property
    def source_model(self) -> Optional[type]:
        """Return the source model (either Pydantic or Dataclass)."""
        return self.pydantic_model or self.dataclass_model
```

```37:41:src/pydantic2django/core/relationships.py
@dataclass
class RelationshipConversionAccessor:
    available_relationships: list[RelationshipMapper] = field(default_factory=list)
```

## Importing and Exporting Mappings

```42:73:src/pydantic2django/core/relationships.py
@classmethod
def from_dict(cls, relationship_mapping_dict: dict) -> "RelationshipConversionAccessor":
    """
    Convert a dictionary of strings representing model qualified names to a RelationshipConversionAccessor
    """
    available_relationships = []
    for pydantic_mqn, django_mqn in relationship_mapping_dict.items():
        ...
        available_relationships.append(RelationshipMapper(pydantic_model, django_model, context=None))
    return cls(available_relationships)
```

```74:92:src/pydantic2django/core/relationships.py
def to_dict(self) -> dict:
    """
    Convert the relationships to a dictionary of strings representing
    model qualified names for bidirectional conversion.
    """
    relationship_mapping_dict = {}
    for relationship in self.available_relationships:
        ...
        relationship_mapping_dict[pydantic_mqn] = django_mqn
    return relationship_mapping_dict
```

Name formats for stable serialization:

```93:104:src/pydantic2django/core/relationships.py
def _get_pydantic_model_qualified_name(self, model: type[BaseModel] | None) -> str:
    if model is None:
        return ""
    return f"{model.__module__}.{model.__name__}"

def _get_django_model_qualified_name(self, model: type[models.Model] | None) -> str:
    if model is None:
        return ""
    return f"{model._meta.app_label}.{model.__name__}"
```

## Discovering and Adding Models

List and query known models:

```105:120:src/pydantic2django/core/relationships.py
@property
def available_source_models(self) -> list[type]:
    ...

@property
def available_django_models(self) -> list[type[models.Model]]:
    return [r.django_model for r in self.available_relationships if r.django_model is not None]
```

Add models incrementally:

```121:129:src/pydantic2django/core/relationships.py
def add_pydantic_model(self, model: type[BaseModel]) -> None:
    ...
    self.available_relationships.append(RelationshipMapper(model, None, context=None))
```

```130:140:src/pydantic2django/core/relationships.py
def add_dataclass_model(self, model: type) -> None:
    ...
    self.available_relationships.append(RelationshipMapper(dataclass_model=model))
```

```141:149:src/pydantic2django/core/relationships.py
def add_django_model(self, model: type[models.Model]) -> None:
    ...
    self.available_relationships.append(RelationshipMapper(None, None, model, context=None))
```

## Mapping and Lookup APIs

Lookups in either direction:

```150:170:src/pydantic2django/core/relationships.py
def get_django_model_for_pydantic(self, pydantic_model: type[BaseModel]) -> Optional[type[models.Model]]:
    for relationship in self.available_relationships:
        if relationship.pydantic_model == pydantic_model and relationship.django_model is not None:
            return relationship.django_model
    return None

def get_pydantic_model_for_django(self, django_model: type[models.Model]) -> Optional[type[BaseModel]]:
    for relationship in self.available_relationships:
        if relationship.django_model == django_model and relationship.pydantic_model is not None:
            return relationship.pydantic_model
    return None
```

Dataclass to Django lookup:

```172:181:src/pydantic2django/core/relationships.py
def get_django_model_for_dataclass(self, dataclass_model: type) -> Optional[type[models.Model]]:
    for relationship in self.available_relationships:
        if relationship.dataclass_model == dataclass_model and relationship.django_model is not None:
            return relationship.django_model
    return None
```

Create or update mappings in a single call:

```183:241:src/pydantic2django/core/relationships.py
def map_relationship(self, source_model: type, django_model: type[models.Model]) -> None:
    source_type = (
        "pydantic" if isinstance(source_model, type) and issubclass(source_model, BaseModel) else
        "dataclass" if dataclasses.is_dataclass(source_model) else
        "unknown"
    )
    if source_type == "unknown":
        logger.warning(...)
        return
    # Update existing or append a new RelationshipMapper
    ...
```

Known source-model checks and name-based lookup:

```242:253:src/pydantic2django/core/relationships.py
def is_source_model_known(self, model: type) -> bool:
    is_pydantic = isinstance(model, type) and issubclass(model, BaseModel)
    is_dataclass = dataclasses.is_dataclass(model)
    ...
```

```254:263:src/pydantic2django/core/relationships.py
def get_source_model_by_name(self, model_name: str) -> Optional[type]:
    for r in self.available_relationships:
        if r.pydantic_model and r.pydantic_model.__name__ == model_name:
            return r.pydantic_model
        if r.dataclass_model and r.dataclass_model.__name__ == model_name:
            return r.dataclass_model
    return None
```

## Typical Usage Patterns

- Initialize empty, add and map on the fly:

```python
from pydantic import BaseModel
from django.db import models
from pydantic2django.core.relationships import RelationshipConversionAccessor

class PostModel(BaseModel):
    ...

class BlogPost(models.Model):
    ...

rel = RelationshipConversionAccessor()
rel.add_pydantic_model(PostModel)
rel.add_django_model(BlogPost)
rel.map_relationship(PostModel, BlogPost)

# Lookups
assert rel.get_django_model_for_pydantic(PostModel) is BlogPost
```

- Serialize/restore mapping (e.g., persisted in JSONField):

```python
mapping_dict = rel.to_dict()
rel2 = RelationshipConversionAccessor.from_dict(mapping_dict)
```

## Integration with Type Mapping

The `RelationshipConversionAccessor` is used by the type mapping system to:
- Check if a source model is known before selecting relationship units (FK/M2M/O2O)
- Resolve `to` targets (including self-references and app labels)

See the relationship checks and resolution within the mapper where it calls into the accessor to determine known models and resolve target Django models.

---

## TimescaleDB constraints, soft references, and FK inversion

TimescaleDB does not support ForeignKeys that point to hypertables. To respect this:

- The XML generator classifies models as either hypertable (time-series facts) or dimension (regular tables).
- Relationship rules during generation:
  - **Hypertable → Regular**: generate a normal `ForeignKey`.
  - **Regular → Regular**: generate a normal `ForeignKey`.
  - **Hypertable → Hypertable**: generate a soft reference (e.g., `UUIDField(db_index=True)`) and leave referential validation to application code or background jobs.
  - **Regular → Hypertable (inverted)**: invert the relationship and emit the `ForeignKey` on the hypertable back to the regular dimension with `on_delete=SET_NULL, null=True, blank=True`.
    - Also auto-generate indexes on the hypertable to preserve performance:
      - `Index(fields=['<dimension_field>'])`
      - `Index(fields=['<dimension_field>', '-time'])` when a `time` field exists on the hypertable.

Heuristics and helpers live under `pydantic2django.django.timescale`:

- `classify_xml_complex_types(...)` produces a `{name: role}` map.
- `should_soft_reference(source_name, target_name, roles)` returns `True` for hypertable→hypertable edges.

This keeps schemas Timescale-safe while preserving joinability to dimensions and the ability to validate soft references at the application layer. The inversion avoids invalid FKs to hypertables after hypertable creation drops the primary key on the base table.
