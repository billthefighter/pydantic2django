# Pydantic2Django Context Storage and Generated Context Classes

This document explains why the context mechanism exists, how `ModelContext` and `FieldContext` work, and how the `context_class.py.j2` template generates per-model context classes used during Django ↔︎ Pydantic/Dataclass conversion.

## Why a context mechanism is required

Some information necessary to round-trip between Django and source models (Pydantic or Dataclass) cannot be represented as first-class Django fields or inferred at runtime solely from the database. Examples include:

- Non-serializable or runtime-only values needed to reconstruct source objects
- Disambiguation for unions/polymorphic fields and special relationship shapes
- Additional metadata required to map back from Django → Pydantic with fidelity

To support lossless conversion, these values are captured as named context fields. They travel alongside model instances (not stored in the database) and are injected back into the source models during reconstruction.

## Core data structures

`FieldContext` describes a single context field: its name, type, optionality, list-ness, metadata, and value.

```27:39:src/pydantic2django/core/context.py
@dataclass
class FieldContext:
    """
    Represents context information for a single field.
    """

    field_name: str
    field_type_str: str  # Renamed from field_type for clarity
    is_optional: bool = False
    is_list: bool = False
    additional_metadata: dict[str, Any] = field(default_factory=dict)
    value: Optional[Any] = None
```

`ModelContext[SourceModelType]` is the in-memory container for all context fields associated with a Django model and its corresponding source model.

```50:58:src/pydantic2django/core/context.py
    django_model: type[models.Model]
    source_class: type[SourceModelType]  # Changed from pydantic_class
    context_fields: dict[str, FieldContext] = field(default_factory=dict)
    context_data: dict[str, Any] = field(default_factory=dict)

    @property
    def required_context_keys(self) -> set[str]:
        required_fields = {fc.field_name for fc in self.context_fields.values() if not fc.is_optional}
        return required_fields
```

Add fields programmatically using `add_field(...)`:

```60:87:src/pydantic2django/core/context.py
    def add_field(
        self,
        field_name: str,
        field_type_str: str,
        is_optional: bool = False,
        is_list: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Add a field to the context storage.
        """
        field_context = FieldContext(
            field_name=field_name,
            field_type_str=field_type_str,
            is_optional=is_optional,
            is_list=is_list,
            additional_metadata=kwargs,
        )
        self.context_fields[field_name] = field_context
```

Validation and value access helpers:

```88:159:src/pydantic2django/core/context.py
    def validate_context(self, context: dict[str, Any]) -> None:
        ...

    def get_field_type_str(self, field_name: str) -> Optional[str]:
        ...

    def get_field_by_name(self, field_name: str) -> Optional[FieldContext]:
        ...

    def to_conversion_dict(self) -> dict[str, Any]:
        ...

    def set_value(self, field_name: str, value: Any) -> None:
        ...

    def get_value(self, field_name: str) -> Optional[Any]:
        ...
```

Automatic import synthesis for generated classes uses the central `TypeHandler`:

```160:201:src/pydantic2django/core/context.py
    def get_required_imports(self) -> dict[str, set[str]]:  # Return sets for auto-dedup
        """
        Get all required imports for the context class fields using TypeHandler.
        """
        imports: dict[str, set[str]] = {"typing": set(), "custom": set()}
        ...
        return imports
```

## Generated context classes

At generation time, a lightweight, per-model dataclass is rendered from `context_class.py.j2`. It mirrors the `ModelContext` instance but provides a concrete, importable Python class specific to the Django model.

```1:46:src/pydantic2django/django/templates/context_class.py.j2
@dataclass
class {{ model_name }}Context(ModelContext):
    """
    Context class for {{ model_name }}.
    Contains non-serializable fields that need to be provided when converting from Django to Pydantic.
    """
    model_name: str = "{{ model_name }}"
    pydantic_class: type = {{ pydantic_class }}
    django_model: type[models.Model]
    context_fields: dict[str, FieldContext] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize context fields after instance creation."""
        {% for field in field_definitions %}
        self.add_field(
            field_name="{{ field.name }}",
            field_type={{ field.literal_type }},
            is_optional={{ field.is_optional }},
            is_list={{ field.is_list }},
            additional_metadata={{ field.metadata }}
        )
        {% endfor %}
{% if field_definitions %}
    @classmethod
    def create(cls,
             django_model: Type[models.Model],
{% for field in field_definitions %}
             {{ field.name }}: {{ field.raw_type }},
{% endfor %}
             ):
        """
        Create a context instance with the required field values.
        """
        context = cls(django_model=django_model)
        {% for field in field_definitions %}
        context.set_value("{{ field.name }}", {{ field.name }})
        {% endfor %}
        return context
{% endif %}
```

The generator responsible for rendering these classes is `ContextClassGenerator`:

```202:217:src/pydantic2django/core/context.py
    @classmethod
    def generate_context_class_code(cls, model_context: "ModelContext", jinja_env: Any | None = None) -> str:
        """
        Generate a string representation of the context class.
        """
        generator = ContextClassGenerator(jinja_env=jinja_env)
        return generator.generate_context_class(model_context)
```

```285:329:src/pydantic2django/core/context.py
    def generate_context_class(self, model_context: ModelContext) -> str:
        """
        Generates the Python code string for a context dataclass.
        """
        template = self._load_template("context_class.py.j2")
        self.imports = model_context.get_required_imports()  # Get imports first
        ...
        return template.render(
            model_name=model_name,
            source_class_name=source_class_name,
            source_module=model_context.source_class.__module__,
            field_definitions="\n".join(field_definitions),
            typing_imports=typing_imports_str,
            custom_imports=custom_imports_list,
        )
```

### What the generated class gives you

- A stable, importable dataclass per Django model that knows which context fields exist
- A `create(...)` helper that constructs the context and sets required values
- Type-annotated fields, with `Optional`/`List` added when applicable
- Imports deduplicated and synthesized automatically via `TypeHandler`

## Typical workflow

1. During generation, the factory builds a `ModelContext` for each model and computes required context fields.
2. `ContextClassGenerator` renders a concrete `<ModelName>Context` class from the template.
3. When converting Django → source model, populate an instance of `<ModelName>Context` with required values.
4. Use `to_conversion_dict()` to pass the captured values into the reconstruction step.

Minimal example sketch:

```python
from pydantic2django.core.context import ModelContext

# model_context is prepared by the generation pipeline
model_context: ModelContext = ...

# At runtime, set values that cannot be read from Django fields alone
model_context.set_value("token", external_token)
model_context.set_value("polymorphic_type", "SubTypeA")

# Later, feed context back into reconstruction
context_values = model_context.to_conversion_dict()
# conversion_layer.reconstruct_from_django(instance, context=context_values)
```

## Design notes

- Context data is not persisted in the database; it is runtime-only.
- Fields are explicit and validated: `required_context_keys` ensures consumers supply all required values.
- Import handling is centralized, so generated classes remain clean and minimal.
- The mechanism is agnostic to source type (Pydantic or Dataclass) via the generic `ModelContext[SourceModelType]`.

---

See also:
- `docs/how_it_works/core_modules_and_concepts.md` for an overview of `context.py` in the core module set.
- `docs/how_it_works/readme_bidirectional_mapper.md` and `docs/how_it_works/readme_relationships.md` for related mapping and relationship infrastructure that may supply or require context values.
