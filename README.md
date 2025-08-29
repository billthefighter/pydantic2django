# Pydantic2Django: The Bridge Between Python Objects and the Django ORM

A powerful utility for generating Django models from Pydantic models, Python dataclasses, and even generic Python classes, and providing seamless, bidirectional data conversion between them.

> [!IMPORTANT]
> Namespace rename and deprecation: The distribution is now published as `django-typed2django` and the new import namespace is `typed2django`. The old `pydantic2django` namespace is deprecated and will be removed in version 1.1.0. Please migrate imports from `pydantic2django.*` to `typed2django.*`.

## Overview

Pydantic2Django bridges the gap between your application's data layer (defined using Pydantic, dataclasses, etc.) and your persistence layer in Django. It allows you to:

1. **Generate `models.py` files automatically** from your existing Python classes.
2. **Convert data seamlessly** between Django model instances and your source Pydantic/dataclass objects.
3. **Persist and manage** application data objects in a Django database without writing boilerplate mapping code.
4. **Handle relationships** between your Python objects and translate them into Django's relational fields (`ForeignKey`, `ManyToManyField`).

This library supports:

- **Pydantic Models**: Robust support for `pydantic.BaseModel`.
- **Python Dataclasses**: Full support for classes decorated with `@dataclasses.dataclass`.
- **Generic Python Classes**: Experimental support for converting plain Python classes that act as data containers (see [Experimental Features](#experimental-features)).

## Core Features

- **Automatic Django Model Generation**: Scans your project to discover data classes and generates a complete, ready-to-use `models.py` file.
- **Bidirectional Data Mapping**: The generated Django models, or models inheriting from the provided base classes, include methods like `.from_pydantic()` and `.to_pydantic()` for easy and reliable data conversion.
- **Relationship Management**: Intelligently detects relationships between your source models and automatically creates the corresponding `ForeignKey` and `ManyToManyField` fields in Django.
- **Timescale-aware Generation (optional)**: Provides heuristics and safe wiring for TimescaleDB so hypertables never use illegal hypertable→hypertable FKs (replaced with indexed soft references), while preserving legal FK patterns.
- **Type Hint Aware**: Leverages type hints to create the most appropriate Django model fields, including support for `Optional`, `Union`, `Literal`, and generic collections.
- **Extensible & Modular**: The library is structured into distinct modules for `pydantic`, `dataclass`, and `typedclass` handling, allowing for clear separation of concerns. You can dive into the core logic in `src/pydantic2django/core/`.

## Full Documentation

For the complete documentation site, visit the GitHub Pages deployment: [pydantic2django docs](https://billthefighter.github.io/pydantic2django/).

## Alternatives

Consider [django-pydantic-field](https://github.com/surenkov/django-pydantic-field) which has lots of cool features but not quite as much scope (and, commensurately, not as much complexity) as this library.

## Installation

```bash
pip install django-typed2django
```

## How It Works

Pydantic2Django operates using a three-stage pipeline:

1. **Discovery**: It scans specified Python packages to find and identify source models (Pydantic, dataclass, etc.).
2. **Factory**: For each discovered model, it analyzes its fields, type hints, and relationships to create an in-memory representation of a Django model.
3. **Generator**: It uses the in-memory representation and Jinja2 templates to generate the final Python code for your `models.py` file, including all necessary imports, model classes, and field definitions.

## Usage

### Example 1: Generating a `models.py` File

The most common use case is to generate a Django `models.py` file from your existing Pydantic models or dataclasses.

Let's say you have Pydantic models in `my_app/pydantic_models.py`:

```python
# my_app/pydantic_models.py
import uuid
from pydantic import BaseModel, Field

class User(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    name: str
    email: str

class Product(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    name: str
    price: float
    owner: User
```

You can generate the Django models with a simple script:

```python
# scripts/generate_django_models.py
from typed2django.pydantic.generator import StaticPydanticModelGenerator

# Create a generator for Pydantic models
generator = StaticPydanticModelGenerator(
    output_path="my_app/models.py",
    packages=["my_app.pydantic_models"],
    app_label="my_app",
    verbose=True
)

# Generate the models file
generator.generate_models_file()
print("Django models generated successfully!")
```

This will create `my_app/models.py` containing `User` and `Product` Django models, complete with a `ForeignKey` relationship.

> **Note**: For dataclasses, you would use `DataclassDjangoModelGenerator` from `typed2django.dataclass`.

### Example 2: Using the Generated Models

The generated models inherit from a base class that provides helpful conversion methods.

```python
# my_app/views.py
from my_app.models import User as DjangoUser
from my_app.pydantic_models import User as PydanticUser

# Assume you get a Pydantic object from an API call
pydantic_user = PydanticUser(name="Jane Doe", email="jane.doe@example.com")

# 1. Create a new Django model instance from the Pydantic object
django_user = DjangoUser.from_pydantic(pydantic_user)
django_user.save()

# 2. Retrieve a Django object and convert it back to a Pydantic object
retrieved_user = DjangoUser.objects.get(name="Jane Doe")
pydantic_version = retrieved_user.to_pydantic()

assert pydantic_version.name == "Jane Doe"
```

This bidirectional conversion makes it trivial to move data between your application logic and the database.

## Advanced Usage

### Filtering Models

You can provide a `filter_function` to the generator to selectively include or exclude models from the generation process.

```python
def user_only_filter(model):
    """Only include models with names containing 'User'"""
    return "User" in model.__name__

generator = StaticPydanticModelGenerator(
    # ... other args
    filter_function=user_only_filter,
)
```

## TimescaleDB Integration (Optional)

Pydantic2Django can help you generate models that follow TimescaleDB’s constraints and best practices.

- **Safe FK policy**
  - Hypertable → Dimension: FK allowed.
  - Dimension → Dimension: FK allowed.
  - Hypertable → Hypertable: Not allowed by TimescaleDB. The generator replaces these with an indexed soft reference field (e.g., `UUIDField(db_index=True, null=True, blank=True)`), leaving referential checks to the application/background jobs.

- **Heuristics to classify hypertables vs dimensions** (XML path)
  - Classified by signals like: presence of time/timestamp/sequence fields; unbounded list growth; event/observation-like names (e.g., `Samples`, `Events`, `Condition`); negative weight for definition/metadata types (e.g., `*Definition*`, `*Parameters*`, `Header`).
  - The XML generator chooses a per-model base:
    - Hypertable types use `pydantic2django.django.timescale.bases.XmlTimescaleBase`.
    - Dimension types use `pydantic2django.django.models.Xml2DjangoBaseClass`.
  - Illegal hypertable→hypertable relations are automatically converted to soft references.

- **Using Timescale bases in other paths**
  - Pydantic and Dataclass paths can opt-in by setting the base explicitly before generation:

```python
from pydantic import BaseModel
from pydantic2django.django.timescale.bases import PydanticTimescaleBase
from pydantic2django.pydantic.generator import StaticPydanticModelGenerator

class SampleObservation(BaseModel):
    device_id: str
    timestamp: int
    value: float

gen = StaticPydanticModelGenerator(output_path="models.py", packages=["my_pkg"], app_label="my_app")
# Force Timescale-enabled base for generated models (treat as hypertables)
gen.base_model_class = PydanticTimescaleBase
gen.generate_models_file()
```

For XML schemas, Timescale handling is automatic via heuristics. For non-XML paths, choose a Timescale base where appropriate.

## Experimental Features

### Generic Python Class Conversion

The library includes an **experimental** generator for converting plain Python classes into Django models, located in `src/pydantic2django/typedclass/`. This is useful for persisting configuration objects or instances of classes from third-party libraries.

Due to its experimental nature, it relies heavily on clear type hints in `__init__` and may require manual adjustments to the generated code. You can read more about its goals and limitations in the [TypedClass README](./src/pydantic2django/typedclass/README.md).

## Testing

Run the tests with:

```bash
uv run pytest -q
```

## Contributing

Contributions are welcome! Please read the contributing guide: [CONTRIBUTING.md](./CONTRIBUTING.md).

- All pull requests must pass the full test suite locally and in CI to demonstrate no regressions, or include a clear explanation for why automated tests are not appropriate for the change (with manual validation steps or rationale).
- Aim to maintain backward compatibility, keep dependencies minimal, and follow PEP 8 with type hints.

## License

MIT
