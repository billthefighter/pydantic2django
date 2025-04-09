# Pydantic2Django

A utility for generating Django models from Pydantic models and Python dataclasses, and providing seamless conversion between them.

## Overview

Pydantic2Django allows you to:

1. Generate Django models from your Pydantic models or Python dataclasses
2. Store Pydantic or dataclass objects in a Django database
3. Convert between Django model instances and Pydantic/dataclass objects

This is particularly useful when you have a codebase that uses Pydantic models or standard Python dataclasses for data validation and representation, and you want to persist those models in a Django database.

## Installation

```bash
pip install pydantic2django
```

## Usage

### Basic Usage

```python
from pydantic2django.static_django_model_generator import StaticDjangoModelGenerator

# Create a generator
generator = StaticDjangoModelGenerator(
    output_path="myapp/models.py",
    packages=["myapp.pydantic_models"],
    app_label="myapp",
    verbose=True
)

# Generate the models file
generator.generate()
```

### Filtering Models

You can specify a filter function to only include certain models:

```python
def filter_models(model):
    """Only include models with names starting with 'User'"""
    return model.__name__.startswith("User")

generator = StaticDjangoModelGenerator(
    output_path="myapp/models.py",
    packages=["myapp.pydantic_models"],
    app_label="myapp",
    filter_function=filter_models,
    verbose=True
)
```

### Using the Generated Models

The generated Django models inherit from `Pydantic2DjangoBaseClass`, which provides methods for converting between Django models and Pydantic objects:

```python
from myapp.models import DjangoUser

# Create a new Django model instance from a Pydantic object
django_user = DjangoUser.from_pydantic(pydantic_user, name="John Doe")
django_user.save()

# Convert back to a Pydantic object
pydantic_user = django_user.to_pydantic()
```

## How It Works

Pydantic2Django works by:

1. Discovering Pydantic models or dataclasses in the specified packages (or through direct registration)
2. Generating Django models that inherit from `Pydantic2DjangoBaseClass` (for Pydantic) or facilitating mapping (for both)
3. Storing the Pydantic/dataclass object's data in a JSONField (in the case of `Pydantic2DjangoBaseClass`)
4. Providing mechanisms (`RelationshipAccessor`) to manage mappings and convert between Django models and source objects (Pydantic/dataclasses)

The `Pydantic2DjangoBaseClass` (primarily for generated models from Pydantic) provides the following fields:

- `id`: UUID primary key
- `name`: A human-readable name for the object
- `object_type`: The name of the Pydantic class
- `data`: JSONField containing the serialized Pydantic object
- `created_at`: Timestamp when the object was created
- `updated_at`: Timestamp when the object was last updated

## Examples

See the `examples` directory for complete examples of how to use Pydantic2Django.

## Testing

Run the tests with:

```bash
python -m unittest discover tests
```

## License

MIT
