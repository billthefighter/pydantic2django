# Pydantic2Django

Generate Django models from Pydantic models, Python dataclasses, or plain typed classes — and convert data back and forth reliably.

> [!IMPORTANT]
> Namespace rename and deprecation: The distribution is now `django-typed2django` and the new import namespace is `typed2django`. The old `pydantic2django` namespace is deprecated and will be removed in version 1.1.0. Please migrate imports to `typed2django.*`.

## Features

- Automatic Django model generation (`models.py`) from your Python types
- Bidirectional mapping: use `.from_pydantic()` and `.to_pydantic()` on generated models
- Relationship handling: `ForeignKey` and `ManyToManyField` inferred from your types
- Type-hint aware: supports `Optional`, `Union`, `Literal`, and common collections
- Modular design: separate pipelines for Pydantic, Dataclasses, and Typed classes

## Quickstart

1. Install

```bash
pip install django-typed2django
```

1. Define your Pydantic models

```python
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

1. Generate a `models.py` file

```python
from typed2django.pydantic.generator import StaticPydanticModelGenerator

generator = StaticPydanticModelGenerator(
    output_path="my_app/models.py",
    packages=["my_app.pydantic_models"],
    app_label="my_app",
    verbose=True,
)

# Write my_app/models.py
generator.generate()
```

1. Use the generated models

```python
from my_app.models import User as DjangoUser
from my_app.pydantic_models import User as PydanticUser

p_user = PydanticUser(name="Jane Doe", email="jane.doe@example.com")

# Create Django instance from Pydantic
dj_user = DjangoUser.from_pydantic(p_user)
dj_user.save()

# Convert back to Pydantic
assert dj_user.to_pydantic().name == "Jane Doe"
```

## How it works

1. Discovery: scan packages to find your source models
2. Factory: analyze fields, type hints, and relationships
3. Generator: render a complete `models.py` with imports, classes, and fields

## Explore next

- Architecture: high-level design and internals
- Docs: documentation files colocated with source code
- API: full reference generated from the code

You can navigate these sections from the left sidebar.
