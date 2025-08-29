## How to generate static Django models and convert bidirectionally

This guide shows how to turn several source types into static Django models and convert data back and forth:
- **Pydantic** ⇄ Django
- **Dataclass** ⇄ Django
- **TypedClass** ⇄ Django
- **XML Schema** ⇄ Django/XML
- **Django** ⇄ Pydantic (dynamic at runtime)

All generators write a single `generated_models.py` that contains Django model classes and convenience helpers. See templates in `src/pydantic2django/django/templates/` for the rendered structure.

### Prerequisites

- Configure Django before using any generator (minimal example shown below).
- Python 3.11+, Pydantic v2, UV/pytest per project guidelines.

```python
import django
from django.conf import settings

if not settings.configured:
    settings.configure(
        INSTALLED_APPS=[
            "django.contrib.contenttypes",
            "django.contrib.auth",
            # Add your app label if needed
        ],
        DATABASES={
            "default": {"ENGINE": "django.db.backends.sqlite3", "NAME": ":memory:"},
        },
        DEFAULT_AUTO_FIELD="django.db.models.BigAutoField",
    )
    django.setup()
```

---

## Pydantic ⇄ Django

Use `StaticPydanticModelGenerator` to scan packages for `BaseModel` classes and generate static Django models. Generated classes inherit `Pydantic2DjangoBaseClass` and include helpers like `.from_pydantic(...)` and `.to_pydantic(...)`.

```python
from pydantic import BaseModel, Field
from pydantic2django.pydantic.generator import StaticPydanticModelGenerator

# 1) Generate static Django models from your Pydantic package
gen = StaticPydanticModelGenerator(
    output_path="generated_models.py",
    packages=["your_pydantic_pkg"],
    app_label="your_app",
    filter_function=None,  # or a callable to include/exclude models
    module_mappings={"__main__": "your_app.models"},  # optional import fixups
    verbose=True,
)
gen.generate()

# 2) Use the generated models
from your_pydantic_pkg import User as PydUser
from generated_models import DjangoUser  # class name prefixed; see output

p = PydUser(id=1, email="john@example.com", name="John")
dj = DjangoUser.from_pydantic(p)
dj.save()

# Convert back
round_trip = dj.to_pydantic()
```

Notes:
- Choices map to `Literal[...]` in Pydantic with original choices preserved in `json_schema_extra`.
- Relationships, JSON, auto PKs, and file/image fields are handled by the bidirectional mapper.

---

## Dataclass ⇄ Django

Use `DataclassDjangoModelGenerator` to generate static Django models from Python dataclasses. Generated classes inherit `Dataclass2DjangoBaseClass` and include `.from_dataclass(...)` and `.to_dataclass(...)`.

```python
import dataclasses
from pydantic2django.dataclass.generator import DataclassDjangoModelGenerator

@dataclasses.dataclass
class Person:
    id: int
    name: str

gen = DataclassDjangoModelGenerator(
    output_path="generated_models.py",
    app_label="dc_app",
    packages=["your_dc_pkg"],
    filter_function=lambda cls: dataclasses.is_dataclass(cls),
    verbose=True,
)
gen.generate()

from your_dc_pkg import Person as DCPerson
from generated_models import DjangoPerson

dc = DCPerson(id=1, name="Ada")
dj = DjangoPerson.from_dataclass(dc)
dj.save()

back = dj.to_dataclass()
```

---

## TypedClass ⇄ Django

Use `TypedClassDjangoModelGenerator` for plain Python classes with type hints you want to map to Django fields. Generated classes inherit `TypedClass2DjangoBaseClass` and include `.from_typedclass(...)` and `.to_typedclass(...)`.

```python
from pydantic2django.typedclass.generator import TypedClassDjangoModelGenerator

class Settings:
    def __init__(self, retries: int, region: str):
        self.retries = retries
        self.region = region

gen = TypedClassDjangoModelGenerator(
    output_path="generated_models.py",
    app_label="typed_app",
    packages=["your_typed_pkg"],
    filter_function=None,
    verbose=True,
)
gen.generate()

from generated_models import DjangoSettings

cfg = Settings(retries=3, region="us-east-1")
dj = DjangoSettings.from_typedclass(cfg)
dj.save()

restored = dj.to_typedclass()
```

---

## XML Schema ⇄ Django/XML

Use `XmlSchemaDjangoModelGenerator` to parse `.xsd` files and generate static Django models that inherit `Xml2DjangoBaseClass`. These provide XML-focused helpers like `.from_xml_dict(...)`, `.to_xml_dict(...)` and `.to_xml_string(...)`.

```python
from pydantic2django.xmlschema.generator import XmlSchemaDjangoModelGenerator

gen = XmlSchemaDjangoModelGenerator(
    schema_files=["tests/xmlschema/fixtures/simple_schema.xsd"],
    output_path="generated_models.py",
    app_label="xsd_app",
    nested_relationship_strategy="auto",  # "fk" | "json" | "auto"
    list_relationship_style="child_fk",   # "child_fk" | "m2m" | "json"
    verbose=True,
)
gen.generate()

from generated_models import Address  # example class from the schema

addr = Address.from_xml_dict({"street": "123 Main", "zipCode": "94107"})
xml = addr.to_xml_string()
```

#### Overriding the base model class

All generators keep their intended base class on `gen.base_model_class`. You can override it before calling `generate()` to inject your own abstract base or enable integrations.

Example: enable TimescaleDB hypertables for XML-generated models using the provided `XmlTimescaleBase` (which combines `Xml2DjangoBaseClass` and `TimescaleModel`):

```python
from pydantic2django.django.models import XmlTimescaleBase
from pydantic2django.xmlschema.generator import XmlSchemaDjangoModelGenerator

gen = XmlSchemaDjangoModelGenerator(
    schema_files=["..."],
    output_path="generated_models.py",
    app_label="xsd_app",
)
gen.base_model_class = XmlTimescaleBase
gen.generate()
```

If a generator exposes a `base_model_class=` constructor argument, you can pass it directly. Otherwise, assigning to `gen.base_model_class` prior to `generate()` is supported.

Note: TimescaleDB expects a `time` column; see the `django-timescaledb` README for details (`https://github.com/jamessewell/django-timescaledb?tab=readme-ov-file`).

---

## Django ⇄ Pydantic (dynamic)

To project a Django model to a Pydantic model (and back) at runtime, use `DjangoPydanticConverter`. This does not write static files; it generates a Pydantic model class on the fly using the same bidirectional mapper.

```python
from pydantic2django.django.conversion import DjangoPydanticConverter
from your_django_app.models import User

conv = DjangoPydanticConverter(User)
PydUser = conv.generated_pydantic_model

u = User.objects.get(pk=1)
p = conv.to_pydantic(u)

# Modify and persist back
u2 = conv.to_django(p)  # updates or creates
```

---

## Common options and tips

- **Filtering**: `filter_function` lets you keep only specific source models.
- **Import fixes**: `module_mappings` rewrites imports (e.g., replace `__main__` with `your_app.models`).
- **Relationships**: Pydantic/Dataclass/TypedClass generators resolve dependencies and order automatically.
- **Context fields**: Pydantic-only context values are surfaced via generated context classes; pass them when calling `.to_pydantic(...)` if required by your model.
- **Templates**: Jinja2 templates under `django/templates/` control the generated file layout.
