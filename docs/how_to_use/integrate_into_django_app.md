## How to integrate generated models into your Django app

This guide shows how to place the generated static models file into your Django app, wire it up, and ship it with migrations.

### 1) Choose where to write the generated file

Recommended: generate into your app package and set the `app_label` to your app name. Example for app `myapp`:

```python
from pydantic2django.pydantic.generator import StaticPydanticModelGenerator

gen = StaticPydanticModelGenerator(
    output_path="myapp/models_generated.py",   # inside your app
    app_label="myapp",                         # matches your app label
    packages=["your_models_pkg"],
    module_mappings={"__main__": "myapp.models_generated"},  # optional import fixups
    verbose=True,
)
gen.generate()
```

Notes:
- `app_label` is embedded in each generated model’s `Meta`. This ensures migrations belong to your app.
- If your source classes are defined in scripts (module `__main__`), use `module_mappings` to rewrite imports to a stable module path.

### 2) Expose generated models from `myapp/models.py`

Keep your hand-written `models.py`, and re-export generated models:

```python
# myapp/models.py
from .models_generated import *  # re-export generated models

# (Optional) Your custom models can live here alongside the generated ones.
```

This lets admin, serializers, and other app code import everything from `myapp.models` as usual.

### 3) Add your app to INSTALLED_APPS

```python
# settings.py
INSTALLED_APPS = [
    # ...
    "myapp",
]
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
```

### 4) Create and apply migrations

Generated models are standard Django models; create migrations like normal:

```bash
python manage.py makemigrations myapp
python manage.py migrate
```

Regenerate models after source changes, then run `makemigrations` again if the schema changed.

### 5) Register in the Django admin (optional)

If you want to auto-register all generated models:

```python
# myapp/admin.py
from django.contrib import admin
from . import models_generated as gm

for name in getattr(gm, "__all__", []):
    admin.site.register(getattr(gm, name))
```

Or import specific classes and register explicitly.

### 6) Use conversions in your app

- Pydantic → Django (with generated model helpers):

```python
from generated_models import DjangoUser
from your_models_pkg import User as PydUser

u = PydUser(id=1, email="j@example.com", name="Jane")
dj = DjangoUser.from_pydantic(u)
dj.save()

u_back = dj.to_pydantic()
```

- Django → Pydantic (dynamic, no files written):

```python
from pydantic2django.django.conversion import DjangoPydanticConverter
from myapp.models import DjangoUser

conv = DjangoPydanticConverter(DjangoUser)
PydUser = conv.generated_pydantic_model

dj = DjangoUser.objects.first()
u = conv.to_pydantic(dj)
```

### 7) Project/CI tips

- Check in the generated file and migrations, or generate as part of your build before `makemigrations` (be consistent across environments).
- If you regenerate frequently, consider a small script or management command to run the generator.
- Keep `pydantic2django` installed in runtime environments if your generated file imports base classes (e.g., `Pydantic2DjangoBaseClass`).

---

### Override the generated base class

By default, generated models inherit a source-specific base (e.g., `Pydantic2DjangoBaseClass`, `Dataclass2DjangoBaseClass`, `Xml2DjangoBaseClass`). You can override the base to add mixins, managers, or TimescaleDB support.

- Define your custom abstract base:

```python
from django.db import models
from pydantic2django.django.models import Xml2DjangoBaseClass

class MyXmlBase(Xml2DjangoBaseClass):
    class Meta:
        abstract = True

    # your common fields/methods here
```

- Use it with a generator before calling `generate()`:

```python
from pydantic2django.xmlschema.generator import XmlSchemaDjangoModelGenerator

gen = XmlSchemaDjangoModelGenerator(
    schema_files=["path/to/schema.xsd"],
    output_path="myapp/models_generated.py",
    app_label="myapp",
)
gen.base_model_class = MyXmlBase  # override the base
gen.generate()
```

- TimescaleDB example (hypertable-ready base):

```python
# Combines Xml2DjangoBaseClass with TimescaleModel from django-timescaledb
from pydantic2django.django.models import XmlTimescaleBase
from pydantic2django.xmlschema.generator import XmlSchemaDjangoModelGenerator

gen = XmlSchemaDjangoModelGenerator(
    schema_files=["path/to/streams.xsd"],
    output_path="myapp/models_generated.py",
    app_label="myapp",
)
gen.base_model_class = XmlTimescaleBase
gen.generate()
```

Notes:
- The Timescale base expects a `time` column; see `django-timescaledb` docs for details (`https://github.com/jamessewell/django-timescaledb?tab=readme-ov-file`).
- If the generator supports a constructor parameter `base_model_class`, you can pass it directly instead of setting the attribute after construction. If not, setting `gen.base_model_class` before `generate()` is sufficient.

---

## TimescaleDB integration (automatic classification)

The XML generator can automatically classify types into hypertables (time-series facts) and dimensions (regular tables) and will:

- Use `XmlTimescaleBase` for hypertables and the regular `Xml2DjangoBaseClass` for dimensions.
- Convert illegal hypertable→hypertable relationships into soft references (`UUIDField(db_index=True)`), keeping schemas Timescale-safe.

What you need:

- Optionally install `django-timescaledb` to enable hypertable behavior; otherwise the Timescale mixin is a no-op.
- Run migrations as usual. Hypertables will be created where supported by `django-timescaledb`.

Tip: If you want to override the classification for specific types, you can pre/post-process the roles map returned by `classify_xml_complex_types(...)` in a custom generator wrapper before invoking `generate()`.
