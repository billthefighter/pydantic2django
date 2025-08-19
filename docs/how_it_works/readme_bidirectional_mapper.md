# Pydantic2Django Bidirectional Mapper Details

This document provides details on how the `BidirectionalTypeMapper` handles specific field type conversions between Django models and Pydantic models.

## How selection works (at a glance)

The mapper discovers and orders mapping units (e.g., `StrFieldMapping`, `IntFieldMapping`, relationships) and then selects the best match per field based on scoring and context.

```94:141:src/pydantic2django/core/bidirectional_mapper.py
def _build_registry(self) -> list[type[TypeMappingUnit]]:
    ordered_units = [
        BigAutoFieldMapping, SmallAutoFieldMapping, AutoFieldMapping,
        PositiveBigIntFieldMapping, PositiveSmallIntFieldMapping, PositiveIntFieldMapping,
        EmailFieldMapping, URLFieldMapping, SlugFieldMapping, IPAddressFieldMapping, FilePathFieldMapping,
        ImageFieldMapping, FileFieldMapping,
        UUIDFieldMapping, JsonFieldMapping,
        ManyToManyFieldMapping, OneToOneFieldMapping, ForeignKeyMapping,
        DecimalFieldMapping, DateTimeFieldMapping, DateFieldMapping, TimeFieldMapping,
        DurationFieldMapping, BinaryFieldMapping, FloatFieldMapping,
        TextFieldMapping, StrFieldMapping,
        BigIntFieldMapping, SmallIntFieldMapping, IntFieldMapping,
        EnumFieldMapping,
    ]
```

## Django `choices` Field Mapping

When mapping a Django field (like `CharField` or `IntegerField`) that has the `choices` attribute set, the `BidirectionalTypeMapper` employs a hybrid approach for the resulting Pydantic field:

1.  **Pydantic Type:** The Python type hint for the Pydantic field is set to `typing.Literal[...]`, where the literal values are the *raw database values* defined in the Django `choices` (e.g., `Literal['S', 'M', 'L']` or `Literal[1, 2, 3]`). If the Django field has `null=True`, the type becomes `Optional[Literal[...]]`.
    *   **Benefit:** This provides strong typing and allows Pydantic to perform validation, ensuring that only the allowed raw values are assigned to the field.

2.  **Metadata:** The original Django `choices` tuple, containing the `(raw_value, human_readable_label)` pairs (e.g., `[('S', 'Small'), ('M', 'Medium'), ('L', 'Large')]`), is preserved within the Pydantic `FieldInfo` associated with the field. Specifically, it's stored under the `json_schema_extra` key:
    ```python
    FieldInfo(..., json_schema_extra={'choices': [('S', 'Small'), ('M', 'Medium')]})
    ```
    *   **Benefit:** This keeps the human-readable labels associated with the field, making them available for other purposes like generating API documentation (e.g., OpenAPI schemas), building UI components (like dropdowns), or custom logic, without sacrificing the validation provided by the `Literal` type.

**Trade-off:** This approach prioritizes data validation using Pydantic's `Literal` type based on the raw stored values. The human-readable labels are available as metadata but are not part of the core Pydantic type validation itself. The Django `get_FOO_display()` method is not directly used during the conversion process, as the focus is on mapping the underlying data values and types.

Relevant code references:

```193:199:src/pydantic2django/core/bidirectional_mapper.py
original_origin = get_origin(original_type_for_cache)
if original_origin is Literal:
    best_unit = EnumFieldMapping
    self._pydantic_cache[cache_key] = best_unit
    return best_unit
```

```739:748:src/pydantic2django/core/bidirectional_mapper.py
if is_choices:
    final_pydantic_type = base_instance_unit.python_type
    if dj_field.choices:
        choice_values = tuple(choice[0] for choice in dj_field.choices)
        if choice_values:
            final_pydantic_type = Literal[choice_values]  # type: ignore
```

```842:850:src/pydantic2django/core/bidirectional_mapper.py
if (
    is_choices
    and "json_schema_extra" in field_info_kwargs
    and "choices" in field_info_kwargs["json_schema_extra"]
):
    logger.debug(f"Kept choices in json_schema_extra for Literal field '{dj_field.name}'")
```

## Other Field Mappings

*(This section can be expanded later with details about other interesting or complex mappings, such as relationships, JSON fields, etc.)*

---

## Pydantic ➜ Django examples

- **int**: `int` ➜ `models.IntegerField`
  - With non-negative constraint (e.g., `ge=0`) ➜ `models.PositiveIntegerField`.
  - Auto PKs are handled separately (see below).

- **float/Decimal**:
  - `float` ➜ `models.FloatField`
  - `Decimal` ➜ `models.DecimalField` (uses `max_digits`/`decimal_places` if provided)

- **bool**: `bool` ➜ `models.BooleanField` (defaults aligned)

- **str-family**:
  - `str` with `max_length` ➜ `models.CharField(max_length=...)`
  - `str` without `max_length` ➜ `models.TextField`
  - `EmailStr` ➜ `models.EmailField`
  - `HttpUrl` ➜ `models.URLField`
  - `IPvAnyAddress` ➜ `models.GenericIPAddressField`
  - `str` with slug pattern ➜ `models.SlugField`
  - `pathlib.Path` ➜ `models.FilePathField`
  - File/Image hints ➜ `models.FileField`/`models.ImageField` (stored as path/URL in Pydantic)

- **UUID/datetime**:
  - `UUID` ➜ `models.UUIDField`
  - `datetime`, `date`, `time`, `timedelta` ➜ respective `DateTimeField`, `DateField`, `TimeField`, `DurationField`

- **bytes**: `bytes` ➜ `models.BinaryField`

- **JSON-like/Any**:
  - `dict`, `list`, `tuple`, `set`, or `Any` ➜ `models.JSONField`

- **Enums/Literals**:
  - `Literal['A','B']` or `Enum` ➜ underlying `CharField`/`IntegerField` with `choices` set.

- **Relationships**:
  - `RelatedModel` ➜ `models.ForeignKey("app.relatedmodel", on_delete=...)`
  - `list[RelatedModel]` ➜ `models.ManyToManyField("app.relatedmodel")`
  - `Union[ModelA, ModelB]` ➜ stored as `JSONField` with `_union_details` metadata for downstream handling.

Key relationship handling:

```631:705:src/pydantic2django/core/bidirectional_mapper.py
if unit_cls in (ForeignKeyMapping, OneToOneFieldMapping, ManyToManyFieldMapping):
    ...
    if is_self_ref:
        model_ref = "self"
    else:
        target_django_model = ...
        model_ref = getattr(target_django_model._meta, "label_lower", target_django_model.__name__)
    kwargs["to"] = model_ref
    django_field_type = unit_cls.django_field_type
    if unit_cls in (ForeignKeyMapping, OneToOneFieldMapping):
        kwargs["on_delete"] = (models.SET_NULL if is_optional else models.CASCADE)
```

M2M/list and unions detection:

```435:488:src/pydantic2django/core/bidirectional_mapper.py
if is_list:
    ...
    if inner_origin in (Union, UnionType) and inner_args:
        ...
        unit_cls = JsonFieldMapping  # GFK-like signal
    elif ... inner_type is known model ...:
        unit_cls = ManyToManyFieldMapping
    else:
        unit_cls = JsonFieldMapping  # list of non-models
```

Union of models (multi-FK signal):

```502:563:src/pydantic2django/core/bidirectional_mapper.py
if simplified_origin in (Union, UnionType) and simplified_args:
    ...
    if union_models and not other_types_in_union:
        union_details = {"type": "multi_fk", "models": union_models, "is_optional": is_optional}
        unit_cls = JsonFieldMapping
```

Null/blank defaults:

```703:709:src/pydantic2django/core/bidirectional_mapper.py
if django_field_type != models.ManyToManyField and not union_details:
    kwargs["null"] = is_optional
    kwargs["blank"] = is_optional
```

Auto-increment PKs (wrapped as Optional in Pydantic when mapping Django ➜ Pydantic):

```804:812:src/pydantic2django/core/bidirectional_mapper.py
is_auto_pk = dj_field.primary_key and isinstance(
    dj_field, (models.AutoField, models.BigAutoField, models.SmallAutoField)
)
if is_auto_pk:
    final_pydantic_type = Optional[int]
    is_optional = True
```

### Minimal Pydantic ➜ Django examples

```python
from pydantic import BaseModel, Field
from pydantic import EmailStr

class User(BaseModel):
    id: int | None = Field(default=None, title="ID")                 # Auto PK ➜ AutoField (null/blank True)
    email: EmailStr = Field(title="Email")                           # ➜ EmailField(max_length=254)
    name: str = Field(title="Name", max_length=120)                  # ➜ CharField(max_length=120)
    bio: str = Field(title="Bio")                                    # ➜ TextField
    settings: dict = Field(title="Settings")                          # ➜ JSONField
```

---

## Django ➜ Pydantic examples

- **Choices ➜ Literal[...] with metadata**: as detailed above.
- **Relationships**:
  - `ForeignKey(User, null=True)` ➜ `Optional[UserModel]`
  - `ManyToManyField(Tag)` ➜ `list[TagModel]`
  - Self-references are represented conservatively; see code reference.

- **Auto PKs**: AutoField/BigAutoField/SmallAutoField ➜ `Optional[int]` with `frozen=True` in `FieldInfo`.

### Minimal Django ➜ Pydantic examples

```python
from django.db import models

class User(models.Model):
    id = models.AutoField(primary_key=True)
    email = models.EmailField()
    name = models.CharField(max_length=120)
    bio = models.TextField(blank=True)
    settings = models.JSONField(default=dict)

# ➜ Pydantic (conceptual):
# class UserModel(BaseModel):
#     id: int | None = Field(default=None, frozen=True)
#     email: EmailStr
#     name: str = Field(max_length=120)
#     bio: str | None = None
#     settings: dict = Field(default_factory=dict)
```

---

## Notes

- Constraint hints (e.g., `max_length`, `ge`, `decimal_places`) are read from `FieldInfo.metadata` when available and mapped accordingly by the specific mapping units.
- Lists of non-models map to `JSONField`; lists of known models map to `ManyToManyField`.
- Unions of known models are represented as `JSONField` with `_union_details` to signal multi-FK semantics to generators.
