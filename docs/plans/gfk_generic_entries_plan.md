# Provider‑agnostic GenericForeignKey (GFK) plan

## What

Introduce a generator‑controlled, provider‑agnostic mode to model polymorphic/repeating nested XML (and multi‑type pydantic/dataclass fields) using Django’s contenttypes (GenericForeignKey) instead of emitting thousands of concrete leaf models or large numbers of JSONFields. This keeps the model surface small, migrations fast, and data queryable across all ingests.

## Why

- Current xmlschema output (e.g., MTConnect Streams) can generate 90k+ line models and lengthy migrations due to substitution groups and deeply nested, repeating leaves.
- Many “leaf” combinations require either per‑leaf classes or JSON fallback; both scale poorly and are hard to query.
- GenericForeignKey lets us store polymorphic entries in a single normalized table with a reverse link to the owning parent, drastically reducing generated code while remaining queryable.
- Approach must be provider‑agnostic to support xmlschema, pydantic, and dataclass paths without domain‑specific types.

Reference: Django contenttypes [The contenttypes framework](https://docs.djangoproject.com/en/5.2/ref/contrib/contenttypes/).

## Scope and behavior

Add core flags (consumed by all generators):
- `enable_gfk: bool` (default: false)
- `gfk_policy: Literal["substitution_only", "repeating_only", "all_nested", "threshold_by_children"]`
- `gfk_threshold_children: int` (used when `threshold_by_children`)
- `gfk_value_mode: Literal["json_only", "typed_columns"]`
- `gfk_normalize_common_attrs: bool` (normalize timestamp/subType‑like attrs; default: false)

When enabled and a field/element meets policy:
- Do not generate a concrete leaf model or schedule child FK injection
- Instead, store instances as rows in a single `GenericEntry` model and expose a `GenericRelation` on the parent for reverse access

## Design details

### Core additions

- Generic entry model (emitted per app in generated output):
  - `content_type`, `object_id`, `content_object` (GenericForeignKey)
  - `element_qname: CharField`
  - `type_qname: CharField | null`
  - `attrs_json: JSONField`
  - Optional typed value columns (if `typed_columns`): `text_value`, `num_value`, `time_value`
  - `order_index: IntegerField`
  - Optional `path_hint: CharField` for debugging/tracing

- Parent reverse access:
  - Add `GenericRelation('GenericEntry', related_query_name='entries')` to parents (wrappers/components) that own qualifying children

- Core policy hook points:
  - In field factories: if policy says GFK, don’t create a concrete field; record a marker on the `ConversionCarrier` (e.g., `carrier.context_data["_pending_gfk_children"]`)
  - In model factory finalize: ensure the `GenericEntry` model exists, inject `GenericRelation` on parents, add indexes (e.g., `(content_type, object_id)`)

### Generator‑specific detection

- xmlschema (`src/pydantic2django/xmlschema/`):
  - Identify substitution groups, repeating complex leaves, deeply nested wrappers; apply chosen policy
  - Ingestor should create `GenericEntry` rows instead of concrete leaf instances

- pydantic (`src/pydantic2django/pydantic/`) and dataclass (`src/pydantic2django/dataclass/`):
  - Detect `Union`/variant fields (e.g., `List[Union[A, B, ...]]`); when enabled, route through GFK instead of many concrete field alternatives

### Tests and docs

- Tests:
  - xmlschema path: verify that enabling `enable_gfk` yields a single `GenericEntry` model, a `GenericRelation` on the parent, and entries persisted by the ingestor
  - pydantic/dataclass path: verify `List[Union[...]]` maps to GFK when enabled and persists entries
  - Negative controls: with GFK disabled, ensure legacy behavior unchanged
  - Performance: smoke test that migrations run significantly faster with GFK enabled for large schemas (coarse check)

- Docs:
  - New “Generic Entries (ContentTypes) mode” page describing flags, trade‑offs, and examples
  - Update xmlschema element handling doc to mention GFK option

## Files to touch (entry points & references)

- Core detection/mapping hooks:
  - `src/pydantic2django/core/bidirectional_mapper.py`
  - `src/pydantic2django/core/mapping_units.py` (has `GenericForeignKeyMappingUnit`)
  - `src/pydantic2django/core/relationships.py`
  - `src/pydantic2django/core/factories.py` (if present) or module‑specific factories
  - `src/pydantic2django/django/conversion.py` (ensure conversion respects GFK; current code path includes `GenericForeignKey` handling around line ~484)

- Generator (xmlschema):
  - `src/pydantic2django/xmlschema/generator.py`
  - `src/pydantic2django/xmlschema/factory.py`
  - `src/pydantic2django/xmlschema/ingestor.py`

- Generator (pydantic):
  - `src/pydantic2django/pydantic/generator.py`
  - `src/pydantic2django/pydantic/factory.py`
  - `src/pydantic2django/pydantic/discovery.py`

- Generator (dataclass):
  - `src/pydantic2django/dataclass/generator.py`
  - `src/pydantic2django/dataclass/factory.py`
  - `src/pydantic2django/dataclass/discovery.py`

- Docs:
  - `docs/how_it_works/xmlschema_element_and_type_handling.md` (add GFK mode reference)
  - `docs/how_it_works/` (new: `generic_entries_contenttypes.md`)

## Migration and runtime considerations

- Ensure `django.contrib.contenttypes` in generated app settings/migrations as needed
- Add DB indexes on `GenericEntry(content_type, object_id)`; optionally on `element_qname`, `type_qname`, `timestamp`/`time_value`
- Keep `attrs_json` as spillover for long‑tail attributes

## Phased TODOs

1) Core plumbing and flags [core]
   - Add `enable_gfk`, `gfk_policy`, `gfk_value_mode`, `gfk_normalize_common_attrs` to generator configs
   - Wire `GenericForeignKeyMappingUnit` as the conceptual trigger; factories record `_pending_gfk_children` instead of emitting a field
   - Implement finalize hook to emit `GenericEntry` model and inject `GenericRelation`

2) xmlschema integration [xmlschema]
   - Policy detection for substitution groups and repeating complex leaves
   - Ingestor: write `GenericEntry` rows (preserve order and attrs) when GFK is enabled
   - Tests: verify model emission and round‑trip ingestion

3) pydantic/dataclass integration [pydantic/dataclass]
   - Detect `List[Union[...]]` (and similar) and route via GFK under policy
   - Tests: ensure persistence and reverse queries via `GenericRelation`

4) Documentation and examples [docs]
   - Author `generic_entries_contenttypes.md` with examples, flags, and trade‑offs
   - Update existing docs to mention the mode as an optional strategy

5) Performance sanity [ops]
   - Add a smoke test or doc snippet comparing migration time/size with/without GFK on a large schema

## Acceptance criteria

- Enabling GFK mode reduces generated model size and migration durations on large schemas
- Reverse access via `GenericRelation` works and is filterable (e.g., by `element_qname`, `attrs_json` keys)
- Legacy behavior unchanged when GFK disabled
- Tests cover xmlschema + pydantic/dataclass paths
