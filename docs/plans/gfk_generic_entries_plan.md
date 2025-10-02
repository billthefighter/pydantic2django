# Provider‑agnostic GenericForeignKey (GFK) plan

## What

Introduce a generator‑controlled, provider‑agnostic mode to model polymorphic/repeating nested XML (and multi‑type pydantic/dataclass fields) using Django’s contenttypes (GenericForeignKey) instead of emitting thousands of concrete leaf models or large numbers of JSONFields. This keeps the model surface small, migrations fast, and data queryable across all ingests.

## Why

- Current xmlschema output (e.g., MTConnect Streams) can generate 90k+ line models and lengthy migrations due to substitution groups and deeply nested, repeating leaves.
- Many “leaf” combinations require either per‑leaf classes or JSON fallback; both scale poorly and are hard to query.
- GenericForeignKey lets us store polymorphic entries in a single normalized table with a reverse link to the owning parent, drastically reducing generated code while remaining queryable.
- Approach must be provider‑agnostic to support xmlschema, pydantic, and dataclass paths without domain‑specific types.

Reference: Django contenttypes [The contenttypes framework](https://docs.djangoproject.com/en/5.2/ref/contrib/contenttypes/).

## Current status (2025-10-01)

- XmlSchema path: initial GFK plumbing implemented.
  - Flags supported on `XmlSchemaDjangoModelGenerator`: `enable_gfk`, `gfk_policy`, `gfk_threshold_children`, `gfk_value_mode`, `gfk_normalize_common_attrs`.
  - Detection: repeating nested complex elements (including wrapper-like containers) can be routed to GFK under `gfk_policy` ∈ {`repeating_only`, `all_nested`, `threshold_by_children`}.
  - Generation finalize: when any parent has GFK-eligible children, a single `GenericEntry` model is emitted for the app and the parent gets `entries = GenericRelation('GenericEntry', related_query_name='entries')`.
  - Ingestion: when a parent exposes `entries` and a repeating nested child is encountered, concrete child instances are skipped and `GenericEntry` rows are persisted with `element_qname`, `type_qname`, `attrs_json` (including attributes and text value), and `order_index`.
- Tests: added a basic xmlschema generation test to assert `GenericEntry` emission and parent `GenericRelation`. Full suite passing with GFK enabled in targeted tests.

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

## XML schema handling – detailed design

This section specifies precisely how XML Schema (XSD) constructs are mapped when `enable_gfk=True`.

### Detection and policy evaluation (discovery/factory)
- Input models come from xmlschema discovery: `XmlSchemaComplexType`, their `elements`, `attributes`, and metadata.
- For each complex type’s content model (sequence/choice/all), inspect each child element E:
  - Repeating leaf: `maxOccurs>1` or `unbounded` and `E` resolves to a simple type or a complex type that functions as a leaf → candidate for GFK when `gfk_policy` ∈ {`repeating_only`, `all_nested`, `threshold_by_children`}.
  - Substitution groups: if E is a head, expand members (already implemented); if E is a member under a wrapper, treat these polymorphic variants as candidates when `gfk_policy` ∈ {`substitution_only`, `all_nested`, `threshold_by_children`}.
  - Threshold policy: for a wrapper/container (TitleCase element name or `*WrapperType`), if the number of distinct child types ≥ `gfk_threshold_children`, use GFK for those children.
- Attribute groups: already resolved into attributes; when GFK is used, attributes are copied into `attrs_json` (or normalized to typed columns when configured).

### Parent (owner) selection
- Owner is typically the nearest non‑leaf container on the path:
  - If a wrapper node (e.g., `Samples`, `Events`, `Condition`) exists, prefer the wrapper as owner.
  - Otherwise, use the next higher parent such as `ComponentStreamType` (or analogous container) to avoid over‑fragmenting owners.
- Implement as: choose the current complex type as parent unless the child is inside a recognized wrapper pattern; in that case, use the wrapper type (if generated) as the owner.

### Mapping from XSD to GenericEntry
- Entry fields:
  - `element_qname`: the element name for the leaf, local name preferred (include prefix/namespace if required by config).
  - `type_qname`: the resolved target type (simple or complex), using local name (store namespace if available).
  - `attrs_json`: all attributes on the leaf element (post attributeGroup expansion). If `gfk_normalize_common_attrs=True`, pull out known keys (e.g., timestamp‑like, subtype‑like, dataItemId‑like) into typed columns.
  - Value mapping (`gfk_value_mode`):
    - `json_only`: put element text (and possibly child simple content) into `attrs_json["value"]`.
    - `typed_columns`: detect and map
      - Element text to `text_value`
      - Numeric text to `num_value` where parsing is unambiguous
      - Date/time text to `time_value` where parsing is unambiguous
    - Any remaining additional content goes to `attrs_json`.
  - `order_index`: increment as entries are encountered under the owner to preserve original order.
  - `path_hint` (optional): dotted path or lightweight xpath within the owner to aid debugging.

### Identity constraints (xs:key / xs:keyref)
- When GFK is chosen, strong FK injection to leaf models isn’t applicable. We handle keys/refs as follows:
  - Represent keyref targets via soft references: copy key values into `attrs_json` and/or into normalized columns (if configured).
  - Optionally add indexes on these soft reference columns for faster lookup.
  - Preserve original key/keyref metadata on the parent context for audit/validation (future validation hook could enforce referential checks on ingest).

### Mixed content and nested simple content
- Mixed content (`mixed=true`) or interleaved text: store raw text in `text_value` (typed mode) or `attrs_json["text"]` (json mode). Child non‑leaf content under a GFK leaf is not further decomposed unless configured; it remains in `attrs_json`.

### Timescale integration
- If Timescale classification marks parent as hypertable, keep GFK on entries while parent remains Timescale‑backed. Suggested indexes:
  - `(owner, -time_value)` composite (owner via `(content_type, object_id)`), if `time_value` is present.
  - Basic `(content_type, object_id)` always.

### Indexing
- Always add `Index(fields=["content_type", "object_id"])` on `GenericEntry`.
- Optional indexes when `typed_columns` enabled:
  - `Index(fields=["element_qname"])`
  - `Index(fields=["type_qname"])`
  - `Index(fields=["time_value"])` and composite `(content_type, object_id, -time_value)` when enabled.

### Generation and migrations
- The generator emits:
  - A single `GenericEntry` model per app when GFK is used anywhere in that app.
  - A `GenericRelation` field on each parent owner model.
  - Ensure `django.contrib.contenttypes` is present in app settings for migration contexts.

### Ingestor algorithm (xml instance → models)
1. Traverse the XML instance using schema guidance (existing ingestor).
2. For each candidate child matching GFK policy:
   - Resolve owner instance (parent/wrapper chain context).
   - Create `GenericEntry` with `content_object=owner`, fill `element_qname`, `type_qname`.
   - Copy attributes; map value to `text_value`/`num_value`/`time_value` or `attrs_json` per `gfk_value_mode`.
   - Assign `order_index` monotonically for that owner.
3. For non‑GFK children, keep legacy behavior (FKs/JSON fallback).

### Overrides and configuration
- Per‑namespace or per‑element overrides: provide a mapping to force or disable GFK for certain element names or paths.
- Per‑run flags: enable/disable GFK globally or by policy (as defined above).
- Backward compatibility: with `enable_gfk=False`, all legacy generation and ingestion paths remain unchanged.

### Example (MTConnect‑style pattern)
- `ComponentStreamType` → wrapper `SamplesType` → many observation leaves (`Angle`, `Position`, …)
  - With `gfk_policy=substitution_only` or `repeating_only`, observation leaves become `GenericEntry` rows with owner=`SamplesType` (preferred) or `ComponentStreamType` (if wrapper is not generated).
  - `attrs_json` contains `dataItemId`, `subType`; `timestamp` goes to `time_value` (typed mode) or `attrs_json["timestamp"]`.
  - Query pattern:
    - `samples.entries.filter(element_qname="Angle", time_value__gte=..., attrs_json__subType="ACTUAL")`

# CURRENT STATUS:
Core
GFK flags added to BaseStaticGenerator with defaults: enable_gfk=True, gfk_policy="threshold_by_children", gfk_threshold_children=8, gfk_value_mode="typed_columns".
Conversion pipeline now records pending_gfk_children and injects GenericRelation('entries') on parents.
GenericEntry model emitted with appropriate indexes; ingestor maps text→typed columns.
xmlschema
GFK precedence moved before JSON fallback; threshold_by_children implemented; nested_relationship_strategy defaulted to 'fk'; per-element overrides supported.
Wrapper owners marked so their nested children route to GFK (structural, not name-based).
Tests verify GenericEntry emission, relation injection, typed columns/indexes, and suppression of JSON placeholders for repeating nested.
pydantic/dataclass
List[Union[...]] routed through GFK; tests added.
Docs
Updated to state GFK is default with recommended defaults.
Reality check (Streams)
Despite passing unit tests, MTConnect Streams still generates very large models with many observation “entry-type” JSONFields alongside GFK.
Why: current structural detection doesn’t yet capture the full MTConnect pattern:
Substitution-group expansions under wrappers (Samples/Events/Condition) and derived Event/DataSet types still travel code paths that emit JSON placeholders and generate many concrete observation models.
included_model_names/discovery still admits many observation complex types; GFK detection doesn’t demote those to “stored in GenericEntry” comprehensively.
Our tests are too simple (single repeating nested/wrapper cases) and don’t replicate substitution groups and the breadth of observation variants in Streams.
Next steps (to make it truly compact)
1) Strengthen structural detection for MTConnect-like polymorphism
Treat substitution-group members under recognized wrapper owners as GFK candidates (policy "substitution_only" as an option, and include in threshold_by_children computation).
When a parent is marked a GFK owner, route all eligible nested complex leaves (including substitution-group variants) to GFK and do not emit placeholders or generate their concrete models.
Gate discovery/inclusion: if a leaf path is GFK-routed, don’t include that leaf’s concrete type in included_model_names.
2) Add representative tests derived from Streams constructs
Build a minimal fixture capturing the pattern:
A wrapper type (e.g., SamplesType) with a head element resolved via substitution group to many member types, including a repeated “dataset-like” element.
Assertions:
GenericEntry emitted; wrapper parent has GenericRelation('entries').
No JSONField placeholders on wrapper/observation models even when element names don’t include “entry”.
Concrete observation/member types are not generated when GFK is applied for that path.
Add a smoke test that counts total models and fields for the fixture and asserts a bounded maximum (proxy for “compactness”).
3) Instrumentation/diagnostics
Log, per parent model, how many children were routed to GFK vs emitted as fields; optionally expose a summary (to make gaps obvious in CI).
4) Optional policy control
Introduce gfk_policy="substitution_only" and test it on the new fixture; confirm it collapses member variants to GenericEntry rows.
How to evolve the tests to be representative
Include substitution groups: head element plus multiple member declarations resolved via discovery.
Include a wrapper/container owner that aggregates many heterogeneous members (mirroring Samples/Events/Condition).
Assert:
No JSON placeholders on the wrapper or on derived observation classes (even with non-“entry” names).
Member complex types are not present in all when routed via GFK.
GenericEntry count/index definitions exist; reverse queries via entries are available.
These changes will align the generator with the Streams reality by structurally routing polymorphic observation leaves into GenericEntry and eliminating dual emission.
