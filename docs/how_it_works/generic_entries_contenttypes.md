## Generic Entries mode (Django ContentTypes)

This optional mode stores polymorphic/repeating nested structures as rows in a single `GenericEntry` model, attached to owning parents via `GenericRelation`. It reduces generated model count and migration size while remaining queryable.

- Defaults (new projects):
  - `enable_gfk=True`
  - `gfk_policy="all_nested"` (favor minimal, GFK‑heavy output)
  - `gfk_threshold_children=4` (only used when `gfk_policy='threshold_by_children'`)
  - `gfk_value_mode="typed_columns"`
  - `gfk_normalize_common_attrs=False`

- Flags (tunable via generators):
  - `enable_gfk: bool`
  - `gfk_policy: "substitution_only" | "repeating_only" | "all_nested" | "threshold_by_children"`
  - `gfk_threshold_children: int` (only for `threshold_by_children`)
  - `gfk_value_mode: "json_only" | "typed_columns"`
  - `gfk_normalize_common_attrs: bool` (reserved)

### Flag reference

- `enable_gfk`
  - Enables Generic Entries mode globally for the generator. If `False`, legacy behavior is preserved and no `GenericEntry` is emitted.

- `gfk_policy`
  - Controls which nested elements are routed to `GenericEntry`.
  - Values:
    - `substitution_only`: Treat elements under wrapper-like owners that participate in substitution groups as GFK entries. Use this when many polymorphic variants share a base type (e.g., MTConnect observations).
    - `repeating_only`: Route repeating complex leaves (`maxOccurs>1` or `unbounded`) through GFK.
    - `all_nested`: Route all eligible nested complex elements through GFK (including single-nested). Useful to aggressively minimize model count.
    - `threshold_by_children`: For wrapper-like containers, route to GFK when the number of distinct child complex types in the wrapper ≥ `gfk_threshold_children`.

- `gfk_threshold_children`
  - Only used with `threshold_by_children`. An integer bound determining when a wrapper is considered polymorphic enough to justify GFK.
  - Example: `gfk_threshold_children=8` means wrappers with ≥8 distinct child complex types are routed via GFK.

- `gfk_value_mode`
  - Controls how element text is stored on `GenericEntry`:
    - `json_only`: Store the element text under `attrs_json["value"]`.
    - `typed_columns`: Attempt to extract text into typed columns: `text_value`, `num_value`, and `time_value` when unambiguous; remaining attributes go to `attrs_json`.

- `gfk_normalize_common_attrs`
  - Reserved for promoting commonly-used attributes (e.g., timestamp, subtype) to normalized columns when `gfk_value_mode="typed_columns"`. Default `False`.

- `gfk_overrides: dict[str, bool]`
  - Optional per-element overrides by local element name. `True` forces GFK for that element; `False` disables it even if policy would apply.


- Behavior:
  - When policy matches, concrete child models are not generated; parents get `entries = GenericRelation('GenericEntry')`.
  - In ingestion, each matching XML element becomes a `GenericEntry` with:
    - `element_qname`, `type_qname`, `attrs_json`, `order_index`, `path_hint`
    - Optional typed value columns: `text_value`, `num_value`, `time_value` (when `gfk_value_mode='typed_columns'`)
  - Indexes are added on `(content_type, object_id)` and, in typed mode, on `element_qname`, `type_qname`, `time_value`, and `(content_type, object_id, -time_value)`.

- Policies:
  - `repeating_only`: route repeating complex leaves through `GenericEntry`.
  - `all_nested`: route all eligible nested complex elements (including single nested) through `GenericEntry`.
  - `threshold_by_children`: for wrapper-like containers, use `GenericEntry` when the number of distinct child complex types ≥ `gfk_threshold_children`.
  - `substitution_only`: for wrapper-like containers, collapse substitution-group members to `GenericEntry` rows regardless of the number of distinct child types.

- Example query:
```python
samples.entries.filter(element_qname="Angle", time_value__gte=..., attrs_json__subType="ACTUAL")
```

### Notes and interactions

- Wrapper detection (generic): Prefer structural checks over names. Owners are selected when:
  - Policy is `all_nested` (all nested routed), or
  - Policy is `threshold_by_children` and the wrapper’s set of distinct child complex types ≥ `gfk_threshold_children`, or
  - Policy is `substitution_only` and the wrapper contains substitution-group members.
  - Name hints (TitleCase element, `*WrapperType`) can assist but are not required.
- Discovery gating: When a path is routed via GFK, the corresponding concrete child complex types may be excluded from the generated model set to keep the surface area small.
- Ingestion: When parents expose `entries`, matching XML child elements are persisted as `GenericEntry` rows, preserving order and attributes. Concrete child instances are not created.

### Either/Or guarantee

- When a model is selected as a GFK owner, its child element fields are suppressed entirely (no JSON placeholders or concrete child relations). If configuration flags would cause both entries and JSON/relations on the same owner, generation fails fast with a clear error.

See also: `docs/plans/gfk_generic_entries_plan.md` and `docs/how_it_works/xmlschema_element_and_type_handling.md`.
