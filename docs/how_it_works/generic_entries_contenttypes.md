## Generic Entries mode (Django ContentTypes)

This optional mode stores polymorphic/repeating nested structures as rows in a single `GenericEntry` model, attached to owning parents via `GenericRelation`. It reduces generated model count and migration size while remaining queryable.

- Defaults (new projects):
  - `enable_gfk=True`
  - `gfk_policy="threshold_by_children"`
  - `gfk_threshold_children=8`
  - `gfk_value_mode="typed_columns"`
  - `gfk_normalize_common_attrs=False`

- Flags (tunable via generators):
  - `enable_gfk: bool`
  - `gfk_policy: "substitution_only" | "repeating_only" | "all_nested" | "threshold_by_children"`
  - `gfk_threshold_children: int` (only for `threshold_by_children`)
  - `gfk_value_mode: "json_only" | "typed_columns"`
  - `gfk_normalize_common_attrs: bool` (reserved)

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

- Example query:
```python
samples.entries.filter(element_qname="Angle", time_value__gte=..., attrs_json__subType="ACTUAL")
```

See also: `docs/plans/gfk_generic_entries_plan.md` and `docs/how_it_works/xmlschema_element_and_type_handling.md`.
