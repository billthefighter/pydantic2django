# XML Schema wrapper children incorrectly attached as parent-side FKs

## Summary
In XML Schema generation, wrapper children (e.g., `Samples`, `Events`) should hold child-side ForeignKeys pointing back to their parent container (e.g., `ComponentStreamType`) when `list_relationship_style="child_fk"`. Currently, the parent gets FKs to the wrappers instead.

## Impact
- Generated models violate intended relationship direction for wrapper container patterns (e.g., MTConnect-like schemas).
- Ingestors relying on child-side FKs for ordering and deletion semantics break or require workarounds.

## Repro
Schema (abridged): `DeviceStreamType` → `ComponentStreamType` contains `SamplesType` and `EventsType`.

Expected:
- `SamplesType.componentstreamtype = models.ForeignKey(ComponentStreamType, ...)`
- `EventsType.componentstreamtype = models.ForeignKey(ComponentStreamType, ...)`

Actual:
- `ComponentStreamType.samples = models.ForeignKey(SamplesType, ...)`
- `ComponentStreamType.events = models.ForeignKey(EventsType, ...)`

## Root cause
`xmlschema/generator.py` builds `carriers_by_name` like this:

```python
carriers_by_name = {
    getattr(c.source_model, "__name__", ""): c for c in self.carriers if c.django_model is not None
}
```

`XmlSchemaComplexType` exposes `.name`, not `__name__`. Keys become empty strings, so the factory’s `finalize_relationships(...)` cannot find the involved models, and the FK flip never happens.

## Fix
Key by XML name when present, fallback to `__name__`:

```python
carriers_by_name = {
    (getattr(c.source_model, "name", None) or getattr(c.source_model, "__name__", "")): c
    for c in self.carriers
    if c.django_model is not None
}
```

## Validation
- Add tests for wrapper-container pattern asserting child-side FKs and absence of parent-side FKs.
- Run existing nested and list tests.

## Notes
- Defaults already prefer child-side FKs; this is a keying bug preventing the relationship finalizer from executing correctly.
