# Fix: XmlInstanceIngestor child FK ordering (parent-first persistence)

## Summary
XmlInstanceIngestor could insert child rows for repeated complex elements (list/"child_fk" strategy) before setting the required ForeignKey to the parent. With non-nullable child→parent FKs, this caused IntegrityError during ingestion.

## Root cause
List children were created without the parent FK set and linked in a subsequent step. Databases enforce NOT NULL at insert time, so the initial INSERT failed.

## Resolution
For list children, include the parent FK in the initial create() call:
- `_build_instance_from_element(...)` now accepts `parent_link_field`.
- The caller passes the FK name (lowercased parent class) so the FK is present in `field_values` before `objects.create(...)`.

Affected file: `src/pydantic2django/xmlschema/ingestor.py`.

## Tests
- Added `tests/xmlschema/test_ingestor_db_fk_order.py` to reproduce the IntegrityError and verify the fix.
- Added `tests/migrations/0009_alter_itemtype_parenttype_not_null.py` to enforce a non-nullable FK for the test.
- Updated `tests/xmlschema/test_ingestor.py` to reuse shared test models and avoid duplicate registration.
- Full suite OK locally.

## Backward compatibility
- No public API changes.
- `save=False` behavior unchanged; links are still set on in-memory objects.

## Notes for MTConnect users
- Resolves IntegrityErrors seen on MTConnect v1.7 Streams/Devices ingestion where list children were inserted before parent linkage.

## Related docs
- `docs/how_it_works/readme_ingestion.md`
- `docs/how_it_works/readme_relationships.md`
