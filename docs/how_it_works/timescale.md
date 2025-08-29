## Timescale integration

This document consolidates how Pydantic2Django integrates with TimescaleDB across model bases, relationship policy, classification heuristics, and migrations/testing.

### Goals
- Treat true time‑series entities as Timescale hypertables.
- Keep container/metadata structures as regular (dimension) tables.
- Prevent fragile/illegal FKs to hypertables while preserving joinability.

### Model bases and constraints
- Hypertables inherit from `XmlTimescaleBase`, which combines the project’s base with `TimescaleModel` (from `django-timescaledb`).
- Because `django-timescaledb` drops the primary key during hypertable conversion, P2D guarantees a separate `UniqueConstraint` on `id` for all Timescale bases. This is attached both via `__init_subclass__` and a `class_prepared` signal to avoid edge cases. This allows downstream `ForeignKey` references to remain valid post‑conversion.

### Relationship policy
- Hypertable → hypertable FKs are disallowed. The generator emits a soft reference (e.g., `UUIDField(db_index=True, null=True, blank=True)`) instead.
- If a dimension would FK to a hypertable, the FK is inverted so the hypertable points to the dimension with `on_delete=SET_NULL` and appropriate indexes are added:
  - `Index(fields=['<dimension>'])`
  - If the hypertable also defines `time`, a composite: `Index(fields=['<dimension>', '-time'])`

### Classification heuristics (XML)
We classify XML complex types into roles using a point system; a score ≥ threshold (default 3) ⇒ hypertable:
- +2 if any time‑like element/attribute appears: names containing `time`, `timestamp`, `sequence`, `effectiveTime`, `sampleRate`, `date`, `datetime`.
- +2 if the type name looks like an observation/event container: `Samples`, `Events`, `Condition`, `*Changed`, `*Removed`, `*Added`, `Streams`.
- +1 if any element indicates unbounded/high‑cardinality growth (`is_list=True` or `maxOccurs="unbounded"`).
- −2 for definition/metadata‑like names: `*Definition*`, `*Definitions*`, `Constraints`, `Properties`, `Parameters`, `Header`, `Counts`, `Configuration`, `Description`, `Location`, `Limits`, `Reference`, `Relationships`.

Container demotion and leaf promotion:
- After the initial score pass, any type that lacks a direct time/date field but has descendant complex types that do contain such fields is treated as a container and demoted to a dimension.
- Those descendant types with direct time/date fields are promoted to hypertables.
- If the container itself has a direct time/date field, it remains a hypertable.

These rules tend to classify top‑level “Streams”/“Devices” containers as dimensions while promoting their leaf observation/time‑bearing children to hypertables.

### Generator flow
1. Discovery loads complex types and determines processing order.
2. We classify each type via the heuristics above.
3. During model assembly, if generated fields collide with base fields, we still assemble a bare model class so the finalize phase can inject inverted FKs and add indexes.
4. Relationship finalization ensures:
   - no hypertable→hypertable FKs (soft refs instead),
   - FK inversion from dimension→hypertable to hypertable→dimension,
   - helpful indexes on hypertables.

### Migrations and Timescale specifics
- Timescale hypertable conversion drops the PK; our unique constraint on `id` preserves FK validity.
- For integration testing, we run migrations against a local TimescaleDB container:
  - `scripts/start_timescaledb.sh` starts/ensures the container, creates the database, and enables the `timescaledb` extension.
  - `make test-timescale-integration` runs the FK/Timescale integration tests with appropriate environment variables (`DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `P2D_INTEGRATION_DB=1`).

### Tests
- Heuristics unit tests (`tests/timescale/test_heuristics_scoring.py`) cover:
  - time/date/datetime field/attribute scoring,
  - container demotion and leaf promotion,
  - containers retaining hypertable role when they have direct time fields,
  - list/unbounded growth scoring,
  - explicit overrides.
- Relationship and generator behavior (`tests/timescale/test_generator_timescale.py`) verifies FK inversion and index creation.
- End‑to‑end generation and migration:
  - Minimal FK to hypertable behavior (`tests/integration/test_timescale_fk.py`).
  - Real XSD (`MTConnectAssets_2.4.xsd`) generation → makemigrations → migrate against TimescaleDB (`tests/integration/test_assets_generate_migrate.py`).

### Edge cases and notes
- Some complex/simple content patterns emit warnings (limited support). They do not affect FK uniqueness or hypertable conversion and can be iterated on independently.
- If your schema contains special containers that truly should remain hypertables, you can force roles via overrides at classification time.
