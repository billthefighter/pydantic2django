# Timescale ingestion: map XML timestamp aliases into canonical `time`

## Summary

Timescale-enabled models (those inheriting a Timescale base that requires a non-null `time` column) now automatically remap common XML timestamp attributes into the canonical `time` field during ingestion. If no acceptable alias is present, ingestion raises a clear, early error instead of failing later with a database NOT NULL violation.

## Motivation

- Hypertables require a non-null `time` column.
- Real-world XML often uses variant names like `creationTime`, `timestamp`, `effectiveTime`, or `dateTime`.
- Previously, missing `time` produced late failures at save time; this change validates and fixes upfront.

## What changed

- Introduced `pydantic2django.core.utils.timescale` with:
  - `TIMESERIES_TIME_ALIASES`: accepted aliases (normalized) — `creation_time`, `timestamp`, `time_stamp`, `effective_time`, `date_time`, `datetime`.
  - `map_time_alias_into_time(data)` and `ensure_time_or_raise(...)` helpers.
  - `TimeseriesTimestampMissingError` with actionable message.
- `Xml2DjangoBaseClass.from_xml_dict` path now:
  - Normalizes XML names to Django-safe identifiers via `sanitize_field_identifier(...)`.
  - If the model declares `time` and the incoming data lacks it, remaps the first present alias into `time`.
  - If still missing, raises `TimeseriesTimestampMissingError` listing attempted aliases.

## Name normalization

Attribute names are normalized to safe Django identifiers: CamelCase is converted to snake_case and punctuation/namespace separators (`-`, `:`, `.`, spaces) become `_`. Examples:
- `creationTime` → `creation_time`
- `time-stamp` → `time_stamp`
- `xlink:timestamp` → `xlink_timestamp`

This is handled by `pydantic2django.core.utils.naming.sanitize_field_identifier` and ensures alias detection works across common variants.

## Affected code

- `src/pydantic2django/django/models.py` — ingestion path remaps aliases and raises on missing `time`.
- `src/pydantic2django/core/utils/timescale.py` — new utilities and error type.
- Docs: `docs/how_it_works/timescale.md`, `docs/how_it_works/xmlschema_notes.md` updated with behavior details.
- Tests: `tests/xmlschema/test_ingestor_timescale_time_mapping.py` covers alias mapping and error cases.

## Compatibility

- Non-Timescale models: no behavior change.
- Timescale models: earlier, clearer failures when `time` is absent; otherwise transparent alias remapping.
- When Timescale is disabled at generation time (see enable flag changenote), models won’t have a `time` field and this mapping is not applied.

## Example

Given MTConnect Streams `Header/creationTime`:
- Incoming XML becomes `{ "creation_time": "2020-01-01T00:00:00Z", ... }` after normalization.
- Ingestion remaps to `{ "time": "2020-01-01T00:00:00Z", ... }` for a Timescale-enabled model.
