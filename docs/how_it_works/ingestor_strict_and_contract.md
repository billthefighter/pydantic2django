## Ingestor Strict Mode and Model/Schema Contract

This page explains the ingestion contract checks and strict mode behavior added to the XML ingestor.

### What strict mode enforces

When `strict=True` on `XmlInstanceIngestor`:

- Unexpected child elements in the XML that are not declared in the complex type cause a hard failure.
- Unmapped XML attributes that don’t correspond to declared attributes cause a hard failure.
- Contract validation (`validate_models(strict=True)`) raises on drift between discovered XML types and installed Django models.
- Timescale-only requirement: models that inherit a Timescale base must have a non-null `time` field at ingestion time; otherwise `TimeseriesTimestampMissingError` is raised. Non-Timescale models are not forced to provide `time`.

All strict failures raise `SchemaSyncError` with a clear remediation message:

“Schema and static model are out of sync; verify schema and/or regenerate static Django models and re-migrate.”

### Contract validation

Use `validate_models()` to verify alignment between schemas and installed models:

- The ingestor computes expected field names for each complex type:
  - all declared attributes (converted to snake_case)
  - all simple elements
  - for single nested complex elements, the parent’s field for the child (snake_case of the element name)
- It then compares these with the installed model’s `model._meta.fields`.
- For Timescale-based models, it also ensures a `time` field is declared.

Behavior:

- `validate_models(strict=False)` returns a list of issues (if any) without raising.
- `validate_models(strict=True)` raises `SchemaSyncError` if issues are found.

### Why enforce Timescale only

The `time` field is required for hypertables. Enforcing it only when the model inherits a Timescale base avoids over-constraining standard models that may have optional timestamps. The ingestor still remaps common aliases (e.g., `creationTime`, `timeStamp`, etc.) into `time` for convenience, but only Timescale models must provide it.

### Failure messages and remediation

On failures, you’ll see a compact summary including the affected model and missing fields. Recommended steps:

- Verify your XSD changes and regenerate static Django models if needed.
- Re-run migrations to apply schema changes.
- For Timescale models, ensure a non-null `time` value is present in the XML (or adjust your mapping).
