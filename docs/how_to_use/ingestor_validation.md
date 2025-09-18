## Using the XML Ingestor: Strict Mode, Contract Checks, and Timescale

This guide shows how to enable strict ingestion, run contract validation, and handle Timescale-specific requirements.

### Enable strict ingestion

```python
from pydantic2django.xmlschema import get_shared_ingestor

ingestor = get_shared_ingestor(
    schema_files=["/path/to/schema.xsd"],
    app_label="myapp",
    dynamic_model_fallback=False,  # prefer installed models in production
    strict=True,                   # fail fast on drift
)
```

Strict mode will:

- Error on unexpected XML child elements or unmapped attributes
- Enforce contract checks (if you call `validate_models(strict=True)`)
- Require `time` only for Timescale models

### Validate models before ingesting

```python
issues = ingestor.validate_models(strict=False)
if issues:
    # Inspect and fix issues, or enforce:
    ingestor.validate_models(strict=True)  # raises SchemaSyncError on drift
```

Typical remediation: “Schema and static model are out of sync; verify schema and/or regenerate static Django models and re-migrate.”

### Timescale-only `time` requirement

The ingestor always tries to remap common timestamp aliases to `time` when the model declares a `time` field. However, missing `time` becomes an error only if the model inherits a Timescale base:

- Timescale model, missing timestamp → raises `TimeseriesTimestampMissingError`
- Non-Timescale model, missing timestamp → no error (still attempts alias remapping)

### Recommended settings for production

- Use `dynamic_model_fallback=False` to ensure only concrete installed models are used
- Use `strict=True` to catch drift early
- Run `validate_models(strict=True)` during startup or CI
- Set a modest ingestor cache TTL or clear cache on deployments if schemas change
