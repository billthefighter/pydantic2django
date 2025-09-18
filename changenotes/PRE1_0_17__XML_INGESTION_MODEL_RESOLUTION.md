# XML ingestion prefers installed Django models over generated stand-ins

## Summary

When ingesting XML instances with `XmlInstanceIngestor`, resolution of Django models now strictly prefers installed (concrete) Django models from the app registry (`apps.get_model`) over any dynamically generated/abstract stand-ins kept in the in-memory registry. This ensures `.objects` managers are available during persistence.

## Root Cause

In some environments (e.g., MTConnect v1.7 schemas), dynamically discovered model stand-ins were present in the internal generated-models registry under the same names as installed models. If code erroneously chose the stand-ins first, persistence failed with `AttributeError: type object '<ModelName>' has no attribute 'objects'`.

## Fix

`XmlInstanceIngestor` resolves models by:

1. Attempting `apps.get_model(f"{app_label}.{model_name}")` first (preloaded and cached across types).
2. Falling back to the generated-models registry only when explicitly enabled with
   `dynamic_model_fallback=True` (default: False).

This behavior is now covered by an explicit test and guards against regressions.

## Tests

- Added `tests/xmlschema/test_ingestor_model_resolution.py::test_ingestor_prefers_installed_models_over_generated` to assert that installed models are used even when a generated stand-in with the same name is present.
- Added `tests/xmlschema/test_ingestor_model_resolution.py::test_ingestor_resolves_nested_types_to_installed_models` to verify nested complex types resolve to installed models and persist correctly.

## Risk and Compatibility

- Behavior change: default now raises a detailed error when no installed model is found. To keep prior
  fallback behavior, pass `dynamic_model_fallback=True`.

## References

- Related ingestion ordering fix: `XML_INGESTION_CHILD_FK_ORDERING.md`
