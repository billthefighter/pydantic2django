## Request: Diagnostics to reproduce XmlInstanceIngestor using abstract stand-ins instead of installed models

### Context
In some environments, ingestion fails because the root/nested models appear to be resolved to abstract/discovered stand-ins lacking `.objects` instead of the installed Django models. We need a concrete, schema-parsing run and targeted diagnostics to reproduce and fix this deterministically.

### 1) Verify schema path visibility (same process that fails)
```python
from pathlib import Path

schema_file = "/app/mtconnect_stream/streamdef/schemas/MTConnectStreams_1.7.xsd"  # replace per endpoint
p = Path(schema_file)
print("exists=", p.exists(), "resolve=", p.resolve())

# Also confirm the exact path passed into get_shared_ingestor in the worker
print("schema_file_used=", schema_file)
```

If `exists=False`, ingestion discovery will find 0 complex types and cannot reach model resolution paths. Please ensure the worker uses a readable, absolute path.

### 2) Discovery success snapshot (must be from a run where schema exists)
```python
from pydantic2django.xmlschema.ingestor import get_shared_ingestor

app_label = "mtconnect_streams_v17_streams"  # replace per endpoint
schema_file = "/abs/path/to/MTConnectStreams_1.7.xsd"

ing = get_shared_ingestor(schema_files=[schema_file], app_label=app_label)
print("schemas_count=", len(ing._schemas))

root_local_name = "MTConnectStreams"  # local name of your XML root element
ct = ing._resolve_root_complex_type(root_local_name)
print("resolved_complex_type=", ct and ct.name)

mc = ing._get_model_for_complex_type(ct) if ct else None
print("model_cls=", mc, "module=", getattr(mc, "__module__", None), "has_objects=", hasattr(mc, "objects") if mc else None)
```

Expected: `schemas_count > 0`, `resolved_complex_type` is your root type name (e.g., `MTConnectStreamsType`).

### 3) App registry vs generated fallback for the same type
```python
from django.apps import apps
import pydantic2django.xmlschema as xm

name = ct.name  # e.g., "MTConnectStreamsType"
installed = apps.get_model(app_label, name)
print("apps.get_model=", installed, "has_objects=", hasattr(installed, "objects"))

generated = xm._GENERATED_MODELS_REGISTRY.get(app_label, {}).get(name)
print("generated_model=", generated, "module=", getattr(generated, "__module__", None), "has_objects=", hasattr(generated, "objects") if generated else None)
```

If `_get_model_for_complex_type(ct)` returns a class without `.objects` while `apps.get_model` returns a concrete class, include these prints.

### 4) App registry snapshot
```python
from django.apps import apps

print("apps_ready=", apps.ready)
print("labels=", [c.label for c in apps.get_app_configs()])

cfg = apps.get_app_config(app_label)
print("models_in_label=", [m.__name__ for m in cfg.get_models()])

M = apps.get_model(app_label, name)
print("get_model=", M, "meta_label=", getattr(getattr(M, "_meta", None), "app_label", None), "has_objects=", hasattr(M, "objects"))

# If your concrete model sets Meta.app_label, print it (optional)
print("class Meta.app_label=", getattr(getattr(M, "_meta", None), "app_label", None))
```

### 5) XSD and XML specifics (for root resolution)
- The XML root element’s QName (e.g., `{ns}MTConnectStreams`).
- The corresponding XSD declaration for the root (either an `<xs:element name=... type=...>` or the `<xs:complexType>` name).
- If possible, attach the exact XSD file for the root and a minimal XML that reproduces the failure.

### 6) Environment summary
- Python, Django, database backend, `pydantic2django` version.
- Installed app labels including the MTConnect apps.
- The exact `app_label` and `schema_file` passed to `get_shared_ingestor`.

### 7) If installed model is available but not used, capture exception
If `_get_model_for_complex_type` is selecting a stand-in, please log any exception raised by `apps.get_model(f"{app_label}.{ct.name}")` in that process (e.g., via temporary logging or a small wrapper) so we can see why the installed model wasn’t selected.

### What we’ll change once reproduced
- Prefer installed models globally: Ensure that for every discovered type (root and nested), the ingestor resolves classes via `apps.get_model` first and only falls back to generated stand-ins if no installed model exists. Cache these decisions for consistency across a run.
- Improve diagnostics: If schema paths are missing, raise a clear error with normalized paths and existence checks to avoid silent 0-type discovery.

### Decision & justification: gate dynamic models behind an explicit flag

To avoid implicit, surprising behavior in production ingestion, dynamic model fallback is now gated
behind an explicit flag:

- `dynamic_model_fallback=True` (default, backward compatible):
  - Allows ephemeral workflows (dry-runs, CI, tooling) without installed models by using generated stand-ins.
  - Still prefers installed models first and caches them for consistency across nested types.

- `dynamic_model_fallback=False` (recommended for production ingestion):
  - Guarantees only installed Django models are used.
  - If a discovered complex type cannot be resolved to an installed model, ingestion raises a
    `ModelResolutionError` including `app_label` and `model_name`, with guidance.

This change removes the implicit fallback path that led to confusing edge cases (e.g., attempting
to persist against abstract/discovered types without managers) and makes ingestion behavior explicit
and predictable in production environments.

### Thank you
With the above outputs from a schema-parsing run (where the path exists), we can write a failing test mirroring your setup, implement the minimal fix, and verify across the full suite to prevent regressions.
