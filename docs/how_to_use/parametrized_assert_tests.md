# Parametrized Single-Assertion Tests for End-to-End Load/Process/Persist

This guide describes a robust testing pattern using parameterized, single-assert test cases (via `pytest.mark.parametrize`) for any end-to-end workflow that loads a file/resource, processes it, and (optionally) persists results.

It applies equally to XML, JSON, CSV, logs, binary artifacts, or remote resources—swap in the appropriate loader/processor.

## Why this pattern?
- Clarity: Each assertion runs as an individual test case with a clear ID, so failures are obvious.
- Faster diagnosis: End-to-end tests often fail at specific steps (load, transform, persist); parameterization pinpoints where.
- Reuse: One setup fixture (arrange) powers many small checks (assert), keeping tests DRY.
- Stability: Centralized setup/teardown avoids brittle per-test scaffolding.

## Core principles
- One assertion per parameterized case.
- Encapsulate preparation and the end-to-end action in a fixture that returns a small context object.
- Keep the context minimal (the result object, a couple of models/helpers) to reduce coupling.
- Name each check by intent (e.g., `loaded_ok`, `has_required_fields`, `persisted_rows`, `links_resolved`).

## Generic template (no framework assumptions)
Use this when your workflow only needs to read/process data and verify results in memory.

```python
from pathlib import Path
import pytest

# Define how to load/process your input (replace with your own)
def load_any(path: str) -> dict:
    # Example loader stub; replace with JSON/CSV/XML/etc.
    # Return a rich result object or dataclass as appropriate.
    return {"path": path, "content": Path(path).read_text(encoding="utf-8")}


@pytest.fixture(scope="function")
def e2e_ctx(tmp_path: Path):
    source_path = "<ABSOLUTE_OR_RELATIVE_SOURCE_PATH>"  # e.g., tests/data/sample.json

    # Arrange: run the real load/process function
    result = load_any(source_path)

    # Optionally write derived files/artifacts under tmp_path if your flow produces outputs
    # (e.g., parsed CSVs, intermediate JSONs)

    # Return only what checks need
    return {"result": result, "source_path": source_path}


@pytest.mark.parametrize(
    "check",
    [
        "loaded_ok",
        "has_content",
        "has_expected_marker",
    ],
)
def test_e2e_checks(e2e_ctx, check: str):
    result = e2e_ctx["result"]

    if check == "loaded_ok":
        assert result is not None
        return

    if check == "has_content":
        assert isinstance(result.get("content"), str) and len(result["content"]) > 0
        return

    if check == "has_expected_marker":
        assert "SOME_MARKER" in result["content"]  # tailor to your data
        return

    raise AssertionError(f"Unknown check: {check}")
```

## Django-backed variant (optional persistence)
If your workflow persists to a database (e.g., Django models), add a setup block that ensures prerequisite tables exist and then perform the save step. This keeps the end-to-end test deterministic in in-memory databases.

Key ideas:
- Ensure system/app prerequisite tables exist when using an in-memory DB (example: `django_content_type`, `auth_permission`).
- Create tables for your installed models via `django_apps.get_model` (avoid creating tables for transient/abstract classes).
- Perform the load/process, then save to the DB; return persisted instances/models in the context.

```python
import importlib.util
import sys
from pathlib import Path

import pytest
from django.apps import apps as django_apps
from django.db import connection

# Domain-specific: your generator/ingestor/persistor
# from yourpackage import run_generation, load_and_persist


def _ensure_tables(models_to_ensure: list[type]) -> None:
    with connection.schema_editor() as se:
        existing = set(connection.introspection.table_names())
        for model in models_to_ensure:
            if model._meta.db_table not in existing:
                se.create_model(model)


@pytest.fixture(scope="function")
@pytest.mark.django_db(transaction=True)
def e2e_ctx(tmp_path: Path):
    app_label = "tests"  # the installed app label that holds your models
    source_path = "<ABSOLUTE_OR_RELATIVE_SOURCE_PATH>"

    # 1) (Optional) Generate/prepare domain models and dynamically import a file that defines them
    # generated_file = run_generation(config=..., output_path=tmp_path / "models.py")
    # spec = importlib.util.spec_from_file_location("tests_generated_models", str(generated_file))
    # module = importlib.util.module_from_spec(spec)
    # sys.modules["tests_generated_models"] = module
    # assert spec and spec.loader
    # spec.loader.exec_module(module)  # type: ignore

    # 2) Ensure framework/system tables (example for Django contenttypes/auth)
    try:
        from django.contrib.contenttypes.models import ContentType
        from django.contrib.auth.models import Permission
        _ensure_tables([ContentType, Permission])
    except Exception:
        # Not required for all projects; safe to ignore if not installed
        pass

    # 3) Ensure your installed model tables exist (example)
    # MyModel = django_apps.get_model(app_label, "MyModel")
    # RelatedModel = django_apps.get_model(app_label, "RelatedModel")
    # _ensure_tables([m for m in (MyModel, RelatedModel) if m is not None])

    # 4) Execute end-to-end load and persist
    # root = load_and_persist(source_path, save=True)
    root = object()  # replace with your persisted root/domain instance

    return {
        "root": root,
        # include key models if your checks need them
        # "MyModel": MyModel,
    }


@pytest.mark.parametrize(
    "check",
    [
        "root_saved",
        # add more checks like: has_children, has_links, counts_match, etc.
    ],
)
@pytest.mark.django_db(transaction=True)
def test_e2e_checks(e2e_ctx, check: str):
    root = e2e_ctx["root"]

    if check == "root_saved":
        # tailor to your model/ORM of choice; for Django, check pk is not None
        assert getattr(root, "pk", None) is not None
        return

    raise AssertionError(f"Unknown check: {check}")
```

## Tips for adapting the pattern
- Replace `load_any` with your real loader (e.g., JSON, CSV, parquet, XML, image, or HTTP fetcher).
- If your flow has hooks (pre/post), keep them inside the fixture—tests remain simple.
- Parameter names should read like behavior, not implementation details.
- When a check fails frequently, promote logging in the fixture to surface context quickly.
- If you have multiple end-to-end flows (e.g., different file types), make the fixture accept a `source_path`/`load_fn` param and `pytest.mark.parametrize` those too.

## Example mappings
- JSON file → `json.load(open(path))` → validate required keys/shape → persist to `MyModel`.
- CSV file → `pandas.read_csv(path)` → normalize columns → bulk create rows.
- Binary artifact → custom parser → checksum/metadata → persist artifact + derived rows.
- Remote resource → `httpx.get(url)` → parse → upsert into tables.

## When to use the framework variant
- Use the framework (e.g., Django) variant when persistence is part of the observable behavior you must verify.
- Otherwise, stick to the generic template for speed and simplicity.
