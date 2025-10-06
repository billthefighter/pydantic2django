from pathlib import Path

import importlib.util
import sys

import pytest
from django.db import connection
from django.apps import apps as django_apps

from pydantic2django.xmlschema import XmlSchemaDjangoModelGenerator, XmlInstanceIngestor


@pytest.fixture(scope="function")
def streams_persist_ctx(tmp_path: Path):
    """Setup: generate Streams models (GFK on), create tables (incl. ContentType, Permission), ingest current.xml with save=True."""
    # Paths to example XSD and XML
    xsd_path = Path(__file__).parent / "example_xml" / "MTConnectStreams_1.7.xsd"
    xml_path = Path(__file__).parent / "example_xml" / "current.xml"

    assert xsd_path.exists(), f"Missing XSD file: {xsd_path}"
    assert xml_path.exists(), f"Missing XML file: {xml_path}"

    # Generate dynamic Django model classes from the XSD under installed app 'tests'
    app_label = "tests"
    out_file = tmp_path / "models.py"

    gen = XmlSchemaDjangoModelGenerator(
        schema_files=[str(xsd_path)],
        output_path=str(out_file),
        app_label=app_label,
        verbose=False,
        nested_relationship_strategy="fk",
        list_relationship_style="child_fk",
        enable_timescale=False,
        enable_gfk=True,
        gfk_policy="all_nested",
        gfk_value_mode="typed_columns",
    )

    gen.generate()

    # Import the generated models module to register models (including GenericEntry) with Django app registry
    spec = importlib.util.spec_from_file_location("tests_streams_models", str(out_file))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["tests_streams_models"] = module
    spec.loader.exec_module(module)  # type: ignore[assignment]

    # Ensure ContentType and Permission tables exist before teardown/post_migrate uses them
    from django.contrib.contenttypes.models import ContentType
    from django.contrib.auth.models import Permission

    with connection.schema_editor() as schema_editor:
        ct_table = ContentType._meta.db_table
        if ct_table not in connection.introspection.table_names():
            schema_editor.create_model(ContentType)

        perm_table = Permission._meta.db_table
        if perm_table not in connection.introspection.table_names():
            schema_editor.create_model(Permission)

        # Create DB tables for all INSTALLED generated models in processing order
        for carrier in gen.carriers:
            dyn_model_cls = getattr(carrier, "django_model", None)
            if dyn_model_cls is None:
                continue
            model_name = getattr(dyn_model_cls, "__name__", None)
            if not model_name:
                continue
            installed_model = django_apps.get_model(app_label, model_name)
            if installed_model is None:
                continue
            is_abstract = getattr(getattr(installed_model, "_meta", None), "abstract", None)
            if is_abstract:
                continue
            table_name = installed_model._meta.db_table  # type: ignore[attr-defined]
            if table_name not in connection.introspection.table_names():
                schema_editor.create_model(installed_model)

        # Ensure GenericEntry model table exists (needed for GFK persistence)
        GenericEntry = django_apps.get_model(app_label, "GenericEntry")
        if GenericEntry is not None:  # type: ignore[truthy-function]
            ge_table = GenericEntry._meta.db_table
            if ge_table not in connection.introspection.table_names():
                schema_editor.create_model(GenericEntry)

    # Ingest the XML instance with persistence
    ingestor = XmlInstanceIngestor(
        schema_files=[str(xsd_path)],
        app_label=app_label,
        dynamic_model_fallback=True,
    )
    root = ingestor.ingest_from_file(str(xml_path), save=True)

    # Provide context to tests
    GenericEntry = django_apps.get_model(app_label, "GenericEntry")
    return {
        "root": root,
        "GenericEntry": GenericEntry,
    }


@pytest.mark.parametrize(
    "check_name",
    [
        "root_saved",
        "genericentry_model",
        "genericentry_columns",
        "root_entries",
        "genericentry_linked",
    ],
)
@pytest.mark.django_db(transaction=True)
def test_streams_persistence_checks(streams_persist_ctx, check_name: str) -> None:
    ctx = streams_persist_ctx
    root = ctx["root"]
    GenericEntry = ctx["GenericEntry"]

    if check_name == "root_saved":
        assert getattr(root, "pk", None) is not None, "Root instance should be saved (pk assigned)"
        return

    if check_name == "genericentry_model":
        assert GenericEntry is not None, "GenericEntry model must be registered under installed app"
        return

    if check_name == "genericentry_columns":
        ge_field_names = {f.name for f in GenericEntry._meta.fields}
        assert {"text_value", "num_value", "time_value"}.issubset(ge_field_names)
        return

    if check_name == "root_entries":
        assert hasattr(root, "entries"), "Root should expose GenericRelation 'entries'"
        assert root.entries.count() > 0, "Expected persisted GenericEntry rows for nested elements"
        return

    if check_name == "genericentry_linked":
        assert GenericEntry.objects.filter(object_id=root.pk).exists()
        return

    raise AssertionError(f"Unknown check: {check_name}")
