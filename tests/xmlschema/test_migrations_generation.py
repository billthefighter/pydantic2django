from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from django.core.management import call_command
from django.test.utils import override_settings

from pydantic2django.xmlschema.generator import XmlSchemaDjangoModelGenerator



@pytest.mark.django_db
def test_makemigrations_runs_on_generated_xml_models(tmp_path: Path):
    """Generate Django models from the MTConnect XSD and ensure makemigrations succeeds.

    This creates a temporary Django app whose AppConfig.label matches the generated app_label,
    injects it into INSTALLED_APPS and MIGRATION_MODULES, and runs system checks and makemigrations.
    """
    # Arrange: generate models into a temp app package
    gen_app_pkg = tmp_path / "gen_mtc_v24"
    gen_app_pkg.mkdir(parents=True, exist_ok=True)
    (gen_app_pkg / "__init__.py").write_text("")
    (gen_app_pkg / "migrations").mkdir(parents=True, exist_ok=True)
    (gen_app_pkg / "migrations" / "__init__.py").write_text("")

    # AppConfig with module name 'gen_mtc_v24' but label matching generated app_label
    apps_py = (
        "from django.apps import AppConfig\n\n"
        "class MtconnectAssetsV24Config(AppConfig):\n"
        "    name = 'gen_mtc_v24'\n"
        "    label = 'mtconnect_assets_v24'\n"
        "    verbose_name = 'mtconnect_assets_v24'\n"
    )
    (gen_app_pkg / "apps.py").write_text(apps_py)

    # Generate models from example XSD
    xsd_path = Path(__file__).parent / "example_xml" / "MTConnectAssets_2.4.xsd"
    assert xsd_path.exists(), "Missing MTConnectAssets_2.4.xsd fixture"
    models_py = gen_app_pkg / "models.py"

    app_label = "mtconnect_assets_v24"
    XmlSchemaDjangoModelGenerator(
        schema_files=[str(xsd_path)],
        output_path=str(models_py),
        app_label=app_label,
        verbose=False,
    ).generate()

    # Ensure Python can import the temp app
    sys.path.insert(0, str(tmp_path))

    # Override Django settings to include our temp app with matching label
    with override_settings(
        INSTALLED_APPS=[
            "django.contrib.contenttypes",
            "django.contrib.auth",
            "tests.apps.TestsConfig",
            "gen_mtc_v24.apps.MtconnectAssetsV24Config",
        ],
        MIGRATION_MODULES={
            "tests": "tests.migrations",
            app_label: "gen_mtc_v24.migrations",
        },
        DEFAULT_AUTO_FIELD="django.db.models.BigAutoField",
    ):
        # Import generated models to surface import errors early
        import importlib

        importlib.invalidate_caches()
        try:
            importlib.import_module("gen_mtc_v24.models")
        except Exception as exc:
            raise AssertionError(f"Failed to import generated models: {exc}")

        # Act: run Django system checks and makemigrations against the generated app
        call_command("check", verbosity=0)
        # Dry run ensures command runs and would create migrations without writing files
        try:
            call_command("makemigrations", "--dry-run", verbosity=0)
        except Exception as exc:
            raise AssertionError(f"makemigrations --dry-run failed: {exc}")

        # Now actually write migrations; success without exceptions is sufficient
        try:
            call_command("makemigrations", verbosity=0)
        except Exception as exc:
            raise AssertionError(f"makemigrations write failed: {exc}")
        # Optional debug: ensure the migrations folder is writable and inspect files (non-fatal if empty)
        _files = [p.name for p in (gen_app_pkg / "migrations").glob("*.py")]
