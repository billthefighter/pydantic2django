import os
import subprocess
import sys
import textwrap
from pathlib import Path
from uuid import uuid4

import pytest


def _integration_ready() -> bool:
    if os.getenv("P2D_INTEGRATION_DB") != "1":
        return False
    try:
        import timescale.db  # type: ignore
    except Exception:
        return False
    try:
        import psycopg  # type: ignore
        _ = psycopg
        return True
    except Exception:
        try:
            import psycopg2  # type: ignore
            _ = psycopg2
            return True
        except Exception:
            return False


pytestmark = pytest.mark.skipif(not _integration_ready(), reason="Integration DB not available")


def _ensure_db_and_extension(dbname: str) -> None:
    try:
        import psycopg  # type: ignore

        host = os.getenv('DB_HOST', '127.0.0.1')
        port = int(os.getenv('DB_PORT', '6543'))
        user = os.getenv('DB_USER', 'postgres')
        password = os.getenv('DB_PASSWORD', 'postgres')

        with psycopg.connect(host=host, port=port, dbname='postgres', user=user, password=password, autocommit=True) as c:
            with c.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (dbname,))
                exists = cur.fetchone() is not None
                if not exists:
                    cur.execute(f'CREATE DATABASE {dbname}')  # type: ignore[arg-type]

        with psycopg.connect(host=host, port=port, dbname=dbname, user=user, password=password, autocommit=True) as c2:
            with c2.cursor() as cur2:
                cur2.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
    except Exception:
        pass


def _write_min_proj(tmp_path: Path, app_label: str) -> Path:
    proj = tmp_path / "proj"
    proj.mkdir()

    (proj / "manage.py").write_text(
        textwrap.dedent(
            '''
            #!/usr/bin/env python
            import os, sys
            if __name__ == '__main__':
                os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'settings')
                from django.core.management import execute_from_command_line
                execute_from_command_line(sys.argv)
            '''
        )
    )

    (proj / "settings.py").write_text(
        textwrap.dedent(
            f'''
            import os
            SECRET_KEY = 'x'
            INSTALLED_APPS = [
                'django.contrib.contenttypes',
                'django.contrib.auth',
                'timescale.db',
                '{app_label}',
            ]
            DATABASES = {{
                'default': {{
                    'ENGINE': 'django.db.backends.postgresql',
                    'HOST': os.environ.get('DB_HOST', 'localhost'),
                    'PORT': int(os.environ.get('DB_PORT', '5432')),
                    'NAME': os.environ.get('DB_NAME', 'lodbrok_db'),
                    'USER': os.environ.get('DB_USER', 'postgres'),
                    'PASSWORD': os.environ.get('DB_PASSWORD', 'postgres'),
                }}
            }}
            ROOT_URLCONF = 'urls'
            ALLOWED_HOSTS = ['*']
            '''
        )
    )

    (proj / "urls.py").write_text("urlpatterns = []\n")

    app = proj / app_label
    app.mkdir()
    (app / "__init__.py").write_text("")

    return proj


def _run(cwd: Path, cmd: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["DJANGO_SETTINGS_MODULE"] = "settings"
    env["PYTHONPATH"] = f"{cwd}:{env.get('PYTHONPATH', '')}"
    return subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, env=env)


def test_generate_models_from_assets_xsd_and_apply_migrations(tmp_path: Path, monkeypatch):
    # DB env
    for k, v in {
        'DB_HOST': os.getenv('DB_HOST', 'localhost'),
        'DB_PORT': os.getenv('DB_PORT', '5432'),
        'DB_NAME': os.getenv('DB_NAME', 'lodbrok_db'),
        'DB_USER': os.getenv('DB_USER', 'postgres'),
        'DB_PASSWORD': os.getenv('DB_PASSWORD', 'postgres'),
    }.items():
        monkeypatch.setenv(k, v)

    # Unique DB to avoid contamination
    test_db = f"p2ddb_{uuid4().hex[:8]}"
    monkeypatch.setenv('DB_NAME', test_db)
    _ensure_db_and_extension(test_db)

    # Prepare minimal project and app
    app_label = "mtassets_app"
    proj = _write_min_proj(tmp_path, app_label)

    # Use the generator to write models into the temp app
    from pydantic2django.xmlschema.generator import XmlSchemaDjangoModelGenerator

    xsd_path = Path(__file__).parent / "references" / "MTConnectAssets_2.4.xsd"
    output_models = proj / app_label / "models.py"

    gen = XmlSchemaDjangoModelGenerator(
        schema_files=[str(xsd_path)],
        output_path=str(output_models),
        app_label=app_label,
        verbose=False,
    )
    gen.generate_models_file()

    # Assert StreamsType is NOT generated as a Timescale model base
    gen_text = output_models.read_text()
    assert "class StreamsType(XmlTimescaleBase)" not in gen_text
    # Prefer explicit non-timescale base inheritance for StreamsType
    assert "class StreamsType(Xml2DjangoBaseClass)" in gen_text

    # Run migrations and ensure success (no FK/unique error)
    r1 = _run(proj, [sys.executable, 'manage.py', 'makemigrations', app_label])
    assert r1.returncode == 0, r1.stderr

    r2 = _run(proj, [sys.executable, 'manage.py', 'migrate', '--skip-checks'])
    stdout_stderr = r2.stdout + "\n" + r2.stderr
    assert r2.returncode == 0, stdout_stderr
    assert 'no unique constraint matching given keys' not in stdout_stderr


def test_hypertable_create_model_embeds_unique_id_and_fk_add_succeeds(tmp_path: Path, monkeypatch):
    # DB env
    for k, v in {
        'DB_HOST': os.getenv('DB_HOST', 'localhost'),
        'DB_PORT': os.getenv('DB_PORT', '5432'),
        'DB_NAME': os.getenv('DB_NAME', 'lodbrok_db'),
        'DB_USER': os.getenv('DB_USER', 'postgres'),
        'DB_PASSWORD': os.getenv('DB_PASSWORD', 'postgres'),
    }.items():
        monkeypatch.setenv(k, v)

    # Unique DB to avoid contamination
    test_db = f"p2ddb_{uuid4().hex[:8]}"
    monkeypatch.setenv('DB_NAME', test_db)
    _ensure_db_and_extension(test_db)

    app_label = "ts_defensive"
    proj = _write_min_proj(tmp_path, app_label)
    output_models = proj / app_label / "models.py"
    # Phase 1: create a hypertable model with constraint embedded in Meta (mirrors template output)
    model_phase1 = textwrap.dedent(
        f'''
        from django.db import models
        from timescale.db.models.fields import TimescaleDateTimeField
        from pydantic2django.django.timescale.bases import XmlTimescaleBase

        class StreamsType(XmlTimescaleBase):
            time = TimescaleDateTimeField(interval='1 day')

            class Meta:
                app_label = '{app_label}'
                abstract = False
                constraints = [
                    models.UniqueConstraint(fields=['id'], name='{app_label}_streamstype_id_unique')
                ]
        '''
    )
    output_models.write_text(model_phase1)

    # Phase 1: make initial migrations and assert UniqueConstraint is part of CreateModel
    r1 = _run(proj, [sys.executable, 'manage.py', 'makemigrations', app_label])
    assert r1.returncode == 0, r1.stderr

    # Locate migration file
    mig_dir = proj / app_label / "migrations"
    mig_files = sorted([p for p in mig_dir.iterdir() if p.name.endswith(".py") and p.name != "__init__.py"])
    assert mig_files, "No migration files generated"
    mig1 = mig_files[0]
    mig_text = mig1.read_text()
    # Expect UniqueConstraint embedded in CreateModel (check by constraint name)
    assert f"{app_label}_streamstype_id_unique" in mig_text

    r2 = _run(proj, [sys.executable, 'manage.py', 'migrate', '--skip-checks'])
    assert r2.returncode == 0, r2.stdout + "\n" + r2.stderr

    # Phase 2: add a new model in the SAME app with FK to the hypertable; ensure migrate succeeds
    ref_model = textwrap.dedent(
        f'''
        from django.db import models
        from pydantic2django.django.models import Xml2DjangoBaseClass

        class RefType(Xml2DjangoBaseClass):
            target = models.ForeignKey('{app_label}.StreamsType', on_delete=models.CASCADE)

            class Meta:
                app_label = '{app_label}'
                abstract = False
        '''
    )
    output_models.write_text(model_phase1 + "\n\n" + ref_model)

    r3 = _run(proj, [sys.executable, 'manage.py', 'makemigrations', app_label])
    assert r3.returncode == 0, r3.stderr
    r4 = _run(proj, [sys.executable, 'manage.py', 'migrate', '--skip-checks'])
    assert r4.returncode == 0, r4.stdout + "\n" + r4.stderr
