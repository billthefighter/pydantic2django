import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from uuid import uuid4


def _integration_ready() -> bool:
    """Return True if integration DB test should run.

    Requires env var P2D_INTEGRATION_DB=1 and presence of timescale + a PG driver.
    """
    if os.getenv("P2D_INTEGRATION_DB") != "1":
        return False
    try:
        import timescale.db  # type: ignore
    except Exception:
        return False
    try:
        import psycopg  # type: ignore  # Django 5 default
        _ = psycopg  # silence linter
        return True
    except Exception:
        try:
            import psycopg2  # type: ignore  # Fallback older driver

            _ = psycopg2  # silence linter
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
        # Best-effort; the make target should have provisioned base DB and extension
        pass


def _write_minimal_proj(tmp_path: Path, model_body: str) -> Path:
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
            '''
            import os
            SECRET_KEY = 'x'
            INSTALLED_APPS = [
                'django.contrib.contenttypes',
                'django.contrib.auth',
                'timescale.db',
                'ts_fk_test',
            ]
            DATABASES = {
                'default': {
                    'ENGINE': 'django.db.backends.postgresql',
                    'HOST': os.environ.get('DB_HOST', 'localhost'),
                    'PORT': int(os.environ.get('DB_PORT', '5432')),
                    'NAME': os.environ.get('DB_NAME', 'lodbrok_db'),
                    'USER': os.environ.get('DB_USER', 'postgres'),
                    'PASSWORD': os.environ.get('DB_PASSWORD', 'postgres'),
                }
            }
            ROOT_URLCONF = 'urls'
            ALLOWED_HOSTS = ['*']
            '''
        )
    )

    (proj / "urls.py").write_text("urlpatterns = []\n")

    app = proj / "ts_fk_test"
    app.mkdir()
    (app / "__init__.py").write_text("")
    (app / "models.py").write_text(model_body)

    return proj


def _run(cwd: Path, cmd: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    # Force our minimal project's settings to avoid pytest-django's --ds override leaking in
    env["DJANGO_SETTINGS_MODULE"] = "settings"
    # Ensure the temp project directory is importable as top-level for 'settings'
    env["PYTHONPATH"] = f"{cwd}:{env.get('PYTHONPATH', '')}"
    return subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, env=env)


def test_fk_migrate_fails_without_unique_on_id(tmp_path: Path, monkeypatch):
    # Set DB env from outer environment
    for k, v in {
        'DB_HOST': os.getenv('DB_HOST', 'localhost'),
        'DB_PORT': os.getenv('DB_PORT', '5432'),
        'DB_NAME': os.getenv('DB_NAME', 'lodbrok_db'),
        'DB_USER': os.getenv('DB_USER', 'postgres'),
        'DB_PASSWORD': os.getenv('DB_PASSWORD', 'postgres'),
    }.items():
        monkeypatch.setenv(k, v)

    # Use a unique database per test run to avoid cross-test contamination
    test_db = f"p2ddb_{uuid4().hex[:8]}"
    monkeypatch.setenv('DB_NAME', test_db)
    _ensure_db_and_extension(test_db)

    # Stage 1: create only the hypertable model and migrate (to trigger hypertable conversion)
    models_src_initial = textwrap.dedent(
        '''
        from django.db import models
        from timescale.db.models.fields import TimescaleDateTimeField
        from pydantic2django.django.models import Xml2DjangoBaseClass, TimescaleModel

        class RawTsBase(Xml2DjangoBaseClass, TimescaleModel):
            class Meta:
                abstract = True

        class StreamsType(RawTsBase):
            time = TimescaleDateTimeField(interval='1 day')

            class Meta:
                app_label = 'ts_fk_test'
                abstract = False
        '''
    )

    proj = _write_minimal_proj(tmp_path, models_src_initial)

    r1 = _run(proj, [sys.executable, 'manage.py', 'makemigrations', 'ts_fk_test'])
    assert r1.returncode == 0, r1.stderr
    r1b = _run(proj, [sys.executable, 'manage.py', 'migrate', '--skip-checks'])
    assert r1b.returncode == 0, r1b.stdout + "\n" + r1b.stderr

    # Stage 2: add the dimension model with FK to the hypertable; then expect migrate to fail
    models_src_full = textwrap.dedent(
        '''
        from django.db import models
        from timescale.db.models.fields import TimescaleDateTimeField
        from pydantic2django.django.models import Xml2DjangoBaseClass, TimescaleModel

        class RawTsBase(Xml2DjangoBaseClass, TimescaleModel):
            class Meta:
                abstract = True

        class StreamsType(RawTsBase):
            time = TimescaleDateTimeField(interval='1 day')

            class Meta:
                app_label = 'ts_fk_test'
                abstract = False

        class DeviceStreamType(Xml2DjangoBaseClass):
            name = models.TextField()
            uuid = models.TextField()
            streamstype = models.ForeignKey('ts_fk_test.StreamsType', on_delete=models.CASCADE, related_name='devicestreams')

            class Meta:
                app_label = 'ts_fk_test'
                abstract = False
        '''
    )
    (proj / 'ts_fk_test' / 'models.py').write_text(models_src_full)

    r2a = _run(proj, [sys.executable, 'manage.py', 'makemigrations', 'ts_fk_test'])
    assert r2a.returncode == 0, r2a.stderr
    r2 = _run(proj, [sys.executable, 'manage.py', 'migrate', '--skip-checks'])
    if r2.returncode == 0:
        # Environment did not reproduce the failure. Inspect constraints to decide whether to xfail.
        try:
            import psycopg  # type: ignore

            conn = psycopg.connect(
                host=os.getenv('DB_HOST', '127.0.0.1'),
                port=int(os.getenv('DB_PORT', '6543')),
                dbname=os.getenv('DB_NAME', 'p2ddb'),
                user=os.getenv('DB_USER', 'postgres'),
                password=os.getenv('DB_PASSWORD', 'postgres'),
            )
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 1
                    FROM pg_constraint c
                    JOIN pg_class t ON c.conrelid = t.oid
                    JOIN pg_namespace n ON n.oid = t.relnamespace
                    JOIN pg_attribute a ON a.attrelid = t.oid AND a.attname = 'id'
                    WHERE t.relname = 'ts_fk_test_streamstype'
                      AND n.nspname = 'public'
                      AND a.attnum = ANY (c.conkey)
                      AND c.contype IN ('u','p')
                    LIMIT 1
                    """
                )
                has_unique = cur.fetchone() is not None
        except Exception:
            has_unique = False
        finally:
            try:
                conn.close()  # type: ignore[name-defined]
            except Exception:
                pass

        if has_unique:
            pytest.xfail("Hypertable retains unique/primary on id in this environment; cannot reproduce failure.")
        else:
            pytest.fail("Expected FK migration to fail due to missing unique id on hypertable, but it succeeded.")
    else:
        assert 'no unique constraint matching given keys' in (r2.stderr + r2.stdout)


def test_fk_migrate_succeeds_with_xmltimescalebase(tmp_path: Path, monkeypatch):
    # Set DB env from outer environment
    for k, v in {
        'DB_HOST': os.getenv('DB_HOST', 'localhost'),
        'DB_PORT': os.getenv('DB_PORT', '5432'),
        'DB_NAME': os.getenv('DB_NAME', 'lodbrok_db'),
        'DB_USER': os.getenv('DB_USER', 'postgres'),
        'DB_PASSWORD': os.getenv('DB_PASSWORD', 'postgres'),
    }.items():
        monkeypatch.setenv(k, v)

    # Use a unique database per test run
    test_db = f"p2ddb_{uuid4().hex[:8]}"
    monkeypatch.setenv('DB_NAME', test_db)
    _ensure_db_and_extension(test_db)

    models_src = textwrap.dedent(
        '''
        from django.db import models
        from timescale.db.models.fields import TimescaleDateTimeField
        from pydantic2django.django.models import Xml2DjangoBaseClass
        from pydantic2django.django.timescale.bases import XmlTimescaleBase

        class StreamsType(XmlTimescaleBase):
            time = TimescaleDateTimeField(interval='1 day')

            class Meta:
                app_label = 'ts_fk_test'
                abstract = False

        class DeviceStreamType(Xml2DjangoBaseClass):
            name = models.TextField()
            uuid = models.TextField()
            streamstype = models.ForeignKey('ts_fk_test.StreamsType', on_delete=models.CASCADE, related_name='devicestreams')

            class Meta:
                app_label = 'ts_fk_test'
                abstract = False
        '''
    )

    proj = _write_minimal_proj(tmp_path, models_src)

    r1 = _run(proj, [sys.executable, 'manage.py', 'makemigrations', 'ts_fk_test'])
    assert r1.returncode == 0, r1.stderr

    r2 = _run(proj, [sys.executable, 'manage.py', 'migrate', '--skip-checks'])
    assert r2.returncode == 0, r2.stdout + "\n" + r2.stderr
