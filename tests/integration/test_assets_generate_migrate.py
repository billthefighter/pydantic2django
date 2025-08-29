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

    # Run migrations and ensure success (no FK/unique error)
    r1 = _run(proj, [sys.executable, 'manage.py', 'makemigrations', app_label])
    assert r1.returncode == 0, r1.stderr

    r2 = _run(proj, [sys.executable, 'manage.py', 'migrate', '--skip-checks'])
    stdout_stderr = r2.stdout + "\n" + r2.stderr
    assert r2.returncode == 0, stdout_stderr
    assert 'no unique constraint matching given keys' not in stdout_stderr
