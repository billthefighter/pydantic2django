"""
Pytest configuration for Django tests.
"""
import os
import sys
from pathlib import Path

import django
import pytest
from django.conf import settings

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Add src directory to Python path
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Configure Django settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tests.settings")


def pytest_configure():
    """Configure Django for tests."""
    django.setup()


@pytest.fixture(scope="session")
def django_db_setup():
    """Configure the test database."""
    settings.DATABASES["default"] = {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    }
