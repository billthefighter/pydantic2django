"""
Pytest configuration for Django tests.
"""
import os
import sys
from pathlib import Path
import logging

import django
import pytest
from django.conf import settings

# Import all fixtures
from .fixtures.fixtures import (
    basic_pydantic_model,
    datetime_pydantic_model,
    optional_fields_model,
    constrained_fields_model,
    relationship_models,
    method_model,
    factory_model,
    product_django_model,
    user_django_model,
    context_django_model,
    context_model_context,
    context_temp_file,
    context_pydantic_model,
    context_with_data,
)

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
def django_db_setup(django_db_blocker):
    """Configure the test database."""
    settings.DATABASES["default"] = {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    }

    from django.core.management import call_command

    with django_db_blocker.unblock():
        call_command("migrate", "tests", verbosity=0)


@pytest.fixture(autouse=True)
def setup_logging():
    """Set up logging for all tests."""
    logging.basicConfig(level=logging.INFO)
    for logger_name in ["tests", "pydantic2django"]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
