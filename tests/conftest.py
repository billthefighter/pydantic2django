"""
Pytest configuration for Django tests.
"""
import os
import sys
from pathlib import Path
import logging
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any, Callable, ClassVar, Optional
from uuid import UUID

import django
import pytest
from django.conf import settings
from django.db import models
from pydantic import BaseModel, EmailStr, Field, ConfigDict
from django.utils.functional import lazy, Promise
from django.utils.translation import gettext_lazy
from django.utils.translation import gettext_lazy as _

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


# Fixtures have been moved to tests/fixtures/fixtures.py

# Helper function and classes remain here


def get_model_fields(django_model):
    """Helper function to get fields from a Django model."""
    return {f.name: f for f in django_model._meta.get_fields()}


class UnserializableType:
    """A type that can't be serialized to JSON."""

    def __init__(self, value: str):
        self.value = value


class ComplexHandler:
    """A complex handler that can't be serialized."""

    def process(self, data: Any) -> Any:
        return data


class SerializableType(BaseModel):
    """A type that can be serialized to JSON."""

    value: str

    model_config = ConfigDict(json_schema_extra={"examples": [{"value": "example"}]})


# --- End of conftest.py content ---


# Define LazyChoiceModel here for discovery
class LazyChoiceModel(models.Model):
    class Meta:
        app_label = "tests"

    class LazyChoices(models.TextChoices):
        CHOICE1 = "C1", _("Lazy Choice One")
        CHOICE2 = "C2", _("Lazy Choice Two")

    lazy_choice_field = models.CharField(
        max_length=2,
        choices=LazyChoices.choices,
        help_text=_("Field with lazy choices"),
        null=True,
        blank=True,
    )
    name = models.CharField(max_length=50, default="Test")


@pytest.fixture(scope="session")
def lazy_choice_model():
    return LazyChoiceModel
