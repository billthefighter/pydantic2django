"""
Pytest configuration and fixtures.
"""
import pytest
from django.conf import settings

@pytest.fixture(autouse=True)
def setup_test_database():
    """Ensure test database is used."""
    settings.DATABASES["default"]["NAME"] = ":memory:" 