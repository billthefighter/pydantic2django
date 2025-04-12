"""
Django settings for running tests.
"""

SECRET_KEY = "test-key-not-for-production"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        # "NAME": "test_db.sqlite3", # Use in-memory DB for tests
        "NAME": ":memory:",
    }
}

INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.auth",
    "tests.apps.TestsConfig",  # Use the explicit app config
]

USE_TZ = True

# Configure test runner
TEST_RUNNER = "django.test.runner.DiscoverRunner"

# Configure migrations
MIGRATION_MODULES = {
    "tests": "tests.migrations",  # Enable migrations for our test app
}

# Configure test database
TESTING = True

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
