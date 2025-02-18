"""
Django settings for running tests.
"""

SECRET_KEY = "test-key-not-for-production"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    }
}

INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.auth",
    "tests",
    "django_pydantic.apps.DjangoPydanticConfig",  # Use the app config
]

USE_TZ = True

# Configure test runner
TEST_RUNNER = "django.test.runner.DiscoverRunner"

# Configure migrations
MIGRATION_MODULES = {
    "django_pydantic": None,  # No migrations for our test app
}

# Configure test database
TESTING = True
