"""
Django app configuration for pydantic2django tests.
"""
from django.apps import AppConfig


class DjangoPydanticConfig(AppConfig):
    """Django app configuration for pydantic2django tests."""

    name = "django_pydantic"
    verbose_name = "Pydantic to Django"
