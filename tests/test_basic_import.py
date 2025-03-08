from pydantic2django.discovery import (
    get_discovered_models,
    get_django_models,
    get_registry,
    setup_dynamic_models,
)


def test_basic_import():
    setup_dynamic_models()
    assert get_discovered_models()
