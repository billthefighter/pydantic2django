from pydantic2django.discovery import (
    ModelDiscovery,
    get_discovered_models,
    get_django_models,
    setup_dynamic_models,
)


def test_basic_import():
    setup_dynamic_models()
    assert get_discovered_models()


def test_model_discovery_class():
    # Test the new class-based approach
    discovery = ModelDiscovery()
    discovery.setup_dynamic_models()
    assert discovery.get_discovered_models()
