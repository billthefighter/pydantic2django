"""
Tests for discovering models from chains.py.

This test specifically focuses on discovering Pydantic models from the chains.py file
and diagnosing issues with field collisions and app label configuration.
"""
import importlib
import logging
import os
import sys
from pathlib import Path

import pytest
from django.db import models
from pydantic import BaseModel

from pydantic2django.discovery import (
    ModelDiscovery,
    discover_models,
    get_discovered_models,
    setup_dynamic_models,
)


@pytest.fixture
def setup_debug_logging():
    """Set up debug logging for the test."""
    logger = logging.getLogger("pydantic2django")
    previous_level = logger.level
    logger.setLevel(logging.DEBUG)
    yield
    logger.setLevel(previous_level)


def test_discover_chain_models(setup_debug_logging):
    """Test discovering models from chains.py."""
    # Get the path to the test_models directory
    test_models_dir = Path(__file__).parent / "test_models"

    # Add the test_models directory to sys.path temporarily
    sys.path.insert(0, str(test_models_dir.parent))

    try:
        # Create a test module with the chains module
        test_module = type(sys)("test_chains_module")

        # Import the chains module
        chains_module = importlib.import_module("tests.test_models.chains")

        # Copy all attributes from chains_module to test_module
        for name in dir(chains_module):
            if not name.startswith("__"):
                setattr(test_module, name, getattr(chains_module, name))

        # Add the module to sys.modules temporarily
        sys.modules["test_chains_module"] = test_module

        # Create a discovery instance with a custom app_label
        discovery = ModelDiscovery()

        # Discover models from the test module
        discovery.discover_models(
            ["test_chains_module"],
            app_label="test_chains",
            filter_function=None,
        )

        # Get the discovered models
        discovered_models = discovery.get_discovered_models()

        # Print discovered models for debugging
        print(f"Discovered {len(discovered_models)} models:")
        for name, model in discovered_models.items():
            print(f"  - {name}: {model}")

        # Try to set up dynamic models
        try:
            django_models = discovery.setup_dynamic_models(app_label="test_chains")
            print(f"Successfully created {len(django_models)} Django models")
            for name, model in django_models.items():
                print(f"  - {name}: {model}")
        except Exception as e:
            print(f"Error setting up dynamic models: {e}")
            # Continue with the test even if this fails

        # Assert that we found at least some models
        assert len(discovered_models) > 0, "No models were discovered"

        # Check for specific models we expect to find
        assert "RetryStrategy" in discovered_models, "RetryStrategy model not found"
        assert "ChainMetadata" in discovered_models, "ChainMetadata model not found"
        assert "ChainState" in discovered_models, "ChainState model not found"
        assert "ChainContext" in discovered_models, "ChainContext model not found"
        assert "ChainStep" in discovered_models, "ChainStep model not found"

    finally:
        # Clean up
        if "test_chains_module" in sys.modules:
            del sys.modules["test_chains_module"]

        # Remove the test_models directory from sys.path
        sys.path.remove(str(test_models_dir.parent))


def test_discover_chain_models_with_global_functions():
    """Test discovering models from chains.py using global functions."""
    # Set up debug logging
    logger = logging.getLogger("pydantic2django")
    previous_level = logger.level
    logger.setLevel(logging.DEBUG)

    # Get the path to the test_models directory
    test_models_dir = Path(__file__).parent / "test_models"

    # Add the test_models directory to sys.path temporarily
    sys.path.insert(0, str(test_models_dir.parent))

    try:
        # Create a test module with the chains module
        test_module = type(sys)("test_chains_module")

        # Import the chains module
        chains_module = importlib.import_module("tests.test_models.chains")

        # Copy all attributes from chains_module to test_module
        for name in dir(chains_module):
            if not name.startswith("__"):
                setattr(test_module, name, getattr(chains_module, name))

        # Add the module to sys.modules temporarily
        sys.modules["test_chains_module"] = test_module

        # Use the global functions with a custom app_label
        discover_models(
            ["test_chains_module"],
            app_label="test_chains",
            filter_function=None,
        )

        # Get the discovered models
        discovered_models = get_discovered_models()

        # Print discovered models for debugging
        print(f"Discovered {len(discovered_models)} models:")
        for name, model in discovered_models.items():
            print(f"  - {name}: {model}")

        # Try to set up dynamic models
        try:
            django_models = setup_dynamic_models(app_label="test_chains")
            print(f"Successfully created {len(django_models)} Django models")
            for name, model in django_models.items():
                print(f"  - {name}: {model}")
        except Exception as e:
            print(f"Error setting up dynamic models: {e}")
            # Continue with the test even if this fails

        # Assert that we found at least some models
        assert len(discovered_models) > 0, "No models were discovered"

    finally:
        # Clean up
        if "test_chains_module" in sys.modules:
            del sys.modules["test_chains_module"]

        # Reset logging level
        logger.setLevel(previous_level)

        # Remove the test_models directory from sys.path
        sys.path.remove(str(test_models_dir.parent))
