"""
Tests for the discovery module behavior.

These tests focus on the behavior of the discovery module, not its implementation details.
They verify that the module can properly discover and register different types of Pydantic models.
"""
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import pytest
from django.db import models
from pydantic import BaseModel

from pydantic2django.discovery import (
    ModelDiscovery,
    get_discovered_models,
    get_django_models,
    get_django_model,
    setup_dynamic_models,
)


def test_model_discovery_finds_pydantic_models(
    basic_pydantic_model, datetime_pydantic_model
):
    """Test that ModelDiscovery can find Pydantic models in a module."""
    # Create a test module with the fixture models
    test_module = type(sys)("test_module")
    test_module.BasicModel = basic_pydantic_model
    test_module.DateTimeModel = datetime_pydantic_model

    # Add some non-model attributes to ensure they're filtered out
    test_module.SOME_CONSTANT = "not a model"
    test_module.some_function = lambda: None

    # Add the module to sys.modules temporarily
    sys.modules["test_module"] = test_module

    try:
        # Create a discovery instance and discover models
        discovery = ModelDiscovery()
        discovery.discover_models(["test_module"])

        # Verify models were discovered
        discovered_models = discovery.get_discovered_models()
        assert "BasicModel" in discovered_models
        assert "DateTimeModel" in discovered_models
        assert len(discovered_models) == 2
    finally:
        # Clean up
        del sys.modules["test_module"]


def test_model_registration_creates_django_models(basic_pydantic_model):
    """Test that discovered models are properly registered as Django models."""
    # Create a test module with the fixture model
    test_module = type(sys)("test_module_reg")
    test_module.BasicModel = basic_pydantic_model

    # Add the module to sys.modules temporarily
    sys.modules["test_module_reg"] = test_module

    try:
        # Create a discovery instance, discover and register models
        discovery = ModelDiscovery()
        discovery.discover_models(["test_module_reg"], app_label="test_app")
        django_models = discovery.setup_dynamic_models(app_label="test_app")

        # Verify Django models were created
        assert "BasicModel" in django_models
        django_model = django_models["BasicModel"]

        # Verify the Django model has the expected fields
        field_names = [
            field.name for field in django_model._meta.fields if field.name != "id"
        ]
        assert "string_field" in field_names
        assert "int_field" in field_names
        assert "float_field" in field_names
        assert "bool_field" in field_names
        assert "decimal_field" in field_names
        assert "email_field" in field_names
    finally:
        # Clean up
        del sys.modules["test_module_reg"]


def test_model_relationships(relationship_models):
    """Test that relationships between models are properly maintained."""
    # Create a test module with the relationship models
    test_module = type(sys)("test_module_rel")
    test_module.Address = relationship_models["Address"]
    test_module.Profile = relationship_models["Profile"]
    test_module.Tag = relationship_models["Tag"]
    test_module.User = relationship_models["User"]

    # Add the module to sys.modules temporarily
    sys.modules["test_module_rel"] = test_module

    try:
        # Create a discovery instance, discover and register models
        discovery = ModelDiscovery()
        discovery.discover_models(["test_module_rel"], app_label="test_rel")
        django_models = discovery.setup_dynamic_models(app_label="test_rel")

        # Verify all models were created
        assert "Address" in django_models
        assert "Profile" in django_models
        assert "Tag" in django_models
        assert "User" in django_models

        # Verify relationships
        user_model = django_models["User"]

        # Check ForeignKey relationship
        address_field = user_model._meta.get_field("address")
        assert isinstance(address_field, models.ForeignKey)

        # The related_model can be either a string representation or the actual model class
        if isinstance(address_field.related_model, str):
            # If it's a string, it should be in the format 'app_label.model_name'
            assert address_field.related_model.endswith("DjangoAddress")
        else:
            # If it's a class, it should be the actual model class
            assert address_field.related_model == django_models["Address"]

        # Check ManyToMany relationship
        tags_field = user_model._meta.get_field("tags")
        assert isinstance(tags_field, models.ManyToManyField)

        # The related_model can be either a string representation or the actual model class
        if isinstance(tags_field.related_model, str):
            # If it's a string, it should be in the format 'app_label.model_name'
            assert tags_field.related_model.endswith("DjangoTag")
        else:
            # If it's a class, it should be the actual model class
            assert tags_field.related_model == django_models["Tag"]

        # Check OneToOne relationship
        profile_field = user_model._meta.get_field("profile")
        assert isinstance(profile_field, models.OneToOneField)

        # The related_model can be either a string representation or the actual model class
        if isinstance(profile_field.related_model, str):
            # If it's a string, it should be in the format 'app_label.model_name'
            assert profile_field.related_model.endswith("DjangoProfile")
        else:
            # If it's a class, it should be the actual model class
            assert profile_field.related_model == django_models["Profile"]
    finally:
        # Clean up
        del sys.modules["test_module_rel"]


def test_optional_fields(optional_fields_model):
    """Test that optional fields are properly handled."""
    # Create a test module with the optional fields model
    test_module = type(sys)("test_module_opt")
    test_module.OptionalModel = optional_fields_model

    # Add the module to sys.modules temporarily
    sys.modules["test_module_opt"] = test_module

    try:
        # Create a discovery instance, discover and register models
        discovery = ModelDiscovery()
        discovery.discover_models(["test_module_opt"], app_label="test_opt")
        django_models = discovery.setup_dynamic_models(app_label="test_opt")

        # Verify model was created
        assert "OptionalModel" in django_models
        django_model = django_models["OptionalModel"]

        # Get field objects
        required_string_field = django_model._meta.get_field("required_string")
        required_int_field = django_model._meta.get_field("required_int")
        optional_string_field = django_model._meta.get_field("optional_string")
        optional_int_field = django_model._meta.get_field("optional_int")

        # Verify required fields are not nullable
        # Use getattr with a default to avoid linter errors
        assert not getattr(required_string_field, "null", False)
        assert not getattr(required_int_field, "null", False)

        # Verify optional fields are nullable
        assert getattr(optional_string_field, "null", False)
        assert getattr(optional_int_field, "null", False)
    finally:
        # Clean up
        del sys.modules["test_module_opt"]


def test_constrained_fields(constrained_fields_model):
    """Test that field constraints are properly transferred to Django models."""
    # Create a test module with the constrained fields model
    test_module = type(sys)("test_module_const")
    test_module.ConstrainedModel = constrained_fields_model

    # Add the module to sys.modules temporarily
    sys.modules["test_module_const"] = test_module

    try:
        # Create a discovery instance, discover and register models
        discovery = ModelDiscovery()
        discovery.discover_models(["test_module_const"], app_label="test_const")
        django_models = discovery.setup_dynamic_models(app_label="test_const")

        # Verify model was created
        assert "ConstrainedModel" in django_models
        django_model = django_models["ConstrainedModel"]

        # Get field objects
        name_field = django_model._meta.get_field("name")
        age_field = django_model._meta.get_field("age")

        # Verify constraints were transferred using getattr with defaults
        assert getattr(name_field, "max_length", 0) == 100
        assert getattr(name_field, "verbose_name", "") == "Full Name"
        assert getattr(name_field, "help_text", "") == "Full name of the user"

        assert getattr(age_field, "verbose_name", "") == "Age"
        assert getattr(age_field, "help_text", "") == "User's age in years"
    finally:
        # Clean up
        del sys.modules["test_module_const"]


def test_model_methods_preserved(method_model):
    """Test that model methods are preserved in the Django model."""
    # Create a test module with the method model
    test_module = type(sys)("test_module_method")
    test_module.MethodModel = method_model

    # Add the module to sys.modules temporarily
    sys.modules["test_module_method"] = test_module

    try:
        # Create a discovery instance, discover and register models
        discovery = ModelDiscovery()
        discovery.discover_models(["test_module_method"], app_label="test_method")
        django_models = discovery.setup_dynamic_models(app_label="test_method")

        # Verify model was created
        assert "MethodModel" in django_models
        django_model = django_models["MethodModel"]

        # Create an instance to test methods
        instance = django_model(name="Test", value=10)

        # Test instance method
        assert hasattr(instance, "instance_method")
        assert callable(instance.instance_method)

        # Test property
        assert hasattr(instance.__class__, "computed_value")

        # Test class method
        assert hasattr(django_model, "class_method")
        assert callable(django_model.class_method)

        # Test static method
        assert hasattr(django_model, "static_method")
        assert callable(django_model.static_method)
    finally:
        # Clean up
        del sys.modules["test_module_method"]


def test_factory_model_functionality(factory_model):
    """Test that factory methods in models work correctly after conversion."""
    # Create a test module with the factory model
    test_module = type(sys)("test_module_factory")

    # Get the Product class from the factory model
    Product = factory_model.create_product.__annotations__["return"]

    test_module.Product = Product
    test_module.ProductFactory = factory_model

    # Add the module to sys.modules temporarily
    sys.modules["test_module_factory"] = test_module

    try:
        # Create a discovery instance, discover and register models
        discovery = ModelDiscovery()
        discovery.discover_models(["test_module_factory"], app_label="test_factory")
        django_models = discovery.setup_dynamic_models(app_label="test_factory")

        # Verify models were created
        assert "Product" in django_models
        assert "ProductFactory" in django_models

        product_model = django_models["Product"]
        factory_model_class = django_models["ProductFactory"]

        # Create a factory instance
        factory = factory_model_class(
            default_price="19.99", default_description="Test product"
        )

        # Test factory methods
        assert hasattr(factory, "create_product")
        assert callable(factory.create_product)

        # Since the factory method returns a Pydantic model, we need to manually convert it to a Django model
        # or modify our test to just verify the Pydantic model is returned correctly
        pydantic_product = factory.create_product(name="Test Product")

        # Verify the product was created correctly as a Pydantic model
        assert isinstance(pydantic_product, Product)
        assert pydantic_product.name == "Test Product"
        assert str(pydantic_product.price) == "19.99"
        assert pydantic_product.description == "Test product"

        # Create a Django model instance manually
        django_product = product_model(
            name=pydantic_product.name,
            price=pydantic_product.price,
            description=pydantic_product.description,
        )

        # Verify the Django model instance
        assert isinstance(django_product, product_model)
        assert django_product.name == "Test Product"
        assert str(django_product.price) == "19.99"
        assert django_product.description == "Test product"
    finally:
        # Clean up
        del sys.modules["test_module_factory"]


def test_global_functions():
    """Test that the global functions work correctly."""
    # Create a test module with a simple model
    test_module = type(sys)("test_module_global")

    class SimpleModel(BaseModel):
        name: str
        value: int

    test_module.SimpleModel = SimpleModel

    # Add the module to sys.modules temporarily
    sys.modules["test_module_global"] = test_module

    try:
        # Use the global functions
        # First, make sure we start with a clean state
        discovery_instance = ModelDiscovery()

        # Monkey patch the global discovery instance
        from pydantic2django.discovery import discovery as global_discovery

        # Save the original instance
        original_discovery = global_discovery
        # Replace with our test instance
        import pydantic2django.discovery

        pydantic2django.discovery.discovery = discovery_instance

        # Discover and register models
        from pydantic2django.discovery import (
            discover_models,
            setup_dynamic_models,
            get_discovered_models,
            get_django_models,
            get_django_model,
        )

        discover_models(["test_module_global"], app_label="test_global")
        setup_dynamic_models(app_label="test_global")

        # Verify models were discovered
        discovered_models = get_discovered_models()
        assert "SimpleModel" in discovered_models
        assert discovered_models["SimpleModel"] == SimpleModel

        # Verify Django models were created
        django_models = get_django_models(app_label="test_global")
        assert "SimpleModel" in django_models
        assert issubclass(django_models["SimpleModel"], models.Model)

        # Test get_django_model function
        django_model = get_django_model(SimpleModel, app_label="test_global")
        assert django_model == django_models["SimpleModel"]
    finally:
        # Clean up
        del sys.modules["test_module_global"]
        # Restore the original discovery instance
        if "original_discovery" in locals():
            pydantic2django.discovery.discovery = original_discovery


def test_multiple_discovery_instances():
    """Test that multiple discovery instances can coexist without interference."""
    # Create two test modules with different models
    test_module1 = type(sys)("test_module_multi1")

    class Model1(BaseModel):
        name: str

    test_module1.Model1 = Model1

    test_module2 = type(sys)("test_module_multi2")

    class Model2(BaseModel):
        value: int

    test_module2.Model2 = Model2

    # Add the modules to sys.modules temporarily
    sys.modules["test_module_multi1"] = test_module1
    sys.modules["test_module_multi2"] = test_module2

    try:
        # Create two separate discovery instances
        discovery1 = ModelDiscovery()
        discovery2 = ModelDiscovery()

        # Discover models in different app labels
        discovery1.discover_models(["test_module_multi1"], app_label="app1")
        discovery2.discover_models(["test_module_multi2"], app_label="app2")

        # Register models
        django_models1 = discovery1.setup_dynamic_models(app_label="app1")
        django_models2 = discovery2.setup_dynamic_models(app_label="app2")

        # Verify models were discovered and registered correctly
        assert "Model1" in discovery1.get_discovered_models()
        assert "Model2" not in discovery1.get_discovered_models()

        assert "Model2" in discovery2.get_discovered_models()
        assert "Model1" not in discovery2.get_discovered_models()

        assert "Model1" in django_models1
        assert "Model2" in django_models2
    finally:
        # Clean up
        del sys.modules["test_module_multi1"]
        del sys.modules["test_module_multi2"]
