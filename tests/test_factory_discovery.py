import sys
import pytest
from pathlib import Path
from typing import Dict, Any, Optional, cast, Type, Union
import shutil
import tempfile
import inspect
import logging

from pydantic import BaseModel
from django.db import models

from pydantic2django import DjangoModelFactory, clear_model_registry
from pydantic2django import discovery

# Set up logging
logging.basicConfig(level=logging.DEBUG)


def create_temp_package(name: str, models_dict: Dict[str, Type[BaseModel]]) -> Path:
    """Create a temporary package with the given models."""
    # Create a temporary directory
    temp_dir = Path(tempfile.mkdtemp())

    # Create package directory
    package_dir = temp_dir / name
    package_dir.mkdir()

    # Create __init__.py with imports
    init_file = package_dir / "__init__.py"
    init_file.write_text("# Package initialization\n")

    # Create models.py with the given models
    models_file = package_dir / "models.py"

    models_content = [
        "from pydantic import BaseModel, Field",
        "",
    ]

    # Add model classes
    for model_name, model_class in models_dict.items():
        # Get model source code
        model_source = f"class {model_name}(BaseModel):"

        # Add field annotations - ensure model_class is a Pydantic model
        if hasattr(model_class, "model_fields"):
            for field_name, field_type in model_class.model_fields.items():
                field_annotation = field_type.annotation
                field_type_name = getattr(
                    field_annotation, "__name__", str(field_annotation)
                )

                # Handle special case for references to other models
                if inspect.isclass(field_annotation) and issubclass(
                    field_annotation, BaseModel
                ):
                    model_source += f"\n    {field_name}: '{field_type_name}'"
                else:
                    model_source += f"\n    {field_name}: {field_type_name}"

        models_content.append(model_source)
        models_content.append("")

    # Write models to models.py
    models_file.write_text("\n".join(models_content))

    print(f"Created package at {temp_dir} with models:")
    print("\n".join(models_content))

    return temp_dir


@pytest.fixture(autouse=True)
def setup_teardown():
    """Clear model registry before and after each test."""
    clear_model_registry()
    yield
    clear_model_registry()


def test_factory_with_discovery():
    """Test using DjangoModelFactory with discovery module."""

    # Define a model
    class User(BaseModel):
        name: str
        email: str

    # Create a temporary package
    temp_dir = create_temp_package("test_package", {"User": User})

    try:
        # Add the temp directory to sys.path
        sys.path.insert(0, str(temp_dir))

        # Use discovery to find models
        discovery_instance = discovery.ModelDiscovery()
        discovery_instance.discover_models(["test_package"])
        discovered_models = discovery_instance.get_discovered_models()

        # Check discovered models - handle potential None
        assert discovered_models is not None, "discover_models returned None"

        # The model might be discovered as DjangoUser or User
        assert any(
            name in ["User", "DjangoUser"] for name in discovered_models
        ), "User model not found in discovered models"

        # Create Django models using factory
        django_models = discovery_instance.setup_dynamic_models(app_label="tests")

        # Check Django models
        assert "DjangoUser" in django_models, "DjangoUser not found in django_models"
        django_user = django_models["DjangoUser"]

        # Check fields
        assert isinstance(django_user._meta.get_field("name"), models.CharField)
        assert isinstance(django_user._meta.get_field("email"), models.CharField)

        # Check Meta attributes
        assert django_user._meta.abstract is False
        assert django_user._meta.managed is True

    finally:
        # Clean up
        if str(temp_dir) in sys.path:
            sys.path.remove(str(temp_dir))

        # Remove the temporary directory
        shutil.rmtree(temp_dir)


def test_factory_with_discovery_relationships():
    """Test using DjangoModelFactory with discovery module for models with relationships."""

    # Define models
    class Category(BaseModel):
        name: str

    class Product(BaseModel):
        name: str
        price: float
        category: Category

    # Create temporary packages
    temp_dir = create_temp_package(
        "test_package", {"Category": Category, "Product": Product}
    )

    try:
        # Add the temp directory to sys.path
        sys.path.insert(0, str(temp_dir))

        # Use discovery to find models
        discovery_instance = discovery.ModelDiscovery()
        discovery_instance.discover_models(["test_package"])
        discovered_models = discovery_instance.get_discovered_models()

        # Print discovered models for debugging
        print("Discovered models:", discovered_models)

        # Check discovered models - handle potential None
        assert discovered_models is not None, "discover_models returned None"

        # For debugging, print the contents of the package
        import importlib

        try:
            test_package = importlib.import_module("test_package")
            print("test_package contents:", dir(test_package))

            test_package_models = importlib.import_module("test_package.models")
            print("test_package.models contents:", dir(test_package_models))

            # Check if Category and Product are in the module
            print(
                "Category in test_package.models:",
                hasattr(test_package_models, "Category"),
            )
            print(
                "Product in test_package.models:",
                hasattr(test_package_models, "Product"),
            )

            # Try to get the Category and Product classes
            if hasattr(test_package_models, "Category"):
                category_class = getattr(test_package_models, "Category")
                print("Category class:", category_class)
                print(
                    "Category is BaseModel subclass:",
                    issubclass(category_class, BaseModel),
                )

            if hasattr(test_package_models, "Product"):
                product_class = getattr(test_package_models, "Product")
                print("Product class:", product_class)
                print(
                    "Product is BaseModel subclass:",
                    issubclass(product_class, BaseModel),
                )
        except Exception as e:
            print(f"Error inspecting test_package: {e}")

        # The models might be discovered with different names
        assert any(
            name in ["Category", "DjangoCategory"] for name in discovered_models
        ), "Category model not found in discovered models"
        assert any(
            name in ["Product", "DjangoProduct"] for name in discovered_models
        ), "Product model not found in discovered models"

        # Create Django models using factory and discovery's setup_dynamic_models
        django_models = discovery_instance.setup_dynamic_models(app_label="tests")

        # Check Django models - handle potential None
        assert django_models is not None, "setup_dynamic_models returned None"
        assert (
            "DjangoCategory" in django_models
        ), "DjangoCategory not found in django_models"
        assert (
            "DjangoProduct" in django_models
        ), "DjangoProduct not found in django_models"

        django_category = django_models["DjangoCategory"]
        django_product = django_models["DjangoProduct"]

        # Check fields
        assert isinstance(django_category._meta.get_field("name"), models.CharField)
        assert isinstance(django_product._meta.get_field("name"), models.CharField)
        assert isinstance(django_product._meta.get_field("price"), models.FloatField)

        # Check relationship
        category_field = django_product._meta.get_field("category")
        assert isinstance(category_field, models.ForeignKey)

        # Check relationship - the related_model might be a string or a class
        related_model = getattr(category_field, "related_model", None)
        assert related_model is not None

        # If it's a string, it should contain "category"
        if isinstance(related_model, str):
            assert "category" in related_model.lower()
        else:
            # If it's a class, it should have the right name
            assert related_model.__name__ == "DjangoCategory"

    finally:
        # Clean up
        if str(temp_dir) in sys.path:
            sys.path.remove(str(temp_dir))

        # Remove the temporary directory
        shutil.rmtree(temp_dir)
