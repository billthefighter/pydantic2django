"""
Tests for the package discovery functionality of the discovery module.

These tests focus on the behavior of the discovery module when discovering models across multiple packages,
ensuring that models are correctly found and registered regardless of their location.
"""
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set

import pytest
from pydantic import BaseModel
from django.apps import apps

from pydantic2django.discovery import ModelDiscovery
from pydantic2django.core import clear_model_registry


def create_temp_package(name: str, models: Dict[str, type]) -> Path:
    """
    Create a temporary package with the given models.

    Args:
        name: The name of the package
        models: Dict mapping model names to model classes

    Returns:
        Path to the package directory
    """
    # Create a temporary directory
    temp_dir = Path(tempfile.mkdtemp())

    # Create the package directory
    package_dir = temp_dir / name
    package_dir.mkdir()

    # Create an __init__.py file
    with open(package_dir / "__init__.py", "w") as f:
        f.write("# Package initialization\n")

    # Create a models.py file with the models
    with open(package_dir / "models.py", "w") as f:
        f.write("from pydantic import BaseModel\n\n")

        # Write each model
        for model_name, model_class in models.items():
            # Get the model fields
            fields = []

            # Use __annotations__ to get field types
            annotations = getattr(model_class, "__annotations__", {})
            for field_name, field_type in annotations.items():
                # Convert the annotation to a string
                if hasattr(field_type, "__name__"):
                    type_str = field_type.__name__
                else:
                    type_str = str(field_type).replace("typing.", "")

                # Add the field
                fields.append(f"    {field_name}: {type_str}")

            # Write the model class
            f.write(f"class {model_name}(BaseModel):\n")
            f.write("\n".join(fields) + "\n\n")

    return temp_dir


def clear_registries():
    """Clear both the model registry and Django app registry."""
    clear_model_registry()
    if "tests" in apps.all_models:
        apps.all_models["tests"].clear()
    if "pydantic2django" in apps.all_models:
        apps.all_models["pydantic2django"].clear()


def test_discover_models_from_package():
    """Test discovering models from a single package."""
    # Clear both registries before starting
    clear_registries()

    # Define some models
    class User(BaseModel):
        name: str
        email: str

    class Product(BaseModel):
        name: str
        price: float

    # Create a temporary package
    temp_dir = create_temp_package("test_package", {"User": User, "Product": Product})

    try:
        # Add the temp directory to sys.path
        sys.path.insert(0, str(temp_dir))

        # Create a discovery instance
        discovery = ModelDiscovery()

        # Discover models - use just the package name, not the module
        discovery.discover_models(["test_package"], app_label="tests")

        # Verify models were discovered
        discovered_models = discovery.get_discovered_models()
        assert "DjangoUser" in discovered_models
        assert "DjangoProduct" in discovered_models

        # Register models
        django_models = discovery.setup_dynamic_models(app_label="tests")

        # Verify models were registered
        assert "DjangoUser" in django_models
        assert "DjangoProduct" in django_models
    finally:
        # Clean up
        if temp_dir in sys.path:
            sys.path.remove(str(temp_dir))

        # Remove the temporary directory
        import shutil

        shutil.rmtree(temp_dir)


def test_discover_models_from_multiple_packages():
    """Test discovering models from multiple packages."""
    # Clear both registries before starting
    clear_registries()

    # Define models for the first package
    class User(BaseModel):
        name: str
        email: str

    # Define models for the second package
    class Product(BaseModel):
        name: str
        price: float

    # Create temporary packages
    temp_dir1 = create_temp_package("package1", {"User": User})
    temp_dir2 = create_temp_package("package2", {"Product": Product})

    try:
        # Add the temp directories to sys.path
        sys.path.insert(0, str(temp_dir1))
        sys.path.insert(0, str(temp_dir2))

        # Create a discovery instance
        discovery = ModelDiscovery()

        # Discover models from both packages - use just the package names
        discovery.discover_models(["package1", "package2"], app_label="tests")

        # Verify models were discovered
        discovered_models = discovery.get_discovered_models()
        assert "DjangoUser" in discovered_models
        assert "DjangoProduct" in discovered_models

        # Register models
        django_models = discovery.setup_dynamic_models(app_label="tests")

        # Verify models were registered
        assert "DjangoUser" in django_models
        assert "DjangoProduct" in django_models
    finally:
        # Clean up
        if str(temp_dir1) in sys.path:
            sys.path.remove(str(temp_dir1))
        if str(temp_dir2) in sys.path:
            sys.path.remove(str(temp_dir2))

        # Remove the temporary directories
        import shutil

        shutil.rmtree(temp_dir1)
        shutil.rmtree(temp_dir2)


def test_discover_models_with_dependencies_across_packages():
    """Test discovering models with dependencies across packages."""
    # Clear both registries before starting
    clear_registries()

    # Define models for the first package
    class Address(BaseModel):
        street: str
        city: str

    # Define models for the second package that depend on the first package
    # We'll create this dynamically in the package

    # Create temporary packages
    temp_dir1 = create_temp_package("pkg_base", {"Address": Address})
    temp_dir2 = create_temp_package("pkg_dependent", {})

    try:
        # Add the temp directories to sys.path
        sys.path.insert(0, str(temp_dir1))
        sys.path.insert(0, str(temp_dir2))

        # Create a User model in the second package that depends on Address
        with open(temp_dir2 / "pkg_dependent" / "models.py", "w") as f:
            f.write("from pydantic import BaseModel\n")
            f.write("from pkg_base.models import Address\n\n")
            f.write("class User(BaseModel):\n")
            f.write("    name: str\n")
            f.write("    address: Address\n")

        # Create a discovery instance
        discovery = ModelDiscovery()

        # Discover models from both packages - use just the package names
        discovery.discover_models(["pkg_base", "pkg_dependent"], app_label="tests")

        # Verify models were discovered
        discovered_models = discovery.get_discovered_models()
        assert "DjangoAddress" in discovered_models
        assert "DjangoUser" in discovered_models

        # Register models
        django_models = discovery.setup_dynamic_models(app_label="tests")

        # Verify models were registered
        assert "DjangoAddress" in django_models
        assert "DjangoUser" in django_models

        # Analyze dependencies after models are registered
        discovery.analyze_dependencies(app_label="tests")

        # Get dependencies
        dependencies = discovery.dependencies

        # Verify dependencies - note that dependencies include the app label
        assert f"tests.DjangoAddress" in dependencies["DjangoUser"]

        # Get registration order
        registration_order = discovery.get_registration_order()

        # Verify Address comes before User
        address_index = registration_order.index("tests.DjangoAddress")
        user_index = registration_order.index("tests.DjangoUser")
        assert address_index < user_index

        # Verify relationship
        user_model = django_models["DjangoUser"]
        address_field = user_model._meta.get_field("address")
        assert address_field.related_model == django_models["DjangoAddress"]
    finally:
        # Clean up
        if str(temp_dir1) in sys.path:
            sys.path.remove(str(temp_dir1))
        if str(temp_dir2) in sys.path:
            sys.path.remove(str(temp_dir2))

        # Remove the temporary directories
        import shutil

        shutil.rmtree(temp_dir1)
        shutil.rmtree(temp_dir2)


def test_discover_models_with_nested_packages():
    """Test discovering models from nested packages."""
    # Clear both registries before starting
    clear_registries()

    # Create a temporary directory structure
    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Create the package structure
        parent_pkg = temp_dir / "parent_pkg"
        parent_pkg.mkdir()

        # Create __init__.py files
        with open(parent_pkg / "__init__.py", "w") as f:
            f.write("# Parent package\n")

        # Create subpackages
        sub_pkg1 = parent_pkg / "sub1"
        sub_pkg1.mkdir()
        with open(sub_pkg1 / "__init__.py", "w") as f:
            f.write("# Subpackage 1\n")

        sub_pkg2 = parent_pkg / "sub2"
        sub_pkg2.mkdir()
        with open(sub_pkg2 / "__init__.py", "w") as f:
            f.write("# Subpackage 2\n")

        # Create models in subpackages
        with open(sub_pkg1 / "models.py", "w") as f:
            f.write("from pydantic import BaseModel\n\n")
            f.write("class Model1(BaseModel):\n")
            f.write("    name: str\n")

        with open(sub_pkg2 / "models.py", "w") as f:
            f.write("from pydantic import BaseModel\n\n")
            f.write("class Model2(BaseModel):\n")
            f.write("    value: int\n")

        # Add the temp directory to sys.path
        sys.path.insert(0, str(temp_dir))

        # Create a discovery instance
        discovery = ModelDiscovery()

        # Discover models from both subpackages - use the parent package name
        discovery.discover_models(["parent_pkg"], app_label="tests")

        # Verify models were discovered
        discovered_models = discovery.get_discovered_models()
        assert "DjangoModel1" in discovered_models
        assert "DjangoModel2" in discovered_models

        # Register models
        django_models = discovery.setup_dynamic_models(app_label="tests")

        # Verify models were registered
        assert "DjangoModel1" in django_models
        assert "DjangoModel2" in django_models
    finally:
        # Clean up
        if str(temp_dir) in sys.path:
            sys.path.remove(str(temp_dir))

        # Remove the temporary directory
        import shutil

        shutil.rmtree(temp_dir)


def test_discover_models_with_wildcard_imports():
    """Test discovering models when using wildcard imports."""
    # Clear both registries before starting
    clear_registries()

    # Create a temporary directory structure
    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Create the package
        pkg = temp_dir / "wildcard_pkg"
        pkg.mkdir()

        # Create __init__.py file
        with open(pkg / "__init__.py", "w") as f:
            f.write("# Package with wildcard imports\n")

        # Create a base_models.py file
        with open(pkg / "base_models.py", "w") as f:
            f.write("from pydantic import BaseModel\n\n")
            f.write("class BaseUser(BaseModel):\n")
            f.write("    name: str\n")
            f.write("    email: str\n")

        # Create a models.py file that imports from base_models
        with open(pkg / "models.py", "w") as f:
            f.write("from pydantic import BaseModel\n")
            f.write("from .base_models import *  # Wildcard import\n\n")
            f.write("class Product(BaseModel):\n")
            f.write("    name: str\n")
            f.write("    price: float\n")
            f.write("\n")
            f.write("class AdminUser(BaseUser):  # Inherits from wildcard import\n")
            f.write("    is_admin: bool = True\n")

        # Add the temp directory to sys.path
        sys.path.insert(0, str(temp_dir))

        # Create a discovery instance
        discovery = ModelDiscovery()

        # Discover models - use just the package name
        discovery.discover_models(["wildcard_pkg"], app_label="tests")

        # Verify models were discovered
        discovered_models = discovery.get_discovered_models()
        assert "DjangoProduct" in discovered_models
        assert "DjangoAdminUser" in discovered_models
        assert "DjangoBaseUser" in discovered_models

        # Register models
        django_models = discovery.setup_dynamic_models(app_label="tests")

        # Verify models were registered
        assert "DjangoProduct" in django_models
        assert "DjangoAdminUser" in django_models
        assert "DjangoBaseUser" in django_models

        # Verify AdminUser has fields from BaseUser
        admin_model = django_models["DjangoAdminUser"]
        field_names = [
            field.name for field in admin_model._meta.fields if field.name != "id"
        ]
        assert "name" in field_names
        assert "email" in field_names
        assert "is_admin" in field_names
    finally:
        # Clean up
        if str(temp_dir) in sys.path:
            sys.path.remove(str(temp_dir))

        # Remove the temporary directory
        import shutil

        shutil.rmtree(temp_dir)
