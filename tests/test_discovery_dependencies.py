"""
Tests for the dependency resolution and topological sorting functionality of the discovery module.

These tests focus on the behavior of the discovery module when dealing with model dependencies,
ensuring that models are registered in the correct order.
"""
import sys
from typing import Dict, Set

import pytest
from pydantic import BaseModel

from pydantic2django.discovery import (
    ModelDiscovery,
    topological_sort,
    validate_model_references,
)


def test_topological_sort_simple():
    """Test topological sort with a simple dependency graph."""
    # Create a simple dependency graph
    dependencies = {
        "A": set(),
        "B": {"A"},
        "C": {"B"},
        "D": {"A", "C"},
    }

    # Sort the graph
    sorted_nodes = topological_sort(dependencies)

    # Verify the order
    # A must come before B, B before C, and A and C before D
    a_index = sorted_nodes.index("A")
    b_index = sorted_nodes.index("B")
    c_index = sorted_nodes.index("C")
    d_index = sorted_nodes.index("D")

    assert a_index < b_index
    assert b_index < c_index
    assert a_index < d_index
    assert c_index < d_index


def test_topological_sort_with_cycle():
    """Test topological sort with a cyclic dependency graph."""
    # Create a dependency graph with a cycle
    dependencies = {
        "A": {"C"},
        "B": {"A"},
        "C": {"B"},
    }

    # Sort the graph - should not raise an exception but log a warning
    sorted_nodes = topological_sort(dependencies)

    # Verify all nodes are in the result
    assert set(sorted_nodes) == {"A", "B", "C"}


def test_validate_model_references():
    """Test validation of model references."""
    # Create a set of models and dependencies
    models = {
        "A": type("A", (), {}),
        "B": type("B", (), {}),
        "C": type("C", (), {}),
    }

    dependencies = {
        "A": set(),
        "B": {"A"},
        "C": {"B", "D"},  # D is missing
    }

    # Validate references
    missing_refs = validate_model_references(models, dependencies)

    # Verify missing references
    assert "D" in missing_refs
    assert len(missing_refs) == 1


def test_model_dependency_resolution():
    """Test that model dependencies are correctly resolved."""
    # Create test modules with interdependent models
    test_module = type(sys)("test_module_deps")

    class ModelA(BaseModel):
        name: str

    class ModelB(BaseModel):
        a_ref: ModelA

    class ModelC(BaseModel):
        b_ref: ModelB
        a_ref: ModelA

    test_module.ModelA = ModelA
    test_module.ModelB = ModelB
    test_module.ModelC = ModelC

    # Add the module to sys.modules temporarily
    sys.modules["test_module_deps"] = test_module

    try:
        # Create a discovery instance
        discovery = ModelDiscovery()

        # Discover models
        discovery.discover_models(["test_module_deps"], app_label="test_deps")

        # Get the registration order
        registration_order = discovery.get_registration_order()

        # Verify the order
        a_index = registration_order.index("ModelA")
        b_index = registration_order.index("ModelB")
        c_index = registration_order.index("ModelC")

        # ModelA should come before ModelB and ModelC
        assert a_index < b_index
        assert a_index < c_index

        # ModelB should come before ModelC
        assert b_index < c_index

        # Register models and verify they were created
        django_models = discovery.setup_dynamic_models(app_label="test_deps")
        assert "ModelA" in django_models
        assert "ModelB" in django_models
        assert "ModelC" in django_models
    finally:
        # Clean up
        del sys.modules["test_module_deps"]


def test_circular_dependency_handling():
    """Test that circular dependencies are handled gracefully."""
    # Create test modules with circular dependencies
    test_module = type(sys)("test_module_circular")

    class ModelX(BaseModel):
        name: str
        # Will be set after ModelY is defined
        y_ref: "ModelY" = None

    class ModelY(BaseModel):
        name: str
        x_ref: ModelX

    # Set the forward reference
    ModelX.model_fields["y_ref"].annotation = ModelY

    test_module.ModelX = ModelX
    test_module.ModelY = ModelY

    # Add the module to sys.modules temporarily
    sys.modules["test_module_circular"] = test_module

    try:
        # Create a discovery instance
        discovery = ModelDiscovery()

        # Discover models
        discovery.discover_models(["test_module_circular"], app_label="test_circular")

        # Analyze dependencies
        discovery.analyze_dependencies(app_label="test_circular")

        # Get dependencies
        dependencies = discovery.dependencies

        # Verify circular dependency is detected
        assert "ModelY" in dependencies["ModelX"]
        assert "ModelX" in dependencies["ModelY"]

        # Get registration order - should not raise an exception
        registration_order = discovery.get_registration_order()

        # Verify both models are in the registration order
        assert "ModelX" in registration_order
        assert "ModelY" in registration_order

        # Register models and verify they were created
        django_models = discovery.setup_dynamic_models(app_label="test_circular")
        assert "ModelX" in django_models
        assert "ModelY" in django_models
    finally:
        # Clean up
        del sys.modules["test_module_circular"]


def test_self_referential_model():
    """Test that self-referential models are handled correctly."""
    # Create a test module with a self-referential model
    test_module = type(sys)("test_module_self_ref")

    class TreeNode(BaseModel):
        name: str
        parent: "TreeNode" = None
        children: list["TreeNode"] = []

    # Set the forward reference
    TreeNode.model_fields["parent"].annotation = TreeNode
    TreeNode.model_fields["children"].annotation = list[TreeNode]

    test_module.TreeNode = TreeNode

    # Add the module to sys.modules temporarily
    sys.modules["test_module_self_ref"] = test_module

    try:
        # Create a discovery instance
        discovery = ModelDiscovery()

        # Discover models
        discovery.discover_models(["test_module_self_ref"], app_label="test_self_ref")

        # Analyze dependencies
        discovery.analyze_dependencies(app_label="test_self_ref")

        # Get dependencies
        dependencies = discovery.dependencies

        # Verify self-reference is detected
        assert (
            "TreeNode" in dependencies["TreeNode"] or "self" in dependencies["TreeNode"]
        )

        # Register models and verify they were created
        django_models = discovery.setup_dynamic_models(app_label="test_self_ref")
        assert "TreeNode" in django_models

        # Verify the model has the correct fields
        tree_node_model = django_models["TreeNode"]
        assert "parent" in [field.name for field in tree_node_model._meta.fields]
        assert "children" in [
            field.name for field in tree_node_model._meta.many_to_many
        ]
    finally:
        # Clean up
        del sys.modules["test_module_self_ref"]
