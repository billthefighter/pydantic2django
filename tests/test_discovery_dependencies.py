"""
Tests for the dependency resolution and topological sorting functionality of the discovery module.

These tests focus on the behavior of the discovery module when dealing with model dependencies,
ensuring that models are registered in the correct order and context-based dependencies are handled.
"""
import sys
from typing import Any, Dict, List, Optional, Set

import pytest
from django.db import models
from pydantic import BaseModel

from pydantic2django.discovery import (
    ModelDiscovery,
    normalize_model_reference,
    topological_sort,
    validate_model_references,
)


def test_normalize_model_reference():
    """Test model reference normalization."""
    # Test string references
    assert normalize_model_reference("Model") == "DjangoModel"
    assert normalize_model_reference("app.Model") == "DjangoModel"
    assert normalize_model_reference("DjangoModel") == "DjangoModel"

    # Test class references
    class TestModel(models.Model):
        pass

    assert normalize_model_reference(TestModel) == "DjangoTestModel"


def test_topological_sort_simple():
    """Test topological sort with a simple dependency graph."""
    # Create a simple dependency graph
    dependencies = {
        "DjangoA": set(),
        "DjangoB": {"DjangoA"},
        "DjangoC": {"DjangoB"},
        "DjangoD": {"DjangoA", "DjangoC"},
    }

    # Sort the graph
    sorted_nodes = topological_sort(dependencies)

    # Verify the order
    # A must come before B, B before C, and A and C before D
    a_index = sorted_nodes.index("DjangoA")
    b_index = sorted_nodes.index("DjangoB")
    c_index = sorted_nodes.index("DjangoC")
    d_index = sorted_nodes.index("DjangoD")

    assert a_index < b_index
    assert b_index < c_index
    assert a_index < d_index
    assert c_index < d_index


def test_topological_sort_with_cycle():
    """Test topological sort with a cyclic dependency graph."""
    # Create a dependency graph with a cycle
    dependencies = {
        "DjangoA": {"DjangoC"},
        "DjangoB": {"DjangoA"},
        "DjangoC": {"DjangoB"},
    }

    # Sort the graph - should not raise an exception but log a warning
    sorted_nodes = topological_sort(dependencies)

    # Verify all nodes are in the result
    assert set(sorted_nodes) == {"DjangoA", "DjangoB", "DjangoC"}


def test_validate_model_references():
    """Test validation of model references."""
    # Create a set of models and dependencies
    models = {
        "DjangoA": type("DjangoA", (), {}),
        "DjangoB": type("DjangoB", (), {}),
        "DjangoC": type("DjangoC", (), {}),
    }

    dependencies = {
        "DjangoA": set(),
        "DjangoB": {"DjangoA"},
        "DjangoC": {"DjangoB", "DjangoD"},  # D is missing
    }

    # Validate references
    errors = validate_model_references(models, dependencies)

    # Verify missing references
    assert len(errors) == 1
    assert "Model 'DjangoC' references non-existent model 'DjangoD'" in errors


def test_model_dependency_resolution():
    """Test that model dependencies are correctly resolved."""
    # Create test modules with interdependent models
    test_module = type(sys)("test_module_deps")

    class ModelA(BaseModel):
        name: str

    class ModelB(BaseModel):
        name: str
        a_ref: ModelA

    class ModelC(BaseModel):
        name: str
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

        # Analyze dependencies
        discovery.analyze_dependencies(app_label="test_deps")

        # Set up Django models
        discovery.setup_dynamic_models(app_label="test_deps")

        # Get the registration order
        registration_order = discovery.get_registration_order()

        # Verify the order
        a_index = registration_order.index("test_deps.DjangoModelA")
        b_index = registration_order.index("test_deps.DjangoModelB")
        c_index = registration_order.index("test_deps.DjangoModelC")

        # ModelA should come before ModelB and ModelC
        assert a_index < b_index
        assert a_index < c_index

        # ModelB should come before ModelC
        assert b_index < c_index

        # Verify Django models were created
        django_models = discovery.get_django_models(app_label="test_deps")
        assert "DjangoModelA" in django_models
        assert "DjangoModelB" in django_models
        assert "DjangoModelC" in django_models

    finally:
        # Clean up
        del sys.modules["test_module_deps"]


@pytest.fixture
def context_test_env():
    """Fixture that sets up the test environment for context-based dependencies."""
    test_module = type(sys)("test_module_context")

    class BaseConfig(BaseModel):
        """Base configuration class."""

        name: str

    class NonSerializableConfig(BaseConfig):
        """Non-serializable configuration."""

        value: str

    class ContextModel(BaseModel):
        """Model with non-serializable fields."""

        name: str
        config: NonSerializableConfig

    class DependentModel(BaseModel):
        """Model that depends on a model with context fields."""

        name: str
        context_ref: ContextModel
        config_ref: Optional[NonSerializableConfig] = None

    test_module.BaseConfig = BaseConfig
    test_module.NonSerializableConfig = NonSerializableConfig
    test_module.ContextModel = ContextModel
    test_module.DependentModel = DependentModel

    sys.modules["test_module_context"] = test_module

    discovery = ModelDiscovery()
    discovery.discover_models(["test_module_context"], app_label="test_context")
    discovery.analyze_dependencies(app_label="test_context")
    discovery.setup_dynamic_models(app_label="test_context")

    yield {
        "discovery": discovery,
        "models": discovery.get_django_models(app_label="test_context"),
        "module": test_module,
    }

    del sys.modules["test_module_context"]


@pytest.mark.parametrize(
    "model_name,expected_fields",
    [
        (
            "DjangoContextModel",
            [
                ("name", models.CharField),
                ("config", models.TextField),
            ],
        ),
        (
            "DjangoDependentModel",
            [
                ("name", models.CharField),
                ("context_ref", models.TextField),
                ("config_ref", models.TextField),
            ],
        ),
    ],
)
def test_context_model_fields(context_test_env, model_name, expected_fields):
    """Test that model fields are created with correct types."""
    django_models = context_test_env["models"]
    model = django_models[model_name]

    for field_name, field_type in expected_fields:
        field = model._meta.get_field(field_name)
        assert isinstance(
            field, field_type
        ), f"Field {field_name} should be {field_type}"


@pytest.mark.parametrize(
    "field_name,expected_is_relationship",
    [
        ("config", True),
        ("context_ref", True),
        ("config_ref", True),
    ],
)
def test_relationship_field_flags(
    context_test_env, field_name, expected_is_relationship
):
    """Test that relationship fields are properly flagged."""
    django_models = context_test_env["models"]

    # Find which model contains the field
    for model_name, model in django_models.items():
        try:
            field = model._meta.get_field(field_name)
            assert (
                getattr(field, "is_relationship", False) == expected_is_relationship
            ), f"Field {field_name} in {model_name} should have is_relationship={expected_is_relationship}"
            break
        except models.FieldDoesNotExist:
            continue


def test_dependency_tracking(context_test_env):
    """Test that model dependencies are correctly tracked."""
    discovery = context_test_env["discovery"]
    deps = discovery.dependencies["DjangoDependentModel"]
    assert "DjangoContextModel" in deps, "DependentModel should depend on ContextModel"


def test_registration_order(context_test_env):
    """Test that models are registered in the correct order."""
    discovery = context_test_env["discovery"]
    registration_order = discovery.get_registration_order()

    # Both models should be in the registration order
    assert "test_context.DjangoContextModel" in registration_order
    assert "test_context.DjangoDependentModel" in registration_order

    # ContextModel should come before DependentModel
    context_index = registration_order.index("test_context.DjangoContextModel")
    dependent_index = registration_order.index("test_context.DjangoDependentModel")
    assert (
        context_index < dependent_index
    ), "ContextModel should be registered before DependentModel"


def test_abstract_base_class_handling():
    """Test that abstract base classes are handled correctly."""
    test_module = type(sys)("test_module_abc")

    from abc import ABC

    class BasePrompt(BaseModel, ABC):
        """Abstract base prompt class."""

        name: str

    class ConcretePrompt(BasePrompt):
        """Concrete implementation of base prompt."""

        content: str

    class PromptUser(BaseModel):
        """Model that uses a prompt."""

        name: str
        prompt: BasePrompt

    test_module.BasePrompt = BasePrompt
    test_module.ConcretePrompt = ConcretePrompt
    test_module.PromptUser = PromptUser

    sys.modules["test_module_abc"] = test_module

    try:
        discovery = ModelDiscovery()
        discovery.discover_models(["test_module_abc"], app_label="test_abc")

        # Verify that abstract base class is not discovered
        assert "DjangoBasePrompt" not in discovery.discovered_models
        assert "DjangoConcretePrompt" in discovery.discovered_models
        assert "DjangoPromptUser" in discovery.discovered_models

        # Analyze dependencies
        discovery.analyze_dependencies(app_label="test_abc")

        # Set up Django models
        discovery.setup_dynamic_models(app_label="test_abc")

        # Verify that the prompt field is handled as a context field
        django_models = discovery.get_django_models(app_label="test_abc")
        prompt_user = django_models["DjangoPromptUser"]
        prompt_field = prompt_user._meta.get_field("prompt")
        assert isinstance(prompt_field, models.TextField)
        assert getattr(prompt_field, "is_relationship", False)

        # Verify that registration order can be determined
        registration_order = discovery.get_registration_order()
        assert "test_abc.DjangoConcretePrompt" in registration_order
        assert "test_abc.DjangoPromptUser" in registration_order

    finally:
        del sys.modules["test_module_abc"]


def test_circular_dependency_handling():
    """Test that circular dependencies are handled gracefully."""
    test_module = type(sys)("test_module_circular")

    class ModelX(BaseModel):
        name: str
        y_ref: Optional["ModelY"] = None

    class ModelY(BaseModel):
        name: str
        x_ref: Optional[ModelX] = None

    ModelX.model_fields["y_ref"].annotation = Optional[ModelY]
    test_module.ModelX = ModelX
    test_module.ModelY = ModelY

    sys.modules["test_module_circular"] = test_module

    try:
        discovery = ModelDiscovery()
        discovery.discover_models(["test_module_circular"], app_label="test_circular")

        # Analyze dependencies
        discovery.analyze_dependencies(app_label="test_circular")

        # Set up Django models
        discovery.setup_dynamic_models(app_label="test_circular")

        # Verify circular dependency is detected
        deps = discovery.dependencies
        assert "DjangoModelY" in deps["DjangoModelX"]
        assert "DjangoModelX" in deps["DjangoModelY"]

        # Get registration order - should not raise an exception
        registration_order = discovery.get_registration_order()
        assert "test_circular.DjangoModelX" in registration_order
        assert "test_circular.DjangoModelY" in registration_order

        # Register models and verify they were created
        django_models = discovery.get_django_models(app_label="test_circular")
        assert "DjangoModelX" in django_models
        assert "DjangoModelY" in django_models

    finally:
        del sys.modules["test_module_circular"]


def test_self_referential_model():
    """Test that self-referential models are handled correctly."""
    test_module = type(sys)("test_module_self_ref")

    class TreeNode(BaseModel):
        name: str
        parent: Optional["TreeNode"] = None
        children: List["TreeNode"] = []

    TreeNode.model_fields["parent"].annotation = Optional[TreeNode]
    TreeNode.model_fields["children"].annotation = List[TreeNode]

    test_module.TreeNode = TreeNode
    sys.modules["test_module_self_ref"] = test_module

    try:
        discovery = ModelDiscovery()
        discovery.discover_models(["test_module_self_ref"], app_label="test_self_ref")

        # Analyze dependencies
        discovery.analyze_dependencies(app_label="test_self_ref")

        # Set up Django models
        discovery.setup_dynamic_models(app_label="test_self_ref")

        # Verify self-reference is detected
        deps = discovery.dependencies
        assert "DjangoTreeNode" in deps["DjangoTreeNode"]

        # Register models and verify they were created
        django_models = discovery.get_django_models(app_label="test_self_ref")
        assert "DjangoTreeNode" in django_models

        # Verify the model has the correct fields
        tree_node_model = django_models["DjangoTreeNode"]
        fields = {f.name: f for f in tree_node_model._meta.get_fields()}

        assert "parent" in fields
        assert isinstance(fields["parent"], models.ForeignKey)
        assert "children" in fields
        assert isinstance(fields["children"], models.ManyToManyField)

    finally:
        del sys.modules["test_module_self_ref"]
