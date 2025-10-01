"""
Deeply nested wrappers: Container → Wrapper → InnerWrapper → Leaf.
Expect the FK to be at LeafType back to ContainerType; intermediates should not carry parent-side FKs.
"""

from pathlib import Path
import re
import pytest


def _xsd() -> Path:
    return Path(__file__).parent / "fixtures" / "deep_nested_wrappers_schema.xsd"


@pytest.fixture(scope="module")
def code(make_generated_code) -> str:
    return make_generated_code(
        schema_paths=_xsd(),
        app_label="deep_wrap_app",
        nested_relationship_strategy="fk",
        list_relationship_style="child_fk",
    )


def _find_model_body(code: str, model: str) -> str:
    m = re.search(rf"class\s+{re.escape(model)}\(.*?\):([\s\S]*?)\n\s*class\s+", code + "\nclass End:\n    pass\n")
    assert m, f"Model {model} missing"
    return m.group(1)


def test_models_present(code: str):
    for name in ["ContainerType", "WrapperType", "InnerWrapperType", "LeafType"]:
        assert f"class {name}(" in code


def test_fk_placement_at_leaf(code: str):
    leaf_body = _find_model_body(code, "LeafType")
    assert re.search(r"^\s*containertype\s*=\s*models\.ForeignKey\(.*to='deep_wrap_app.ContainerType'", leaf_body, re.MULTILINE)

    # Intermediates should not have FKs to ContainerType
    for model in ["WrapperType", "InnerWrapperType", "ContainerType"]:
        body = _find_model_body(code, model)
        assert "ForeignKey(" not in body or "to='deep_wrap_app.LeafType'" not in body
