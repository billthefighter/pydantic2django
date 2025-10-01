"""
Reused wrapper type under multiple parents: ensure child FKs have distinct related_name
and point back to the correct parent model.
"""

from pathlib import Path
import re
import pytest


def _xsd() -> Path:
    return Path(__file__).parent / "fixtures" / "reused_wrapper_schema.xsd"


@pytest.fixture(scope="module")
def code(make_generated_code) -> str:
    return make_generated_code(
        schema_paths=_xsd(),
        app_label="reused_wrap_app",
        nested_relationship_strategy="fk",
        list_relationship_style="child_fk",
    )


def _model_body(code: str, model: str) -> str:
    m = re.search(rf"class\s+{re.escape(model)}\(.*?\):([\s\S]*?)\n\s*class\s+", code + "\nclass End:\n    pass\n")
    assert m, f"Model {model} missing"
    return m.group(1)


def test_models_present(code: str):
    for name in ["SamplesType", "ParentAType", "ParentBType"]:
        assert f"class {name}(" in code


def test_child_fk_to_correct_parent(code: str):
    body = _model_body(code, "SamplesType")
    # There should be two FKs back to two different parents with distinct related_names
    fk_lines = [l for l in body.splitlines() if "ForeignKey(" in l]
    assert len(fk_lines) == 2
    assert any("to='reused_wrap_app.ParentAType'" in l for l in fk_lines)
    assert any("to='reused_wrap_app.ParentBType'" in l for l in fk_lines)
    # related_name uniqueness
    assert len({re.search(r"related_name='(\w+)'", l).group(1) for l in fk_lines if "related_name=" in l}) == len(fk_lines)
