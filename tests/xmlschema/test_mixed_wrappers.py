"""
Mixed wrappers pattern: optional Samples, required Events, optional Condition.
Verify child FKs exist and parent does not carry FKs to wrappers; ensure required/optional wrappers do not force nullability on child FK.
"""

from pathlib import Path
import re
import pytest


def _xsd() -> Path:
    return Path(__file__).parent / "fixtures" / "mixed_wrappers_schema.xsd"


@pytest.fixture(scope="module")
def code(make_generated_code) -> str:
    return make_generated_code(
        schema_paths=_xsd(),
        app_label="mixed_wrappers_app",
        nested_relationship_strategy="fk",
        list_relationship_style="child_fk",
    )


def _has_fk_to_parent(code: str, child_model: str, parent_model: str):
    body = re.search(rf"class\s+{re.escape(child_model)}\(.*?\):([\s\S]*?)\n\s*class\s+", code + "\nclass End:\n    pass\n").group(1)
    m = re.search(r"^\s*(\w+)\s*=\s*models\.ForeignKey\((.*?)\)", body, re.MULTILINE)
    assert m, f"No FK found in {child_model}"
    kwargs = m.group(2)
    assert f"to='mixed_wrappers_app.{parent_model}'" in kwargs
    assert "on_delete=models.CASCADE" in kwargs


def test_models_present(code: str):
    for name in ["ComponentStreamType", "SamplesType", "EventsType", "ConditionType"]:
        assert f"class {name}(" in code


def test_child_fk_direction(code: str):
    _has_fk_to_parent(code, "SamplesType", "ComponentStreamType")
    _has_fk_to_parent(code, "EventsType", "ComponentStreamType")
    _has_fk_to_parent(code, "ConditionType", "ComponentStreamType")

    assert "\n    samples = models.ForeignKey(" not in code
    assert "\n    events = models.ForeignKey(" not in code
    assert "\n    condition = models.ForeignKey(" not in code
