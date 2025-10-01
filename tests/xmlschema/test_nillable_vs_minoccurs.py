"""
Nillable vs minOccurs=0 on wrappers: ensure FK direction is correct and nullability
differences do not cause parent-side FKs to be emitted.
"""

from pathlib import Path
import re
import pytest


def _xsd() -> Path:
    return Path(__file__).parent / "fixtures" / "nillable_vs_minoccurs_schema.xsd"


@pytest.fixture(scope="module")
def code(make_generated_code) -> str:
    return make_generated_code(
        schema_paths=_xsd(),
        app_label="nilmin_app",
        nested_relationship_strategy="fk",
        list_relationship_style="child_fk",
    )


def test_models_present(code: str):
    assert "class ContainerType(" in code
    assert "class WrapperAType(" in code
    assert "class WrapperBType(" in code


def test_no_parent_side_fks(code: str):
    body = re.search(r"class\s+ContainerType\(.*?\):([\s\S]*?)\n\s*class\s+", code + "\nclass End:\n    pass\n").group(1)
    assert "models.ForeignKey(" not in body
