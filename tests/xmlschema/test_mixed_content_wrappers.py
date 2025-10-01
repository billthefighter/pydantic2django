"""
Mixed content wrapper: wrapper has text and an element. Ensure generation still places FK at child level appropriately and text is captured (likely as CharField/TextField) without breaking FK logic.
"""

from pathlib import Path
import re
import pytest


def _xsd() -> Path:
    return Path(__file__).parent / "fixtures" / "mixed_content_wrapper_schema.xsd"


@pytest.fixture(scope="module")
def code(make_generated_code) -> str:
    return make_generated_code(
        schema_paths=_xsd(),
        app_label="mixed_content_app",
        nested_relationship_strategy="fk",
        list_relationship_style="child_fk",
    )


def test_models_present(code: str):
    assert "class ContainerType(" in code
    assert "class MixedWrapperType(" in code


def test_no_parent_side_fk(code: str):
    # Parent should not have FK to Wrapper
    container_body = re.search(r"class\s+ContainerType\(.*?\):([\s\S]*?)\n\s*class\s+", code + "\nclass End:\n    pass\n").group(1)
    assert "models.ForeignKey(" not in container_body
