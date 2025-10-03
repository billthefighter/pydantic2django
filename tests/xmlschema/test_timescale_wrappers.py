"""
Timescale roles with wrapper children: ensure child FKs remain correct and presence of timestamp allows hypertable classification when enabled.
This test only asserts FK direction; classification is covered elsewhere.
"""

from pathlib import Path
import re
import pytest


def _xsd() -> Path:
    return Path(__file__).parent / "fixtures" / "timescale_wrappers_schema.xsd"


@pytest.fixture(scope="module")
def code(make_generated_code) -> str:
    return make_generated_code(
        schema_paths=_xsd(),
        app_label="ts_wrap_app",
        nested_relationship_strategy="fk",
        list_relationship_style="child_fk",
        enable_gfk=False,
    )


def test_models_present(code: str):
    assert "class ComponentStreamType(" in code
    assert "class SamplesType(" in code


def test_child_fk_direction(code: str):
    assert "to='ts_wrap_app.ComponentStreamType'" in code
