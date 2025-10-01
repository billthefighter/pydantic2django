"""
Namespace/imported wrappers: parent in one namespace imports wrapper types from another.
Ensure child FKs back to parent even across namespaces.
"""

from pathlib import Path
import re
import pytest


def _xsds() -> list[Path]:
    base = Path(__file__).parent / "fixtures"
    return [base / "ns_wrappers_parent.xsd", base / "ns_wrappers_child.xsd"]


@pytest.fixture(scope="module")
def code(make_generated_code) -> str:
    return make_generated_code(
        schema_paths=_xsds(),
        app_label="ns_wrap_app",
        nested_relationship_strategy="fk",
        list_relationship_style="child_fk",
    )


def test_models_present(code: str):
    assert "class ComponentStreamType(" in code
    assert "class SamplesType(" in code
    assert "class EventsType(" in code


def test_child_fk_across_namespace(code: str):
    # Leaf wrappers should still FK to ComponentStreamType
    assert "to='ns_wrap_app.ComponentStreamType'" in code
