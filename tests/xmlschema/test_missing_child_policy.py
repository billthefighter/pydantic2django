"""
Tests for finalize_relationships behavior when child/leaf targets are not generated.

Expect: no FK injection attempts to missing children; logs indicate skip; generation completes.
"""

from pathlib import Path
import pytest


def _xsd() -> Path:
    # Reuse namespaces wrappers where wrappers are generated but no separate leaf is present
    base = Path(__file__).parent / "fixtures"
    return base / "ns_wrappers_parent.xsd"


@pytest.fixture(scope="module")
def code(make_generated_code) -> str:
    # Only pass parent; omit child schema file to force missing child carriers
    return make_generated_code(
        schema_paths=_xsd(),
        app_label="ns_missing_app",
        nested_relationship_strategy="fk",
        list_relationship_style="child_fk",
    )


def test_generation_succeeds_without_missing_child_fk(code: str):
    assert "class ComponentStreamType(" in code
    # No FK to child should be present; JSON fallback likely
    assert "ForeignKey(" not in code or "to='ns_missing_app.SamplesType'" not in code
