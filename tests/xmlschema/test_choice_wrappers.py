"""
Choice wrappers: ComponentStream has either Samples or Events.
Validate that whichever wrapper is present gets a child FK to the parent, and parent does not get FKs to wrappers.
"""

from pathlib import Path
import pytest


def _xsd() -> Path:
    return Path(__file__).parent / "fixtures" / "choice_wrappers_schema.xsd"


@pytest.fixture(scope="module")
def code(make_generated_code) -> str:
    return make_generated_code(
        schema_paths=_xsd(),
        app_label="choice_wrap_app",
        nested_relationship_strategy="fk",
        list_relationship_style="child_fk",
    )


def test_models_present(code: str):
    assert "class ComponentStreamType(" in code
    assert "class SamplesType(" in code
    assert "class EventsType(" in code


def test_child_fk_direction(code: str):
    # Each wrapper should have an FK back to ComponentStreamType
    assert "to='choice_wrap_app.ComponentStreamType'" in code
    # Parent should not carry FKs to wrappers
    assert "\n    samples = models.ForeignKey(" not in code
    assert "\n    events = models.ForeignKey(" not in code
