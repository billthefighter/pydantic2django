"""
Unit tests for wrapper-container schema patterns.

Validates that wrapper children (Samples/Events) relate back to the parent
ComponentStream via child foreign keys (child_fk), and that the parent does
not hold FKs to the wrappers.
"""

from pathlib import Path
import re
import pytest


def _fixture_xsd() -> Path:
    return Path(__file__).parent / "fixtures" / "wrapper_container_schema.xsd"


@pytest.fixture(scope="module")
def generated_code(make_generated_code) -> str:
    """Generate models from the wrapper-container XSD once per module."""
    return make_generated_code(
        schema_paths=_fixture_xsd(),
        app_label="wrapper_app",
        nested_relationship_strategy="fk",
        list_relationship_style="child_fk",
    )


def _assert_field_definition(code: str, model_name: str, field_name: str, type_name: str, contains: list[str], not_contains: list[str] | None = None) -> None:
    pattern = re.compile(rf"class\s+{re.escape(model_name)}\(.*?\):([\s\S]*?)\n\s*class\s+", re.MULTILINE)
    match = pattern.search(code + "\nclass EndSentinel:\n    pass\n")
    assert match, f"Model {model_name} not found"
    body = match.group(1)
    field_pattern = re.compile(rf"^\s*{re.escape(field_name)}\s*=\s*models\.{re.escape(type_name)}\((.*?)\)", re.MULTILINE)
    m2 = field_pattern.search(body)
    assert m2, f"Field {field_name} of type {type_name} not found in {model_name}"
    kwargs_str = m2.group(1)
    for frag in contains:
        assert frag in kwargs_str, f"Expected '{frag}' in kwargs for {field_name} in {model_name}, got: {kwargs_str}"
    if not_contains:
        for frag in not_contains:
            assert frag not in kwargs_str, f"Did not expect '{frag}' in kwargs for {field_name} in {model_name}, got: {kwargs_str}"


def test_models_exist(generated_code: str):
    assert "class ComponentStreamType(" in generated_code
    assert "class SamplesType(" in generated_code
    assert "class EventsType(" in generated_code


def test_child_fk_direction(generated_code: str):
    """SamplesType and EventsType should have FKs pointing to ComponentStreamType.
    Parent should NOT have FKs pointing to wrapper children.
    """
    # Child-side FKs
    _assert_field_definition(
        generated_code,
        model_name="SamplesType",
        field_name="componentstreamtype",
        type_name="ForeignKey",
        contains=["to='wrapper_app.ComponentStreamType'", "on_delete=models.CASCADE", "related_name='samples'"],
    )
    _assert_field_definition(
        generated_code,
        model_name="EventsType",
        field_name="componentstreamtype",
        type_name="ForeignKey",
        contains=["to='wrapper_app.ComponentStreamType'", "on_delete=models.CASCADE", "related_name='events'"],
    )

    # Ensure parent does NOT have FKs to wrappers
    assert "\n    samples = models.ForeignKey(" not in generated_code
    assert "\n    events = models.ForeignKey(" not in generated_code
