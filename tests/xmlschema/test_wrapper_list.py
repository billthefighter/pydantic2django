"""
Wrapper list pattern tests: ComponentStream → SamplesList (wrapper) → many Sample.
Expect child FKs from SampleType to ComponentStreamType via related_name 'sampletype_set' or 'sampleslist' indirection.
"""

from pathlib import Path
import re
import pytest


def _xsd() -> Path:
    return Path(__file__).parent / "fixtures" / "wrapper_list_schema.xsd"


@pytest.fixture(scope="module")
def code(make_generated_code) -> str:
    return make_generated_code(
        schema_paths=_xsd(),
        app_label="wrapper_list_app",
        nested_relationship_strategy="fk",
        list_relationship_style="child_fk",
    )


def _has_field(code: str, model: str, field: str, type_name: str, contains: list[str]) -> None:
    pattern = re.compile(rf"class\s+{re.escape(model)}\(.*?\):([\s\S]*?)\n\s*class\s+", re.MULTILINE)
    m = pattern.search(code + "\nclass End:\n    pass\n")
    assert m, f"Model {model} missing"
    body = m.group(1)
    fpat = re.compile(rf"^\s*{re.escape(field)}\s*=\s*models\.{re.escape(type_name)}\((.*?)\)", re.MULTILINE)
    fm = fpat.search(body)
    assert fm, f"Field {field} not found in {model}"
    kwargs = fm.group(1)
    for c in contains:
        assert c in kwargs, f"Expected '{c}' in {field} kwargs; got {kwargs}"


def test_models_present(code: str):
    assert "class ComponentStreamType(" in code
    assert "class SamplesListType(" in code
    assert "class SampleType(" in code


def test_child_fk_from_sample_to_componentstream(code: str):
    # At leaf level, SampleType should have FK to ComponentStreamType
    _has_field(
        code,
        model="SampleType",
        field="componentstreamtype",
        type_name="ForeignKey",
        contains=["to='wrapper_list_app.ComponentStreamType'", "on_delete=models.CASCADE"],
    )

    # Parent should not have FK to SamplesList or Sample
    assert "\n    sampleslist = models.ForeignKey(" not in code
    assert "\n    sample = models.ForeignKey(" not in code
