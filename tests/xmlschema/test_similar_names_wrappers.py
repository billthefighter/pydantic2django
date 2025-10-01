"""
Similar names on wrappers and attributes: ensure related_name uniqueness and normalization.
"""

from pathlib import Path
import re
import pytest


def _xsd() -> Path:
    return Path(__file__).parent / "fixtures" / "similar_names_wrappers_schema.xsd"


@pytest.fixture(scope="module")
def code(make_generated_code) -> str:
    return make_generated_code(
        schema_paths=_xsd(),
        app_label="similar_names_app",
        nested_relationship_strategy="fk",
        list_relationship_style="child_fk",
    )


def test_models_present(code: str):
    assert "class ContainerType(" in code
    assert "class DataWrapperType(" in code


def test_related_name_uniqueness(code: str):
    body = re.search(r"class\s+DataWrapperType\(.*?\):([\s\S]*?)\n\s*class\s+", code + "\nclass End:\n    pass\n").group(1)
    fks = [l for l in body.splitlines() if "ForeignKey(" in l]
    # Two FKs back to ContainerType with distinct related_names
    assert len(fks) == 2
    assert all("to='similar_names_app.ContainerType'" in l for l in fks)
    names = {re.search(r"related_name='(\w+)'", l).group(1) for l in fks if "related_name=" in l}
    assert len(names) == len(fks)
