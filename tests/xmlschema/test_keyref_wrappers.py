"""
Key/keyref within a wrapper container: ensure keyref produces FK from BookType to AuthorType, and wrapper children still link correctly to parent if applicable.
"""

from pathlib import Path
import re
import pytest


def _xsd() -> Path:
    return Path(__file__).parent / "fixtures" / "keyref_wrappers_schema.xsd"


@pytest.fixture(scope="module")
def code(make_generated_code) -> str:
    return make_generated_code(
        schema_paths=_xsd(),
        app_label="keyref_wrap_app",
        nested_relationship_strategy="fk",
        list_relationship_style="child_fk",
    )


def test_models_present(code: str):
    for name in ["AuthorType", "BookType", "LibraryWrapperType", "ContainerType"]:
        assert f"class {name}(" in code


def test_keyref_fk_from_book_to_author(code: str):
    body = re.search(r"class\s+BookType\(.*?\):([\s\S]*?)\n\s*class\s+", code + "\nclass End:\n    pass\n").group(1)
    assert re.search(r"^\s*author_ref\s*=\s*models\.ForeignKey\(.*to='keyref_wrap_app.AuthorType'", body, re.MULTILINE)
