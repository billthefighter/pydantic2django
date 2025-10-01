"""
M2M wrapper variation: tags wrapper under Article should become ManyToManyField when configured.
"""

from pathlib import Path
import re
import pytest


def _xsd() -> Path:
    return Path(__file__).parent / "fixtures" / "m2m_wrapper_schema.xsd"


@pytest.fixture(scope="module")
def code(make_generated_code) -> str:
    return make_generated_code(
        schema_paths=_xsd(),
        app_label="m2m_wrap_app",
        nested_relationship_strategy="fk",
        list_relationship_style="m2m",
    )


def test_models_present(code: str):
    assert "class ArticleType(" in code
    assert "class TagType(" in code


def test_m2m_emitted(code: str):
    article_body = re.search(r"class\s+ArticleType\(.*?\):([\s\S]*?)\n\s*class\s+", code + "\nclass End:\n    pass\n").group(1)
    # Expect an M2M field from Article to TagType
    assert re.search(r"^\s*tags\s*=\s*models\.ManyToManyField\(.*to='m2m_wrap_app.TagType'", article_body, re.MULTILINE)
