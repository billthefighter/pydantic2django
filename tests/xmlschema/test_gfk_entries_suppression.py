from pathlib import Path

from pydantic2django.xmlschema.generator import XmlSchemaDjangoModelGenerator


def test_gfk_suppresses_entry_like_json_placeholders(tmp_path: Path):
    xsd_path = Path(__file__).parent / "fixtures" / "entry_wrapper_schema.xsd"
    out_file = tmp_path / "models.py"

    gen = XmlSchemaDjangoModelGenerator(
        schema_files=[str(xsd_path)],
        output_path=str(out_file),
        app_label="gfk_suppress",
        enable_gfk=True,
        gfk_policy="repeating_only",
        nested_relationship_strategy="fk",
    )
    gen.generate()

    code = out_file.read_text()

    # GenericEntry emitted and parent has GenericRelation
    assert "class GenericEntry(" in code
    assert "entries = GenericRelation('GenericEntry', related_query_name='entries')" in code

    # WrapperType should NOT include placeholder JSON fields for entry-like elements
    assert "class WrapperType(" in code
    # No 'entry' JSON field for the repeating Entry (routed via GFK)
    assert "entry = models.JSONField" not in code
    # No FooEntry/BarEntry placeholders (suppressed when GFK is enabled)
    assert "foo_entry = models.JSONField" not in code
    assert "bar_entry = models.JSONField" not in code
