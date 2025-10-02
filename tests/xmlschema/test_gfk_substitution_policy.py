from pathlib import Path

from pydantic2django.xmlschema.generator import XmlSchemaDjangoModelGenerator


def test_gfk_substitution_only_collapses_members(tmp_path: Path):
    xsd_path = Path(__file__).parent / "fixtures" / "subst_wrapper_schema.xsd"
    out_file = tmp_path / "models.py"

    gen = XmlSchemaDjangoModelGenerator(
        schema_files=[str(xsd_path)],
        output_path=str(out_file),
        app_label="gfk_subst",
        enable_gfk=True,
        gfk_policy="substitution_only",
        nested_relationship_strategy="fk",
        list_relationship_style="child_fk",
    )
    gen.generate()

    code = out_file.read_text()

    # GenericEntry emitted and wrapper has GenericRelation
    assert "class GenericEntry(" in code
    assert "class SamplesType(" in code
    assert "entries = GenericRelation('GenericEntry', related_query_name='entries')" in code

    # Ensure concrete member types are not generated when routed via GFK
    assert "class Angle(" not in code
    assert "class Position(" not in code

    # Suppress JSON placeholders under wrapper
    assert "observation = models.JSONField" not in code
