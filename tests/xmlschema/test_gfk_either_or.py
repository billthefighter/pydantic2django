from pathlib import Path
import pytest

from pydantic2django.xmlschema.generator import XmlSchemaDjangoModelGenerator


def test_either_or_no_dual_emission_entry_wrapper(tmp_path: Path):
    """Wrapper with entry-like children must emit either GenericEntry or JSON placeholders, not both."""
    xsd_path = Path(__file__).parent / "fixtures" / "entry_wrapper_schema.xsd"
    out_file = tmp_path / "models.py"

    gen = XmlSchemaDjangoModelGenerator(
        schema_files=[str(xsd_path)],
        output_path=str(out_file),
        app_label="either_or",
        enable_gfk=True,
        gfk_policy="all_nested",
        nested_relationship_strategy="fk",
        list_relationship_style="child_fk",
    )
    gen.generate()

    code = out_file.read_text()

    # GenericEntry emitted and parent has GenericRelation
    assert "class GenericEntry(" in code
    assert "entries = GenericRelation('GenericEntry', related_query_name='entries')" in code

    # Ensure no JSON placeholders under the wrapper when entries are present
    assert "class WrapperType(" in code
    assert "entry = models.JSONField" not in code
    assert "foo_entry = models.JSONField" not in code
    assert "bar_entry = models.JSONField" not in code


def test_conflict_flags_fail_generation(tmp_path: Path):
    """When GFK owners are detected but JSON relationship strategies are requested, generation should fail."""
    xsd_path = Path(__file__).parent / "fixtures" / "entry_wrapper_schema.xsd"
    out_file = tmp_path / "models.py"

    gen = XmlSchemaDjangoModelGenerator(
        schema_files=[str(xsd_path)],
        output_path=str(out_file),
        app_label="either_or_conflict",
        enable_gfk=True,
        gfk_policy="all_nested",
        nested_relationship_strategy="json",  # Conflicts with GFK owners
    )

    with pytest.raises(ValueError):
        gen.generate()
