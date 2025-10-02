from pathlib import Path

from pydantic2django.xmlschema.generator import XmlSchemaDjangoModelGenerator


def test_generic_entry_emitted_and_parent_has_generic_relation(tmp_path: Path):
    xsd_path = Path(__file__).parent / "fixtures" / "nested_schema.xsd"
    out_file = tmp_path / "models.py"

    gen = XmlSchemaDjangoModelGenerator(
        schema_files=[str(xsd_path)],
        output_path=str(out_file),
        app_label="gfk_app",
        nested_relationship_strategy="fk",
        list_relationship_style="child_fk",
        enable_gfk=True,
        gfk_policy="repeating_only",
    )
    gen.generate()

    code = out_file.read_text()

    # GenericEntry model emitted
    assert "class GenericEntry(" in code

    # ParentType should expose GenericRelation 'entries'
    assert "class ParentType(" in code
    assert "entries = GenericRelation('GenericEntry', related_query_name='entries')" in code


def test_generic_entry_with_overrides_arg(tmp_path: Path):
    xsd_path = Path(__file__).parent / "fixtures" / "nested_schema.xsd"
    out_file = tmp_path / "models.py"

    # Provide overrides to ensure constructor path is exercised
    gen = XmlSchemaDjangoModelGenerator(
        schema_files=[str(xsd_path)],
        output_path=str(out_file),
        app_label="gfk_app",
        nested_relationship_strategy="fk",
        list_relationship_style="child_fk",
        enable_gfk=True,
        gfk_policy="repeating_only",
        gfk_overrides={"items": True},
    )
    gen.generate()

    code = out_file.read_text()
    assert "class GenericEntry(" in code

def test_generic_entry_typed_columns_and_indexes(tmp_path: Path):
    """When gfk_value_mode='typed_columns', GenericEntry should include typed columns and indexes."""
    xsd_path = Path(__file__).parent / "fixtures" / "nested_schema.xsd"
    out_file = tmp_path / "models.py"

    gen = XmlSchemaDjangoModelGenerator(
        schema_files=[str(xsd_path)],
        output_path=str(out_file),
        app_label="gfk_app",
        nested_relationship_strategy="fk",
        list_relationship_style="child_fk",
        enable_gfk=True,
        gfk_policy="repeating_only",
        gfk_value_mode="typed_columns",
    )
    gen.generate()

    code = out_file.read_text()

    assert "class GenericEntry(" in code
    assert "text_value = models.TextField(" in code
    assert "num_value = models.DecimalField(" in code
    assert "time_value = models.DateTimeField(" in code
    # Indexes
    assert "models.Index(fields=['content_type', 'object_id'])" in code
    assert "models.Index(fields=['element_qname'])" in code
    assert "models.Index(fields=['type_qname'])" in code
    assert "models.Index(fields=['time_value'])" in code
    assert "models.Index(fields=['content_type', 'object_id', '-time_value'])" in code


def test_gfk_suppresses_json_placeholder_for_repeating_nested(tmp_path: Path):
    """When GFK is enabled for repeating nested, do not emit JSON placeholder on parent."""
    xsd_path = Path(__file__).parent / "fixtures" / "nested_schema.xsd"
    out_file = tmp_path / "models.py"

    gen = XmlSchemaDjangoModelGenerator(
        schema_files=[str(xsd_path)],
        output_path=str(out_file),
        app_label="gfk_app",
        nested_relationship_strategy="fk",
        list_relationship_style="child_fk",
        enable_gfk=True,
        gfk_policy="repeating_only",
    )
    gen.generate()

    code = out_file.read_text()
    # Parent has GenericRelation
    assert "entries = GenericRelation('GenericEntry', related_query_name='entries')" in code
    # Ensure there is no JSONField placeholder for the repeating element on the parent
    assert "items = models.JSONField" not in code
