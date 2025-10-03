from pathlib import Path

from pydantic2django.xmlschema.generator import XmlSchemaDjangoModelGenerator
from pydantic2django.xmlschema.ingestor import XmlInstanceIngestor


def test_gfk_ingestion_skips_concrete_children_entry_wrapper(tmp_path: Path):
    xsd_path = Path(__file__).parent / "fixtures" / "entry_wrapper_schema.xsd"
    out_file = tmp_path / "models.py"

    gen = XmlSchemaDjangoModelGenerator(
        schema_files=[str(xsd_path)],
        output_path=str(out_file),
        app_label="gfk_ingest_app",
        enable_gfk=True,
        gfk_policy="all_nested",
        nested_relationship_strategy="fk",
        list_relationship_style="child_fk",
    )
    gen.generate()

    # Minimal XML instance aligned to the schema
    xml = (
        "<tns:RootType xmlns:tns=\"http://example.com/entries\">"
        "  <tns:Wrapper>"
        "    <tns:Entry><tns:value>10</tns:value></tns:Entry>"
        "    <tns:Entry><tns:value>11</tns:value></tns:Entry>"
        "    <tns:FooEntry><tns:value>X</tns:value></tns:FooEntry>"
        "  </tns:Wrapper>"
        "</tns:RootType>"
    )

    ing = XmlInstanceIngestor(schema_files=[str(xsd_path)], app_label="gfk_ingest_app", dynamic_model_fallback=True)
    root = ing.ingest_from_string(xml, save=False)

    # Owner containers should be created
    assert root is not None
    created_names = [obj.__class__.__name__ for obj in ing.created_instances]
    assert "WrapperType" in created_names or any("WrapperType" in n for n in created_names)

    # Concrete child entry types should NOT be instantiated when GFK is enabled
    assert all("ChildEntryType" not in n for n in created_names)
